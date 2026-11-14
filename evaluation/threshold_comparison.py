"""Compare semantic cache similarity thresholds.

Uses:
- Real all-MiniLM-L6-v2 embeddings
- Actual semantic cache and LRU optimizer
- Fresh cache for every threshold
- MockLLM (no real Gemini API calls)

Ground truth is based on expected_semantic_group labels.
"""

import csv
import json
from pathlib import Path

from src.cache_optimizer import CacheOptimizer
from src.config import config
from src.embeddings import EmbeddingGenerator
from src.llm import MockLLM
from src.pipeline import SemanticCachingPipeline
from src.semantic_cache import SemanticCache
from src.similarity import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "datasets" / "evaluation_queries.json"
RESULTS_DIR = ROOT / "evaluation" / "results"

THRESHOLDS = [0.85, 0.82, 0.80, 0.75]
CACHE_CAPACITY = 3
CACHE_TTL_SECONDS = 3600


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Dataset must be a non-empty list.")

    return dataset


def calculate_metrics(rows):
    tp = sum(row["classification"] == "TP" for row in rows)
    fp = sum(row["classification"] == "FP" for row in rows)
    fn = sum(row["classification"] == "FN" for row in rows)
    tn = sum(row["classification"] == "TN" for row in rows)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    total = len(rows)

    return {
        "total_queries": total,
        "cache_hits": tp + fp,
        "llm_calls": fn + tn,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "hit_rate": round((tp + fp) / total, 4) if total else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "false_positive_rate": (
            round(fp / (fp + tn), 4) if fp + tn else 0.0
        ),
    }


def run_experiment(dataset, threshold, embedding_generator):
    """Run one independent threshold experiment."""

    base_cache = SemanticCache(
        embedding_generator=embedding_generator
    )

    cache = CacheOptimizer(
        cache=base_cache,
        max_size=CACHE_CAPACITY,
        ttl_seconds=CACHE_TTL_SECONDS,
    )

    # Use the configured app settings but override only the threshold.
    experiment_config = config
    original_threshold = experiment_config.similarity_threshold

    # Pipeline reads the threshold from its config.
    class ThresholdConfig:
        def __getattr__(self, name):
            if name == "similarity_threshold":
                return threshold
            return getattr(experiment_config, name)

    pipeline = SemanticCachingPipeline(
        app_config=ThresholdConfig(),
        cache=cache,
        llm=MockLLM(),
    )

    rows = []

    for index, item in enumerate(dataset, start=1):
        query = item["query"]
        expected_group = item["expected_semantic_group"]

        # Inspect cache before lookup. This establishes whether any
        # same-group entry could have been retrieved at this threshold.
        query_embedding = embedding_generator.encode(query)
        candidates = []

        for entry in cache.entries():
            if entry.embedding is None:
                continue

            score = cosine_similarity(
                query_embedding,
                entry.embedding,
            )
            candidates.append((score, entry))

        relevant_available = any(
            entry.metadata.get("expected_semantic_group") == expected_group
            and score >= threshold
            for score, entry in candidates
        )

        response = pipeline.process_query(query)

        if response.source == "error":
            raise RuntimeError(
                f"Pipeline error for query: {query}"
            )

        retrieved_query = ""
        retrieved_group = ""

        if response.source == "cache":
            # Match the actual response to its originating cache entry.
            matched_entry = next(
                (
                    entry for entry in cache.entries()
                    if entry.response == response.response_text
                    and entry.query in {
                        candidate.query for _, candidate in candidates
                    }
                ),
                None,
            )

            if matched_entry is None:
                raise RuntimeError(
                    "Could not identify the cache entry that supplied "
                    f"the response for: {query}"
                )

            retrieved_query = matched_entry.query
            retrieved_group = matched_entry.metadata.get(
                "expected_semantic_group", ""
            )

            classification = (
                "TP" if retrieved_group == expected_group else "FP"
            )

        else:
            classification = "FN" if relevant_available else "TN"

        rows.append({
            "id": item.get("id", index),
            "query": query,
            "expected_group": expected_group,
            "source": response.source,
            "retrieved_query": retrieved_query,
            "retrieved_group": retrieved_group,
            "similarity_score": (
                round(response.similarity_score, 4)
                if response.similarity_score is not None
                else ""
            ),
            "classification": classification,
        })

        # Attach ground-truth label to the newly cached response.
        if response.source == "llm":
            for entry in cache.entries():
                if entry.query == query:
                    entry.metadata["expected_semantic_group"] = (
                        expected_group
                    )
                    break

    return calculate_metrics(rows), rows


def main():
    print("=" * 72)
    print("             SIMILARITY THRESHOLD COMPARISON")
    print("=" * 72)

    dataset = load_dataset()

    print(f"Dataset queries : {len(dataset)}")
    print(f"Embedding model : {config.embedding_model_name}")
    print(f"Cache capacity  : {CACHE_CAPACITY}")
    print("Eviction policy : LRU")
    print("LLM mode        : MockLLM")
    print()

    embedding_generator = EmbeddingGenerator(
        config.embedding_model_name
    )

    summaries = []
    all_rows = []

    for threshold in THRESHOLDS:
        print(f"\nRunning threshold: {threshold:.2f}")

        metrics, rows = run_experiment(
            dataset,
            threshold,
            embedding_generator,
        )

        summaries.append({
            "threshold": threshold,
            **metrics,
        })

        for row in rows:
            all_rows.append({
                "threshold": threshold,
                **row,
            })

        print(
            f"Hits: {metrics['cache_hits']} | "
            f"TP: {metrics['true_positives']} | "
            f"FP: {metrics['false_positives']} | "
            f"FN: {metrics['false_negatives']} | "
            f"TN: {metrics['true_negatives']} | "
            f"F1: {metrics['f1_score']:.4f}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / "threshold_comparison.json"
    csv_path = RESULTS_DIR / "threshold_comparison.csv"

    output = {
        "experiment": {
            "dataset": str(DATASET_PATH.relative_to(ROOT)),
            "embedding_model": config.embedding_model_name,
            "thresholds": THRESHOLDS,
            "cache_capacity": CACHE_CAPACITY,
            "cache_ttl_seconds": CACHE_TTL_SECONDS,
            "eviction_policy": "LRU",
            "llm_mode": "MockLLM",
            "ground_truth": "expected_semantic_group labels",
            "note": (
                "Metrics measure semantic retrieval relevance, "
                "not factual correctness of generated answers."
            ),
        },
        "results": summaries,
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "threshold",
                "id",
                "query",
                "expected_group",
                "source",
                "retrieved_query",
                "retrieved_group",
                "similarity_score",
                "classification",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n" + "=" * 72)
    print("                    COMPARISON SUMMARY")
    print("=" * 72)

    print(
        f"{'Threshold':<12}{'Hits':<8}{'TP':<7}{'FP':<7}"
        f"{'FN':<7}{'TN':<7}{'Precision':<12}{'Recall':<10}{'F1':<10}"
    )

    for result in summaries:
        print(
            f"{result['threshold']:<12.2f}"
            f"{result['cache_hits']:<8}"
            f"{result['true_positives']:<7}"
            f"{result['false_positives']:<7}"
            f"{result['false_negatives']:<7}"
            f"{result['true_negatives']:<7}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<10.4f}"
            f"{result['f1_score']:<10.4f}"
        )

    print("\nSaved results:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
