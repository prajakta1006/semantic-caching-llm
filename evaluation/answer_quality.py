"""Evaluate semantic cache retrieval quality using labelled queries.

Uses the real embedding model, actual pipeline and CacheOptimizer.
MockLLM is used so no paid API calls are made.

Ground truth is based on expected_semantic_group labels.
This evaluates retrieval relevance, not factual answer correctness.
"""

import csv
import json
from collections import defaultdict, Counter
from pathlib import Path

from src.config import config
from src.embeddings import EmbeddingGenerator
from src.llm import MockLLM
from src.pipeline import SemanticCachingPipeline
from src.semantic_cache import SemanticCache
from src.cache_optimizer import CacheOptimizer
from src.similarity import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "datasets" / "evaluation_queries.json"
RESULTS_DIR = ROOT / "evaluation" / "results"
CSV_PATH = RESULTS_DIR / "answer_quality_results.csv"
JSON_PATH = RESULTS_DIR / "answer_quality_summary.json"

# Match the optimizer settings used by the main application.
CACHE_CAPACITY = 3
CACHE_TTL_SECONDS = 3600


def load_dataset():
    """Load and validate labelled evaluation queries."""
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list) or not data:
        raise ValueError("Evaluation dataset must be a non-empty JSON list.")

    for item in data:
        if not item.get("query") or not item.get("expected_semantic_group"):
            raise ValueError(
                "Every item must contain query and expected_semantic_group."
            )

    return data


def inspect_cache(cache, query, embedding_generator, threshold):
    """Inspect available entries without changing LRU access metadata.

    Returns:
        Best matching entry and its similarity score.
        Whether a relevant same-group entry meets the threshold.
    """
    query_embedding = embedding_generator.encode(query)
    candidates = []

    for entry in cache.entries():
        if entry.embedding is None:
            continue

        score = cosine_similarity(query_embedding, entry.embedding)

        candidates.append((score, entry))

    if not candidates:
        return None, None, False

    best_score, best_entry = max(candidates, key=lambda item: item[0])

    # The expected group is attached to each entry when it is first
    # inserted by the evaluator.
    expected_group = None

    return (
        best_entry,
        best_score,
        candidates,
    )


def calculate_metrics(rows):
    """Calculate binary retrieval relevance metrics."""
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

    total = tp + fp + fn + tn

    accuracy = (
        (tp + tn) / total
        if total
        else 0.0
    )

    return {
        "total_queries": total,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "false_positive_rate": (
            round(fp / (fp + tn), 4)
            if fp + tn
            else 0.0
        ),
    }


def calculate_answer_consistency(rows):
    """Measure response consistency among same-group cache hits.

    For each group, calculate the fraction of cache-hit responses
    matching that group's most frequently returned response.
    """
    group_responses = defaultdict(list)

    for row in rows:
        if row["classification"] == "TP":
            group_responses[row["expected_group"]].append(
                row["response"]
            )

    group_scores = {}

    for group, responses in group_responses.items():
        counts = Counter(responses)
        most_common_count = counts.most_common(1)[0][1]

        group_scores[group] = round(
            most_common_count / len(responses),
            4,
        )

    overall = (
        sum(group_scores.values()) / len(group_scores)
        if group_scores
        else 0.0
    )

    return round(overall, 4), group_scores


def main():
    """Run end-to-end semantic retrieval quality evaluation."""
    print("=" * 68)
    print("          SEMANTIC CACHE ANSWER QUALITY EVALUATION")
    print("=" * 68)

    dataset = load_dataset()

    print(f"Dataset queries       : {len(dataset)}")
    print(f"Embedding model       : {config.embedding_model_name}")
    print(f"Similarity threshold  : {config.similarity_threshold}")
    print(f"Cache capacity        : {CACHE_CAPACITY}")
    print(f"Cache TTL             : {CACHE_TTL_SECONDS} seconds")
    print("LLM mode              : MockLLM (no real API calls)")
    print()

    embedding_generator = EmbeddingGenerator(
        config.embedding_model_name
    )

    semantic_cache = SemanticCache(
        embedding_generator=embedding_generator
    )

    optimized_cache = CacheOptimizer(
        cache=semantic_cache,
        max_size=CACHE_CAPACITY,
        ttl_seconds=CACHE_TTL_SECONDS,
    )

    pipeline = SemanticCachingPipeline(
        cache=optimized_cache,
        llm=MockLLM(),
    )

    rows = []

    for index, item in enumerate(dataset, start=1):
        query = item["query"]
        expected_group = item["expected_semantic_group"]

        # Inspect cache contents before the pipeline processes the query.
        # This independently establishes whether a relevant answer
        # was available at or above the configured similarity threshold.
        query_embedding = embedding_generator.encode(query)

        candidates = []

        for entry in optimized_cache.entries():
            if entry.embedding is None:
                continue

            score = cosine_similarity(
                query_embedding,
                entry.embedding,
            )

            candidates.append((score, entry))

        best_entry = None
        best_score = None

        if candidates:
            best_score, best_entry = max(
                candidates,
                key=lambda candidate: candidate[0],
            )

        relevant_entry_available = any(
            entry.metadata.get("expected_semantic_group")
            == expected_group
            and score >= config.similarity_threshold
            for score, entry in candidates
        )

        response = pipeline.process_query(query)

        if response.source == "error":
            raise RuntimeError(
                f"MockLLM pipeline failed for query: {query}"
            )

        retrieved_group = ""
        retrieved_query = ""

        if response.source == "cache":
            if best_entry is None:
                raise RuntimeError(
                    "Pipeline returned a cache hit, but no candidate "
                    "was found during independent inspection."
                )

            retrieved_group = (
                best_entry.metadata.get("expected_semantic_group", "")
            )
            retrieved_query = best_entry.query

            if retrieved_group == expected_group:
                classification = "TP"
            else:
                classification = "FP"

        else:
            classification = (
                "FN"
                if relevant_entry_available
                else "TN"
            )

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
                else (
                    round(best_score, 4)
                    if best_score is not None
                    else ""
                )
            ),
            "classification": classification,
            "response": response.response_text,
        })

        # Tag the newly inserted cache entry with its ground-truth group.
        if response.source == "llm":
            for entry in optimized_cache.entries():
                if entry.query == query:
                    entry.metadata["expected_semantic_group"] = (
                        expected_group
                    )
                    break

        print(
            f"[{index:02d}/{len(dataset)}] "
            f"{query[:39]:39} | "
            f"{response.source.upper():5} | "
            f"{classification}"
        )

    metrics = calculate_metrics(rows)

    consistency, consistency_by_group = (
        calculate_answer_consistency(rows)
    )

    summary = {
        "experiment": {
            "dataset": str(DATASET_PATH.relative_to(ROOT)),
            "embedding_model": config.embedding_model_name,
            "similarity_threshold": config.similarity_threshold,
            "cache_capacity": CACHE_CAPACITY,
            "cache_ttl_seconds": CACHE_TTL_SECONDS,
            "eviction_policy": "LRU",
            "llm_mode": "MockLLM",
            "evaluation_type": "semantic retrieval relevance",
            "note": (
                "TP/FP/FN/TN use expected semantic group labels. "
                "This experiment evaluates retrieval relevance, not "
                "independent factual correctness of generated answers. "
                "The cache uses the same capacity and TTL settings as "
                "the main application."
            ),
        },
        "metrics": metrics,
        "answer_consistency": consistency,
        "answer_consistency_by_group": consistency_by_group,
        "query_results": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "query",
                "expected_group",
                "source",
                "retrieved_query",
                "retrieved_group",
                "similarity_score",
                "classification",
                "response",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 68)
    print("                 QUALITY EVALUATION RESULTS")
    print("=" * 68)

    for key, value in metrics.items():
        print(f"{key.replace('_', ' ').title():25}: {value}")

    print(f"{'Answer Consistency':25}: {consistency:.4f}")

    print("\nAnswer consistency by group:")
    for group, score in consistency_by_group.items():
        print(f"  {group:22}: {score:.4f}")

    print("\nResults saved:")
    print(CSV_PATH)
    print(JSON_PATH)


if __name__ == "__main__":
    main()
