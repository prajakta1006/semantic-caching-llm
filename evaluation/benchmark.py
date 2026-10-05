"""Benchmark script for semantic cache evaluation owned by Andi.

Runs baseline (NO CACHE) vs cached experiment on evaluation dataset,
computes comparative metrics, prints tabular reports, and saves JSON/CSV results.
"""

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evaluation.baseline import run_baseline
from src.cache_optimizer import CacheOptimizer
from src.config import config
from src.evaluator import EvaluationTracker
from src.pipeline import SemanticCachingPipeline
from src.semantic_cache import SemanticCache


def load_dataset(dataset_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load evaluation queries dataset."""
    if dataset_path is None:
        dataset_path = (
            Path(__file__).resolve().parent.parent
            / "datasets"
            / "evaluation_queries.json"
        )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found at {dataset_path}"
        )

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Dataset must be a non-empty list of query items.")

    return data


def create_benchmark_cache(
    capacity: int = 3,
    similarity_threshold: float = 0.85,
) -> Tuple[CacheOptimizer, str]:
    """Initialize semantic cache and optimizer for benchmark.

    Returns:
        Tuple of (CacheOptimizer, embedding_mode_string)
    """
    try:
        semantic_cache = SemanticCache(
            embedding_model_name=config.embedding_model_name
        )
        embedding_mode = f"REAL ({config.embedding_model_name})"
    except Exception as exc:
        print(f"[!] Warning: Could not initialize real embedding model: {exc}")
        print("[!] Benchmark requires real embeddings for accurate results.")
        raise RuntimeError(
            f"Benchmark failed to load real SentenceTransformer model ({config.embedding_model_name}): {exc}"
        ) from exc

    optimizer = CacheOptimizer(
        cache=semantic_cache,
        max_size=capacity,
        ttl_seconds=None,
    )

    return optimizer, embedding_mode


def run_benchmark(
    dataset_path: Optional[Path] = None,
    cache_capacity: int = 3,
    similarity_threshold: float = 0.85,
    save_results: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete baseline vs cached benchmark suite."""
    queries = load_dataset(dataset_path)

    # 1. Run Baseline (NO CACHE)
    baseline_results = run_baseline(queries)

    # 2. Run Cached Experiment (WITH SEMANTIC CACHE)
    cache, embedding_mode = create_benchmark_cache(
        capacity=cache_capacity,
        similarity_threshold=similarity_threshold,
    )
    evaluator = EvaluationTracker()

    cached_pipeline = SemanticCachingPipeline(
        app_config=config,
        cache=cache,
        evaluator=evaluator,
    )

    cached_query_details: List[Dict[str, Any]] = []

    for item in queries:
        q_str = item["query"]
        q_id = item.get("id")
        group = item.get("group", "unknown")

        response = cached_pipeline.process_query(q_str)

        cached_query_details.append(
            {
                "id": q_id,
                "group": group,
                "query": q_str,
                "source": response.source,
                "decision_route": (
                    response.decision.route if response.decision else "UNKNOWN"
                ),
                "similarity_score": response.similarity_score,
                "latency_ms": response.latency_ms,
                "estimated_cost": (
                    response.decision.estimated_cost
                    if response.decision
                    else response.metadata.get("cost", 0.0)
                ),
                "model": response.metadata.get("model", "n/a"),
            }
        )

    cached_metrics = evaluator.get_metrics()

    # 3. Calculate Comparison Metrics
    b_queries = baseline_results["total_queries"]
    b_llm_calls = baseline_results["llm_calls"]
    b_avg_lat = baseline_results["avg_latency_ms"]
    b_total_lat = baseline_results["total_latency_ms"]
    b_cost = baseline_results["estimated_cost"]

    c_queries = cached_metrics.total_queries
    c_hits = cached_metrics.cache_hits
    c_misses = cached_metrics.cache_misses
    c_hit_rate_pct = cached_metrics.hit_rate * 100.0
    c_llm_calls = cached_metrics.cache_misses
    c_llm_avoided = cached_metrics.cache_hits
    c_avg_lat = cached_metrics.avg_latency_ms
    c_total_lat = sum(d["latency_ms"] for d in cached_query_details)
    c_cost = (
        cached_metrics.metadata.get("total_estimated_cost")
        or cached_metrics.metadata.get("estimated_cost_with_caching", 0.0)
    )

    llm_calls_reduced_pct = (
        ((b_llm_calls - c_llm_calls) / b_llm_calls * 100.0)
        if b_llm_calls > 0
        else 0.0
    )

    latency_change_pct = (
        ((c_avg_lat - b_avg_lat) / b_avg_lat * 100.0)
        if b_avg_lat > 0
        else 0.0
    )

    estimated_savings = max(0.0, b_cost - c_cost)

    cost_reduction_pct = (
        (estimated_savings / b_cost * 100.0)
        if b_cost > 0
        else 0.0
    )

    # Combine query results
    per_query_comparison: List[Dict[str, Any]] = []
    for b_q, c_q in zip(baseline_results["query_results"], cached_query_details):
        per_query_comparison.append(
            {
                "id": b_q["id"],
                "group": b_q["group"],
                "query": b_q["query"],
                "baseline_source": b_q["source"],
                "baseline_latency_ms": b_q["latency_ms"],
                "baseline_cost": b_q["estimated_cost"],
                "cached_source": c_q["source"],
                "cached_similarity_score": c_q["similarity_score"],
                "cached_latency_ms": c_q["latency_ms"],
                "cached_cost": c_q["estimated_cost"],
            }
        )

    results_data = {
        "experiment": {
            "embedding_model": config.embedding_model_name,
            "embedding_mode": embedding_mode,
            "similarity_threshold": similarity_threshold,
            "cache_capacity": cache_capacity,
            "eviction_policy": "LRU",
            "number_of_queries": len(queries),
        },
        "baseline": {
            "total_queries": b_queries,
            "llm_calls": b_llm_calls,
            "avg_latency_ms": round(b_avg_lat, 2),
            "total_latency_ms": round(b_total_lat, 2),
            "estimated_cost": round(b_cost, 6),
        },
        "cached": {
            "total_queries": c_queries,
            "cache_hits": c_hits,
            "cache_misses": c_misses,
            "hit_rate_pct": round(c_hit_rate_pct, 2),
            "llm_calls": c_llm_calls,
            "llm_calls_avoided": c_llm_avoided,
            "avg_latency_ms": round(c_avg_lat, 2),
            "total_latency_ms": round(c_total_lat, 2),
            "estimated_cost": round(c_cost, 6),
        },
        "comparison": {
            "llm_calls_reduced_pct": round(llm_calls_reduced_pct, 2),
            "latency_change_pct": round(latency_change_pct, 2),
            "cost_reduction_pct": round(cost_reduction_pct, 2),
            "estimated_savings": round(estimated_savings, 6),
        },
        "query_results": per_query_comparison,
    }

    if save_results:
        if output_dir is None:
            output_dir = (
                Path(__file__).resolve().parent.parent
                / "evaluation"
                / "results"
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "benchmark_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        csv_path = output_dir / "benchmark_results.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "id",
                "group",
                "query",
                "baseline_source",
                "baseline_latency_ms",
                "baseline_cost",
                "cached_source",
                "cached_similarity_score",
                "cached_latency_ms",
                "cached_cost",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_query_comparison:
                writer.writerow(
                    {
                        "id": row["id"],
                        "group": row["group"],
                        "query": row["query"],
                        "baseline_source": row["baseline_source"],
                        "baseline_latency_ms": f"{row['baseline_latency_ms']:.2f}",
                        "baseline_cost": f"{row['baseline_cost']:.6f}",
                        "cached_source": row["cached_source"],
                        "cached_similarity_score": (
                            f"{row['cached_similarity_score']:.4f}"
                            if row["cached_similarity_score"] is not None
                            else "N/A"
                        ),
                        "cached_latency_ms": f"{row['cached_latency_ms']:.2f}",
                        "cached_cost": f"{row['cached_cost']:.6f}",
                    }
                )

    return results_data


def print_benchmark_report(results: Dict[str, Any]) -> None:
    """Print human-readable benchmark report."""
    exp = results["experiment"]
    base = results["baseline"]
    cached = results["cached"]
    comp = results["comparison"]

    print()
    print("=" * 60)
    print("SEMANTIC CACHE BENCHMARK")
    print("=" * 60)
    print(f"Dataset          : {exp['number_of_queries']} queries")
    print(f"Embedding Model  : {exp['embedding_model']}")
    print(f"Threshold        : {exp['similarity_threshold']}")
    print(f"Cache Capacity   : {exp['cache_capacity']}")
    print(f"Eviction Policy  : {exp['eviction_policy']}")

    print("-" * 60)
    print("WITHOUT CACHE")
    print("-" * 60)
    print(f"Total Queries    : {base['total_queries']}")
    print(f"LLM Calls        : {base['llm_calls']}")
    print(f"Average Latency  : {base['avg_latency_ms']:.2f} ms")
    print(f"Estimated Cost   : ${base['estimated_cost']:.4f}")

    print("-" * 60)
    print("WITH SEMANTIC CACHE")
    print("-" * 60)
    print(f"Total Queries    : {cached['total_queries']}")
    print(f"Cache Hits       : {cached['cache_hits']}")
    print(f"Cache Misses     : {cached['cache_misses']}")
    print(f"Hit Rate         : {cached['hit_rate_pct']:.2f}%")
    print(f"LLM Calls        : {cached['llm_calls']}")
    print(f"LLM Calls Avoided: {cached['llm_calls_avoided']}")
    print(f"Average Latency  : {cached['avg_latency_ms']:.2f} ms")
    print(f"Estimated Cost   : ${cached['estimated_cost']:.4f}")

    print("-" * 60)
    print("IMPACT")
    print("-" * 60)
    print(f"LLM Calls Reduced: {comp['llm_calls_reduced_pct']:.2f}%")
    print(f"Latency Change   : {comp['latency_change_pct']:.2f}%")
    print(f"Cost Reduction   : {comp['cost_reduction_pct']:.2f}%")
    print(f"Estimated Savings: ${comp['estimated_savings']:.4f}")
    print("=" * 60)
    print()


def main():
    """CLI entrypoint for running benchmark: py -m evaluation.benchmark"""
    results = run_benchmark()
    print_benchmark_report(results)


if __name__ == "__main__":
    main()
