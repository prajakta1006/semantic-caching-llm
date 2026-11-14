
"""Benchmark script for semantic cache evaluation.

Compares:
1. Baseline: all queries go directly to MockLLM (no cache).
2. Cached: queries use real semantic embeddings and the semantic cache,
   with MockLLM used only when a cache miss occurs.

No live Gemini/API calls are made by this benchmark.
"""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evaluation.baseline import run_baseline
from src.cache_optimizer import CacheOptimizer
from src.config import config
from src.evaluator import EvaluationTracker
from src.llm import MockLLM
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

    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list) or not data:
        raise ValueError("Dataset must be a non-empty list of query items.")

    for index, item in enumerate(data):
        if not isinstance(item, dict) or not isinstance(item.get("query"), str):
            raise ValueError(
                f"Dataset item {index} must be an object containing a string query."
            )

    return data


def create_benchmark_cache(
    capacity: int = 3,
    similarity_threshold: float = 0.85,
) -> Tuple[CacheOptimizer, str]:
    """Initialize semantic cache and optimizer using real embeddings."""
    try:
        semantic_cache = SemanticCache(
            embedding_model_name=config.embedding_model_name
        )
        embedding_mode = f"REAL ({config.embedding_model_name})"

    except Exception as exc:
        print(f"[!] Could not initialize real embedding model: {exc}")
        raise RuntimeError(
            "Benchmark requires the configured SentenceTransformer model."
        ) from exc

    optimizer = CacheOptimizer(
        cache=semantic_cache,
        max_size=capacity,
        ttl_seconds=None,
    )

    return optimizer, embedding_mode


def percentile(values: List[float], percentile_value: float) -> float:
    """Calculate a percentile using linear interpolation."""
    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile_value / 100.0
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)

    fraction = position - lower_index

    return (
        ordered[lower_index]
        + fraction * (ordered[upper_index] - ordered[lower_index])
    )


def calculate_latency_metrics(
    latencies: List[float],
    total_elapsed_seconds: float,
) -> Dict[str, float]:
    """Calculate average, P50, P95, and throughput."""
    total_queries = len(latencies)
    total_latency_ms = sum(latencies)

    return {
        "avg_latency_ms": (
            total_latency_ms / total_queries if total_queries else 0.0
        ),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "total_latency_ms": total_latency_ms,
        "total_elapsed_seconds": total_elapsed_seconds,
        "throughput_queries_per_second": (
            total_queries / total_elapsed_seconds
            if total_elapsed_seconds > 0
            else 0.0
        ),
    }


def get_llm_metrics(metrics: Any) -> Dict[str, int]:
    """Extract LLM telemetry from EvaluationTracker metrics."""
    metadata = metrics.metadata or {}

    return {
        "successful_llm_calls": metadata.get("llm_calls", 0),
        "llm_call_attempts": metadata.get("llm_call_attempts", 0),
        "failed_llm_calls": metadata.get("llm_calls_failed", 0),
        "http_attempts": metadata.get("http_attempts", 0),
        "retry_count": metadata.get("retry_count", 0),
        "fallback_attempt_count": metadata.get("fallback_attempt_count", 0),
        "fallback_count": metadata.get("fallback_count", 0),
    }


def run_benchmark(
    dataset_path: Optional[Path] = None,
    cache_capacity: int = 3,
    similarity_threshold: float = 0.85,
    save_results: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the baseline versus semantic-cache benchmark."""

    queries = load_dataset(dataset_path)

    # Explicit MockLLM injection prevents accidental Gemini/API calls.
    print("[1/4] Running baseline experiment (no cache)...")

    baseline_results = run_baseline(
        queries,
        llm=MockLLM(default_model=config.default_llm_model),
    )

    print("[2/4] Initializing real semantic embedding cache...")

    cache, embedding_mode = create_benchmark_cache(
        capacity=cache_capacity,
        similarity_threshold=similarity_threshold,
    )

    evaluator = EvaluationTracker()

    # Explicitly inject MockLLM. app_config is retained for other settings.
    cached_pipeline = SemanticCachingPipeline(
        app_config=config,
        cache=cache,
        llm=MockLLM(default_model=config.default_llm_model),
        evaluator=evaluator,
    )

    print("[3/4] Running cached experiment...")

    cached_query_details: List[Dict[str, Any]] = []

    cached_experiment_start = time.perf_counter()

    for index, item in enumerate(queries, start=1):
        query = item["query"]

        start_time = time.perf_counter()
        response = cached_pipeline.process_query(query)
        measured_latency_ms = (time.perf_counter() - start_time) * 1000.0

        cached_query_details.append(
            {
                "id": item.get("id"),
                "group": item.get("group", "unknown"),
                "query": query,
                "source": response.source,
                "decision_route": (
                    response.decision.route
                    if response.decision
                    else "UNKNOWN"
                ),
                "similarity_score": response.similarity_score,
                "latency_ms": measured_latency_ms,
                "estimated_cost": (
                    response.decision.estimated_cost
                    if response.decision
                    else response.metadata.get("cost", 0.0)
                ),
                "model": response.metadata.get("model", "n/a"),
            }
        )

        if index % 10 == 0 or index == len(queries):
            print(f"    Processed {index}/{len(queries)} queries")

    cached_elapsed_seconds = (
        time.perf_counter() - cached_experiment_start
    )

    cached_metrics = evaluator.get_metrics()

    # ---------------------------------------------------------
    # 1. Extract latency and LLM metrics
    # ---------------------------------------------------------

    baseline_latencies = [
        item["latency_ms"]
        for item in baseline_results["query_results"]
    ]

    cached_latencies = [
        item["latency_ms"]
        for item in cached_query_details
    ]

    baseline_latency_metrics = calculate_latency_metrics(
        baseline_latencies,
        baseline_results.get("total_elapsed_seconds", 0.0),
    )

    cached_latency_metrics = calculate_latency_metrics(
        cached_latencies,
        cached_elapsed_seconds,
    )

    baseline_llm_metrics = {
        "successful_llm_calls": baseline_results.get("llm_calls", 0),
        "llm_call_attempts": baseline_results.get("llm_call_attempts", 0),
        "failed_llm_calls": baseline_results.get("llm_calls_failed", 0),
        "http_attempts": baseline_results.get("http_attempts", 0),
        "retry_count": baseline_results.get("retry_count", 0),
        "fallback_attempt_count": baseline_results.get(
            "fallback_attempt_count", 0
        ),
        "fallback_count": baseline_results.get("fallback_count", 0),
    }

    cached_llm_metrics = get_llm_metrics(cached_metrics)

    # ---------------------------------------------------------
    # 2. Cache statistics
    # ---------------------------------------------------------

    cached_queries = cached_metrics.total_queries
    cache_hits = cached_metrics.cache_hits
    cache_misses = cached_metrics.cache_misses

    hit_rate_pct = cached_metrics.hit_rate * 100.0

    # Successful LLM generations are different from misses:
    # a failed generation is still a miss but not a successful call.
    successful_cached_calls = cached_llm_metrics["successful_llm_calls"]
    attempted_cached_calls = cached_llm_metrics["llm_call_attempts"]
    failed_cached_calls = cached_llm_metrics["failed_llm_calls"]

    # ---------------------------------------------------------
    # 3. Cost and comparison metrics
    # ---------------------------------------------------------

    baseline_cost = baseline_results.get("estimated_cost", 0.0)

    cached_cost = (
        cached_metrics.metadata.get("total_estimated_cost")
        or cached_metrics.metadata.get("estimated_cost_with_caching", 0.0)
    )

    baseline_successful_calls = baseline_llm_metrics["successful_llm_calls"]

    calls_reduced_pct = (
        (
            (baseline_successful_calls - successful_cached_calls)
            / baseline_successful_calls
        ) * 100.0
        if baseline_successful_calls > 0
        else 0.0
    )

    baseline_avg_latency = baseline_latency_metrics["avg_latency_ms"]
    cached_avg_latency = cached_latency_metrics["avg_latency_ms"]

    latency_change_pct = (
        (
            (cached_avg_latency - baseline_avg_latency)
            / baseline_avg_latency
        ) * 100.0
        if baseline_avg_latency > 0
        else 0.0
    )

    cost_savings = max(0.0, baseline_cost - cached_cost)

    cost_reduction_pct = (
        (cost_savings / baseline_cost) * 100.0
        if baseline_cost > 0
        else 0.0
    )

    baseline_throughput = baseline_latency_metrics[
        "throughput_queries_per_second"
    ]
    cached_throughput = cached_latency_metrics[
        "throughput_queries_per_second"
    ]

    throughput_change_pct = (
        (
            (cached_throughput - baseline_throughput)
            / baseline_throughput
        ) * 100.0
        if baseline_throughput > 0
        else 0.0
    )

    # ---------------------------------------------------------
    # 4. Per-query comparison
    # ---------------------------------------------------------

    per_query_comparison: List[Dict[str, Any]] = []

    for baseline_item, cached_item in zip(
        baseline_results["query_results"],
        cached_query_details,
    ):
        per_query_comparison.append(
            {
                "id": baseline_item["id"],
                "group": baseline_item["group"],
                "query": baseline_item["query"],
                "baseline_source": baseline_item["source"],
                "baseline_latency_ms": baseline_item["latency_ms"],
                "baseline_cost": baseline_item["estimated_cost"],
                "cached_source": cached_item["source"],
                "cached_similarity_score": cached_item["similarity_score"],
                "cached_latency_ms": cached_item["latency_ms"],
                "cached_cost": cached_item["estimated_cost"],
            }
        )

    # ---------------------------------------------------------
    # 5. Final result object
    # ---------------------------------------------------------

    results_data = {
        "experiment": {
            "llm_mode": "MockLLM (simulated; no real API calls)",
            "embedding_model": config.embedding_model_name,
            "embedding_mode": embedding_mode,
            "similarity_threshold": similarity_threshold,
            "cache_capacity": cache_capacity,
            "eviction_policy": "LRU",
            "number_of_queries": len(queries),
            "execution_order": "Dataset order; single sequential pass",
            "note": (
                "Cached latency includes real embedding and cache lookup "
                "overhead. LLM generation uses MockLLM in both experiments."
            ),
        },
        "baseline": {
            "total_queries": baseline_results["total_queries"],
            "llm_calls": baseline_llm_metrics["successful_llm_calls"],
            "llm_call_attempts": baseline_llm_metrics["llm_call_attempts"],
            "failed_llm_calls": baseline_llm_metrics["failed_llm_calls"],
            "http_attempts": baseline_llm_metrics["http_attempts"],
            "retry_count": baseline_llm_metrics["retry_count"],
            "fallback_attempt_count": baseline_llm_metrics[
                "fallback_attempt_count"
            ],
            "fallback_count": baseline_llm_metrics["fallback_count"],
            **{
                key: round(value, 4)
                for key, value in baseline_latency_metrics.items()
            },
            "estimated_cost": round(baseline_cost, 6),
        },
        "cached": {
            "total_queries": cached_queries,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "hit_rate_pct": round(hit_rate_pct, 2),
            "llm_calls": successful_cached_calls,
            "llm_call_attempts": attempted_cached_calls,
            "failed_llm_calls": failed_cached_calls,
            "http_attempts": cached_llm_metrics["http_attempts"],
            "retry_count": cached_llm_metrics["retry_count"],
            "fallback_attempt_count": cached_llm_metrics[
                "fallback_attempt_count"
            ],
            "fallback_count": cached_llm_metrics["fallback_count"],
            "llm_calls_avoided": cache_hits,
            **{
                key: round(value, 4)
                for key, value in cached_latency_metrics.items()
            },
            "estimated_cost": round(cached_cost, 6),
        },
        "comparison": {
            "llm_calls_reduced_pct": round(calls_reduced_pct, 2),
            "latency_change_pct": round(latency_change_pct, 2),
            "throughput_change_pct": round(throughput_change_pct, 2),
            "cost_reduction_pct": round(cost_reduction_pct, 2),
            "estimated_savings": round(cost_savings, 6),
        },
        "query_results": per_query_comparison,
    }

    # ---------------------------------------------------------
    # 6. Save JSON and CSV
    # ---------------------------------------------------------

    if save_results:
        if output_dir is None:
            output_dir = (
                Path(__file__).resolve().parent.parent
                / "evaluation"
                / "results"
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "benchmark_results.json"

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(results_data, file, indent=2)

        csv_path = output_dir / "benchmark_results.csv"

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

        with open(
            csv_path,
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for row in per_query_comparison:
                writer.writerow(
                    {
                        **row,
                        "baseline_latency_ms": (
                            f"{row['baseline_latency_ms']:.4f}"
                        ),
                        "baseline_cost": f"{row['baseline_cost']:.6f}",
                        "cached_latency_ms": (
                            f"{row['cached_latency_ms']:.4f}"
                        ),
                        "cached_cost": f"{row['cached_cost']:.6f}",
                        "cached_similarity_score": (
                            f"{row['cached_similarity_score']:.4f}"
                            if row["cached_similarity_score"] is not None
                            else "N/A"
                        ),
                    }
                )

        print(f"\nResults saved:")
        print(f"  JSON: {json_path}")
        print(f"  CSV : {csv_path}")

    return results_data


def print_benchmark_report(results: Dict[str, Any]) -> None:
    """Print a human-readable benchmark report."""

    experiment = results["experiment"]
    baseline = results["baseline"]
    cached = results["cached"]
    comparison = results["comparison"]

    print()
    print("=" * 66)
    print("             SEMANTIC CACHE BENCHMARK")
    print("=" * 66)

    print(f"Dataset             : {experiment['number_of_queries']} queries")
    print(f"Embedding Model     : {experiment['embedding_model']}")
    print(f"Embedding Mode      : {experiment['embedding_mode']}")
    print(f"Similarity Threshold: {experiment['similarity_threshold']}")
    print(f"Cache Capacity      : {experiment['cache_capacity']}")
    print(f"Eviction Policy     : {experiment['eviction_policy']}")
    print(f"LLM Mode            : {experiment['llm_mode']}")

    print("\nWITHOUT CACHE")
    print("-" * 66)
    print(f"Total Queries       : {baseline['total_queries']}")
    print(f"Successful LLM Calls: {baseline['llm_calls']}")
    print(f"LLM Call Attempts   : {baseline['llm_call_attempts']}")
    print(f"Failed LLM Calls    : {baseline['failed_llm_calls']}")
    print(f"Average Latency     : {baseline['avg_latency_ms']:.2f} ms")
    print(f"P50 Latency         : {baseline['p50_latency_ms']:.2f} ms")
    print(f"P95 Latency         : {baseline['p95_latency_ms']:.2f} ms")
    print(
        f"Throughput          : "
        f"{baseline['throughput_queries_per_second']:.2f} queries/sec"
    )
    print(f"Estimated Cost      : ${baseline['estimated_cost']:.6f}")

    print("\nWITH SEMANTIC CACHE")
    print("-" * 66)
    print(f"Total Queries       : {cached['total_queries']}")
    print(f"Cache Hits          : {cached['cache_hits']}")
    print(f"Cache Misses        : {cached['cache_misses']}")
    print(f"Hit Rate            : {cached['hit_rate_pct']:.2f}%")
    print(f"Successful LLM Calls: {cached['llm_calls']}")
    print(f"LLM Call Attempts   : {cached['llm_call_attempts']}")
    print(f"Failed LLM Calls    : {cached['failed_llm_calls']}")
    print(f"Retries             : {cached['retry_count']}")
    print(f"Fallback Attempts   : {cached['fallback_attempt_count']}")
    print(f"Fallbacks Used      : {cached['fallback_count']}")
    print(f"Average Latency     : {cached['avg_latency_ms']:.2f} ms")
    print(f"P50 Latency         : {cached['p50_latency_ms']:.2f} ms")
    print(f"P95 Latency         : {cached['p95_latency_ms']:.2f} ms")
    print(
        f"Throughput          : "
        f"{cached['throughput_queries_per_second']:.2f} queries/sec"
    )
    print(f"Estimated Cost      : ${cached['estimated_cost']:.6f}")

    print("\nCOMPARATIVE IMPACT")
    print("-" * 66)
    print(f"LLM Calls Reduced   : {comparison['llm_calls_reduced_pct']:.2f}%")
    print(f"Latency Change      : {comparison['latency_change_pct']:.2f}%")
    print(
        f"Throughput Change   : "
        f"{comparison['throughput_change_pct']:.2f}%"
    )
    print(f"Cost Reduction      : {comparison['cost_reduction_pct']:.2f}%")
    print(f"Estimated Savings   : ${comparison['estimated_savings']:.6f}")

    print("-" * 66)
    print("Note: MockLLM is used for both experiments.")
    print("Cached latency includes real embedding and cache lookup overhead.")
    print("=" * 66)
    print()


def main():
    """CLI entrypoint: python -m evaluation.benchmark"""
    results = run_benchmark()
    print_benchmark_report(results)


if __name__ == "__main__":
    main()
