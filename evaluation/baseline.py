"""Baseline runner (NO CACHE experiment).

Evaluates pipeline performance when all queries bypass the semantic cache
and are routed directly to the LLM.
"""

import time
from typing import Any, Dict, List, Optional, Union

from src.evaluator import EvaluationTracker
from src.interfaces import EvaluatorInterface, LLMInterface
from src.llm import MockLLM
from src.pipeline import SemanticCachingPipeline


def run_baseline(
    queries: List[Union[str, Dict[str, Any]]],
    llm: Optional[LLMInterface] = None,
    evaluator: Optional[EvaluatorInterface] = None,
) -> Dict[str, Any]:
    """Run the evaluation dataset without caching."""

    tracker = evaluator if evaluator is not None else EvaluationTracker()
    llm_service = llm if llm is not None else MockLLM()

    pipeline = SemanticCachingPipeline(
        cache=None,
        llm=llm_service,
        evaluator=tracker,
    )

    query_details: List[Dict[str, Any]] = []
    experiment_start = time.perf_counter()

    for item in queries:
        if isinstance(item, dict):
            query_str = item.get("query", "")
            group = item.get("group", "unknown")
            q_id = item.get("id")
        else:
            query_str = str(item)
            group = "unknown"
            q_id = None

        start_time = time.perf_counter()
        response = pipeline.process_query(query_str)
        measured_latency_ms = (time.perf_counter() - start_time) * 1000.0

        query_details.append(
            {
                "id": q_id,
                "group": group,
                "query": query_str,
                "source": response.source,
                "decision_route": (
                    response.decision.route if response.decision else "UNKNOWN"
                ),
                "latency_ms": measured_latency_ms,
                "estimated_cost": (
                    response.decision.estimated_cost
                    if response.decision
                    else response.metadata.get("cost", 0.0)
                ),
                "model": response.metadata.get("model", "mock-llm-v1"),
            }
        )

    total_elapsed_seconds = time.perf_counter() - experiment_start
    metrics = tracker.get_metrics()
    metric_data = metrics.metadata

    total_latency_ms = sum(d["latency_ms"] for d in query_details)
    total_queries = len(query_details)

    return {
        "total_queries": total_queries,

        # Successful generations, not merely cache misses.
        "llm_calls": metric_data.get("llm_calls", 0),
        "llm_call_attempts": metric_data.get("llm_call_attempts", 0),
        "llm_calls_failed": metric_data.get("llm_calls_failed", 0),
        "http_attempts": metric_data.get("http_attempts", 0),
        "retry_count": metric_data.get("retry_count", 0),
        "fallback_attempt_count": metric_data.get("fallback_attempt_count", 0),
        "fallback_count": metric_data.get("fallback_count", 0),

        "total_latency_ms": total_latency_ms,
        "avg_latency_ms": (
            total_latency_ms / total_queries if total_queries else 0.0
        ),
        "total_elapsed_seconds": total_elapsed_seconds,
        "throughput_queries_per_second": (
            total_queries / total_elapsed_seconds
            if total_elapsed_seconds > 0
            else 0.0
        ),

        "estimated_cost": (
            metric_data.get("total_estimated_cost")
            or metric_data.get("estimated_cost_with_caching", 0.0)
        ),
        "query_results": query_details,
        "metrics": metrics,
    }