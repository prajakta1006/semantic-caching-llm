"""Baseline runner (NO CACHE experiment).

Evaluates pipeline performance when all queries bypass the semantic cache
and are routed directly to the LLM.
"""

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
    """Run evaluation dataset without caching (direct LLM calls).

    Args:
        queries: List of query strings or dataset query dicts.
        llm: Optional LLM implementation (defaults to MockLLM).
        evaluator: Optional evaluator instance (defaults to fresh EvaluationTracker).

    Returns:
        Dict containing baseline metrics and per-query execution details.
    """
    tracker = evaluator if evaluator is not None else EvaluationTracker()
    llm_service = llm if llm is not None else MockLLM()

    # Create pipeline with NO cache connected (cache=None)
    pipeline = SemanticCachingPipeline(
        cache=None,
        llm=llm_service,
        evaluator=tracker,
    )

    query_details: List[Dict[str, Any]] = []

    for item in queries:
        if isinstance(item, dict):
            query_str = item.get("query", "")
            group = item.get("group", "unknown")
            q_id = item.get("id", None)
        else:
            query_str = str(item)
            group = "unknown"
            q_id = None

        response = pipeline.process_query(query_str)

        query_details.append(
            {
                "id": q_id,
                "group": group,
                "query": query_str,
                "source": response.source,
                "decision_route": (
                    response.decision.route if response.decision else "UNKNOWN"
                ),
                "latency_ms": response.latency_ms,
                "estimated_cost": (
                    response.decision.estimated_cost
                    if response.decision
                    else response.metadata.get("cost", 0.0)
                ),
                "model": response.metadata.get("model", "mock-llm-v1"),
            }
        )

    metrics = tracker.get_metrics()

    return {
        "total_queries": metrics.total_queries,
        "llm_calls": metrics.cache_misses,
        "total_latency_ms": sum(d["latency_ms"] for d in query_details),
        "avg_latency_ms": metrics.avg_latency_ms,
        "estimated_cost": (
            metrics.metadata.get("total_estimated_cost")
            or metrics.metadata.get("estimated_cost_with_caching", 0.0)
        ),
        "query_results": query_details,
        "metrics": metrics,
    }
