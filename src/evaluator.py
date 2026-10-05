"""Evaluation module owned by Andi.

Responsibilities:
    - Observe and record pipeline executions (QueryRequest, PipelineResponse, latency)
    - Track metrics: hits, misses, hit rate, LLM calls, LLM calls avoided, latencies, costs
    - Compute cost savings and latency improvements (simulated/estimated)
    - Provide structured EvaluationMetrics conforming to EvaluatorInterface
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional

from src.interfaces import EvaluatorInterface
from src.models import EvaluationMetrics, PipelineResponse, QueryRequest


@dataclass
class QueryRecord:
    """Record of a single processed query."""

    query: str
    source: str  # "cache" or "llm"
    decision_route: str
    similarity_score: Optional[float]
    latency_ms: float
    estimated_cost: float
    llm_call_occurred: bool
    llm_call_avoided: bool
    model: str
    session_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvaluationTracker(EvaluatorInterface):
    """Concrete implementation of Andi's EvaluatorInterface.

    Observes pipeline results without interfering with cache lookup,
    embedding generation, LLM generation, or eviction policies.
    """

    def __init__(self, default_llm_cost_per_query: float = 0.002) -> None:
        """Initialize the evaluation tracker.

        Args:
            default_llm_cost_per_query: Fallback estimated cost per LLM call
                used when calculating hypothetical non-cached baseline cost
                for cache hits.
        """
        self.default_llm_cost_per_query = default_llm_cost_per_query
        self._records: List[QueryRecord] = []

    def record(
        self,
        request: QueryRequest,
        response: PipelineResponse,
        latency_ms: float,
    ) -> None:
        """Record one pipeline execution."""
        source = (response.source or "unknown").lower()
        is_hit = source == "cache"
        is_miss = source == "llm"

        decision_route = (
            response.decision.route
            if response.decision
            else ("CACHE_HIT" if is_hit else "LLM_AFTER_CACHE_MISS")
        )

        cost = 0.0
        if response.decision and response.decision.estimated_cost:
            cost = response.decision.estimated_cost
        elif "cost" in response.metadata:
            try:
                cost = float(response.metadata["cost"])
            except (ValueError, TypeError):
                cost = 0.0
        elif is_miss:
            cost = self.default_llm_cost_per_query

        model = response.metadata.get(
            "model",
            "mock-llm-v1" if is_miss else "cache",
        )

        rec = QueryRecord(
            query=request.query,
            source=source,
            decision_route=decision_route,
            similarity_score=response.similarity_score,
            latency_ms=latency_ms,
            estimated_cost=cost,
            llm_call_occurred=is_miss,
            llm_call_avoided=is_hit,
            model=str(model),
            session_id=request.session_id,
            timestamp=request.timestamp,
            metadata=dict(response.metadata),
        )

        self._records.append(rec)

    def get_metrics(self) -> EvaluationMetrics:
        """Return accumulated evaluation metrics."""
        total_queries = len(self._records)
        if total_queries == 0:
            return EvaluationMetrics(
                total_queries=0,
                cache_hits=0,
                cache_misses=0,
                hit_rate=0.0,
                total_cost_saved=0.0,
                avg_latency_ms=0.0,
                metadata={
                    "llm_calls": 0,
                    "llm_calls_avoided": 0,
                    "llm_reduction_pct": 0.0,
                    "avg_hit_latency_ms": 0.0,
                    "avg_miss_latency_ms": 0.0,
                    "total_estimated_cost": 0.0,
                    "estimated_cost_with_caching": 0.0,
                    "estimated_cost_without_caching": 0.0,
                    "estimated_cost_savings": 0.0,
                    "cost_reduction_pct": 0.0,
                },
            )

        cache_hits = sum(1 for r in self._records if r.llm_call_avoided)
        cache_misses = sum(1 for r in self._records if r.llm_call_occurred)
        hit_rate = cache_hits / total_queries

        total_latency = sum(r.latency_ms for r in self._records)
        avg_latency_ms = total_latency / total_queries

        hit_latencies = [r.latency_ms for r in self._records if r.llm_call_avoided]
        avg_hit_latency_ms = (
            sum(hit_latencies) / len(hit_latencies) if hit_latencies else 0.0
        )

        miss_latencies = [r.latency_ms for r in self._records if r.llm_call_occurred]
        avg_miss_latency_ms = (
            sum(miss_latencies) / len(miss_latencies) if miss_latencies else 0.0
        )

        # Actual cost with caching
        cost_with_caching = sum(r.estimated_cost for r in self._records)

        # Estimated cost without caching (if every query called LLM)
        cost_without_caching = 0.0
        for r in self._records:
            if r.llm_call_occurred:
                cost_without_caching += r.estimated_cost
            else:
                cost_without_caching += self.default_llm_cost_per_query

        cost_saved = max(0.0, cost_without_caching - cost_with_caching)
        cost_reduction_pct = (
            (cost_saved / cost_without_caching * 100.0)
            if cost_without_caching > 0
            else 0.0
        )

        llm_reduction_pct = (cache_hits / total_queries) * 100.0

        metadata = {
            "llm_calls": cache_misses,
            "llm_calls_avoided": cache_hits,
            "llm_reduction_pct": llm_reduction_pct,
            "avg_hit_latency_ms": avg_hit_latency_ms,
            "avg_miss_latency_ms": avg_miss_latency_ms,
            "total_estimated_cost": cost_with_caching,
            "estimated_cost_with_caching": cost_with_caching,
            "estimated_cost_without_caching": cost_without_caching,
            "estimated_cost_savings": cost_saved,
            "cost_reduction_pct": cost_reduction_pct,
        }

        return EvaluationMetrics(
            total_queries=total_queries,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            hit_rate=hit_rate,
            total_cost_saved=cost_saved,
            avg_latency_ms=avg_latency_ms,
            metadata=metadata,
        )

    def records(self) -> List[QueryRecord]:
        """Return a copy of all recorded query records."""
        return list(self._records)

    def clear(self) -> None:
        """Clear all stored evaluation records."""
        self._records.clear()


# Alias for compatibility with default naming conventions
DefaultEvaluator = EvaluationTracker
