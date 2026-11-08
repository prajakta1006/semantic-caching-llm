
"""Evaluation module owned by Andi.

Responsibilities:
    - Observe and record pipeline executions.
    - Track cache hits, misses and LLM generation outcomes.
    - Track provider HTTP attempts, retries and fallback usage.
    - Track latency, costs and estimated savings.
    - Provide structured EvaluationMetrics conforming to EvaluatorInterface.
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
    source: str
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

    # LLM generation outcomes.
    llm_call_attempted: bool = False
    llm_call_succeeded: bool = False
    llm_call_failed: bool = False

    # Individual provider request metrics.
    http_attempts: int = 0
    retry_count: int = 0
    fallback_attempted: bool = False
    fallback_used: bool = False


class EvaluationTracker(EvaluatorInterface):
    """Track pipeline execution and performance metrics."""

    def __init__(
        self,
        default_llm_cost_per_query: float = 0.002,
    ) -> None:
        self.default_llm_cost_per_query = (
            default_llm_cost_per_query
        )
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

        response_metadata = response.metadata or {}
        llm_metadata = response_metadata.get("llm_metadata") or {}

        if not isinstance(llm_metadata, dict):
            llm_metadata = {}

        attempted = bool(
            response_metadata.get(
                "llm_call_attempted",
                source in ("llm", "error"),
            )
        )

        succeeded = bool(
            response_metadata.get(
                "llm_call_succeeded",
                source == "llm",
            )
        )

        failed = bool(
            response_metadata.get(
                "llm_call_failed",
                source == "error",
            )
        )

        decision_route = (
            response.decision.route
            if response.decision
            else (
                "CACHE_HIT"
                if is_hit
                else (
                    "LLM_ERROR"
                    if source == "error"
                    else "LLM_AFTER_CACHE_MISS"
                )
            )
        )

        # Determine the known generation cost.
        cost = 0.0

        if succeeded:
            if (
                response.decision
                and response.decision.estimated_cost
            ):
                cost = float(
                    response.decision.estimated_cost
                )

            elif "cost" in response_metadata:
                try:
                    cost = float(
                        response_metadata["cost"]
                    )
                except (ValueError, TypeError):
                    cost = 0.0

            else:
                cost = self.default_llm_cost_per_query

        model = response_metadata.get(
            "model",
            llm_metadata.get(
                "model_used",
                "mock-llm-v1" if succeeded else (
                    "cache" if is_hit else "unknown"
                ),
            ),
        )

        # Read provider-level HTTP attempt information.
        try:
            http_attempts = int(
                llm_metadata.get(
                    "request_attempts",
                    response_metadata.get(
                        "http_attempts",
                        1 if attempted else 0,
                    ),
                )
            )
        except (ValueError, TypeError):
            http_attempts = 1 if attempted else 0

        try:
            retry_count = int(
                llm_metadata.get(
                    "retry_count",
                    response_metadata.get("retry_count", 0),
                )
            )
        except (ValueError, TypeError):
            retry_count = 0

        fallback_attempted = bool(
            llm_metadata.get(
                "fallback_attempted",
                response_metadata.get(
                    "fallback_attempted",
                    False,
                ),
            )
        )

        fallback_used = bool(
            llm_metadata.get(
                "fallback_used",
                response_metadata.get("fallback_used", False),
            )
        )

        # Preserve sensible defaults for older pipeline implementations.
        if not attempted:
            http_attempts = 0
            retry_count = 0
            fallback_attempted = False
            fallback_used = False

        rec = QueryRecord(
            query=request.query,
            source=source,
            decision_route=decision_route,
            similarity_score=response.similarity_score,
            latency_ms=latency_ms,
            estimated_cost=cost,
            llm_call_occurred=succeeded,
            llm_call_avoided=is_hit,
            model=str(model),
            session_id=request.session_id,
            timestamp=request.timestamp,
            metadata=dict(response_metadata),
            llm_call_attempted=attempted,
            llm_call_succeeded=succeeded,
            llm_call_failed=failed,
            http_attempts=max(0, http_attempts),
            retry_count=max(0, retry_count),
            fallback_attempted=fallback_attempted,
            fallback_used=fallback_used,
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
                    "llm_call_attempts": 0,
                    "llm_calls_failed": 0,
                    "llm_calls_avoided": 0,
                    "http_attempts": 0,
                    "retry_count": 0,
                    "fallback_attempt_count": 0,
                    "fallback_count": 0,
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

        cache_hits = sum(
            1 for r in self._records
            if r.llm_call_avoided
        )

        cache_misses = sum(
            1 for r in self._records
            if r.source in ("llm", "error")
        )

        successful_calls = sum(
            1 for r in self._records
            if r.llm_call_succeeded
        )

        failed_calls = sum(
            1 for r in self._records
            if r.llm_call_failed
        )

        attempted_calls = sum(
            1 for r in self._records
            if r.llm_call_attempted
        )

        total_http_attempts = sum(
            r.http_attempts for r in self._records
        )

        total_retries = sum(
            r.retry_count for r in self._records
        )

        fallback_attempt_count = sum(
            1 for r in self._records
            if r.fallback_attempted
        )

        fallback_count = sum(
            1 for r in self._records
            if r.fallback_used
        )

        hit_rate = cache_hits / total_queries

        total_latency = sum(
            r.latency_ms for r in self._records
        )

        avg_latency_ms = total_latency / total_queries

        hit_latencies = [
            r.latency_ms
            for r in self._records
            if r.llm_call_avoided
        ]

        avg_hit_latency_ms = (
            sum(hit_latencies) / len(hit_latencies)
            if hit_latencies
            else 0.0
        )

        miss_latencies = [
            r.latency_ms
            for r in self._records
            if r.source in ("llm", "error")
        ]

        avg_miss_latency_ms = (
            sum(miss_latencies) / len(miss_latencies)
            if miss_latencies
            else 0.0
        )

        cost_with_caching = sum(
            r.estimated_cost for r in self._records
        )

        cost_without_caching = 0.0

        for record in self._records:
            if record.llm_call_succeeded:
                cost_without_caching += record.estimated_cost

            elif record.llm_call_avoided:
                cost_without_caching += (
                    self.default_llm_cost_per_query
                )

        cost_saved = max(
            0.0,
            cost_without_caching - cost_with_caching,
        )

        cost_reduction_pct = (
            cost_saved / cost_without_caching * 100.0
            if cost_without_caching > 0
            else 0.0
        )

        llm_reduction_pct = (
            cache_hits / total_queries * 100.0
        )

        metadata = {
            "llm_calls": successful_calls,
            "llm_call_attempts": attempted_calls,
            "llm_calls_failed": failed_calls,
            "llm_calls_avoided": cache_hits,
            "http_attempts": total_http_attempts,
            "retry_count": total_retries,
            "fallback_attempt_count": fallback_attempt_count,
            "fallback_count": fallback_count,
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
        """Return a copy of all stored query records."""

        return list(self._records)

    def clear(self) -> None:
        """Clear all stored evaluation records."""

        self._records.clear()


# Alias for compatibility with existing naming conventions.
DefaultEvaluator = EvaluationTracker
