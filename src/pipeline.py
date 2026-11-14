
"""LLM pipeline orchestration.

Responsibilities:
- query preprocessing
- cache -> LLM flow
- LLM invocation
- storing successful LLM responses
- pipeline response construction
- graceful LLM failure handling
- propagation of LLM execution metadata to evaluator

Cache algorithms and evaluation logic remain in their respective modules.
"""

import time
from typing import Optional

from src.config import AppConfig, config as default_config
from src.interfaces import (
    CacheInterface,
    EvaluatorInterface,
    LLMInterface,
    QueryPreprocessorInterface,
)
from src.llm import LLMService
from src.models import (
    CacheEntry,
    DecisionResult,
    PipelineResponse,
    QueryRequest,
)


class DefaultPreprocessor(QueryPreprocessorInterface):
    """Basic query normalization."""

    def preprocess(self, query: str) -> str:
        return query.strip()


class SemanticCachingPipeline:
    """Coordinates the end-to-end semantic caching workflow."""

    def __init__(
        self,
        app_config: Optional[AppConfig] = None,
        preprocessor: Optional[QueryPreprocessorInterface] = None,
        cache: Optional[CacheInterface] = None,
        llm: Optional[LLMInterface] = None,
        evaluator: Optional[EvaluatorInterface] = None,
    ) -> None:

        self.config = app_config or default_config

        self.preprocessor = (
            preprocessor or DefaultPreprocessor()
        )

        self.cache = cache

        self.llm = (
            llm
            if llm is not None
            else LLMService(app_config=self.config)
        )

        self.evaluator = evaluator

    def process_query(
        self,
        raw_query: str,
        session_id: Optional[str] = None,
    ) -> PipelineResponse:
        """Process a query through the semantic caching pipeline."""

        start_time = time.perf_counter()

        clean_query = self._preprocess(raw_query)

        request = QueryRequest(
            query=clean_query,
            session_id=session_id,
        )

        metadata: dict[str, object] = {}

        # ---------------------------------------------------------
        # 1. CACHE LOOKUP
        # ---------------------------------------------------------

        if self.cache is not None:
            cache_match = self._cache_get(
                clean_query,
                metadata,
            )

            if cache_match is not None:
                latency_ms = self._elapsed_ms(start_time)

                response = PipelineResponse(
                    query=raw_query,
                    response_text=cache_match.response,
                    source="cache",
                    similarity_score=cache_match.similarity_score,
                    decision=self._cache_hit_decision(cache_match),
                    latency_ms=latency_ms,
                    metadata={
                        **metadata,
                        "llm_call_attempted": False,
                        "llm_call_succeeded": False,
                        "llm_call_failed": False,
                        "http_attempts": 0,
                        "retry_count": 0,
                        "fallback_attempted": False,
                        "fallback_used": False,
                    },
                )

                self._record_evaluation(
                    request,
                    response,
                    latency_ms,
                )

                return response

        else:
            metadata["cache_status"] = (
                "semantic cache not connected"
            )

        # ---------------------------------------------------------
        # 2. CACHE MISS -> LLM
        # ---------------------------------------------------------

        try:
            llm_result = self._generate_llm_response(
                clean_query
            )

        except Exception as exc:
            # Failed responses must never be stored in the cache.
            latency_ms = self._elapsed_ms(start_time)

            error_message = str(exc)

            error_metadata = self._extract_llm_error_metadata(exc)
            llm_metadata = error_metadata["llm_metadata"]

            response = PipelineResponse(
                query=raw_query,
                response_text=(
                    "The LLM could not generate a response. "
                    "Please try again later."
                ),
                source="error",
                similarity_score=None,
                decision=DecisionResult(
                    route="LLM_ERROR",
                    estimated_cost=0.0,
                    confidence=0.0,
                    reason=(
                        "The cache did not provide a response "
                        "and LLM generation failed."
                    ),
                    metadata={
                        "provider": llm_metadata.get(
                            "provider",
                            self.config.llm_provider,
                        ),
                        "error": error_message,
                        **error_metadata,
                    },
                ),
                latency_ms=latency_ms,
                metadata={
                    **metadata,
                    "provider": llm_metadata.get(
                        "provider",
                        self.config.llm_provider,
                    ),
                    "llm_call_attempted": True,
                    "llm_call_succeeded": False,
                    "llm_call_failed": True,
                    "llm_error": error_message,
                    "llm_metadata": llm_metadata,
                    "http_attempts": error_metadata["http_attempts"],
                    "retry_count": error_metadata["retry_count"],
                    "fallback_attempted": error_metadata[
                        "fallback_attempted"
                    ],
                    "fallback_used": error_metadata["fallback_used"],
                },
            )

            self._record_evaluation(
                request,
                response,
                latency_ms,
            )

            return response

        # ---------------------------------------------------------
        # 3. STORE SUCCESSFUL LLM RESPONSE
        # ---------------------------------------------------------

        if self.cache is not None:
            self._cache_put(
                query=clean_query,
                response=llm_result.text,
                metadata={
                    "model": llm_result.model_name,
                    "tokens": llm_result.tokens_used,
                    "cost": llm_result.cost,
                    "provider": llm_result.metadata.get(
                        "provider",
                        self.config.llm_provider,
                    ),
                },
                response_metadata=metadata,
            )

        # ---------------------------------------------------------
        # 4. BUILD SUCCESSFUL RESPONSE
        # ---------------------------------------------------------

        latency_ms = self._elapsed_ms(start_time)

        llm_metadata = dict(
            llm_result.metadata or {}
        )

        provider = llm_metadata.get(
            "provider",
            self.config.llm_provider,
        )

        http_attempts = self._safe_int(
            llm_metadata.get("request_attempts", 1),
            default=1,
        )

        retry_count = self._safe_int(
            llm_metadata.get("retry_count", 0),
            default=0,
        )

        fallback_attempted = bool(
            llm_metadata.get(
                "fallback_attempted",
                False,
            )
        )

        fallback_used = bool(
            llm_metadata.get(
                "fallback_used",
                False,
            )
        )

        decision = DecisionResult(
            route="LLM_AFTER_CACHE_MISS",
            estimated_cost=llm_result.cost,
            confidence=1.0,
            reason=(
                "No sufficiently similar cached response "
                "was available; query was sent to the LLM."
            ),
            metadata={
                "selected_model": llm_result.model_name,
                "provider": provider,
            },
        )

        response = PipelineResponse(
            query=raw_query,
            response_text=llm_result.text,
            source="llm",
            similarity_score=None,
            decision=decision,
            latency_ms=latency_ms,
            metadata={
                **metadata,
                "model": llm_result.model_name,
                "cost": llm_result.cost,
                "tokens_used": llm_result.tokens_used,
                "provider": provider,
                "llm_call_attempted": True,
                "llm_call_succeeded": True,
                "llm_call_failed": False,
                "http_attempts": max(0, http_attempts),
                "retry_count": max(0, retry_count),
                "fallback_attempted": fallback_attempted,
                "fallback_used": fallback_used,
                "llm_metadata": llm_metadata,
            },
        )

        self._record_evaluation(
            request,
            response,
            latency_ms,
        )

        return response

    # =============================================================
    # QUERY
    # =============================================================

    def _preprocess(
        self,
        raw_query: str,
    ) -> str:
        """Validate and normalize a user query."""

        if raw_query is None:
            raise ValueError("Query cannot be None")

        clean_query = self.preprocessor.preprocess(raw_query)

        if not clean_query:
            raise ValueError("Query cannot be empty")

        return clean_query

    # =============================================================
    # CACHE
    # =============================================================

    def _cache_get(
        self,
        query: str,
        metadata: dict[str, object],
    ) -> Optional[CacheEntry]:
        """Call the semantic cache."""

        try:
            return self.cache.get(
                query=query,
                threshold=self.config.similarity_threshold,
            )

        except Exception as exc:
            metadata["cache_lookup_error"] = str(exc)
            return None

    def _cache_put(
        self,
        query: str,
        response: str,
        metadata: dict[str, object],
        response_metadata: dict[str, object],
    ) -> None:
        """Store a successful LLM response through CacheInterface."""

        try:
            self.cache.put(
                query=query,
                response=response,
                metadata=metadata,
            )

        except Exception as exc:
            response_metadata["cache_store_error"] = str(exc)

    # =============================================================
    # LLM
    # =============================================================

    def _generate_llm_response(self, query: str):
        """Generate a response using the configured LLM."""

        return self.llm.generate(
            query=query,
            model=self.config.default_llm_model,
        )

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        """Convert metadata to an integer safely."""

        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _extract_llm_error_metadata(
        exc: Exception,
    ) -> dict:
        """Extract telemetry attached to an LLM exception."""

        raw_metadata = getattr(
            exc,
            "metadata",
            {},
        )

        if not isinstance(raw_metadata, dict):
            raw_metadata = {}

        try:
            attempts = int(
                raw_metadata.get("request_attempts", 1)
            )
        except (ValueError, TypeError):
            attempts = 1

        try:
            retries = int(
                raw_metadata.get("retry_count", 0)
            )
        except (ValueError, TypeError):
            retries = 0

        return {
            "http_attempts": max(0, attempts),
            "retry_count": max(0, retries),
            "fallback_attempted": bool(
                raw_metadata.get("fallback_attempted", False)
            ),
            "fallback_used": bool(
                raw_metadata.get("fallback_used", False)
            ),
            "llm_metadata": dict(raw_metadata),
        }

    # =============================================================
    # EVALUATION
    # =============================================================

    def _record_evaluation(
        self,
        request: QueryRequest,
        response: PipelineResponse,
        latency_ms: float,
    ) -> None:
        """Send telemetry to the evaluator."""

        if self.evaluator is None:
            return

        try:
            self.evaluator.record(
                request,
                response,
                latency_ms,
            )

        except Exception as exc:
            response.metadata["evaluator_error"] = str(exc)

    # =============================================================
    # DECISION METADATA
    # =============================================================

    @staticmethod
    def _cache_hit_decision(
        cache_match: CacheEntry,
    ) -> DecisionResult:
        """Create metadata describing a cache hit."""

        return DecisionResult(
            route="CACHE_HIT",
            estimated_cost=0.0,
            confidence=cache_match.similarity_score,
            reason=(
                "Semantic cache similarity met the "
                "configured threshold."
            ),
            metadata={
                "selected_model": None,
                "provider": "cache",
            },
        )

    # =============================================================
    # TIMING
    # =============================================================

    @staticmethod
    def _elapsed_ms(start_time: float) -> float:
        """Return elapsed time in milliseconds."""

        return (
            time.perf_counter() - start_time
        ) * 1000.0
