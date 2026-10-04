"""LLM pipeline orchestration owned by Dhano.

The pipeline coordinates the components but does not implement their
algorithms.

Responsibilities delegated to other modules:

    TanTan
        - semantic cache
        - embeddings
        - similarity search

    Praj
        - cache optimization policies such as LRU/TTL/frequency

    Andi
        - evaluation and benchmarking

Dhano owns:

    - query preprocessing
    - cache -> LLM decision flow
    - LLM invocation
    - storing LLM responses in the cache
    - returning the final PipelineResponse
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
from src.llm import MockLLM
from src.models import (
    CacheEntry,
    DecisionResult,
    PipelineResponse,
    QueryRequest,
)


class DefaultPreprocessor(QueryPreprocessorInterface):
    """Basic query normalization.

    More advanced preprocessing can be added later without changing
    the pipeline contract.
    """

    def preprocess(self, query: str) -> str:
        return query.strip()


class SemanticCachingPipeline:
    """Main query-processing pipeline.

    Flow:

        User Query
            ↓
        Preprocess
            ↓
        Semantic Cache
            ↓
        HIT ───────────────→ return cached response
            │
            MISS
            ↓
        LLM
            ↓
        Store response in cache
            ↓
        Evaluation
            ↓
        Return response
    """

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
            preprocessor
            or DefaultPreprocessor()
        )

        # TanTan's semantic cache is injected here.
        # A temporary fallback is used only so the repository continues
        # to run before TanTan's semantic cache module is connected.
        self.cache = cache

        # Dhano owns the LLM integration.
        self.llm = (
            llm
            or MockLLM(
                default_model=self.config.default_llm_model
            )
        )

        # Andi's evaluator is injected here.
        self.evaluator = evaluator

    def process_query(
        self,
        raw_query: str,
        session_id: Optional[str] = None,
    ) -> PipelineResponse:
        """Process one user query end-to-end."""

        start_time = time.perf_counter()

        clean_query = self._preprocess(raw_query)

        request = QueryRequest(
            query=clean_query,
            session_id=session_id,
        )

        metadata: dict[str, object] = {}

        # ---------------------------------------------------------
        # STEP 1: SEMANTIC CACHE LOOKUP
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
                    decision=self._cache_hit_decision(
                        cache_match
                    ),
                    latency_ms=latency_ms,
                    metadata=metadata,
                )

                self._record_evaluation(
                    request,
                    response,
                    latency_ms,
                )

                return response

        # ---------------------------------------------------------
        # STEP 2: CACHE MISS → LLM
        # ---------------------------------------------------------

        llm_result = self._generate_llm_response(
            clean_query
        )

        # ---------------------------------------------------------
        # STEP 3: STORE LLM RESPONSE
        # ---------------------------------------------------------

        if self.cache is not None:
            self._cache_put(
                query=clean_query,
                response=llm_result.text,
                metadata={
                    "model": llm_result.model_name,
                    "tokens": llm_result.tokens_used,
                    "cost": llm_result.cost,
                },
                response_metadata=metadata,
            )

        # ---------------------------------------------------------
        # STEP 4: BUILD FINAL RESPONSE
        # ---------------------------------------------------------

        latency_ms = self._elapsed_ms(start_time)

        decision = DecisionResult(
            route="DIRECT_LLM",
            estimated_cost=llm_result.cost,
            confidence=1.0,
            reason="Semantic cache miss; request sent to LLM.",
            metadata={
                "selected_model": llm_result.model_name,
                "provider": llm_result.metadata.get(
                    "provider",
                    "unknown",
                ),
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
            },
        )

        self._record_evaluation(
            request,
            response,
            latency_ms,
        )

        return response

    # =============================================================
    # INTERNAL HELPERS
    # =============================================================

    def _preprocess(self, raw_query: str) -> str:
        """Validate and normalize the user query."""

        if raw_query is None:
            raise ValueError("Query cannot be None")

        clean_query = self.preprocessor.preprocess(raw_query)

        if not clean_query:
            raise ValueError("Query cannot be empty")

        return clean_query

    def _cache_get(
        self,
        query: str,
        metadata: dict[str, object],
    ) -> Optional[CacheEntry]:
        """Ask TanTan's semantic cache for a match."""

        try:
            return self.cache.get(
                query=query,
                threshold=self.config.similarity_threshold,
            )

        except Exception as exc:
            # A cache failure should not prevent the LLM from
            # answering the user.
            metadata["cache_lookup_error"] = str(exc)
            return None

    def _cache_put(
        self,
        query: str,
        response: str,
        metadata: dict[str, object],
        response_metadata: dict[str, object],
    ) -> None:
        """Store an LLM response in TanTan's cache."""

        try:
            self.cache.put(
                query=query,
                response=response,
                metadata=metadata,
            )

        except Exception as exc:
            response_metadata["cache_store_error"] = str(exc)

    def _generate_llm_response(self, query: str):
        """Generate an answer using Dhano's LLM service."""

        try:
            return self.llm.generate(
                query=query,
                model=self.config.default_llm_model,
            )

        except Exception as exc:
            raise RuntimeError(
                f"LLM generation failed: {exc}"
            ) from exc

    def _record_evaluation(
        self,
        request: QueryRequest,
        response: PipelineResponse,
        latency_ms: float,
    ) -> None:
        """Send telemetry to Andi's evaluator when available."""

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

    def _cache_hit_decision(
        self,
        cache_match: CacheEntry,
    ) -> DecisionResult:
        """Create the decision metadata for a cache hit."""

        return DecisionResult(
            route="CACHE_HIT",
            estimated_cost=0.0,
            confidence=cache_match.similarity_score,
            reason=(
                "Semantic cache similarity met the configured "
                "threshold."
            ),
            metadata={
                "selected_model": None,
                "provider": "cache",
            },
        )

    @staticmethod
    def _elapsed_ms(start_time: float) -> float:
        """Return elapsed execution time in milliseconds."""

        return (
            time.perf_counter() - start_time
        ) * 1000.0
