"""Pipeline coordinator for Semantic Caching with Cost-Benefit Cascade Flow.

TanTan owns this orchestration layer. Teammate algorithms plug in through the
interfaces imported below; this module does not implement their algorithms.
"""

import time
from typing import Optional

from src.components import (
    BasicEvaluator,
    DefaultCascade,
    DefaultPreprocessor,
    ExactMatchCache,
    MockCache,
    MockDecisionEngine,
    MockEvaluator,
    MockLLM,
)
from src.config import AppConfig, config as default_config
from src.interfaces import (
    CacheInterface,
    CascadeInterface,
    DecisionInterface,
    EvaluatorInterface,
    LLMInterface,
    QueryPreprocessorInterface,
)
from src.models import CacheEntry, DecisionResult, PipelineResponse, QueryRequest


class SemanticCachingPipeline:
    """Core pipeline coordinator connecting cache, cascade, LLM, and evaluator."""

    def __init__(
        self,
        app_config: Optional[AppConfig] = None,
        preprocessor: Optional[QueryPreprocessorInterface] = None,
        cache: Optional[CacheInterface] = None,
        cascade: Optional[CascadeInterface] = None,
        decision_engine: Optional[DecisionInterface] = None,
        llm: Optional[LLMInterface] = None,
        evaluator: Optional[EvaluatorInterface] = None,
    ) -> None:
        self.config = app_config or default_config
        self.preprocessor = preprocessor or DefaultPreprocessor()
        self.cache = cache or ExactMatchCache()
        self.cascade = cascade or decision_engine or DefaultCascade()
        self.decision_engine = self.cascade
        self.llm = llm or MockLLM(default_model=self.config.default_llm_model)
        self.evaluator = evaluator or BasicEvaluator()

    def process_query(self, raw_query: str, session_id: Optional[str] = None) -> PipelineResponse:
        """Execute the end-to-end foundation flow.

        User Query -> preprocessing -> cache.get -> on miss cascade.decide ->
        llm.generate -> cache.put -> evaluator.record -> response.
        """
        start_time = time.perf_counter()
        metadata: dict[str, object] = {}

        clean_query = self._preprocess(raw_query)
        request = QueryRequest(query=clean_query, session_id=session_id)

        cache_match = self._safe_cache_get(clean_query, metadata)
        if cache_match is not None:
            latency_ms = self._elapsed_ms(start_time)
            decision = self._cache_hit_decision(cache_match)
            response = PipelineResponse(
                query=raw_query,
                response_text=cache_match.response,
                source="cache",
                similarity_score=cache_match.similarity_score,
                decision=decision,
                latency_ms=latency_ms,
                metadata=metadata,
            )
            self._safe_record(request, response, latency_ms)
            return response

        decision = self._decide(request)
        selected_model = self._selected_model(decision)
        llm_result = self._generate(clean_query, selected_model)

        cache_metadata = {
            "tokens": llm_result.tokens_used,
            "cost": llm_result.cost,
            "model": llm_result.model_name,
        }
        self._safe_cache_put(clean_query, llm_result.text, cache_metadata, metadata)

        latency_ms = self._elapsed_ms(start_time)
        response = PipelineResponse(
            query=raw_query,
            response_text=llm_result.text,
            source="llm",
            similarity_score=0.0,
            decision=decision,
            latency_ms=latency_ms,
            metadata={
                **metadata,
                "model": llm_result.model_name,
                "cost": llm_result.cost,
            },
        )
        self._safe_record(request, response, latency_ms)
        return response

    def _preprocess(self, raw_query: str) -> str:
        if raw_query is None:
            raise ValueError("Query cannot be None")

        clean_query = self.preprocessor.preprocess(raw_query)
        if not clean_query:
            raise ValueError("Query cannot be empty")
        return clean_query

    def _safe_cache_get(self, query: str, metadata: dict[str, object]) -> Optional[CacheEntry]:
        try:
            return self.cache.get(query, threshold=self.config.similarity_threshold)
        except Exception as exc:
            metadata["cache_lookup_error"] = str(exc)
            return None

    def _safe_cache_put(
        self,
        query: str,
        response: str,
        cache_metadata: dict[str, object],
        response_metadata: dict[str, object],
    ) -> None:
        try:
            self.cache.put(query=query, response=response, metadata=cache_metadata)
        except Exception as exc:
            response_metadata["cache_store_error"] = str(exc)

    def _decide(self, request: QueryRequest) -> DecisionResult:
        try:
            return self.cascade.decide(request, self.config)
        except Exception as exc:
            raise RuntimeError(f"Cascade decision failed: {exc}") from exc

    def _generate(self, query: str, selected_model: Optional[str]):
        try:
            return self.llm.generate(query, model=selected_model)
        except Exception as exc:
            raise RuntimeError(f"LLM generation failed: {exc}") from exc

    def _safe_record(
        self,
        request: QueryRequest,
        response: PipelineResponse,
        latency_ms: float,
    ) -> None:
        try:
            self.evaluator.record(request, response, latency_ms)
        except Exception as exc:
            response.metadata["evaluator_error"] = str(exc)

    def _cache_hit_decision(self, cache_match: CacheEntry) -> DecisionResult:
        return DecisionResult(
            route="CACHE_HIT",
            estimated_cost=0.0,
            confidence=cache_match.similarity_score,
            reason=(
                f"Cache similarity ({cache_match.similarity_score:.2f}) meets "
                f"threshold ({self.config.similarity_threshold:.2f})"
            ),
            metadata={"selected_model": None, "provider": "cache"},
        )

    def _selected_model(self, decision: DecisionResult) -> str:
        model = decision.metadata.get("selected_model") if decision.metadata else None
        return str(model or self.config.default_llm_model)

    @staticmethod
    def _elapsed_ms(start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000.0
