"""Pipeline Foundation for Semantic Caching with Cost-Benefit Cascade Flow.

This module provides the modular architecture and extension interfaces for:
- Person 2 (Dhano): Semantic Cache Module
- Person 3 (Praj): Cascade & Decision Logic
- Person 4 (Andi): Evaluation & Metrics

Maintained by Person 1 (TanTan).
"""

from abc import ABC, abstractmethod
import time
from typing import Optional

from src.config import AppConfig, config as default_config
from src.models import (
    CacheEntry,
    DecisionResult,
    EvaluationMetrics,
    LLMResponse,
    PipelineResponse,
    QueryRequest,
)


# =============================================================================
# INTERFACES & EXTENSION POINTS
# =============================================================================


class QueryPreprocessorInterface(ABC):
    """Abstract interface for query normalization and preprocessing."""

    @abstractmethod
    def preprocess(self, query: str) -> str:
        """Clean and normalize the input query."""
        pass


class CacheInterface(ABC):
    """Abstract interface for Semantic Cache Module (Dhano's extension point)."""

    @abstractmethod
    def lookup(self, query: str, threshold: float) -> Optional[CacheEntry]:
        """Search cache for semantically similar query above threshold."""
        pass

    @abstractmethod
    def insert(
        self,
        query: str,
        response: str,
        embedding: Optional[list] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Store query and response into semantic cache."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return current number of items in cache."""
        pass


class DecisionInterface(ABC):
    """Abstract interface for Cascade / Cost-Benefit Decision Engine (Praj's extension point)."""

    @abstractmethod
    def decide(
        self,
        request: QueryRequest,
        cache_match: Optional[CacheEntry],
        config: AppConfig,
    ) -> DecisionResult:
        """Evaluate cost-benefit trade-offs and select execution path."""
        pass


class LLMInterface(ABC):
    """Abstract interface for LLM / Generation provider."""

    @abstractmethod
    def generate(self, query: str, decision: Optional[DecisionResult] = None) -> LLMResponse:
        """Generate response via LLM inference or fallback."""
        pass


class EvaluatorInterface(ABC):
    """Abstract interface for Evaluation and Metrics Module (Andi's extension point)."""

    @abstractmethod
    def record(
        self,
        request: QueryRequest,
        response: PipelineResponse,
        latency_ms: float,
    ) -> None:
        """Record telemetry, latency, hit/miss metrics."""
        pass

    @abstractmethod
    def get_metrics(self) -> EvaluationMetrics:
        """Compute and return accumulated evaluation metrics."""
        pass


# =============================================================================
# PLACEHOLDER / MOCK IMPLEMENTATIONS (FOUNDATION RUNTIME)
# =============================================================================


class DefaultPreprocessor(QueryPreprocessorInterface):
    """Basic query cleaning and whitespace normalization."""

    def preprocess(self, query: str) -> str:
        return query.strip()


class MockCache(CacheInterface):
    """Placeholder cache foundation.

    TODO (Dhano): Replace with full vector embedding search, FAISS/cosine similarity,
    and cache eviction strategies in the semantic cache module.
    """

    def __init__(self) -> None:
        self._storage: dict[str, CacheEntry] = {}

    def lookup(self, query: str, threshold: float) -> Optional[CacheEntry]:
        # Exact match placeholder until Dhano implements vector embeddings
        normalized = query.lower().strip()
        if normalized in self._storage:
            entry = self._storage[normalized]
            entry.similarity_score = 1.0
            return entry
        return None

    def insert(
        self,
        query: str,
        response: str,
        embedding: Optional[list] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        normalized = query.lower().strip()
        self._storage[normalized] = CacheEntry(
            query=query,
            response=response,
            embedding=embedding,
            similarity_score=1.0,
            metadata=metadata or {},
        )

    def size(self) -> int:
        return len(self._storage)


class MockDecisionEngine(DecisionInterface):
    """Placeholder cascade decision engine.

    TODO (Praj): Implement cost-benefit cascade algorithms, query complexity scoring,
    and adaptive routing thresholds.
    """

    def decide(
        self,
        request: QueryRequest,
        cache_match: Optional[CacheEntry],
        config: AppConfig,
    ) -> DecisionResult:
        if cache_match is not None and cache_match.similarity_score >= config.similarity_threshold:
            return DecisionResult(
                route="CACHE_HIT",
                estimated_cost=0.0,
                confidence=cache_match.similarity_score,
                reason=f"Semantic cache similarity ({cache_match.similarity_score:.2f}) meets threshold ({config.similarity_threshold:.2f})",
            )
        return DecisionResult(
            route="DIRECT_LLM",
            estimated_cost=0.01,
            confidence=1.0,
            reason="Cache miss or low similarity; routing to LLM",
        )


class MockLLM(LLMInterface):
    """Deterministic mock LLM for foundation testing without external API requirements."""

    def generate(self, query: str, decision: Optional[DecisionResult] = None) -> LLMResponse:
        mock_answers = {
            "what is machine learning?": "Machine learning is a field of AI focusing on building applications that learn from data and improve accuracy over time without being explicitly programmed.",
            "what is semantic caching?": "Semantic caching stores query-response pairs using vector embeddings so that semantically similar queries can reuse cached answers.",
        }
        normalized = query.lower().strip().rstrip("?")
        normalized_with_q = normalized + "?"
        
        answer = (
            mock_answers.get(normalized)
            or mock_answers.get(normalized_with_q)
            or f"Simulated LLM response for query: '{query}'"
        )
        return LLMResponse(
            text=answer,
            model_name="mock-llm-v1",
            tokens_used=len(answer.split()),
            latency_ms=12.5,
            cost=0.002,
        )


class MockEvaluator(EvaluatorInterface):
    """Placeholder metrics aggregator.

    TODO (Andi): Implement comprehensive benchmarking, latency distribution tracking,
    cost reduction calculations, and statistical reports.
    """

    def __init__(self) -> None:
        self.metrics = EvaluationMetrics()

    def record(
        self,
        request: QueryRequest,
        response: PipelineResponse,
        latency_ms: float,
    ) -> None:
        self.metrics.total_queries += 1
        if response.source == "cache":
            self.metrics.cache_hits += 1
            self.metrics.total_cost_saved += 0.002
        else:
            self.metrics.cache_misses += 1

        self.metrics.hit_rate = (
            self.metrics.cache_hits / self.metrics.total_queries
            if self.metrics.total_queries > 0
            else 0.0
        )
        self.metrics.avg_latency_ms = (
            (self.metrics.avg_latency_ms * (self.metrics.total_queries - 1) + latency_ms)
            / self.metrics.total_queries
        )

    def get_metrics(self) -> EvaluationMetrics:
        return self.metrics


# =============================================================================
# MAIN PIPELINE
# =============================================================================


class SemanticCachingPipeline:
    """Core pipeline coordinator connecting all modules."""

    def __init__(
        self,
        app_config: Optional[AppConfig] = None,
        preprocessor: Optional[QueryPreprocessorInterface] = None,
        cache: Optional[CacheInterface] = None,
        decision_engine: Optional[DecisionInterface] = None,
        llm: Optional[LLMInterface] = None,
        evaluator: Optional[EvaluatorInterface] = None,
    ) -> None:
        self.config = app_config or default_config
        self.preprocessor = preprocessor or DefaultPreprocessor()
        self.cache = cache or MockCache()
        self.decision_engine = decision_engine or MockDecisionEngine()
        self.llm = llm or MockLLM()
        self.evaluator = evaluator or MockEvaluator()

    def process_query(self, raw_query: str, session_id: Optional[str] = None) -> PipelineResponse:
        """Execute the end-to-end semantic caching pipeline flow:

        1. Preprocess Query
        2. Semantic Cache Check (Dhano's module)
        3. Cache Hit Check -> Return if hit
        4. Cascade / Cost-Benefit Decision (Praj's module)
        5. LLM Call
        6. Store Result in Cache (Dhano's module)
        7. Evaluation / Metrics (Andi's module)
        """
        start_time = time.perf_counter()

        # Step 1: Query Preprocessing
        clean_query = self.preprocessor.preprocess(raw_query)
        request = QueryRequest(query=clean_query, session_id=session_id)

        # Step 2: Semantic Cache Lookup (TODO: Dhano)
        cache_match = self.cache.lookup(clean_query, threshold=self.config.similarity_threshold)

        # Step 3 & 4: Cascade / Cost-Benefit Decision (TODO: Praj)
        decision = self.decision_engine.decide(request, cache_match, self.config)

        if decision.route == "CACHE_HIT" and cache_match is not None:
            # Step 3: Cache Hit Path
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            response = PipelineResponse(
                query=raw_query,
                response_text=cache_match.response,
                source="cache",
                similarity_score=cache_match.similarity_score,
                decision=decision,
                latency_ms=latency_ms,
            )
        else:
            # Step 5: Cache Miss -> LLM Call
            llm_result = self.llm.generate(clean_query, decision=decision)

            # Step 6: Store Result in Cache (TODO: Dhano)
            self.cache.insert(
                query=clean_query,
                response=llm_result.text,
                metadata={"tokens": llm_result.tokens_used, "cost": llm_result.cost},
            )

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            response = PipelineResponse(
                query=raw_query,
                response_text=llm_result.text,
                source="llm",
                similarity_score=cache_match.similarity_score if cache_match else 0.0,
                decision=decision,
                latency_ms=latency_ms,
                metadata={"model": llm_result.model_name, "cost": llm_result.cost},
            )

        # Step 7: Evaluation & Metrics Recording (TODO: Andi)
        self.evaluator.record(request, response, latency_ms)

        return response
