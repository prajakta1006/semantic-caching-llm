"""Default foundation components used by TanTan's pipeline.

These are simple mocks/adapters only. They deliberately do not implement
semantic vector search, real cost-benefit routing, or benchmark analysis.
"""

from typing import Optional

from src.config import AppConfig
from src.interfaces import CacheInterface, CascadeInterface, EvaluatorInterface, LLMInterface, QueryPreprocessorInterface
from src.models import CacheEntry, DecisionResult, EvaluationMetrics, LLMResponse, PipelineResponse, QueryRequest


class DefaultPreprocessor(QueryPreprocessorInterface):
    """Basic query cleaning and whitespace normalization."""

    def preprocess(self, query: str) -> str:
        return query.strip()


class ExactMatchCache(CacheInterface):
    """In-memory exact-match cache adapter.

    TODO (Dhano): Replace this with a semantic cache that creates embeddings,
    searches by cosine/FAISS similarity, applies the threshold, and manages
    eviction. The pipeline should continue calling get/put unchanged.
    """

    def __init__(self) -> None:
        self._storage: dict[str, CacheEntry] = {}

    def get(self, query: str, threshold: float) -> Optional[CacheEntry]:
        normalized = query.lower().strip()
        if normalized not in self._storage:
            return None

        entry = self._storage[normalized]
        entry.similarity_score = 1.0
        return entry if entry.similarity_score >= threshold else None

    def put(
        self,
        query: str,
        response: str,
        embedding: Optional[list[float]] = None,
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


class DefaultCascade(CascadeInterface):
    """Placeholder cache-miss routing decision.

    TODO (Praj): Replace this with the cost-benefit cascade algorithm that
    scores query complexity, estimates model costs/quality, and selects the
    appropriate path/model.
    """

    def decide(self, request: QueryRequest, config: AppConfig) -> DecisionResult:
        return DecisionResult(
            route="DIRECT_LLM",
            estimated_cost=0.01,
            confidence=1.0,
            reason="Default TanTan cascade placeholder; cache miss routes to mock/default LLM",
            metadata={
                "selected_model": config.default_llm_model,
                "provider": config.llm_provider,
            },
        )


class MockLLM(LLMInterface):
    """Deterministic mock LLM for demos and tests without external APIs."""

    def __init__(self, default_model: str = "mock-llm-v1") -> None:
        self.default_model = default_model

    def generate(self, query: str, model: Optional[str] = None) -> LLMResponse:
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
            model_name=model or self.default_model,
            tokens_used=len(answer.split()),
            latency_ms=12.5,
            cost=0.002,
        )


class BasicEvaluator(EvaluatorInterface):
    """Small metrics recorder used until Andi's evaluator is plugged in.

    TODO (Andi): Replace this with benchmarking, latency distributions, cost
    analysis, reports, and threshold experiments.
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


# Historical names retained so current demos/tests and teammate imports work.
MockCache = ExactMatchCache
MockDecisionEngine = DefaultCascade
MockEvaluator = BasicEvaluator
