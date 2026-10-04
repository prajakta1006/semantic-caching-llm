"""Component contracts for the semantic caching pipeline.

These interfaces are intentionally small so each teammate can replace one
component without rewriting the pipeline coordinator.
"""

from abc import ABC, abstractmethod
from typing import Optional

from src.config import AppConfig
from src.models import CacheEntry, DecisionResult, EvaluationMetrics, LLMResponse, PipelineResponse, QueryRequest


class QueryPreprocessorInterface(ABC):
    """Contract for query normalization and preprocessing."""

    @abstractmethod
    def preprocess(self, query: str) -> str:
        """Clean and normalize the input query."""


class CacheInterface(ABC):
    """Contract for Dhano's semantic cache implementation."""

    @abstractmethod
    def get(self, query: str, threshold: float) -> Optional[CacheEntry]:
        """Return a sufficiently similar cached entry, or None on miss."""

    @abstractmethod
    def put(
        self,
        query: str,
        response: str,
        embedding: Optional[list[float]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Store a query-response pair."""

    @abstractmethod
    def size(self) -> int:
        """Return the current number of cached entries."""

    def lookup(self, query: str, threshold: float) -> Optional[CacheEntry]:
        """Backward-compatible alias for older foundation code."""
        return self.get(query, threshold)

    def insert(
        self,
        query: str,
        response: str,
        embedding: Optional[list[float]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Backward-compatible alias for older foundation code."""
        self.put(query, response, embedding=embedding, metadata=metadata)


class CascadeInterface(ABC):
    """Contract for Praj's cascade / cost-benefit decision engine."""

    @abstractmethod
    def decide(self, request: QueryRequest, config: AppConfig) -> DecisionResult:
        """Choose the model/path for a cache miss."""


class LLMInterface(ABC):
    """Contract for mock or future API-backed LLM providers."""

    @abstractmethod
    def generate(self, query: str, model: Optional[str] = None) -> LLMResponse:
        """Generate a response for the query."""


class EvaluatorInterface(ABC):
    """Contract for Andi's evaluation and metrics module."""

    @abstractmethod
    def record(
        self,
        request: QueryRequest,
        response: PipelineResponse,
        latency_ms: float,
    ) -> None:
        """Record execution telemetry."""

    @abstractmethod
    def get_metrics(self) -> EvaluationMetrics:
        """Return accumulated metrics."""


# Historical name retained so existing teammate imports continue to work.
DecisionInterface = CascadeInterface
