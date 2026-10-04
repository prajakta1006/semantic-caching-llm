"""Shared component contracts for the semantic caching pipeline.

These interfaces define stable boundaries between team members.

Ownership:
    - TanTan: semantic cache, embeddings, similarity
    - Dhano: LLM integration and pipeline orchestration
    - Praj: cache optimization policies
    - Andi: evaluation and benchmarking
"""

from abc import ABC, abstractmethod
from typing import Optional

from src.config import AppConfig
from src.models import (
    CacheEntry,
    EvaluationMetrics,
    LLMResponse,
    PipelineResponse,
    QueryRequest,
)


class QueryPreprocessorInterface(ABC):
    """Contract for query normalization and preprocessing."""

    @abstractmethod
    def preprocess(self, query: str) -> str:
        """Clean and normalize the input query."""


class CacheInterface(ABC):
    """Stable contract for TanTan's semantic cache.

    The pipeline only knows this interface.
    The implementation may internally use:
        - embeddings
        - cosine similarity
        - FAISS
        - LRU/TTL/frequency policies
    """

    @abstractmethod
    def get(
        self,
        query: str,
        threshold: float,
    ) -> Optional[CacheEntry]:
        """Return the best sufficiently similar cached entry."""

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
        """Return the current cache size."""

    # Compatibility aliases
    def lookup(
        self,
        query: str,
        threshold: float,
    ) -> Optional[CacheEntry]:
        """Backward-compatible alias for get()."""
        return self.get(query, threshold)

    def insert(
        self,
        query: str,
        response: str,
        embedding: Optional[list[float]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Backward-compatible alias for put()."""
        self.put(
            query=query,
            response=response,
            embedding=embedding,
            metadata=metadata,
        )


class LLMInterface(ABC):
    """Contract owned by Dhano for LLM integration."""

    @abstractmethod
    def generate(
        self,
        query: str,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """Generate an answer for a query."""


class EvaluatorInterface(ABC):
    """Contract for Andi's evaluation module."""

    @abstractmethod
    def record(
        self,
        request: QueryRequest,
        response: PipelineResponse,
        latency_ms: float,
    ) -> None:
        """Record one pipeline execution."""

    @abstractmethod
    def get_metrics(self) -> EvaluationMetrics:
        """Return accumulated evaluation metrics."""


# Kept only for compatibility with the existing foundation.
class CascadeInterface(ABC):
    """Optional decision/routing extension.

    This is not a separate core team responsibility anymore.
    It can remain as an extension point without forcing cascade
    logic into Dhano's pipeline.
    """

    @abstractmethod
    def decide(
        self,
        request: QueryRequest,
        config: AppConfig,
    ):
        """Return an optional routing decision."""


DecisionInterface = CascadeInterface
