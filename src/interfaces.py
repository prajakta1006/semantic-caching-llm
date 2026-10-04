"""Shared interfaces for the semantic caching pipeline.

These interfaces define the stable boundaries between team members.

Ownership:
    - TanTan: semantic cache, embeddings, similarity search
    - Dhano: LLM integration and pipeline orchestration
    - Praj: cache optimization policies
    - Andi: evaluation, datasets, and benchmarking

Important:
    The pipeline depends only on these interfaces. It does not depend on
    another teammate's implementation details.
"""

from abc import ABC, abstractmethod
from typing import Optional

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

    The implementation may internally use:
        - embeddings
        - cosine similarity
        - FAISS
        - LRU
        - TTL
        - frequency
        - eviction policies

    None of those implementation details should appear in the pipeline.
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
        """Return the current number of cached entries."""


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
