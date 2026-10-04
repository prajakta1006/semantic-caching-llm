"""Semantic cache implementation.

Owned by TanTan.

Responsibilities:
    - Generate query embeddings
    - Store query-response pairs
    - Perform semantic similarity search
    - Return the best matching cached response
    - Apply similarity threshold

This module intentionally does NOT implement:
    - LRU
    - TTL
    - frequency-based eviction
    - evaluation
    - LLM calls

Those responsibilities belong to Praj, Andi, and Dhano.
"""

from typing import List, Optional

from src.config import config as default_config
from src.embeddings import EmbeddingGenerator
from src.interfaces import CacheInterface
from src.models import CacheEntry
from src.similarity import cosine_similarity


class SemanticCache(CacheInterface):
    """In-memory semantic cache using vector similarity."""

    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
    ) -> None:
        """Initialize the semantic cache.

        Args:
            embedding_model_name:
                Name of the SentenceTransformer model.

            embedding_generator:
                Optional injected embedding generator.
                This is useful for testing and future extensions.
        """

        model_name = (
            embedding_model_name
            or default_config.embedding_model_name
        )

        self.embedding_generator = (
            embedding_generator
            or EmbeddingGenerator(model_name)
        )

        self._entries: List[CacheEntry] = []

    def get(
        self,
        query: str,
        threshold: float,
    ) -> Optional[CacheEntry]:
        """Return the best sufficiently similar cached entry.

        Args:
            query:
                User query to search for.

            threshold:
                Minimum similarity required for a cache hit.

        Returns:
            The best matching CacheEntry when similarity >= threshold.
            None when no sufficiently similar entry exists.
        """

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0.0 and 1.0"
            )

        if not self._entries:
            return None

        query_embedding = (
            self.embedding_generator.encode(query)
        )

        best_entry: Optional[CacheEntry] = None
        best_score = -1.0

        for entry in self._entries:

            if entry.embedding is None:
                continue

            score = cosine_similarity(
                query_embedding,
                entry.embedding,
            )

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is None:
            return None

        if best_score < threshold:
            return None

        # Return a new CacheEntry so the stored entry is not
        # modified when reporting the similarity score.
        return CacheEntry(
            query=best_entry.query,
            response=best_entry.response,
            embedding=best_entry.embedding,
            similarity_score=best_score,
            timestamp=best_entry.timestamp,
            metadata=dict(best_entry.metadata),
        )

    def put(
        self,
        query: str,
        response: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Store a query-response pair in the semantic cache.

        If an embedding is not supplied, it is generated automatically.
        """

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if not isinstance(response, str):
            raise TypeError(
                "response must be a string"
            )

        if embedding is None:
            embedding = (
                self.embedding_generator.encode(query)
            )

        entry = CacheEntry(
            query=query,
            response=response,
            embedding=list(embedding),
            similarity_score=0.0,
            metadata=dict(metadata or {}),
        )

        self._entries.append(entry)

    def size(self) -> int:
        """Return the number of cached entries."""

        return len(self._entries)

    def clear(self) -> None:
        """Remove all cached entries.

        This helper is useful for tests and experiments.
        """

        self._entries.clear()

    def entries(self) -> List[CacheEntry]:
        """Return a copy of the current cache entries.

        This is primarily useful for inspection and experiments.
        """

        return list(self._entries)
