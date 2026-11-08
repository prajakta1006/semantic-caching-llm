
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

Those responsibilities belong to Praj, Anuja, and Dhano.
"""

from typing import List, Optional

from src.config import config as default_config
from src.embeddings import EmbeddingGenerator
from src.interfaces import CacheInterface
from src.models import CacheEntry
from src.similarity import cosine_similarity
import math

class SemanticCache(CacheInterface):
    """In-memory semantic cache using vector similarity."""

    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        store=None,
    ) -> None:
        """Initialize the semantic cache with optional persistence."""

        model_name = (
            embedding_model_name
            or default_config.embedding_model_name
        )

        self.embedding_generator = (
            embedding_generator
            or EmbeddingGenerator(model_name)
        )

        # Optional persistence backend, such as MongoCacheStore.
        # When no store is supplied, the cache remains in memory only.
        self.store = store
        self._entries: List[CacheEntry] = []

        # Restore previously persisted entries when a store is supplied.
        if self.store is not None:
            self._entries = self.store.load_entries()

    def _persist_entry(self, entry: CacheEntry) -> None:
        """Persist only the newly added or updated cache entry."""
        if self.store is not None:
            self.store.save_entry(entry)

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
        """Store a validated query-response pair in the semantic cache.

        If an embedding is not supplied, it is generated automatically.
        """

    # Validate query
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query cannot be empty")

    # Validate response
        if not isinstance(response, str):
            raise TypeError("response must be a string")

        if not response.strip():
            raise ValueError("response cannot be empty")

    # Generate embedding if not supplied
        if embedding is None:
            embedding = self.embedding_generator.encode(query)

    # Validate embedding
        if not isinstance(embedding, (list, tuple)):
            raise TypeError("embedding must be a list or tuple of numbers")

        if not embedding:
            raise ValueError("embedding cannot be empty")

        validated_embedding = []

        for value in embedding:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("embedding values must be numeric")

            if not math.isfinite(value):
                raise ValueError("embedding values must be finite")

            validated_embedding.append(float(value))

    # Validate metadata
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary")

    # Create cache entry
        entry = CacheEntry(
            query=query,
            response=response,
            embedding=validated_embedding,
            similarity_score=0.0,
            metadata=dict(metadata or {}),
        )

    # Store in memory and MongoDB
        self._entries.append(entry)
        self._persist_entry(entry)


    def size(self) -> int:
        """Return the number of cached entries."""

        return len(self._entries)

    def clear(self) -> None:
        """Remove all cached entries.

        This helper is useful for tests and experiments.
        """

        self._entries.clear()

        if self.store is not None:
            self.store.clear()


    def remove(self, query: str) -> None:
        """Remove an exact query from memory and persistent storage."""
        self._entries = [
            entry for entry in self._entries
            if entry.query != query
        ]

        if self.store is not None:
            self.store.delete_entry(query)

    def entries(self) -> List[CacheEntry]:
        """Return a copy of the current cache entries.

        This is primarily useful for inspection and experiments.
        """

        return list(self._entries)

    def update_metadata(
        self,
        query: str,
        last_accessed: float,
        access_count: int,
    ) -> None:
        """Update and persist the optimizer metadata for an entry."""

        for entry in self._entries:
            if entry.query == query:
                entry.last_accessed = last_accessed
                entry.access_count = access_count
                self._persist_entry(entry)
                return