"""Cache optimization layer owned by Praj.

This module adds cache-management policies around TanTan's
SemanticCache without modifying TanTan's semantic-search logic.

Responsibilities:
    - Maximum cache capacity
    - LRU tracking
    - TTL expiration
    - Access-frequency tracking
    - Eviction policy
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, List, Optional

from src.interfaces import CacheInterface
from src.models import CacheEntry


@dataclass
class CacheMetadata:
    """Optimization metadata maintained separately from CacheEntry."""

    created_at: float
    last_accessed: float
    access_count: int = 0


class CacheOptimizer(CacheInterface):
    """Wrap a semantic cache with LRU, TTL, frequency and capacity policies."""

    def __init__(
        self,
        cache: CacheInterface,
        max_size: int = 100,
        ttl_seconds: Optional[float] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be greater than 0")

        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError(
                "ttl_seconds must be greater than 0 or None"
            )

        self.cache = cache
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.clock = clock

        self._metadata: dict[str, CacheMetadata] = {}

        # Oldest/LRU entry is at the beginning.
        # Newest/MRU entry is at the end.
        self._lru: OrderedDict[str, None] = OrderedDict()

    # ------------------------------------------------------------------
    # CacheInterface
    # ------------------------------------------------------------------

    def get(
        self,
        query: str,
        threshold: float,
    ) -> Optional[CacheEntry]:
        """Return a semantic cache hit after TTL validation."""

        self._remove_expired_entries()

        entry = self.cache.get(
            query=query,
            threshold=threshold,
        )

        if entry is None:
            return None

        key = entry.query
        now = self.clock()

        metadata = self._metadata.get(key)

        if metadata is None:
            metadata = CacheMetadata(
                created_at=now,
                last_accessed=now,
                access_count=0,
            )

            self._metadata[key] = metadata
            self._lru[key] = None

        metadata.access_count += 1
        metadata.last_accessed = now

        self._lru.move_to_end(key)

        return entry

    def put(
        self,
        query: str,
        response: str,
        embedding: Optional[list[float]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Store an entry and enforce optimization policies."""

        self._remove_expired_entries()

        existing_key = self._find_exact_query(query)

        if existing_key is not None:
            self._remove_entry(existing_key)

        self.cache.put(
            query=query,
            response=response,
            embedding=embedding,
            metadata=metadata,
        )

        now = self.clock()

        self._metadata[query] = CacheMetadata(
            created_at=now,
            last_accessed=now,
            access_count=0,
        )

        self._lru[query] = None
        self._lru.move_to_end(query)

        self._enforce_capacity()

    def size(self) -> int:
        """Return the number of currently valid cached entries."""

        self._remove_expired_entries()

        return self.cache.size()

    # ------------------------------------------------------------------
    # Public optimization helpers
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear the wrapped cache and optimization metadata."""

        clear_method = getattr(self.cache, "clear", None)

        if not callable(clear_method):
            raise AttributeError(
                "Wrapped cache must provide clear()"
            )

        clear_method()

        self._metadata.clear()
        self._lru.clear()

    def entries(self) -> List[CacheEntry]:
        """Return currently valid cache entries."""

        self._remove_expired_entries()

        entries_method = getattr(self.cache, "entries", None)

        if not callable(entries_method):
            raise AttributeError(
                "Wrapped cache must provide entries()"
            )

        return list(entries_method())

    def get_frequency(self, query: str) -> int:
        """Return the successful access count for a cached query."""

        self._remove_expired_entries()

        metadata = self._metadata.get(query)

        if metadata is None:
            return 0

        return metadata.access_count

    def get_last_accessed(
        self,
        query: str,
    ) -> Optional[float]:
        """Return the last successful-access timestamp."""

        metadata = self._metadata.get(query)

        if metadata is None:
            return None

        return metadata.last_accessed

    def get_cache_metadata(
        self,
        query: str,
    ) -> Optional[CacheMetadata]:
        """Return a copy of optimization metadata."""

        metadata = self._metadata.get(query)

        if metadata is None:
            return None

        return CacheMetadata(
            created_at=metadata.created_at,
            last_accessed=metadata.last_accessed,
            access_count=metadata.access_count,
        )

    # ------------------------------------------------------------------
    # TTL
    # ------------------------------------------------------------------

    def _is_expired(
        self,
        query: str,
        now: Optional[float] = None,
    ) -> bool:
        """Return whether an entry has exceeded its TTL."""

        if self.ttl_seconds is None:
            return False

        metadata = self._metadata.get(query)

        if metadata is None:
            return False

        current_time = (
            self.clock()
            if now is None
            else now
        )

        return (
            current_time - metadata.created_at
            >= self.ttl_seconds
        )

    def _remove_expired_entries(self) -> None:
        """Remove every entry whose TTL has expired."""

        if self.ttl_seconds is None:
            return

        now = self.clock()

        expired_queries = [
            query
            for query in self._metadata
            if self._is_expired(query, now)
        ]

        for query in expired_queries:
            self._remove_entry(query)

    # ------------------------------------------------------------------
    # LRU / eviction
    # ------------------------------------------------------------------

    def _enforce_capacity(self) -> None:
        """Evict entries until cache size is within max_size."""

        self._remove_expired_entries()

        while self.cache.size() > self.max_size:

            victim = self._select_eviction_candidate()

            if victim is None:
                break

            self._remove_entry(victim)

    def _select_eviction_candidate(self) -> Optional[str]:
        """Return the least recently used query."""

        if not self._lru:
            return None

        return next(iter(self._lru))

    def _remove_entry(self, query: str) -> None:
        """Remove an entry from both optimization and cache storage."""

        entries_method = getattr(self.cache, "entries", None)

        if not callable(entries_method):
            raise AttributeError(
                "Wrapped cache must provide entries()"
            )

        entries = list(entries_method())

        remaining = [
            entry
            for entry in entries
            if entry.query != query
        ]

        clear_method = getattr(self.cache, "clear", None)

        if not callable(clear_method):
            raise AttributeError(
                "Wrapped cache must provide clear()"
            )

        clear_method()

        for entry in remaining:
            self.cache.put(
                query=entry.query,
                response=entry.response,
                embedding=entry.embedding,
                metadata=entry.metadata,
            )

        self._metadata.pop(query, None)
        self._lru.pop(query, None)

    def _find_exact_query(
        self,
        query: str,
    ) -> Optional[str]:
        """Find an exact query currently stored in the cache."""

        entries_method = getattr(self.cache, "entries", None)

        if not callable(entries_method):
            return None

        for entry in entries_method():
            if entry.query == query:
                return entry.query

        return None