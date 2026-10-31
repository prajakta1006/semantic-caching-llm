
"""Integration tests for persistent semantic caching.

These tests use an in-memory MongoDB substitute, so they run offline.
They verify persistence across cache instances, semantic matching,
LRU eviction, TTL expiration and database failure handling.
"""

import time

import pytest

from src.cache_optimizer import CacheOptimizer
from src.mongo_cache_store import MongoCacheStore
from src.semantic_cache import SemanticCache


class FakeCollection:
    """In-memory substitute for a MongoDB collection."""

    def __init__(self):
        self.documents = {}

    def replace_one(self, filter_, document, upsert=False):
        key = filter_["_id"]

        if upsert or key in self.documents:
            self.documents[key] = dict(document)

    def find(self, filter_):
        return list(self.documents.values())

    def delete_one(self, filter_):
        self.documents.pop(filter_["_id"], None)

    def delete_many(self, filter_):
        if not filter_:
            self.documents.clear()


class FakeClock:
    """Controllable clock for deterministic TTL testing."""

    def __init__(self):
        self.value = time.time()

    def now(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class FakeEmbeddingGenerator:
    """Deterministic embeddings for semantic cache tests."""

    def encode(self, text):
        embeddings = {
            "what is machine learning?": [1.0, 0.0],
            "explain machine learning": [0.99, 0.01],
            "what is a binary search tree?": [0.0, 1.0],
            "what is operating system scheduling?": [0.0, 0.8],
            "explain dijkstra algorithm": [0.8, 0.6],
        }

        return embeddings.get(
            text.strip().lower(),
            [0.0, 0.0],
        )


def make_persistent_cache(
    collection,
    max_size=3,
    ttl_seconds=None,
    clock=None,
):
    """Create a complete persistent cache using a shared collection."""

    store = MongoCacheStore(collection=collection)

    semantic_cache = SemanticCache(
        embedding_generator=FakeEmbeddingGenerator(),
        store=store,
    )

    optimizer_args = {
        "cache": semantic_cache,
        "max_size": max_size,
        "ttl_seconds": ttl_seconds,
    }

    if clock is not None:
        optimizer_args["clock"] = clock.now

    optimizer = CacheOptimizer(**optimizer_args)

    return optimizer, semantic_cache, store


def test_cache_entries_survive_restart():
    """Saved entries should be restored by a new cache instance."""

    collection = FakeCollection()

    cache1, _, _ = make_persistent_cache(collection)

    cache1.put(
        query="what is machine learning?",
        response="Machine learning is a field of AI.",
    )

    assert cache1.size() == 1

    # Simulate restarting the application.
    cache2, _, _ = make_persistent_cache(collection)

    assert cache2.size() == 1

    entries = cache2.entries()

    assert entries[0].query == "what is machine learning?"
    assert entries[0].response == "Machine learning is a field of AI."


def test_semantic_matching_works_after_restart():
    """Restored embeddings should still support semantic matching."""

    collection = FakeCollection()

    cache1, _, _ = make_persistent_cache(collection)

    cache1.put(
        query="what is machine learning?",
        response="Machine learning is a field of AI.",
    )

    # Create a new cache to simulate application restart.
    cache2, _, _ = make_persistent_cache(collection)

    result = cache2.get(
        query="explain machine learning",
        threshold=0.90,
    )

    assert result is not None
    assert result.response == "Machine learning is a field of AI."
    assert result.similarity_score >= 0.90


def test_access_frequency_survives_restart():
    """Access frequency should be restored from MongoDB."""

    collection = FakeCollection()

    cache1, _, _ = make_persistent_cache(collection)

    cache1.put(
        query="what is machine learning?",
        response="ML answer",
    )

    # Access the entry twice.
    cache1.get(
        query="what is machine learning?",
        threshold=0.90,
    )

    cache1.get(
        query="what is machine learning?",
        threshold=0.90,
    )

    assert cache1.get_frequency(
        "what is machine learning?"
    ) == 2

    # Simulate restart.
    cache2, _, _ = make_persistent_cache(collection)

    assert cache2.get_frequency(
        "what is machine learning?"
    ) == 2


def test_lru_order_survives_restart_and_evicts_correct_entry():
    """The least recently used entry should be evicted after restart."""

    collection = FakeCollection()

    cache1, _, _ = make_persistent_cache(
        collection,
        max_size=2,
    )

    cache1.put(
        query="what is machine learning?",
        response="ML answer",
    )

    cache1.put(
        query="what is a binary search tree?",
        response="BST answer",
    )

    # Make machine learning the most recently used entry.
    cache1.get(
        query="what is machine learning?",
        threshold=0.90,
    )

    # Restart the cache.
    cache2, _, _ = make_persistent_cache(
        collection,
        max_size=2,
    )

    # Adding a third entry should evict the BST entry.
    cache2.put(
        query="what is operating system scheduling?",
        response="OS scheduling answer",
    )

    remaining_queries = {
        entry.query for entry in cache2.entries()
    }

    assert "what is machine learning?" in remaining_queries

    assert (
        "what is operating system scheduling?"
        in remaining_queries
    )

    assert (
        "what is a binary search tree?"
        not in remaining_queries
    )

    # Verify that MongoDB also reflects the eviction.
    persisted_queries = {
        document["query"]
        for document in collection.documents.values()
    }

    assert persisted_queries == remaining_queries


def test_ttl_expiration_deletes_entry_from_persistent_storage():
    """Expired entries should be removed from memory and MongoDB."""

    collection = FakeCollection()
    clock = FakeClock()

    cache1, _, _ = make_persistent_cache(
        collection,
        max_size=3,
        ttl_seconds=10,
        clock=clock,
    )

    cache1.put(
        query="what is machine learning?",
        response="ML answer",
    )

    assert len(collection.documents) == 1

    # Advance beyond the TTL.
    clock.advance(11)

    assert cache1.size() == 0

    # Verify deletion from MongoDB.
    assert len(collection.documents) == 0

    # Simulate restart.
    cache2, _, _ = make_persistent_cache(
        collection,
        max_size=3,
        ttl_seconds=10,
        clock=clock,
    )

    assert cache2.size() == 0


def test_missing_mongodb_uri_raises_clear_error(monkeypatch):
    """Missing MongoDB configuration should fail explicitly."""

    monkeypatch.delenv("MONGODB_URI", raising=False)

    with pytest.raises(
        ValueError,
        match="MONGODB_URI is missing",
    ):
        MongoCacheStore(uri=None)


class BrokenCollection:
    """Collection that simulates a database connection failure."""

    def find(self, filter_):
        raise ConnectionError("Simulated MongoDB connection failure")


def test_database_read_failure_is_not_silently_ignored():
    """A failed database read should propagate during initialization."""

    store = MongoCacheStore(
        collection=BrokenCollection()
    )

    with pytest.raises(
        ConnectionError,
        match="Simulated MongoDB connection failure",
    ):
        SemanticCache(
            embedding_generator=FakeEmbeddingGenerator(),
            store=store,
        )
