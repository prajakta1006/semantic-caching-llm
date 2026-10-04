"""Tests for Praj's cache optimization layer."""

from src.cache_optimizer import CacheOptimizer
from src.semantic_cache import SemanticCache


class FakeClock:
    """Deterministic clock for TTL tests."""

    def __init__(self, value: float = 0.0):
        self.value = value

    def now(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeEmbeddingGenerator:
    """Deterministic embeddings for optimization tests."""

    def __init__(self):
        self.embeddings = {
            "query one": [1.0, 0.0],
            "query two": [0.0, 1.0],
            "query three": [0.7071, 0.7071],
            "query four": [0.7071, -0.7071],
        }

    def encode(self, text: str):
        return self.embeddings.get(
            text.strip().lower(),
            [0.0, 0.0],
        )


def make_cache(
    max_size=3,
    ttl_seconds=None,
    clock=None,
):
    semantic_cache = SemanticCache(
        embedding_generator=FakeEmbeddingGenerator()
    )

    if clock is None:
        return CacheOptimizer(
            cache=semantic_cache,
            max_size=max_size,
            ttl_seconds=ttl_seconds,
        )

    return CacheOptimizer(
        cache=semantic_cache,
        max_size=max_size,
        ttl_seconds=ttl_seconds,
        clock=clock.now,
    )


def test_optimizer_starts_empty():
    cache = make_cache()

    assert cache.size() == 0


def test_put_increases_size():
    cache = make_cache()

    cache.put(
        query="query one",
        response="answer one",
    )

    assert cache.size() == 1


def test_semantic_cache_still_works_through_optimizer():
    cache = make_cache()

    cache.put(
        query="query one",
        response="answer one",
    )

    result = cache.get(
        query="query one",
        threshold=0.90,
    )

    assert result is not None
    assert result.response == "answer one"


def test_frequency_increases_on_cache_hit():
    cache = make_cache()

    cache.put(
        query="query one",
        response="answer one",
    )

    assert cache.get_frequency("query one") == 0

    cache.get(
        query="query one",
        threshold=0.90,
    )

    assert cache.get_frequency("query one") == 1

    cache.get(
        query="query one",
        threshold=0.90,
    )

    assert cache.get_frequency("query one") == 2


def test_lru_evicts_least_recently_used_entry():
    cache = make_cache(max_size=2)

    cache.put(
        query="query one",
        response="answer one",
    )

    cache.put(
        query="query two",
        response="answer two",
    )

    # Make query one recently used.
    result = cache.get(
        query="query one",
        threshold=0.90,
    )

    assert result is not None

    # query two is now the LRU entry.
    cache.put(
        query="query three",
        response="answer three",
    )

    assert cache.size() == 2

    assert cache.get(
        query="query two",
        threshold=0.90,
    ) is None

    assert cache.get(
        query="query one",
        threshold=0.90,
    ) is not None

    assert cache.get(
        query="query three",
        threshold=0.90,
    ) is not None


def test_ttl_removes_expired_entry():
    clock = FakeClock()

    cache = make_cache(
        max_size=10,
        ttl_seconds=10,
        clock=clock,
    )

    cache.put(
        query="query one",
        response="answer one",
    )

    assert cache.size() == 1

    clock.advance(11)

    assert cache.get(
        query="query one",
        threshold=0.90,
    ) is None

    assert cache.size() == 0


def test_ttl_does_not_remove_fresh_entry():
    clock = FakeClock()

    cache = make_cache(
        max_size=10,
        ttl_seconds=10,
        clock=clock,
    )

    cache.put(
        query="query one",
        response="answer one",
    )

    clock.advance(5)

    assert cache.get(
        query="query one",
        threshold=0.90,
    ) is not None

    assert cache.size() == 1


def test_expired_entries_are_removed_before_capacity_eviction():
    clock = FakeClock()

    cache = make_cache(
        max_size=2,
        ttl_seconds=10,
        clock=clock,
    )

    cache.put(
        query="query one",
        response="answer one",
    )

    cache.put(
        query="query two",
        response="answer two",
    )

    clock.advance(11)

    # Both existing entries are expired.
    cache.put(
        query="query three",
        response="answer three",
    )

    assert cache.size() == 1

    assert cache.get(
        query="query three",
        threshold=0.90,
    ) is not None


def test_capacity_is_never_exceeded():
    cache = make_cache(max_size=2)

    cache.put(
        query="query one",
        response="answer one",
    )

    cache.put(
        query="query two",
        response="answer two",
    )

    cache.put(
        query="query three",
        response="answer three",
    )

    assert cache.size() == 2


def test_clear_removes_entries_and_metadata():
    cache = make_cache()

    cache.put(
        query="query one",
        response="answer one",
    )

    cache.get(
        query="query one",
        threshold=0.90,
    )

    assert cache.get_frequency("query one") == 1

    cache.clear()

    assert cache.size() == 0
    assert cache.get_frequency("query one") == 0


def test_last_accessed_changes_after_hit():
    clock = FakeClock()

    cache = make_cache(
        max_size=10,
        clock=clock,
    )

    cache.put(
        query="query one",
        response="answer one",
    )

    initial = cache.get_last_accessed("query one")

    clock.advance(5)

    cache.get(
        query="query one",
        threshold=0.90,
    )

    updated = cache.get_last_accessed("query one")

    assert initial is not None
    assert updated is not None
    assert updated > initial