"""Tests for TanTan's semantic cache."""

import pytest

from src.embeddings import EmbeddingGenerator
from src.semantic_cache import SemanticCache


class FakeEmbeddingGenerator:
    """Deterministic embedding generator for unit tests."""

    def __init__(self):
        self.embeddings = {
            "what is machine learning?": [1.0, 0.0],
            "explain machine learning": [0.99, 0.01],
            "what is pizza?": [0.0, 1.0],
        }

    def encode(self, text: str):
        """Return a deterministic vector."""

        if text in self.embeddings:
            return self.embeddings[text]

        return [0.0, 0.0]


def test_semantic_cache_implements_cache_interface():

    cache = SemanticCache(
        embedding_generator=FakeEmbeddingGenerator()
    )

    assert cache.size() == 0


def test_cache_put_increases_size():

    cache = SemanticCache(
        embedding_generator=FakeEmbeddingGenerator()
    )

    cache.put(
        query="what is machine learning?",
        response="Machine learning is a field of AI.",
    )

    assert cache.size() == 1


def test_semantic_cache_returns_similar_query():

    cache = SemanticCache(
        embedding_generator=FakeEmbeddingGenerator()
    )

    cache.put(
        query="what is machine learning?",
        response="Machine learning is a field of AI.",
    )

    result = cache.get(
        query="explain machine learning",
        threshold=0.90,
    )

    assert result is not None

    assert result.response == (
        "Machine learning is a field of AI."
    )

    assert result.similarity_score >= 0.90


def test_semantic_cache_misses_when_similarity_is_low():

    cache = SemanticCache(
        embedding_generator=FakeEmbeddingGenerator()
    )

    cache.put(
        query="what is machine learning?",
        response="Machine learning is a field of AI.",
    )

    result = cache.get(
        query="what is pizza?",
        threshold=0.90,
    )

    assert result is None


def test_semantic_cache_returns_best_match():

    cache = SemanticCache(
        embedding_generator=FakeEmbeddingGenerator()
    )

    cache.put(
        query="what is machine learning?",
        response="ML answer",
    )

    cache.put(
        query="what is pizza?",
        response="Pizza answer",
    )

    result = cache.get(
        query="explain machine learning",
        threshold=0.90,
    )

    assert result is not None

    assert result.response == "ML answer"


def test_empty_cache_returns_miss():

    cache = SemanticCache(
        embedding_generator=FakeEmbeddingGenerator()
    )

    result = cache.get(
        query="hello",
        threshold=0.85,
    )

    assert result is None


def test_invalid_threshold_is_rejected():

    cache = SemanticCache(
        embedding_generator=FakeEmbeddingGenerator()
    )

    with pytest.raises(ValueError):

        cache.get(
            query="hello",
            threshold=1.5,
        )


def test_clear_removes_entries():

    cache = SemanticCache(
        embedding_generator=FakeEmbeddingGenerator()
    )

    cache.put(
        query="what is machine learning?",
        response="ML answer",
    )

    assert cache.size() == 1

    cache.clear()

    assert cache.size() == 0
