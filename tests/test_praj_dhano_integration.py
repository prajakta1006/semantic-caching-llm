"""Integration tests for Praj's cache optimizer with Dhano's pipeline."""

from src.cache_optimizer import CacheOptimizer
from src.models import LLMResponse
from src.pipeline import SemanticCachingPipeline
from src.semantic_cache import SemanticCache


class FakeEmbeddingGenerator:
    """Deterministic embeddings for integration testing."""

    def __init__(self):
        self.embeddings = {
            "what is caching?": [1.0, 0.0],
            "explain caching": [1.0, 0.0],
            "what is machine learning?": [0.0, 1.0],
        }

    def encode(self, text: str):
        return self.embeddings.get(
            text.strip().lower(),
            [0.0, 0.0],
        )


class SpyLLM:
    """Fake LLM that tracks how many times generation occurs."""

    def __init__(self):
        self.calls = 0

    def generate(self, query: str, model=None) -> LLMResponse:
        self.calls += 1

        return LLMResponse(
            text=f"generated answer for {query}",
            model_name=model or "spy-model",
            tokens_used=5,
            cost=0.002,
            metadata={
                "provider": "test",
            },
        )


def make_pipeline(max_size=10):
    """Build Dhano's pipeline using Praj's optimizer."""

    semantic_cache = SemanticCache(
        embedding_generator=FakeEmbeddingGenerator()
    )

    optimized_cache = CacheOptimizer(
        cache=semantic_cache,
        max_size=max_size,
    )

    llm = SpyLLM()

    pipeline = SemanticCachingPipeline(
        cache=optimized_cache,
        llm=llm,
    )

    return pipeline, optimized_cache, llm


def test_praj_optimizer_works_inside_dhano_pipeline():
    """Pipeline should use the optimized cache transparently."""

    pipeline, cache, llm = make_pipeline()

    first = pipeline.process_query(
        "What is caching?"
    )

    second = pipeline.process_query(
        "What is caching?"
    )

    assert first.source == "llm"
    assert second.source == "cache"

    assert first.response_text == second.response_text

    assert llm.calls == 1

    assert cache.size() == 1


def test_semantic_hit_through_optimizer():
    """Semantic similarity should still work through CacheOptimizer."""

    pipeline, cache, llm = make_pipeline()

    first = pipeline.process_query(
        "What is caching?"
    )

    second = pipeline.process_query(
        "Explain caching"
    )

    assert first.source == "llm"
    assert second.source == "cache"

    assert llm.calls == 1
    assert cache.size() == 1


def test_optimizer_capacity_is_respected_by_pipeline():
    """Pipeline should respect Praj's maximum cache size."""

    pipeline, cache, llm = make_pipeline(
        max_size=2
    )

    pipeline.process_query(
        "What is caching?"
    )

    pipeline.process_query(
        "What is machine learning?"
    )

    pipeline.process_query(
        "Explain caching"
    )

    assert cache.size() <= 2