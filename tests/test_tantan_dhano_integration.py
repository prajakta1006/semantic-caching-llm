"""Integration test for TanTan semantic cache + Dhano pipeline."""

from src.models import LLMResponse
from src.pipeline import SemanticCachingPipeline
from src.semantic_cache import SemanticCache


class FakeEmbeddingGenerator:
    """Deterministic embeddings for integration testing."""

    def __init__(self):
        self.embeddings = {
            "what is machine learning?": [1.0, 0.0],
            "explain machine learning": [0.99, 0.01],
        }

    def encode(self, text: str):
        """Return a deterministic embedding.

        Normalizing the input makes the fake embedding generator
        behave consistently with the query preprocessing used
        by the pipeline.
        """

        normalized_text = text.strip().lower()

        return self.embeddings.get(
            normalized_text,
            [0.0, 1.0],
        )


class SpyLLM:
    """LLM spy used to verify whether the LLM is called."""

    def __init__(self):
        self.calls = 0

    def generate(
        self,
        query: str,
        model=None,
    ) -> LLMResponse:
        """Simulate an LLM response."""

        self.calls += 1

        return LLMResponse(
            text="Machine learning answer",
            model_name=model or "test-model",
            tokens_used=3,
            cost=0.002,
            metadata={
                "provider": "test"
            },
        )


def test_tantan_cache_integrates_with_dhano_pipeline():
    """Verify semantic cache and pipeline integration."""

    cache = SemanticCache(
        embedding_generator=FakeEmbeddingGenerator()
    )

    llm = SpyLLM()

    pipeline = SemanticCachingPipeline(
        cache=cache,
        llm=llm,
    )

    # ---------------------------------------------------------
    # First query:
    # Cache MISS → LLM → Store
    # ---------------------------------------------------------

    first = pipeline.process_query(
        "What is machine learning?"
    )

    assert first.source == "llm"

    assert llm.calls == 1

    assert cache.size() == 1

    assert first.response_text == (
        "Machine learning answer"
    )

    # ---------------------------------------------------------
    # Second query:
    # Semantically similar → Cache HIT
    # ---------------------------------------------------------

    second = pipeline.process_query(
        "explain machine learning"
    )

    assert second.source == "cache"

    assert second.response_text == (
        "Machine learning answer"
    )

    assert second.similarity_score is not None

    assert second.similarity_score >= 0.85

    # ---------------------------------------------------------
    # LLM must NOT be called again.
    # ---------------------------------------------------------

    assert llm.calls == 1

    # ---------------------------------------------------------
    # Cache should still contain only one entry.
    # ---------------------------------------------------------

    assert cache.size() == 1
