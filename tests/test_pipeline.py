"""Tests for Dhano's LLM pipeline."""

from src.interfaces import (
    CacheInterface,
    EvaluatorInterface,
)
from src.models import (
    CacheEntry,
    EvaluationMetrics,
    LLMResponse,
    PipelineResponse,
    QueryRequest,
)
from src.pipeline import SemanticCachingPipeline


class SpyCache(CacheInterface):
    """Minimal fake cache used only for pipeline testing."""

    def __init__(
        self,
        entry=None,
    ):
        self.entry = entry
        self.get_calls = 0
        self.put_calls = 0

    def get(
        self,
        query: str,
        threshold: float,
    ):
        self.get_calls += 1
        return self.entry

    def put(
        self,
        query: str,
        response: str,
        embedding=None,
        metadata=None,
    ) -> None:

        self.put_calls += 1

        self.entry = CacheEntry(
            query=query,
            response=response,
            similarity_score=1.0,
            embedding=embedding,
            metadata=metadata or {},
        )

    def size(self) -> int:
        return 1 if self.entry else 0


class SpyLLM:
    """Fake LLM used to verify pipeline behavior."""

    def __init__(self):
        self.calls = 0
        self.last_query = None
        self.last_model = None

    def generate(
        self,
        query: str,
        model=None,
    ) -> LLMResponse:

        self.calls += 1
        self.last_query = query
        self.last_model = model

        return LLMResponse(
            text=f"answer for {query}",
            model_name=model or "spy-model",
            tokens_used=3,
            cost=0.002,
            metadata={
                "provider": "test"
            },
        )


class SpyEvaluator(EvaluatorInterface):
    """Fake evaluator used to verify integration."""

    def __init__(self):
        self.calls = 0
        self.last_request = None
        self.last_response = None

    def record(
        self,
        request: QueryRequest,
        response: PipelineResponse,
        latency_ms: float,
    ) -> None:

        self.calls += 1
        self.last_request = request
        self.last_response = response

    def get_metrics(
        self,
    ) -> EvaluationMetrics:

        return EvaluationMetrics(
            total_queries=self.calls
        )


def test_pipeline_rejects_empty_query():

    pipeline = SemanticCachingPipeline()

    try:
        pipeline.process_query("   ")
        assert False

    except ValueError as exc:

        assert str(exc) == (
            "Query cannot be empty"
        )


def test_cache_miss_calls_llm():

    cache = SpyCache()
    llm = SpyLLM()

    pipeline = SemanticCachingPipeline(
        cache=cache,
        llm=llm,
    )

    response = pipeline.process_query(
        "new query"
    )

    assert response.source == "llm"

    assert llm.calls == 1

    assert llm.last_query == (
        "new query"
    )

    assert cache.get_calls == 1

    assert cache.put_calls == 1

    assert cache.entry.response == (
        "answer for new query"
    )


def test_cache_hit_does_not_call_llm():

    cache = SpyCache(
        CacheEntry(
            query="hello",
            response="cached answer",
            similarity_score=0.94,
        )
    )

    llm = SpyLLM()

    pipeline = SemanticCachingPipeline(
        cache=cache,
        llm=llm,
    )

    response = pipeline.process_query(
        "hello"
    )

    assert response.source == "cache"

    assert response.response_text == (
        "cached answer"
    )

    assert response.similarity_score == 0.94

    assert response.decision.route == (
        "CACHE_HIT"
    )

    assert llm.calls == 0

    assert cache.get_calls == 1


def test_cache_miss_then_hit():

    cache = SpyCache()
    llm = SpyLLM()

    pipeline = SemanticCachingPipeline(
        cache=cache,
        llm=llm,
    )

    first = pipeline.process_query(
        "What is caching?"
    )

    second = pipeline.process_query(
        "What is caching?"
    )

    assert first.source == "llm"

    assert second.source == "cache"

    assert first.response_text == (
        second.response_text
    )

    assert llm.calls == 1

    assert cache.get_calls == 2

    assert cache.put_calls == 1


def test_evaluator_receives_pipeline_response():

    evaluator = SpyEvaluator()

    pipeline = SemanticCachingPipeline(
        evaluator=evaluator,
    )

    response = pipeline.process_query(
        "What is machine learning?"
    )

    assert evaluator.calls == 1

    assert evaluator.last_response is response

    assert (
        evaluator.last_request.query
        == "What is machine learning?"
    )


def test_pipeline_can_run_without_cache():

    llm = SpyLLM()

    pipeline = SemanticCachingPipeline(
        llm=llm,
    )

    response = pipeline.process_query(
        "test query"
    )

    assert response.source == "llm"

    assert llm.calls == 1

    assert (
        response.metadata["cache_status"]
        == "semantic cache not connected"
    )
