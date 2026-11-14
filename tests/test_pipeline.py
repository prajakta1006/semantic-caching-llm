
"""Tests for Dhanashree's LLM pipeline."""

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
from src.evaluator import EvaluationTracker


class SpyCache(CacheInterface):
    """Minimal fake cache used only for pipeline testing."""

    def __init__(self, entry=None):
        self.entry = entry
        self.get_calls = 0
        self.put_calls = 0

    def get(self, query: str, threshold: float):
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

    def generate(self, query: str, model=None) -> LLMResponse:
        self.calls += 1
        self.last_query = query
        self.last_model = model

        return LLMResponse(
            text=f"answer for {query}",
            model_name=model or "spy-model",
            tokens_used=3,
            cost=0.002,
            metadata={
                "provider": "test",
                "request_attempts": 1,
                "retry_count": 0,
                "fallback_attempted": False,
                "fallback_used": False,
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

    def get_metrics(self) -> EvaluationMetrics:
        return EvaluationMetrics(
            total_queries=self.calls
        )


def test_pipeline_rejects_empty_query():
    pipeline = SemanticCachingPipeline()

    try:
        pipeline.process_query("   ")
        assert False

    except ValueError as exc:
        assert str(exc) == "Query cannot be empty"


def test_cache_miss_calls_llm():
    cache = SpyCache()
    llm = SpyLLM()

    pipeline = SemanticCachingPipeline(
        cache=cache,
        llm=llm,
    )

    response = pipeline.process_query("new query")

    assert response.source == "llm"
    assert llm.calls == 1
    assert llm.last_query == "new query"
    assert cache.get_calls == 1
    assert cache.put_calls == 1
    assert cache.entry.response == "answer for new query"


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

    response = pipeline.process_query("hello")

    assert response.source == "cache"
    assert response.response_text == "cached answer"
    assert response.similarity_score == 0.94
    assert response.decision.route == "CACHE_HIT"
    assert llm.calls == 0
    assert cache.get_calls == 1


def test_cache_miss_then_hit():
    cache = SpyCache()
    llm = SpyLLM()

    pipeline = SemanticCachingPipeline(
        cache=cache,
        llm=llm,
    )

    first = pipeline.process_query("What is caching?")
    second = pipeline.process_query("What is caching?")

    assert first.source == "llm"
    assert second.source == "cache"
    assert first.response_text == second.response_text
    assert llm.calls == 1
    assert cache.get_calls == 2
    assert cache.put_calls == 1


def test_evaluator_receives_pipeline_response():
    evaluator = SpyEvaluator()
    llm = SpyLLM()

    pipeline = SemanticCachingPipeline(
        evaluator=evaluator,
        llm=llm,
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
    assert llm.calls == 1


def test_pipeline_can_run_without_cache():
    llm = SpyLLM()

    pipeline = SemanticCachingPipeline(llm=llm)

    response = pipeline.process_query("test query")

    assert response.source == "llm"
    assert llm.calls == 1
    assert (
        response.metadata["cache_status"]
        == "semantic cache not connected"
    )


class FailingLLM:
    """Fake LLM that simulates a provider failure."""

    def __init__(self):
        self.calls = 0

    def generate(self, query: str, model=None):
        self.calls += 1
        raise RuntimeError("Simulated Gemini timeout")


def test_pipeline_handles_llm_failure():
    cache = SpyCache()
    llm = FailingLLM()
    evaluator = SpyEvaluator()

    pipeline = SemanticCachingPipeline(
        cache=cache,
        llm=llm,
        evaluator=evaluator,
    )

    response = pipeline.process_query("test query")

    assert response.source == "error"
    assert response.decision.route == "LLM_ERROR"
    assert response.metadata["llm_call_attempted"] is True
    assert response.metadata["llm_call_succeeded"] is False
    assert "Simulated Gemini timeout" in response.metadata["llm_error"]

    assert llm.calls == 1
    assert cache.get_calls == 1
    assert cache.put_calls == 0
    assert evaluator.calls == 1


def test_pipeline_can_continue_after_llm_failure():
    cache = SpyCache()
    llm = FailingLLM()

    pipeline = SemanticCachingPipeline(
        cache=cache,
        llm=llm,
    )

    first = pipeline.process_query("first query")
    second = pipeline.process_query("second query")

    assert first.source == "error"
    assert second.source == "error"
    assert llm.calls == 2
    assert cache.put_calls == 0


class TelemetryFailingLLM:
    """Fake LLM that fails with realistic Gemini telemetry."""

    def generate(self, query: str, model=None):
        error = RuntimeError("Simulated Gemini HTTP 503 failures")

        error.metadata = {
            "provider": "gemini",
            "requested_model": "primary-model",
            "model_used": None,
            "request_attempts": 3,
            "retry_count": 1,
            "fallback_attempted": True,
            "fallback_used": False,
        }

        raise error


class TelemetryFallbackLLM:
    """Fake LLM that succeeds after using a fallback model."""

    def generate(self, query: str, model=None):
        return LLMResponse(
            text="Answer generated by fallback model",
            model_name="fallback-model",
            tokens_used=25,
            latency_ms=100.0,
            cost=0.002,
            metadata={
                "provider": "gemini",
                "requested_model": "primary-model",
                "model_used": "fallback-model",
                "request_attempts": 3,
                "retry_count": 1,
                "fallback_attempted": True,
                "fallback_used": True,
            },
        )


def test_pipeline_and_evaluator_track_failed_fallback_end_to_end():
    cache = SpyCache()
    evaluator = EvaluationTracker()
    llm = TelemetryFailingLLM()

    pipeline = SemanticCachingPipeline(
        cache=cache,
        llm=llm,
        evaluator=evaluator,
    )

    response = pipeline.process_query("failure integration query")

    assert response.source == "error"
    assert response.metadata["llm_call_attempted"] is True
    assert response.metadata["llm_call_failed"] is True

    assert response.metadata["http_attempts"] == 3
    assert response.metadata["retry_count"] == 1
    assert response.metadata["fallback_attempted"] is True
    assert response.metadata["fallback_used"] is False

    assert response.metadata["llm_metadata"]["provider"] == "gemini"

    # Failed generations must not be cached.
    assert cache.put_calls == 0

    metrics = evaluator.get_metrics().metadata

    assert metrics["llm_call_attempts"] == 1
    assert metrics["llm_calls"] == 0
    assert metrics["llm_calls_failed"] == 1
    assert metrics["http_attempts"] == 3
    assert metrics["retry_count"] == 1
    assert metrics["fallback_attempt_count"] == 1
    assert metrics["fallback_count"] == 0


def test_pipeline_and_evaluator_track_successful_fallback_end_to_end():
    cache = SpyCache()
    evaluator = EvaluationTracker()
    llm = TelemetryFallbackLLM()

    pipeline = SemanticCachingPipeline(
        cache=cache,
        llm=llm,
        evaluator=evaluator,
    )

    response = pipeline.process_query("fallback success query")

    assert response.source == "llm"
    assert response.response_text == (
        "Answer generated by fallback model"
    )

    assert response.metadata["model"] == "fallback-model"
    assert response.metadata["http_attempts"] == 3
    assert response.metadata["retry_count"] == 1
    assert response.metadata["fallback_attempted"] is True
    assert response.metadata["fallback_used"] is True

    # Successful responses should be cached.
    assert cache.put_calls == 1

    metrics = evaluator.get_metrics().metadata

    assert metrics["llm_call_attempts"] == 1
    assert metrics["llm_calls"] == 1
    assert metrics["llm_calls_failed"] == 0
    assert metrics["http_attempts"] == 3
    assert metrics["retry_count"] == 1
    assert metrics["fallback_attempt_count"] == 1
    assert metrics["fallback_count"] == 1
