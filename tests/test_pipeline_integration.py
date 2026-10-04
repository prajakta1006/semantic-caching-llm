"""Integration tests for TanTan's modular pipeline foundation."""

import pytest

from src.config import AppConfig
from src.interfaces import CacheInterface, CascadeInterface, EvaluatorInterface, LLMInterface
from src.models import CacheEntry, DecisionResult, EvaluationMetrics, LLMResponse, PipelineResponse, QueryRequest
from src.pipeline import SemanticCachingPipeline


class SpyCache(CacheInterface):
    def __init__(self, initial_entry=None):
        self.entry = initial_entry
        self.get_calls = 0
        self.put_calls = 0
        self.stored_query = None
        self.stored_response = None

    def get(self, query: str, threshold: float):
        self.get_calls += 1
        return self.entry

    def put(self, query: str, response: str, embedding=None, metadata=None) -> None:
        self.put_calls += 1
        self.stored_query = query
        self.stored_response = response
        self.entry = CacheEntry(query=query, response=response, similarity_score=1.0)

    def size(self) -> int:
        return 1 if self.entry else 0


class SpyCascade(CascadeInterface):
    def __init__(self):
        self.calls = 0

    def decide(self, request: QueryRequest, config: AppConfig) -> DecisionResult:
        self.calls += 1
        return DecisionResult(
            route="DIRECT_LLM",
            estimated_cost=0.01,
            confidence=1.0,
            reason="spy cascade",
            metadata={"selected_model": "spy-model"},
        )


class SpyLLM(LLMInterface):
    def __init__(self):
        self.calls = 0
        self.last_model = None

    def generate(self, query: str, model=None) -> LLMResponse:
        self.calls += 1
        self.last_model = model
        return LLMResponse(
            text=f"answer for {query}",
            model_name=model or "spy-model",
            tokens_used=3,
            cost=0.002,
        )


class SpyEvaluator(EvaluatorInterface):
    def __init__(self):
        self.calls = 0
        self.last_response = None

    def record(self, request: QueryRequest, response: PipelineResponse, latency_ms: float) -> None:
        self.calls += 1
        self.last_response = response

    def get_metrics(self) -> EvaluationMetrics:
        return EvaluationMetrics(total_queries=self.calls)


def test_pipeline_accepts_query_with_default_configuration():
    response = SemanticCachingPipeline().process_query("What is semantic caching?")

    assert response.source == "llm"
    assert response.decision.route == "DIRECT_LLM"
    assert "Semantic caching" in response.response_text


def test_cache_hit_avoids_llm_and_cascade_calls():
    cache = SpyCache(CacheEntry(query="hello", response="cached answer", similarity_score=1.0))
    cascade = SpyCascade()
    llm = SpyLLM()

    response = SemanticCachingPipeline(cache=cache, cascade=cascade, llm=llm).process_query("hello")

    assert response.source == "cache"
    assert response.response_text == "cached answer"
    assert response.decision.route == "CACHE_HIT"
    assert llm.calls == 0
    assert cascade.calls == 0


def test_cache_miss_calls_cascade_llm_and_stores_result():
    cache = SpyCache()
    cascade = SpyCascade()
    llm = SpyLLM()

    response = SemanticCachingPipeline(cache=cache, cascade=cascade, llm=llm).process_query("new query")

    assert response.source == "llm"
    assert cascade.calls == 1
    assert llm.calls == 1
    assert llm.last_model == "spy-model"
    assert cache.put_calls == 1
    assert cache.stored_query == "new query"
    assert cache.stored_response == "answer for new query"


def test_pipeline_can_work_with_injected_evaluator():
    evaluator = SpyEvaluator()

    response = SemanticCachingPipeline(evaluator=evaluator).process_query("What is machine learning?")

    assert response.source == "llm"
    assert evaluator.calls == 1
    assert evaluator.last_response is response


def test_empty_query_is_rejected_cleanly():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        SemanticCachingPipeline().process_query("   ")
