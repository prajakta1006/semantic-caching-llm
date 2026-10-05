"""Unit and integration tests for Andi's evaluation module."""

from pathlib import Path
import pytest

from evaluation.baseline import run_baseline
from evaluation.benchmark import load_dataset, run_benchmark
from src.config import config
from src.evaluator import DefaultEvaluator, EvaluationTracker, QueryRecord
from src.interfaces import CacheInterface
from src.models import (
    CacheEntry,
    DecisionResult,
    LLMResponse,
    PipelineResponse,
    QueryRequest,
)
from src.pipeline import SemanticCachingPipeline


class FakeCache(CacheInterface):
    """Simple fake cache for evaluation tests."""

    def __init__(self, hit_entry: CacheEntry | None = None):
        self.hit_entry = hit_entry

    def get(self, query: str, threshold: float) -> CacheEntry | None:
        return self.hit_entry

    def put(
        self,
        query: str,
        response: str,
        embedding: list[float] | None = None,
        metadata: dict | None = None,
    ) -> None:
        pass

    def size(self) -> int:
        return 1 if self.hit_entry else 0


class FakeLLM:
    """Fake LLM for testing."""

    def __init__(self, cost: float = 0.002, latency_ms: float = 12.5):
        self.cost = cost
        self.latency_ms = latency_ms

    def generate(self, query: str, model: str | None = None) -> LLMResponse:
        return LLMResponse(
            text=f"Response to {query}",
            model_name=model or "fake-llm",
            tokens_used=10,
            latency_ms=self.latency_ms,
            cost=self.cost,
        )


def test_evaluator_records_cache_hit():
    tracker = EvaluationTracker()
    req = QueryRequest(query="What is AI?")
    resp = PipelineResponse(
        query="What is AI?",
        response_text="AI is artificial intelligence.",
        source="cache",
        similarity_score=0.95,
        decision=DecisionResult(route="CACHE_HIT", estimated_cost=0.0),
        latency_ms=2.5,
        metadata={"model": "cache"},
    )

    tracker.record(req, resp, 2.5)
    records = tracker.records()

    assert len(records) == 1
    rec = records[0]
    assert rec.query == "What is AI?"
    assert rec.source == "cache"
    assert rec.decision_route == "CACHE_HIT"
    assert rec.similarity_score == 0.95
    assert rec.latency_ms == 2.5
    assert rec.estimated_cost == 0.0
    assert rec.llm_call_occurred is False
    assert rec.llm_call_avoided is True


def test_evaluator_records_cache_miss():
    tracker = EvaluationTracker()
    req = QueryRequest(query="What is quantum computing?")
    resp = PipelineResponse(
        query="What is quantum computing?",
        response_text="Quantum computing uses qubits.",
        source="llm",
        similarity_score=None,
        decision=DecisionResult(route="LLM_AFTER_CACHE_MISS", estimated_cost=0.002),
        latency_ms=15.0,
        metadata={"model": "mock-llm-v1", "cost": 0.002},
    )

    tracker.record(req, resp, 15.0)
    records = tracker.records()

    assert len(records) == 1
    rec = records[0]
    assert rec.query == "What is quantum computing?"
    assert rec.source == "llm"
    assert rec.decision_route == "LLM_AFTER_CACHE_MISS"
    assert rec.similarity_score is None
    assert rec.latency_ms == 15.0
    assert rec.estimated_cost == 0.002
    assert rec.llm_call_occurred is True
    assert rec.llm_call_avoided is False


def test_evaluator_hit_rate_calculation():
    tracker = EvaluationTracker()

    # 2 hits, 1 miss -> 2/3 = 0.6666...
    req = QueryRequest(query="q1")
    resp_hit = PipelineResponse(
        query="q1", response_text="r1", source="cache", latency_ms=1.0
    )
    resp_miss = PipelineResponse(
        query="q3",
        response_text="r3",
        source="llm",
        latency_ms=10.0,
        metadata={"cost": 0.002},
    )

    tracker.record(req, resp_hit, 1.0)
    tracker.record(req, resp_hit, 1.0)
    tracker.record(req, resp_miss, 10.0)

    metrics = tracker.get_metrics()
    assert metrics.total_queries == 3
    assert metrics.cache_hits == 2
    assert metrics.cache_misses == 1
    assert pytest.approx(metrics.hit_rate, 0.01) == 2 / 3


def test_evaluator_llm_call_counts_and_avoided():
    tracker = EvaluationTracker()
    req = QueryRequest(query="q")

    tracker.record(req, PipelineResponse(query="q", response_text="r", source="cache"), 1.0)
    tracker.record(req, PipelineResponse(query="q", response_text="r", source="llm", metadata={"cost": 0.002}), 10.0)
    tracker.record(req, PipelineResponse(query="q", response_text="r", source="cache"), 1.0)

    metrics = tracker.get_metrics()
    assert metrics.metadata["llm_calls"] == 1
    assert metrics.metadata["llm_calls_avoided"] == 2
    assert pytest.approx(metrics.metadata["llm_reduction_pct"], 0.01) == 66.67


def test_evaluator_average_latency_calculation():
    tracker = EvaluationTracker()
    req = QueryRequest(query="q")

    tracker.record(req, PipelineResponse(query="q", response_text="r", source="cache"), 2.0)
    tracker.record(req, PipelineResponse(query="q", response_text="r", source="llm", metadata={"cost": 0.002}), 10.0)

    metrics = tracker.get_metrics()
    assert metrics.avg_latency_ms == 6.0
    assert metrics.metadata["avg_hit_latency_ms"] == 2.0
    assert metrics.metadata["avg_miss_latency_ms"] == 10.0


def test_evaluator_cost_calculations():
    tracker = EvaluationTracker(default_llm_cost_per_query=0.002)
    req = QueryRequest(query="q")

    # 1 miss ($0.002), 2 hits ($0.000 actual, but would be $0.002 each without cache)
    tracker.record(req, PipelineResponse(query="q", response_text="r", source="llm", metadata={"cost": 0.002}), 10.0)
    tracker.record(req, PipelineResponse(query="q", response_text="r", source="cache"), 2.0)
    tracker.record(req, PipelineResponse(query="q", response_text="r", source="cache"), 2.0)

    metrics = tracker.get_metrics()
    meta = metrics.metadata

    assert meta["total_estimated_cost"] == 0.002
    assert meta["estimated_cost_with_caching"] == 0.002
    assert meta["estimated_cost_without_caching"] == 0.006
    assert pytest.approx(meta["estimated_cost_savings"], 0.0001) == 0.004
    assert pytest.approx(meta["cost_reduction_pct"], 0.01) == 66.67


def test_evaluator_empty_dataset_behavior():
    tracker = EvaluationTracker()
    metrics = tracker.get_metrics()

    assert metrics.total_queries == 0
    assert metrics.cache_hits == 0
    assert metrics.cache_misses == 0
    assert metrics.hit_rate == 0.0
    assert metrics.avg_latency_ms == 0.0
    assert metrics.total_cost_saved == 0.0


def test_evaluator_one_query_dataset_behavior():
    tracker = EvaluationTracker()
    req = QueryRequest(query="Single query")
    resp = PipelineResponse(
        query="Single query",
        response_text="Ans",
        source="llm",
        latency_ms=12.0,
        metadata={"cost": 0.002},
    )

    tracker.record(req, resp, 12.0)
    metrics = tracker.get_metrics()

    assert metrics.total_queries == 1
    assert metrics.cache_hits == 0
    assert metrics.cache_misses == 1
    assert metrics.hit_rate == 0.0
    assert metrics.avg_latency_ms == 12.0


def test_evaluator_repeated_semantic_queries():
    tracker = EvaluationTracker()
    pipeline = SemanticCachingPipeline(
        cache=FakeCache(
            hit_entry=CacheEntry(query="What is ML?", response="ML is Machine Learning")
        ),
        llm=FakeLLM(),
        evaluator=tracker,
    )

    for _ in range(5):
        pipeline.process_query("What is ML?")

    metrics = tracker.get_metrics()
    assert metrics.total_queries == 5
    assert metrics.cache_hits == 5
    assert metrics.cache_misses == 0
    assert metrics.hit_rate == 1.0


def test_evaluator_integration_with_pipeline():
    tracker = EvaluationTracker()
    fake_cache = FakeCache(hit_entry=None)
    fake_llm = FakeLLM(cost=0.005)

    pipeline = SemanticCachingPipeline(
        cache=fake_cache,
        llm=fake_llm,
        evaluator=tracker,
    )

    resp = pipeline.process_query("Test integration query")

    assert resp.source == "llm"
    assert len(tracker.records()) == 1

    metrics = tracker.get_metrics()
    assert metrics.total_queries == 1
    assert metrics.cache_misses == 1
    assert metrics.metadata["total_estimated_cost"] == 0.005


def test_baseline_runner():
    queries = [
        {"id": 1, "group": "g1", "query": "q1"},
        {"id": 2, "group": "g1", "query": "q2"},
    ]

    res = run_baseline(queries, llm=FakeLLM(cost=0.002))

    assert res["total_queries"] == 2
    assert res["llm_calls"] == 2
    assert res["estimated_cost"] == 0.004
    assert len(res["query_results"]) == 2


def test_benchmark_dataset_loading():
    dataset = load_dataset()
    assert len(dataset) >= 30
    assert any(item["group"] == "machine_learning" for item in dataset)
    assert any(item["group"] == "semantic_caching" for item in dataset)
