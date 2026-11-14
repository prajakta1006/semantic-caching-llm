
"""Tests for LLM integration and Gemini reliability."""

import pytest

from src.config import AppConfig
from src.llm import (
    GeminiLLM,
    LLMService,
    MockLLM,
    _GeminiRequestError,
)


def test_mock_llm_returns_response():
    llm = MockLLM()

    response = llm.generate(
        "What is machine learning?"
    )

    assert response.text
    assert response.model_name == "mock-llm-v1"
    assert response.tokens_used > 0
    assert response.cost >= 0


def test_mock_llm_accepts_model():
    llm = MockLLM()

    response = llm.generate(
        "test query",
        model="custom-model",
    )

    assert response.model_name == "custom-model"


def test_llm_service_uses_mock_for_now():
    test_config = AppConfig(
        llm_provider="mock",
        default_llm_model="mock-llm-v1",
    )

    llm = LLMService(
        app_config=test_config,
    )

    response = llm.generate(
        "What is semantic caching?"
    )

    assert response.text
    assert response.metadata["provider"] == "mock"
    assert response.model_name == "mock-llm-v1"


def make_gemini_response(answer="Test Gemini answer"):
    """Create a simulated successful Gemini API response."""

    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": answer}
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 20,
            "totalTokenCount": 30,
        },
    }


def create_gemini():
    """Create Gemini without making a real API request."""

    return GeminiLLM(
        api_key="test-key",
        default_model="primary-model",
        max_retries=1,
        retry_delay=0,
        fallback_models=("fallback-model",),
    )


def test_gemini_primary_succeeds_after_retry(monkeypatch):
    llm = create_gemini()

    calls = {"count": 0}

    def fake_request(query, model):
        calls["count"] += 1

        if calls["count"] == 1:
            raise _GeminiRequestError(
                message="Simulated HTTP 503",
                retryable=True,
            )

        return make_gemini_response()

    monkeypatch.setattr(
        llm,
        "_make_request",
        fake_request,
    )

    monkeypatch.setattr(
        "src.llm.time.sleep",
        lambda _: None,
    )

    response = llm.generate("Test query")

    assert response.text == "Test Gemini answer"
    assert response.metadata["request_attempts"] == 2
    assert response.metadata["retry_count"] == 1
    assert response.metadata["fallback_attempted"] is False
    assert response.metadata["fallback_used"] is False
    assert response.model_name == "primary-model"


def test_gemini_fallback_succeeds(monkeypatch):
    llm = create_gemini()

    calls = {"primary": 0, "fallback": 0}

    def fake_request(query, model):
        if model == "primary-model":
            calls["primary"] += 1

            raise _GeminiRequestError(
                message="Simulated primary HTTP 503",
                retryable=True,
            )

        if model == "fallback-model":
            calls["fallback"] += 1
            return make_gemini_response(
                "Fallback Gemini answer"
            )

        raise AssertionError(f"Unexpected model: {model}")

    monkeypatch.setattr(
        llm,
        "_make_request",
        fake_request,
    )

    monkeypatch.setattr(
        "src.llm.time.sleep",
        lambda _: None,
    )

    response = llm.generate("Test query")

    assert response.text == "Fallback Gemini answer"
    assert response.model_name == "fallback-model"

    assert calls["primary"] == 2
    assert calls["fallback"] == 1

    assert response.metadata["request_attempts"] == 3
    assert response.metadata["retry_count"] == 1
    assert response.metadata["fallback_attempted"] is True
    assert response.metadata["fallback_used"] is True


def test_gemini_failure_preserves_attempt_metadata(monkeypatch):
    llm = create_gemini()

    calls = {"primary": 0, "fallback": 0}

    def fake_request(query, model):
        if model == "primary-model":
            calls["primary"] += 1

            raise _GeminiRequestError(
                message="Simulated primary HTTP 503",
                retryable=True,
            )

        if model == "fallback-model":
            calls["fallback"] += 1

            raise _GeminiRequestError(
                message="Simulated fallback HTTP 503",
                retryable=True,
            )

        raise AssertionError(f"Unexpected model: {model}")

    monkeypatch.setattr(
        llm,
        "_make_request",
        fake_request,
    )

    monkeypatch.setattr(
        "src.llm.time.sleep",
        lambda _: None,
    )

    with pytest.raises(_GeminiRequestError) as error:
        llm.generate("Test query")

    metadata = error.value.metadata

    assert calls["primary"] == 2
    assert calls["fallback"] == 1

    assert metadata["provider"] == "gemini"
    assert metadata["request_attempts"] == 3
    assert metadata["retry_count"] == 1
    assert metadata["fallback_attempted"] is True
    assert metadata["fallback_used"] is False
