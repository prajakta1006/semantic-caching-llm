"""Tests for Dhano's LLM integration."""

from src.llm import LLMService, MockLLM


def test_mock_llm_returns_response():

    llm = MockLLM()

    response = llm.generate(
        "What is machine learning?"
    )

    assert response.text

    assert response.model_name == (
        "mock-llm-v1"
    )

    assert response.tokens_used > 0

    assert response.cost >= 0


def test_mock_llm_accepts_model():

    llm = MockLLM()

    response = llm.generate(
        "test query",
        model="custom-model",
    )

    assert response.model_name == (
        "custom-model"
    )


def test_llm_service_uses_mock_for_now():

    llm = LLMService()

    response = llm.generate(
        "What is semantic caching?"
    )

    assert response.text

    assert response.metadata[
        "provider"
    ] == "mock"
