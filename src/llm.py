"""LLM integration layer owned by Dhano.

This module is responsible only for generating an answer after the
semantic cache reports a miss.

The implementation is intentionally provider-independent so a real
LLM API can be plugged in later without changing the pipeline.
"""

from typing import Optional

from src.interfaces import LLMInterface
from src.models import LLMResponse


class MockLLM(LLMInterface):
    """Deterministic local LLM used for development and testing.

    No external API or API key is required.
    """

    def __init__(self, default_model: str = "mock-llm-v1") -> None:
        self.default_model = default_model

    def generate(
        self,
        query: str,
        model: Optional[str] = None,
    ) -> LLMResponse:
        normalized = query.lower().strip()

        known_answers = {
            "what is machine learning?": (
                "Machine learning is a field of artificial intelligence "
                "that enables systems to learn patterns from data and "
                "improve their performance without being explicitly "
                "programmed for every task."
            ),
            "what is semantic caching?": (
                "Semantic caching stores query-response pairs together "
                "with vector representations so that semantically "
                "similar future queries can reuse previous responses."
            ),
            "what is an algorithm?": (
                "An algorithm is a finite sequence of well-defined steps "
                "used to solve a problem or perform a computation."
            ),
        }

        answer = known_answers.get(
            normalized,
            f"Simulated LLM response for query: '{query}'",
        )

        return LLMResponse(
            text=answer,
            model_name=model or self.default_model,
            tokens_used=len(answer.split()),
            latency_ms=12.5,
            cost=0.002,
            metadata={
                "provider": "mock",
            },
        )


class LLMService(LLMInterface):
    """Provider-independent LLM service placeholder.

    Replace the generate() implementation with a real provider when
    required. The pipeline does not need to change.
    """

    def __init__(
        self,
        default_model: str = "mock-llm-v1",
    ) -> None:
        self.default_model = default_model

    def generate(
        self,
        query: str,
        model: Optional[str] = None,
    ) -> LLMResponse:
        # For the current project milestone, use the deterministic
        # mock implementation so the project works without API keys.
        mock = MockLLM(default_model=self.default_model)
        return mock.generate(query, model=model)
