
"""LLM integration layer.

Supports:
- MockLLM: deterministic local responses for testing.
- GeminiLLM: Google Gemini through its generateContent REST API.
- LLMService: selects the provider using AppConfig.

Gemini reliability and performance features:
- Bounded request timeouts.
- One retry for temporary primary-model failures.
- Automatic fallback to another compatible model.
- Fallback is attempted only once to limit worst-case latency.
- Accurate request-attempt, retry, fallback, token and latency metadata.
"""

import json
import os
import time
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.config import AppConfig
from src.interfaces import LLMInterface
from src.models import LLMResponse


class MockLLM(LLMInterface):
    """Deterministic local LLM used for development and testing."""

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

        selected_model = model or self.default_model

        return LLMResponse(
            text=answer,
            model_name=selected_model,
            tokens_used=len(answer.split()),
            latency_ms=12.5,
            cost=0.002,
            metadata={
                "provider": "mock",
                "requested_model": selected_model,
                "model_used": selected_model,
                "retry_count": 0,
                "request_attempts": 1,
                "fallback_attempted": False,
                "fallback_used": False,
            },
        )


class _GeminiRequestError(RuntimeError):
    """Internal error containing Gemini failure and request metadata."""

    def __init__(
        self,
        message: str,
        retryable: bool,
        retry_after: Optional[float] = None,
        attempts: int = 1,
        retry_count: int = 0,
        metadata: Optional[dict] = None,
    ) -> None:
        super().__init__(message)

        self.retryable = retryable
        self.retry_after = retry_after
        self.attempts = attempts
        self.metadata = metadata or {}

        self.metadata.setdefault("request_attempts", attempts)
        self.metadata.setdefault("retry_count", retry_count)
        self.metadata.setdefault("provider", "gemini")


class GeminiLLM(LLMInterface):
    """Google Gemini integration using the generateContent REST API."""

    API_BASE_URL = (
        "https://generativelanguage.googleapis.com/v1beta/models"
    )

    RETRYABLE_STATUS_CODES = {
        408,
        429,
        500,
        502,
        503,
        504,
    }

    def __init__(
        self,
        api_key: str,
        default_model: str = "gemini-3.8-flash",
        timeout: float = 20.0,
        max_retries: int = 1,
        retry_delay: float = 1.0,
        fallback_models: Optional[tuple] = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError(
                "Gemini API key is missing. Set the environment "
                "variable configured by LLM_API_KEY_ENV."
            )

        if timeout <= 0:
            raise ValueError(
                "Gemini timeout must be greater than zero."
            )

        if max_retries < 0:
            raise ValueError(
                "max_retries cannot be negative."
            )

        if retry_delay < 0:
            raise ValueError(
                "retry_delay cannot be negative."
            )

        self.api_key = api_key.strip()
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.fallback_models = (
            fallback_models
            if fallback_models is not None
            else ("gemini-3.7-flash",)
        )

    def _make_request(
        self,
        query: str,
        model: str,
    ) -> dict:
        """Send one HTTP request to the selected Gemini model."""

        url = f"{self.API_BASE_URL}/{model}:generateContent"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": query}
                    ]
                }
            ]
        }

        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                return json.loads(
                    response.read().decode("utf-8")
                )

        except HTTPError as exc:
            error_body = exc.read().decode(
                "utf-8",
                errors="replace",
            )

            retry_after = None

            try:
                retry_after_header = exc.headers.get("Retry-After")

                if retry_after_header:
                    retry_after = float(retry_after_header)

            except (TypeError, ValueError):
                retry_after = None

            raise _GeminiRequestError(
                message=(
                    f"Gemini API returned HTTP {exc.code}: "
                    f"{error_body}"
                ),
                retryable=(
                    exc.code in self.RETRYABLE_STATUS_CODES
                ),
                retry_after=retry_after,
            ) from exc

        except (URLError, TimeoutError, OSError) as exc:
            raise _GeminiRequestError(
                message=f"Could not complete Gemini API request: {exc}",
                retryable=True,
            ) from exc

        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise _GeminiRequestError(
                message=f"Gemini returned an invalid response: {exc}",
                retryable=False,
            ) from exc

    def _request_with_retries(
        self,
        query: str,
        model: str,
        max_retries: Optional[int] = None,
    ) -> tuple:
        """Make a bounded number of attempts for one model."""

        retry_limit = (
            self.max_retries
            if max_retries is None
            else max_retries
        )

        attempts = 0
        retry_count = 0

        for attempt in range(retry_limit + 1):
            attempts += 1

            try:
                response_data = self._make_request(
                    query=query,
                    model=model,
                )

                return response_data, attempts, retry_count

            except _GeminiRequestError as exc:
                is_last_attempt = attempt == retry_limit

                if not exc.retryable or is_last_attempt:
                    exc.attempts = attempts
                    exc.metadata.update({
                        "request_attempts": attempts,
                        "retry_count": retry_count,
                    })
                    raise

                retry_count += 1

                delay = min(
                    self.retry_delay * (2 ** attempt),
                    5.0,
                )

                if exc.retry_after is not None:
                    delay = min(
                        max(delay, exc.retry_after),
                        5.0,
                    )

                print(
                    f"Temporary Gemini failure for {model}. "
                    f"Retrying in {delay:.1f} seconds "
                    f"({attempt + 1}/{retry_limit} retries)."
                )

                time.sleep(delay)

        raise RuntimeError(
            f"Gemini request failed unexpectedly for {model}."
        )

    def _build_metadata(
        self,
        requested_model: str,
        actual_model: Optional[str],
        total_attempts: int,
        total_retries: int,
        fallback_attempted: bool,
        fallback_used: bool,
    ) -> dict:
        """Build consistent metadata for success and failure cases."""

        return {
            "provider": "gemini",
            "requested_model": requested_model,
            "model_used": actual_model,
            "request_attempts": total_attempts,
            "retry_count": total_retries,
            "fallback_attempted": fallback_attempted,
            "fallback_used": fallback_used,
        }

    def generate(
        self,
        query: str,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """Generate a response with bounded retry and fallback support."""

        selected_model = (
            model or self.default_model
        ).strip().removeprefix("models/")

        if not selected_model:
            raise ValueError(
                "Gemini model name cannot be empty."
            )

        models_to_try = [selected_model]

        for fallback_model in self.fallback_models:
            normalized_fallback = (
                fallback_model.strip().removeprefix("models/")
            )

            if (
                normalized_fallback
                and normalized_fallback not in models_to_try
            ):
                models_to_try.append(normalized_fallback)

        start_time = time.perf_counter()

        total_attempts = 0
        total_retries = 0
        response_data = None
        actual_model = None
        last_error = None
        fallback_attempted = False

        for index, current_model in enumerate(models_to_try):
            is_fallback = index > 0

            # Primary model gets the configured retry limit.
            # Each fallback model gets only one attempt.
            retry_limit = 0 if is_fallback else self.max_retries

            if is_fallback:
                fallback_attempted = True

            try:
                print(f"Attempting Gemini model: {current_model}")

                response_data, attempts, retries = (
                    self._request_with_retries(
                        query=query,
                        model=current_model,
                        max_retries=retry_limit,
                    )
                )

                total_attempts += attempts
                total_retries += retries
                actual_model = current_model
                break

            except _GeminiRequestError as exc:
                total_attempts += exc.attempts
                total_retries += exc.metadata.get(
                    "retry_count",
                    0,
                )
                last_error = exc

                fallback_available = (
                    index < len(models_to_try) - 1
                )

                if not exc.retryable or not fallback_available:
                    metadata = self._build_metadata(
                        requested_model=selected_model,
                        actual_model=None,
                        total_attempts=total_attempts,
                        total_retries=total_retries,
                        fallback_attempted=fallback_attempted,
                        fallback_used=False,
                    )

                    raise _GeminiRequestError(
                        message=str(exc),
                        retryable=exc.retryable,
                        retry_after=exc.retry_after,
                        attempts=total_attempts,
                        retry_count=total_retries,
                        metadata=metadata,
                    ) from exc

                print(
                    f"Model {current_model} failed temporarily. "
                    "Trying the fallback model."
                )

        if response_data is None or actual_model is None:
            metadata = self._build_metadata(
                requested_model=selected_model,
                actual_model=None,
                total_attempts=total_attempts,
                total_retries=total_retries,
                fallback_attempted=fallback_attempted,
                fallback_used=False,
            )

            raise _GeminiRequestError(
                message=(
                    f"All Gemini models failed. Last error: {last_error}"
                ),
                retryable=False,
                attempts=total_attempts,
                retry_count=total_retries,
                metadata=metadata,
            )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        # The HTTP request succeeded, but the returned content
        # still needs validation.
        candidates = response_data.get("candidates", [])

        if not candidates:
            metadata = self._build_metadata(
                requested_model=selected_model,
                actual_model=actual_model,
                total_attempts=total_attempts,
                total_retries=total_retries,
                fallback_attempted=fallback_attempted,
                fallback_used=(actual_model != selected_model),
            )

            raise _GeminiRequestError(
                message=(
                    "Gemini returned no response candidates. "
                    "Check promptFeedback for a possible block reason."
                ),
                retryable=False,
                attempts=total_attempts,
                retry_count=total_retries,
                metadata=metadata,
            )

        candidate = candidates[0]

        parts = candidate.get(
            "content",
            {},
        ).get("parts", [])

        answer = "".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and "text" in part
        ).strip()

        if not answer:
            metadata = self._build_metadata(
                requested_model=selected_model,
                actual_model=actual_model,
                total_attempts=total_attempts,
                total_retries=total_retries,
                fallback_attempted=fallback_attempted,
                fallback_used=(actual_model != selected_model),
            )

            raise _GeminiRequestError(
                message="Gemini returned an empty text response.",
                retryable=False,
                attempts=total_attempts,
                retry_count=total_retries,
                metadata=metadata,
            )

        usage = response_data.get("usageMetadata", {})
        total_tokens = usage.get("totalTokenCount", 0)

        response_metadata = self._build_metadata(
            requested_model=selected_model,
            actual_model=actual_model,
            total_attempts=total_attempts,
            total_retries=total_retries,
            fallback_attempted=fallback_attempted,
            fallback_used=(actual_model != selected_model),
        )

        response_metadata.update({
            "prompt_tokens": usage.get(
                "promptTokenCount",
                0,
            ),
            "response_tokens": usage.get(
                "candidatesTokenCount",
                0,
            ),
            "total_tokens": total_tokens,
            "finish_reason": candidate.get("finishReason"),
        })

        return LLMResponse(
            text=answer,
            model_name=actual_model,
            tokens_used=total_tokens,
            latency_ms=latency_ms,
            cost=0.0,
            metadata=response_metadata,
        )


class LLMService(LLMInterface):
    """Selects and manages the configured LLM provider."""

    def __init__(
        self,
        app_config: Optional[AppConfig] = None,
        default_model: Optional[str] = None,
    ) -> None:
        self.config = app_config or AppConfig()

        self.default_model = (
            default_model or self.config.default_llm_model
        )

        provider = self.config.llm_provider.lower().strip()

        if provider == "mock":
            self.provider = MockLLM(
                default_model=self.default_model
            )

        elif provider == "gemini":
            api_key = os.getenv(
                self.config.llm_api_key_env
            )

            self.provider = GeminiLLM(
                api_key=api_key or "",
                default_model=self.default_model,
                timeout=self.config.llm_timeout_seconds,
                max_retries=1,
                retry_delay=1.0,
            )

        else:
            raise ValueError(
                f"Unsupported LLM provider: '{provider}'. "
                "Supported providers are 'mock' and 'gemini'."
            )

    def generate(
        self,
        query: str,
        model: Optional[str] = None,
    ) -> LLMResponse:
        return self.provider.generate(
            query=query,
            model=model,
        )
