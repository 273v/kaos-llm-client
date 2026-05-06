"""Tests for kaos_llm_client.errors — exception hierarchy."""

from __future__ import annotations

from kaos_core.exceptions import KaosCoreError

from kaos_llm_client.errors import (
    KaosLLMAuthError,
    KaosLLMError,
    KaosLLMProviderError,
    KaosLLMRetryExhaustedError,
    KaosLLMTransportError,
    KaosLLMValidationError,
)


class TestErrorHierarchy:
    def test_base_inherits_kaos_core(self):
        assert issubclass(KaosLLMError, KaosCoreError)

    def test_auth_inherits_llm_error(self):
        assert issubclass(KaosLLMAuthError, KaosLLMError)

    def test_transport_inherits_llm_error(self):
        assert issubclass(KaosLLMTransportError, KaosLLMError)

    def test_provider_inherits_llm_error(self):
        assert issubclass(KaosLLMProviderError, KaosLLMError)

    def test_retry_inherits_transport(self):
        assert issubclass(KaosLLMRetryExhaustedError, KaosLLMTransportError)

    def test_validation_inherits_llm_error(self):
        assert issubclass(KaosLLMValidationError, KaosLLMError)


class TestKaosLLMError:
    def test_message_and_details(self):
        err = KaosLLMError("test error", provider="openai", code=42)
        assert err.message == "test error"
        assert err.details["provider"] == "openai"
        assert err.details["code"] == 42

    def test_str_representation(self):
        err = KaosLLMError("test error")
        assert str(err) == "test error"

        err2 = KaosLLMError("test error", detail="extra")
        assert "extra" in str(err2)


class TestProviderError:
    def test_structured_fields(self):
        err = KaosLLMProviderError(
            "bad request",
            provider="anthropic",
            model="claude-sonnet-4-6",
            status_code=400,
            raw_error={"type": "error", "error": {"message": "max_tokens required"}},
            fix="Add max_tokens to request",
        )
        assert err.provider == "anthropic"
        assert err.model == "claude-sonnet-4-6"
        assert err.status_code == 400
        assert err.raw_error is not None
        assert err.fix is not None

    def test_details_dict_populated(self):
        err = KaosLLMProviderError("error", provider="openai", status_code=500)
        assert err.details["provider"] == "openai"
        assert err.details["status_code"] == 500


class TestRetryExhaustedError:
    def test_attempt_count(self):
        err = KaosLLMRetryExhaustedError("all retries exhausted", attempts=3)
        assert err.attempts == 3
        assert err.details["attempts"] == 3

    def test_last_error_chaining(self):
        original = KaosLLMProviderError("rate limited", provider="openai", status_code=429)
        err = KaosLLMRetryExhaustedError("all retries exhausted", attempts=3, last_error=original)
        assert err.__cause__ is original
