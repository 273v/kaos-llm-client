"""Provider contract tests for GroqClient.

Tests instantiation, base URL, headers, and request/response parsing.
Follows the same pattern as test_openai.py.
"""

from __future__ import annotations

import pytest

from kaos_llm_client.errors import KaosLLMAuthError
from kaos_llm_client.providers.groq import GroqClient
from kaos_llm_client.types import ProviderRequest


def _make_groq_client(model: str = "llama-3.3-70b-versatile") -> GroqClient:
    """Create a GroqClient with a test key (no settings resolution)."""
    return GroqClient(model=model, api_key="test-groq-key")


def _make_request(request_id: str = "req-test") -> ProviderRequest:
    """Create a minimal ProviderRequest for parse tests."""
    return ProviderRequest(
        provider="groq",
        model="llama-3.3-70b-versatile",
        endpoint="/v1/chat/completions",
        body={},
        request_id=request_id,
    )


class TestGroqInstantiation:
    """Tests for GroqClient construction."""

    def test_provider_name(self):
        client = _make_groq_client()
        assert client._provider_name == "groq"

    def test_model_stored(self):
        client = _make_groq_client("llama-3.3-70b-versatile")
        assert client.model == "llama-3.3-70b-versatile"

    def test_api_key_override(self):
        client = _make_groq_client()
        assert client._api_key_override == "test-groq-key"


class TestGroqBaseUrl:
    """Tests for GroqClient base URL resolution."""

    def test_default_base_url(self):
        client = _make_groq_client()
        assert client._base_url == "https://api.groq.com/openai"

    def test_base_url_override(self):
        client = GroqClient(
            model="llama-3.3-70b-versatile",
            api_key="test-key",
            base_url="http://localhost:8080",
        )
        assert client._base_url == "http://localhost:8080"


class TestGroqHeaders:
    """Tests for GroqClient header generation."""

    def test_headers_bearer_token(self):
        client = _make_groq_client()
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer test-groq-key"
        assert headers["Content-Type"] == "application/json"


class TestGroqBuildRequest:
    """Tests for GroqClient._build_request()."""

    def test_build_request_basic(self):
        client = _make_groq_client()
        messages = [{"role": "user", "content": "hello"}]
        req = client._build_request(messages)

        assert req.body["model"] == "llama-3.3-70b-versatile"
        assert req.body["messages"] == messages
        assert req.provider == "groq"
        assert req.endpoint == "/v1/chat/completions"


class TestGroqParseResponse:
    """Tests for GroqClient._parse_response()."""

    def test_parse_response_text(self):
        client = _make_groq_client()
        raw = {
            "choices": [
                {
                    "message": {"content": "hello", "role": "assistant"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "total_tokens": 6,
            },
            "id": "resp-1",
            "model": "llama-3.3-70b-versatile",
        }
        request = _make_request()
        resp = client._parse_response(raw, request)

        assert resp.text == "hello"
        assert resp.stop_reason == "stop"
        assert resp.usage.input_tokens == 5
        assert resp.usage.output_tokens == 1


class TestGroqAuthError:
    """Tests for GroqClient auth error handling."""

    def test_missing_api_key_raises(self):
        client = GroqClient(model="llama-3.3-70b-versatile")
        with pytest.raises(KaosLLMAuthError, match="Groq API key is not configured"):
            client._get_api_key_from_settings()
