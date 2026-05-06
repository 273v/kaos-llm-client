"""Provider contract tests for OpenRouterClient.

Tests instantiation, base URL, headers (including HTTP-Referer),
and request/response parsing. Follows the same pattern as test_openai.py.
"""

from __future__ import annotations

import pytest

from kaos_llm_client.errors import KaosLLMAuthError
from kaos_llm_client.providers.openrouter import OpenRouterClient
from kaos_llm_client.settings import KaosLLMSettings
from kaos_llm_client.types import ProviderRequest


def _make_openrouter_client(model: str = "openai/gpt-5") -> OpenRouterClient:
    """Create an OpenRouterClient with a test key (no settings resolution)."""
    return OpenRouterClient(model=model, api_key="test-openrouter-key")


def _make_request(request_id: str = "req-test") -> ProviderRequest:
    """Create a minimal ProviderRequest for parse tests."""
    return ProviderRequest(
        provider="openrouter",
        model="openai/gpt-5",
        endpoint="/v1/chat/completions",
        body={},
        request_id=request_id,
    )


class TestOpenRouterInstantiation:
    """Tests for OpenRouterClient construction."""

    def test_provider_name(self):
        client = _make_openrouter_client()
        assert client._provider_name == "openrouter"

    def test_model_stored(self):
        client = _make_openrouter_client("openai/gpt-5")
        assert client.model == "openai/gpt-5"

    def test_api_key_override(self):
        client = _make_openrouter_client()
        assert client._api_key_override == "test-openrouter-key"


class TestOpenRouterBaseUrl:
    """Tests for OpenRouterClient base URL resolution."""

    def test_default_base_url(self):
        client = _make_openrouter_client()
        assert client._base_url == "https://openrouter.ai/api"

    def test_base_url_override(self):
        client = OpenRouterClient(
            model="openai/gpt-5",
            api_key="test-key",
            base_url="http://localhost:8080",
        )
        assert client._base_url == "http://localhost:8080"


class TestOpenRouterHeaders:
    """Tests for OpenRouterClient header generation."""

    def test_headers_bearer_token(self):
        client = _make_openrouter_client()
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer test-openrouter-key"
        assert headers["Content-Type"] == "application/json"

    def test_headers_no_referer_by_default(self):
        client = _make_openrouter_client()
        headers = client._build_headers()
        assert "HTTP-Referer" not in headers

    def test_headers_with_referer(self):
        settings = KaosLLMSettings(openrouter_site_url="https://myapp.example.com")
        client = OpenRouterClient(
            model="openai/gpt-5",
            api_key="test-key",
            settings=settings,
        )
        headers = client._build_headers()
        assert headers["HTTP-Referer"] == "https://myapp.example.com"
        assert headers["Authorization"] == "Bearer test-key"


class TestOpenRouterBuildRequest:
    """Tests for OpenRouterClient._build_request()."""

    def test_build_request_basic(self):
        client = _make_openrouter_client()
        messages = [{"role": "user", "content": "hello"}]
        req = client._build_request(messages)

        assert req.body["model"] == "openai/gpt-5"
        assert req.body["messages"] == messages
        assert req.provider == "openrouter"
        assert req.endpoint == "/v1/chat/completions"


class TestOpenRouterParseResponse:
    """Tests for OpenRouterClient._parse_response()."""

    def test_parse_response_text(self):
        client = _make_openrouter_client()
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
            "model": "openai/gpt-5",
        }
        request = _make_request()
        resp = client._parse_response(raw, request)

        assert resp.text == "hello"
        assert resp.stop_reason == "stop"
        assert resp.usage.input_tokens == 5
        assert resp.usage.output_tokens == 1


class TestOpenRouterAuthError:
    """Tests for OpenRouterClient auth error handling."""

    def test_missing_api_key_raises(self):
        client = OpenRouterClient(model="openai/gpt-5")
        with pytest.raises(KaosLLMAuthError, match="OpenRouter API key is not configured"):
            client._get_api_key_from_settings()
