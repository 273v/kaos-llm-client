"""Provider contract tests for MistralClient.

Tests instantiation, base URL, headers, and request/response parsing.
Follows the same pattern as test_openai.py.
"""

from __future__ import annotations

import pytest

from kaos_llm_client.errors import KaosLLMAuthError
from kaos_llm_client.providers.mistral import MistralClient
from kaos_llm_client.types import ProviderRequest


def _make_mistral_client(model: str = "mistral-large-latest") -> MistralClient:
    """Create a MistralClient with a test key (no settings resolution)."""
    return MistralClient(model=model, api_key="test-mistral-key")


def _make_request(request_id: str = "req-test") -> ProviderRequest:
    """Create a minimal ProviderRequest for parse tests."""
    return ProviderRequest(
        provider="mistral",
        model="mistral-large-latest",
        endpoint="/v1/chat/completions",
        body={},
        request_id=request_id,
    )


class TestMistralInstantiation:
    """Tests for MistralClient construction."""

    def test_provider_name(self):
        client = _make_mistral_client()
        assert client._provider_name == "mistral"

    def test_model_stored(self):
        client = _make_mistral_client("mistral-large-latest")
        assert client.model == "mistral-large-latest"

    def test_api_key_override(self):
        client = _make_mistral_client()
        assert client._api_key_override == "test-mistral-key"


class TestMistralBaseUrl:
    """Tests for MistralClient base URL resolution."""

    def test_default_base_url(self):
        client = _make_mistral_client()
        assert client._base_url == "https://api.mistral.ai"

    def test_base_url_override(self):
        client = MistralClient(
            model="mistral-large-latest",
            api_key="test-key",
            base_url="http://localhost:8080",
        )
        assert client._base_url == "http://localhost:8080"


class TestMistralHeaders:
    """Tests for MistralClient header generation."""

    def test_headers_bearer_token(self):
        client = _make_mistral_client()
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer test-mistral-key"
        assert headers["Content-Type"] == "application/json"


class TestMistralBuildRequest:
    """Tests for MistralClient._build_request()."""

    def test_build_request_basic(self):
        client = _make_mistral_client()
        messages = [{"role": "user", "content": "hello"}]
        req = client._build_request(messages)

        assert req.body["model"] == "mistral-large-latest"
        assert req.body["messages"] == messages
        assert req.provider == "mistral"
        assert req.endpoint == "/v1/chat/completions"


class TestMistralParseResponse:
    """Tests for MistralClient._parse_response()."""

    def test_parse_response_text(self):
        client = _make_mistral_client()
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
            "model": "mistral-large-latest",
        }
        request = _make_request()
        resp = client._parse_response(raw, request)

        assert resp.text == "hello"
        assert resp.stop_reason == "stop"
        assert resp.usage.input_tokens == 5
        assert resp.usage.output_tokens == 1


class TestMistralAuthError:
    """Tests for MistralClient auth error handling."""

    def test_missing_api_key_raises(self):
        client = MistralClient(model="mistral-large-latest")
        with pytest.raises(KaosLLMAuthError, match="Mistral API key is not configured"):
            client._get_api_key_from_settings()
