"""Tests for the embeddings API (embed_async / embed).

Tests the OpenAI-compatible embeddings endpoint using MockTransport,
as well as the EmbeddingResponse type and NotImplementedError for
unsupported providers.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from kaos_llm_client.providers.anthropic import AnthropicClient
from kaos_llm_client.providers.openai import OpenAIClient
from kaos_llm_client.types import EmbeddingResponse, UsageInfo

# ---------------------------------------------------------------------------
# Canned embedding response
# ---------------------------------------------------------------------------

OPENAI_EMBEDDING_RESPONSE: dict[str, Any] = {
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "index": 0,
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    ],
    "model": "text-embedding-3-small",
    "usage": {
        "prompt_tokens": 5,
        "total_tokens": 5,
    },
}

OPENAI_EMBEDDING_MULTI_RESPONSE: dict[str, Any] = {
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "index": 0,
            "embedding": [0.1, 0.2, 0.3],
        },
        {
            "object": "embedding",
            "index": 1,
            "embedding": [0.4, 0.5, 0.6],
        },
    ],
    "model": "text-embedding-3-small",
    "usage": {
        "prompt_tokens": 10,
        "total_tokens": 10,
    },
}


# ---------------------------------------------------------------------------
# Mock transport helpers
# ---------------------------------------------------------------------------


def _inject_mock_transport(
    client: Any,
    payload: dict[str, Any],
    status: int = 200,
) -> None:
    """Replace the client's httpx clients with mock transports."""

    async def async_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    def sync_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    base_url = client._base_url
    client._async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(async_handler),
        base_url=base_url,
    )
    client._sync_client = httpx.Client(
        transport=httpx.MockTransport(sync_handler),
        base_url=base_url,
    )


# ---------------------------------------------------------------------------
# EmbeddingResponse type tests
# ---------------------------------------------------------------------------


class TestEmbeddingResponseType:
    """Tests for the EmbeddingResponse type."""

    def test_embedding_convenience_property(self):
        resp = EmbeddingResponse(
            provider="openai",
            model="text-embedding-3-small",
            embeddings=[[0.1, 0.2, 0.3]],
        )
        assert resp.embedding == [0.1, 0.2, 0.3]

    def test_embedding_convenience_empty(self):
        resp = EmbeddingResponse(
            provider="openai",
            model="text-embedding-3-small",
            embeddings=[],
        )
        assert resp.embedding == []

    def test_embedding_response_fields(self):
        resp = EmbeddingResponse(
            provider="openai",
            model="text-embedding-3-small",
            embeddings=[[1.0, 2.0]],
            usage=UsageInfo(input_tokens=5, total_tokens=5),
            request_id="req-123",
        )
        assert resp.provider == "openai"
        assert resp.model == "text-embedding-3-small"
        assert resp.request_id == "req-123"
        assert resp.usage.input_tokens == 5


# ---------------------------------------------------------------------------
# OpenAI embed round-trip tests
# ---------------------------------------------------------------------------


class TestOpenAIEmbeddings:
    """Tests for OpenAI-compatible embed_async / embed."""

    async def test_embed_async_single_string(self) -> None:
        """Single string input is normalized to a list."""
        client = OpenAIClient(model="text-embedding-3-small", api_key="test-key")
        _inject_mock_transport(client, OPENAI_EMBEDDING_RESPONSE)

        result = await client.embed_async("hello world")

        assert isinstance(result, EmbeddingResponse)
        assert result.provider == "openai"
        assert result.model == "text-embedding-3-small"
        assert len(result.embeddings) == 1
        assert result.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert result.usage.input_tokens == 5

    async def test_embed_async_list_input(self) -> None:
        """List of strings input."""
        client = OpenAIClient(model="text-embedding-3-small", api_key="test-key")
        _inject_mock_transport(client, OPENAI_EMBEDDING_MULTI_RESPONSE)

        result = await client.embed_async(["hello", "world"])

        assert len(result.embeddings) == 2
        assert result.embeddings[0] == [0.1, 0.2, 0.3]
        assert result.embeddings[1] == [0.4, 0.5, 0.6]
        assert result.usage.input_tokens == 10

    def test_embed_sync(self) -> None:
        """Sync embed() wraps embed_async()."""
        client = OpenAIClient(model="text-embedding-3-small", api_key="test-key")
        _inject_mock_transport(client, OPENAI_EMBEDDING_RESPONSE)

        result = client.embed("hello world")

        assert isinstance(result, EmbeddingResponse)
        assert result.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

    async def test_embed_async_with_dimensions(self) -> None:
        """Dimensions parameter is passed through to the request body."""
        captured_body: dict[str, Any] = {}

        async def capture_handler(request: httpx.Request) -> httpx.Response:
            import json

            captured_body.update(json.loads(request.content.decode()))
            return httpx.Response(200, json=OPENAI_EMBEDDING_RESPONSE)

        client = OpenAIClient(model="text-embedding-3-small", api_key="test-key")
        client._async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(capture_handler),
            base_url=client._base_url,
        )

        await client.embed_async("hello", dimensions=256)

        assert captured_body["dimensions"] == 256

    async def test_embed_async_model_override(self) -> None:
        """Model override is passed through to the request body."""
        captured_body: dict[str, Any] = {}

        async def capture_handler(request: httpx.Request) -> httpx.Response:
            import json

            captured_body.update(json.loads(request.content.decode()))
            return httpx.Response(200, json=OPENAI_EMBEDDING_RESPONSE)

        client = OpenAIClient(model="gpt-5", api_key="test-key")
        client._async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(capture_handler),
            base_url=client._base_url,
        )

        await client.embed_async("hello", model="text-embedding-3-large")

        assert captured_body["model"] == "text-embedding-3-large"


# ---------------------------------------------------------------------------
# Unsupported provider tests
# ---------------------------------------------------------------------------


class TestEmbeddingNotSupported:
    """Tests for providers that don't support embeddings."""

    async def test_anthropic_embed_raises(self) -> None:
        """Anthropic does not support embeddings."""
        client = AnthropicClient(model="claude-sonnet-4-6", api_key="test-key")
        with pytest.raises(NotImplementedError, match="anthropic does not support embeddings"):
            await client.embed_async("hello")

    def test_anthropic_embed_sync_raises(self) -> None:
        """Sync embed also raises NotImplementedError."""
        client = AnthropicClient(model="claude-sonnet-4-6", api_key="test-key")
        with pytest.raises(NotImplementedError, match="anthropic does not support embeddings"):
            client.embed("hello")
