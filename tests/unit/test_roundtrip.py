"""Round-trip tests: chat() -> HTTP -> response -> parse through the full pipeline.

Uses httpx.MockTransport to inject fake HTTP responses, then verifies the complete
code path through chat(), chat_async(), json(), pydantic(), and streaming.

Every test exercises the real code path:
  chat() -> run_sync(chat_async()) -> request_async() -> _build_request()
  -> httpx POST (mocked) -> raise_for_status -> _parse_response() -> cache -> return
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from kaos_llm_client.cache import FileCache
from kaos_llm_client.errors import KaosLLMAuthError, KaosLLMProviderError
from kaos_llm_client.providers.anthropic import AnthropicClient
from kaos_llm_client.providers.google import GoogleClient
from kaos_llm_client.providers.openai import OpenAIClient
from kaos_llm_client.transport import RetryPolicy
from kaos_llm_client.types import (
    CachePolicy,
    RequestOptions,
    ToolChoice,
    ToolDefinition,
)

# ---------------------------------------------------------------------------
# Canned provider response payloads
# ---------------------------------------------------------------------------

OPENAI_CHAT_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "model": "gpt-5",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

OPENAI_TOOL_CALL_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-456",
    "object": "chat.completion",
    "model": "gpt-5",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "London"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
}

OPENAI_JSON_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-789",
    "object": "chat.completion",
    "model": "gpt-5",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": '{"name": "Alice", "age": 30}',
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
}

ANTHROPIC_CHAT_RESPONSE: dict[str, Any] = {
    "id": "msg-123",
    "type": "message",
    "model": "claude-sonnet-4-6",
    "content": [{"type": "text", "text": "Hello!"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}

ANTHROPIC_THINKING_RESPONSE: dict[str, Any] = {
    "id": "msg-456",
    "type": "message",
    "model": "claude-sonnet-4-6",
    "content": [
        {"type": "thinking", "thinking": "Let me think about this..."},
        {"type": "text", "text": "The answer is 42."},
    ],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 15, "output_tokens": 20},
}

ANTHROPIC_TOOL_USE_RESPONSE: dict[str, Any] = {
    "id": "msg-789",
    "type": "message",
    "model": "claude-sonnet-4-6",
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_abc",
            "name": "get_weather",
            "input": {"city": "Paris"},
        }
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 12, "output_tokens": 8},
}

GOOGLE_CHAT_RESPONSE: dict[str, Any] = {
    "candidates": [
        {
            "content": {"parts": [{"text": "Hello!"}], "role": "model"},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 10,
        "candidatesTokenCount": 5,
        "totalTokenCount": 15,
    },
}


# ---------------------------------------------------------------------------
# Mock transport helpers
# ---------------------------------------------------------------------------


def _make_async_handler(payload: dict[str, Any], status: int = 200) -> Any:
    """Return an async transport handler that returns a fixed JSON response."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


def _make_sync_handler(payload: dict[str, Any], status: int = 200) -> Any:
    """Return a sync transport handler that returns a fixed JSON response."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


def _inject_mock_transport(
    client: Any,
    payload: dict[str, Any],
    status: int = 200,
) -> None:
    """Replace the client's async and sync httpx clients with mock transports.

    Must include base_url so relative endpoints (e.g., /v1/chat/completions)
    resolve correctly through httpx.
    """
    base_url = client._base_url
    client._async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_make_async_handler(payload, status)),
        base_url=base_url,
    )
    client._sync_client = httpx.Client(
        transport=httpx.MockTransport(_make_sync_handler(payload, status)),
        base_url=base_url,
    )


# ---------------------------------------------------------------------------
# OpenAI round-trip tests
# ---------------------------------------------------------------------------


class TestOpenAIRoundtrips:
    """Full pipeline round-trips through the OpenAI client."""

    def test_openai_chat_roundtrip(self) -> None:
        """Sync chat() through the full pipeline with a mocked transport."""
        client = OpenAIClient(model="gpt-5", api_key="test-key")
        _inject_mock_transport(client, OPENAI_CHAT_RESPONSE)

        result = client.chat([{"role": "user", "content": "Hi"}])

        assert result.text == "Hello!"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.usage.total_tokens == 15
        assert result.stop_reason == "stop"
        assert result.provider == "openai"
        assert result.model == "gpt-5"
        assert result.response_id == "chatcmpl-123"

    async def test_openai_chat_async_roundtrip(self) -> None:
        """Async chat_async() through the full pipeline."""
        client = OpenAIClient(model="gpt-5", api_key="test-key")
        _inject_mock_transport(client, OPENAI_CHAT_RESPONSE)

        result = await client.chat_async([{"role": "user", "content": "Hi"}])

        assert result.text == "Hello!"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.stop_reason == "stop"
        assert result.response_id == "chatcmpl-123"

    def test_openai_tool_calling_roundtrip(self) -> None:
        """Tool calling via chat() -- mock returns tool_calls, verify structured parse."""
        client = OpenAIClient(model="gpt-5", api_key="test-key")
        _inject_mock_transport(client, OPENAI_TOOL_CALL_RESPONSE)

        weather_tool = ToolDefinition(
            name="get_weather",
            description="Get current weather",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )

        result = client.chat(
            [{"role": "user", "content": "Weather in London?"}],
            tools=[weather_tool],
            tool_choice=ToolChoice(type="auto"),
        )

        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "London"}
        assert tc.id == "call_abc"
        assert result.stop_reason == "tool_calls"

    def test_openai_json_roundtrip(self) -> None:
        """json() call with schema, verify response.output_json."""
        client = OpenAIClient(model="gpt-5", api_key="test-key")
        _inject_mock_transport(client, OPENAI_JSON_RESPONSE)

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }

        result = client.json(
            [{"role": "user", "content": "Give me a person record"}],
            schema=schema,
        )

        assert result.output_json == {"name": "Alice", "age": 30}
        assert result.text == '{"name": "Alice", "age": 30}'

    def test_openai_pydantic_roundtrip(self) -> None:
        """pydantic() call with a Pydantic model, verify typed result."""

        class Person(BaseModel):
            name: str
            age: int

        client = OpenAIClient(model="gpt-5", api_key="test-key")
        _inject_mock_transport(client, OPENAI_JSON_RESPONSE)

        person = client.pydantic(
            [{"role": "user", "content": "Give me a person record"}],
            output_type=Person,
        )

        assert isinstance(person, Person)
        assert person.name == "Alice"
        assert person.age == 30
        # Verify the internal _response was attached
        assert hasattr(person, "_response")

    async def test_openai_streaming_roundtrip(self) -> None:
        """Full async streaming: mock returns SSE data lines, verify accumulated text."""
        sse_chunks = [
            {
                "id": "chatcmpl-s1",
                "choices": [{"delta": {"role": "assistant", "content": "Hello"}}],
            },
            {"id": "chatcmpl-s1", "choices": [{"delta": {"content": " world"}}]},
            {
                "id": "chatcmpl-s1",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        ]

        sse_body = ""
        for chunk in sse_chunks:
            sse_body += f"data: {json.dumps(chunk)}\n\n"
        sse_body += "data: [DONE]\n\n"

        async def stream_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_body.encode(),
                headers={"content-type": "text/event-stream"},
            )

        client = OpenAIClient(model="gpt-5", api_key="test-key")
        client._async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(stream_handler),
            base_url=client._base_url,
        )

        collected_text = []
        async for chunk in client.chat_stream_async(
            [{"role": "user", "content": "Hi"}],
        ):
            if chunk.type == "text_delta" and chunk.text:
                collected_text.append(chunk.text)

        assert "".join(collected_text) == "Hello world"


# ---------------------------------------------------------------------------
# Anthropic round-trip tests
# ---------------------------------------------------------------------------


class TestAnthropicRoundtrips:
    """Full pipeline round-trips through the Anthropic client."""

    def test_anthropic_chat_roundtrip(self) -> None:
        """Sync chat() with Anthropic content blocks format."""
        client = AnthropicClient(model="claude-sonnet-4-6", api_key="test-key")
        _inject_mock_transport(client, ANTHROPIC_CHAT_RESPONSE)

        result = client.chat([{"role": "user", "content": "Hi"}])

        assert result.text == "Hello!"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.stop_reason == "end_turn"
        assert result.provider == "anthropic"
        assert result.response_id == "msg-123"

    def test_anthropic_thinking_roundtrip(self) -> None:
        """Mock returns thinking + text blocks, verify both .thinking and .text."""
        client = AnthropicClient(model="claude-sonnet-4-6", api_key="test-key")
        _inject_mock_transport(client, ANTHROPIC_THINKING_RESPONSE)

        result = client.chat([{"role": "user", "content": "What is 6 * 7?"}])

        assert result.thinking == "Let me think about this..."
        assert result.text == "The answer is 42."
        assert result.usage.input_tokens == 15
        assert result.usage.output_tokens == 20

    def test_anthropic_tool_use_roundtrip(self) -> None:
        """Mock returns tool_use blocks, verify .tool_calls."""
        client = AnthropicClient(model="claude-sonnet-4-6", api_key="test-key")
        _inject_mock_transport(client, ANTHROPIC_TOOL_USE_RESPONSE)

        weather_tool = ToolDefinition(
            name="get_weather",
            description="Get current weather",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )

        result = client.chat(
            [{"role": "user", "content": "Weather in Paris?"}],
            tools=[weather_tool],
        )

        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "Paris"}
        assert tc.id == "toolu_abc"
        assert result.stop_reason == "tool_use"


# ---------------------------------------------------------------------------
# Google round-trip tests
# ---------------------------------------------------------------------------


class TestGoogleRoundtrips:
    """Full pipeline round-trips through the Google client."""

    def test_google_chat_roundtrip(self) -> None:
        """Sync chat() with Google candidates format."""
        client = GoogleClient(model="gemini-2.5-pro", api_key="test-key")
        _inject_mock_transport(client, GOOGLE_CHAT_RESPONSE)

        result = client.chat([{"role": "user", "content": "Hi"}])

        assert result.text == "Hello!"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.usage.total_tokens == 15
        assert result.stop_reason == "STOP"
        assert result.provider == "google"


# ---------------------------------------------------------------------------
# Error handling round-trip tests
# ---------------------------------------------------------------------------


class TestErrorRoundtrips:
    """Error handling through the full pipeline."""

    async def test_retry_on_429(self) -> None:
        """Mock returns 429 first, then 200 on retry -- verify success."""
        attempt = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                return httpx.Response(
                    429,
                    json={"error": {"message": "rate limited"}},
                )
            return httpx.Response(200, json=OPENAI_CHAT_RESPONSE)

        client = OpenAIClient(model="gpt-5", api_key="test-key", max_retries=2)
        # Very short backoff so the test is fast
        client._retry_policy = RetryPolicy(max_retries=2, backoff_base=0.01)
        client._async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url=client._base_url,
        )

        result = await client.chat_async([{"role": "user", "content": "Hi"}])

        assert result.text == "Hello!"
        assert attempt == 2

    async def test_auth_error_no_retry(self) -> None:
        """Mock returns 401 -- verify KaosLLMAuthError raised immediately, no retry."""
        attempt = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempt
            attempt += 1
            return httpx.Response(
                401,
                json={"error": {"message": "invalid api key"}},
            )

        client = OpenAIClient(model="gpt-5", api_key="test-key", max_retries=3)
        client._async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url=client._base_url,
        )

        with pytest.raises(KaosLLMAuthError):
            await client.chat_async([{"role": "user", "content": "Hi"}])

        # Auth errors are never retried
        assert attempt == 1

    async def test_provider_error_400(self) -> None:
        """Mock returns 400 with error body -- verify KaosLLMProviderError with details."""
        error_body = {
            "error": {"message": "Invalid model specified", "type": "invalid_request_error"}
        }

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json=error_body)

        client = OpenAIClient(model="gpt-5", api_key="test-key")
        client._async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url=client._base_url,
        )

        with pytest.raises(KaosLLMProviderError) as exc_info:
            await client.chat_async([{"role": "user", "content": "Hi"}])

        assert exc_info.value.status_code == 400
        assert exc_info.value.raw_error == error_body


# ---------------------------------------------------------------------------
# Cache round-trip tests
# ---------------------------------------------------------------------------


class TestCacheRoundtrips:
    """Cache behavior through the full pipeline."""

    def test_cache_hit_skips_http(self, tmp_path: Any) -> None:
        """Two identical requests with FileCache: second must not hit transport."""
        call_count = 0

        def sync_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, json=OPENAI_CHAT_RESPONSE)

        async def async_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, json=OPENAI_CHAT_RESPONSE)

        cache = FileCache(tmp_path / "llm_cache")
        client = OpenAIClient(model="gpt-5", api_key="test-key", cache=cache)
        base = client._base_url
        client._async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(async_handler),
            base_url=base,
        )
        client._sync_client = httpx.Client(
            transport=httpx.MockTransport(sync_handler),
            base_url=base,
        )

        messages = [{"role": "user", "content": "Hi"}]
        options = RequestOptions(cache_policy=CachePolicy.FORCE)

        # First call -- hits transport
        r1 = client.chat(messages, options=options)
        assert call_count == 1
        assert r1.text == "Hello!"

        # Second call -- should come from cache
        r2 = client.chat(messages, options=options)
        assert call_count == 1  # no additional HTTP call
        assert r2.text == "Hello!"

    def test_cache_skip_policy(self, tmp_path: Any) -> None:
        """CachePolicy.SKIP always hits transport, even with FileCache."""
        call_count = 0

        def sync_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, json=OPENAI_CHAT_RESPONSE)

        async def async_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, json=OPENAI_CHAT_RESPONSE)

        cache = FileCache(tmp_path / "llm_cache")
        client = OpenAIClient(model="gpt-5", api_key="test-key", cache=cache)
        base = client._base_url
        client._async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(async_handler),
            base_url=base,
        )
        client._sync_client = httpx.Client(
            transport=httpx.MockTransport(sync_handler),
            base_url=base,
        )

        messages = [{"role": "user", "content": "Hi"}]
        options = RequestOptions(cache_policy=CachePolicy.SKIP)

        # Both calls should hit transport
        client.chat(messages, options=options)
        assert call_count == 1

        client.chat(messages, options=options)
        assert call_count == 2


# ---------------------------------------------------------------------------
# Sync wrapper round-trip test
# ---------------------------------------------------------------------------


class TestSyncWrapper:
    """Verify the sync wrapper correctly drives the async pipeline."""

    def test_sync_wrapper_works(self) -> None:
        """chat() (sync) must traverse the same path as chat_async(), end to end."""
        client = OpenAIClient(model="gpt-5", api_key="test-key")
        _inject_mock_transport(client, OPENAI_CHAT_RESPONSE)

        # Verify the sync path (chat -> run_sync -> request_async -> _build_request
        # -> httpx POST -> _parse_response)
        result = client.chat([{"role": "user", "content": "Hi"}])

        assert result.text == "Hello!"
        assert result.provider == "openai"
        assert result.model == "gpt-5"
        assert result.usage.total_tokens == 15
        assert result.stop_reason == "stop"
        assert result.raw == OPENAI_CHAT_RESPONSE
