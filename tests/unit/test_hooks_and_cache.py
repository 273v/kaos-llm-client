"""Tests for RequestHooks lifecycle callbacks and CachePoint preprocessing.

Part 1: Full pipeline round-trips with mock HTTP verifying hook invocations.
Part 2: Unit tests for CachePoint -> cache_control conversion (no HTTP).
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from kaos_llm_client.errors import KaosLLMProviderError
from kaos_llm_client.providers.anthropic import AnthropicClient
from kaos_llm_client.providers.openai import OpenAIClient
from kaos_llm_client.transport import RetryPolicy
from kaos_llm_client.types import ProviderRequest, ProviderResponse, RequestHooks

# ---------------------------------------------------------------------------
# Canned responses (reused from test_roundtrip.py pattern)
# ---------------------------------------------------------------------------

OPENAI_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-hooks-1",
    "object": "chat.completion",
    "model": "gpt-5",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from hooks test!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
}

OPENAI_ERROR_BODY: dict[str, Any] = {
    "error": {"message": "Bad request: invalid model", "type": "invalid_request_error"}
}


# ---------------------------------------------------------------------------
# Mock transport helpers
# ---------------------------------------------------------------------------


def _make_async_handler(payload: dict[str, Any], status: int = 200) -> Any:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


def _make_sync_handler(payload: dict[str, Any], status: int = 200) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


def _inject_mock(
    client: Any,
    payload: dict[str, Any],
    status: int = 200,
) -> None:
    base_url = client._base_url
    client._async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_make_async_handler(payload, status)),
        base_url=base_url,
    )
    client._sync_client = httpx.Client(
        transport=httpx.MockTransport(_make_sync_handler(payload, status)),
        base_url=base_url,
    )


# ===========================================================================
# Part 1: RequestHooks tests (full pipeline round-trips with mock HTTP)
# ===========================================================================


class TestRequestHooks:
    """Verify RequestHooks lifecycle callbacks fire through the full pipeline."""

    def test_on_request_fires(self) -> None:
        """on_request callback receives the ProviderRequest before HTTP."""
        captured: list[ProviderRequest] = []

        def on_req(req: ProviderRequest) -> None:
            captured.append(req)

        hooks = RequestHooks(on_request=on_req)
        client = OpenAIClient(model="gpt-5", api_key="test-key", hooks=hooks)
        _inject_mock(client, OPENAI_RESPONSE)

        client.chat([{"role": "user", "content": "Hi"}])

        assert len(captured) == 1
        assert isinstance(captured[0], ProviderRequest)
        assert captured[0].provider == "openai"

    def test_on_response_fires(self) -> None:
        """on_response callback receives (ProviderRequest, ProviderResponse) on success."""
        captured: list[tuple[ProviderRequest, ProviderResponse]] = []

        def on_resp(req: ProviderRequest, resp: ProviderResponse) -> None:
            captured.append((req, resp))

        hooks = RequestHooks(on_response=on_resp)
        client = OpenAIClient(model="gpt-5", api_key="test-key", hooks=hooks)
        _inject_mock(client, OPENAI_RESPONSE)

        client.chat([{"role": "user", "content": "Hi"}])

        assert len(captured) == 1
        req, resp = captured[0]
        assert isinstance(req, ProviderRequest)
        assert isinstance(resp, ProviderResponse)
        assert resp.text == "Hello from hooks test!"

    def test_on_error_fires(self) -> None:
        """on_error callback receives (ProviderRequest, exception) on HTTP error."""
        captured: list[tuple[ProviderRequest, Exception]] = []

        def on_err(req: ProviderRequest, exc: Exception) -> None:
            captured.append((req, exc))

        hooks = RequestHooks(on_error=on_err)
        client = OpenAIClient(model="gpt-5", api_key="test-key", hooks=hooks)
        # Disable retries so we get the error immediately
        client._retry_policy = RetryPolicy(max_retries=0, backoff_base=0.01)
        _inject_mock(client, OPENAI_ERROR_BODY, status=400)

        with pytest.raises(KaosLLMProviderError):
            client.chat([{"role": "user", "content": "Hi"}])

        assert len(captured) == 1
        req, exc = captured[0]
        assert isinstance(req, ProviderRequest)
        assert isinstance(exc, Exception)

    def test_hooks_none_is_safe(self) -> None:
        """Client with hooks=None does not crash -- baseline safety check."""
        client = OpenAIClient(model="gpt-5", api_key="test-key", hooks=None)
        _inject_mock(client, OPENAI_RESPONSE)

        result = client.chat([{"role": "user", "content": "Hi"}])

        assert result.text == "Hello from hooks test!"

    def test_on_request_sees_correct_body(self) -> None:
        """The ProviderRequest in on_request contains model and messages in the body."""
        captured: list[ProviderRequest] = []

        def on_req(req: ProviderRequest) -> None:
            captured.append(req)

        hooks = RequestHooks(on_request=on_req)
        client = OpenAIClient(model="gpt-5", api_key="test-key", hooks=hooks)
        _inject_mock(client, OPENAI_RESPONSE)

        client.chat([{"role": "user", "content": "Test message"}])

        req = captured[0]
        assert req.body["model"] == "gpt-5"
        assert isinstance(req.body["messages"], list)
        assert any(m.get("content") == "Test message" for m in req.body["messages"])

    def test_on_response_sees_usage(self) -> None:
        """The ProviderResponse in on_response contains parsed usage with input_tokens > 0."""
        captured_resp: list[ProviderResponse] = []

        def on_resp(req: ProviderRequest, resp: ProviderResponse) -> None:
            captured_resp.append(resp)

        hooks = RequestHooks(on_response=on_resp)
        client = OpenAIClient(model="gpt-5", api_key="test-key", hooks=hooks)
        _inject_mock(client, OPENAI_RESPONSE)

        client.chat([{"role": "user", "content": "Hi"}])

        resp = captured_resp[0]
        assert resp.usage.input_tokens == 12
        assert resp.usage.output_tokens == 7
        assert resp.usage.total_tokens == 19

    def test_multiple_hooks(self) -> None:
        """When all hooks are set, on_request and on_response both fire on success."""
        req_log: list[ProviderRequest] = []
        resp_log: list[ProviderResponse] = []
        err_log: list[Exception] = []
        retry_log: list[Any] = []

        hooks = RequestHooks(
            on_request=lambda r: req_log.append(r),
            on_response=lambda r, resp: resp_log.append(resp),
            on_error=lambda r, e: err_log.append(e),
            on_retry=lambda *a: retry_log.append(a),
        )
        client = OpenAIClient(model="gpt-5", api_key="test-key", hooks=hooks)
        _inject_mock(client, OPENAI_RESPONSE)

        client.chat([{"role": "user", "content": "Hi"}])

        # on_request and on_response should fire; on_error and on_retry should not
        assert len(req_log) == 1
        assert len(resp_log) == 1
        assert len(err_log) == 0
        assert len(retry_log) == 0

    async def test_on_retry_fires_on_429(self) -> None:
        """on_retry fires when a non-streaming request retries on 429."""
        retry_log: list[tuple[Any, ...]] = []
        state = {"attempt": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            state["attempt"] += 1
            if state["attempt"] == 1:
                return httpx.Response(429, json={"error": {"message": "rate limited"}})
            return httpx.Response(200, json=OPENAI_RESPONSE)

        hooks = RequestHooks(
            on_retry=lambda req, attempt, exc: retry_log.append((attempt, str(exc)))
        )
        client = OpenAIClient(model="gpt-5", api_key="test-key", hooks=hooks, max_retries=2)
        client._retry_policy = RetryPolicy(max_retries=2, backoff_base=0.0)
        client._async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url=client._base_url
        )

        result = await client.chat_async([{"role": "user", "content": "Hi"}])

        assert result.text == "Hello from hooks test!"
        assert len(retry_log) == 1
        assert retry_log[0][0] == 0  # attempt index
        assert "429" in retry_log[0][1] or "rate" in retry_log[0][1].lower()


# ===========================================================================
# Part 2: CachePoint preprocessing tests (unit, no HTTP)
# ===========================================================================


class TestCachePointPreprocessing:
    """Verify CachePoint marker handling in _preprocess_messages()."""

    def test_base_strips_cache_points(self) -> None:
        """OpenAI-compatible base strips cache_point role messages entirely."""
        client = OpenAIClient(model="gpt-5", api_key="test-key")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "cache_point"},
            {"role": "user", "content": "Hello"},
        ]

        result = client._preprocess_messages(messages)

        assert len(result) == 2
        assert all(m["role"] != "cache_point" for m in result)
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_anthropic_converts_cache_point_string_content(self) -> None:
        """Anthropic converts CachePoint after a string-content message to cache_control list."""
        client = AnthropicClient(model="claude-sonnet-4-6", api_key="test-key")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "Long system prompt for caching."},
            {"role": "cache_point"},
            {"role": "user", "content": "What is the answer?"},
        ]

        result = client._preprocess_messages(messages)

        # Should have 2 messages (cache_point stripped)
        assert len(result) == 2

        # System message content should now be a list with cache_control
        system_msg = result[0]
        assert system_msg["role"] == "system"
        assert isinstance(system_msg["content"], list)
        assert len(system_msg["content"]) == 1
        block = system_msg["content"][0]
        assert block["type"] == "text"
        assert block["text"] == "Long system prompt for caching."
        assert block["cache_control"] == {"type": "ephemeral"}

        # User message should be unchanged
        assert result[1] == {"role": "user", "content": "What is the answer?"}

    def test_anthropic_converts_cache_point_list_content(self) -> None:
        """Anthropic adds cache_control to last block when content is already a list."""
        client = AnthropicClient(model="claude-sonnet-4-6", api_key="test-key")
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Part one."},
                    {"type": "text", "text": "Part two."},
                ],
            },
            {"role": "cache_point"},
            {"role": "user", "content": "Question"},
        ]

        result = client._preprocess_messages(messages)

        assert len(result) == 2
        system_content = result[0]["content"]
        assert isinstance(system_content, list)
        # cache_control added to the LAST block only
        assert "cache_control" not in system_content[0]
        assert system_content[1]["cache_control"] == {"type": "ephemeral"}

    def test_anthropic_cache_point_at_start_ignored(self) -> None:
        """CachePoint as first message (nothing before it) does not crash."""
        client = AnthropicClient(model="claude-sonnet-4-6", api_key="test-key")
        messages: list[dict[str, Any]] = [
            {"role": "cache_point"},
            {"role": "user", "content": "Hello"},
        ]

        result = client._preprocess_messages(messages)

        # CachePoint stripped, no crash
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_anthropic_multiple_cache_points(self) -> None:
        """Two CachePoints at different positions both add cache_control."""
        client = AnthropicClient(model="claude-sonnet-4-6", api_key="test-key")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "System prompt."},
            {"role": "cache_point"},
            {"role": "user", "content": "First user message."},
            {"role": "cache_point"},
            {"role": "user", "content": "Second user message."},
        ]

        result = client._preprocess_messages(messages)

        # 3 messages: system, user, user
        assert len(result) == 3

        # System message should have cache_control (from first cache_point)
        system_content = result[0]["content"]
        assert isinstance(system_content, list)
        assert system_content[0]["cache_control"] == {"type": "ephemeral"}

        # First user message should have cache_control (from second cache_point)
        # String content gets converted to list
        user1_content = result[1]["content"]
        assert isinstance(user1_content, list)
        assert user1_content[0]["cache_control"] == {"type": "ephemeral"}

        # Second user message should be untouched
        assert result[2]["content"] == "Second user message."
