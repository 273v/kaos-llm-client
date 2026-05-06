"""Provider contract tests for OpenAIResponsesClient.

Tests _messages_to_input_items conversion, _build_request, _parse_response
for text/tool_calls/reasoning, builtin_tools, previous_response_id, and
round-trip with MockTransport.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from kaos_llm_client.errors import KaosLLMAuthError
from kaos_llm_client.providers.openai_responses import (
    OpenAIResponsesClient,
    _messages_to_input_items,
)
from kaos_llm_client.types import (
    ProviderRequest,
    ProviderResponse,
    StreamChunk,
    ToolDefinition,
)


def _single(result: StreamChunk | list[StreamChunk]) -> StreamChunk:
    """Extract single chunk from parse result (may be list)."""
    return result[0] if isinstance(result, list) else result


# ---------------------------------------------------------------------------
# Canned responses
# ---------------------------------------------------------------------------

RESPONSES_TEXT_RESPONSE: dict[str, Any] = {
    "id": "resp-abc123",
    "model": "gpt-5",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello! How can I help?"}],
        }
    ],
    "usage": {
        "input_tokens": 10,
        "output_tokens": 8,
        "total_tokens": 18,
    },
}

RESPONSES_TOOL_CALL_RESPONSE: dict[str, Any] = {
    "id": "resp-def456",
    "model": "gpt-5",
    "status": "completed",
    "output": [
        {
            "type": "function_call",
            "call_id": "call_xyz",
            "name": "get_weather",
            "arguments": '{"city": "NYC"}',
        }
    ],
    "usage": {
        "input_tokens": 15,
        "output_tokens": 10,
        "total_tokens": 25,
    },
}

RESPONSES_REASONING_RESPONSE: dict[str, Any] = {
    "id": "resp-ghi789",
    "model": "o3",
    "status": "completed",
    "output": [
        {
            "type": "reasoning",
            "summary": [
                {"type": "summary_text", "text": "Let me think about this..."},
            ],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "The answer is 42."}],
        },
    ],
    "usage": {
        "input_tokens": 20,
        "output_tokens": 30,
        "total_tokens": 50,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(model: str = "gpt-5") -> OpenAIResponsesClient:
    """Create an OpenAIResponsesClient with a test key."""
    return OpenAIResponsesClient(model=model, api_key="test-key")


def _make_request(request_id: str = "req-test") -> ProviderRequest:
    """Create a minimal ProviderRequest for parse tests."""
    return ProviderRequest(
        provider="openai-responses",
        model="gpt-5",
        endpoint="/v1/responses",
        body={},
        request_id=request_id,
    )


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
# Tests: _messages_to_input_items
# ---------------------------------------------------------------------------


class TestMessagesToInputItems:
    """Tests for _messages_to_input_items conversion."""

    def test_user_message(self):
        items = _messages_to_input_items([{"role": "user", "content": "hello"}])
        assert items == [{"type": "message", "role": "user", "content": "hello"}]

    def test_system_becomes_developer(self):
        items = _messages_to_input_items([{"role": "system", "content": "be helpful"}])
        assert items == [{"type": "message", "role": "developer", "content": "be helpful"}]

    def test_assistant_message(self):
        items = _messages_to_input_items([{"role": "assistant", "content": "hi there"}])
        assert items == [{"type": "message", "role": "assistant", "content": "hi there"}]

    def test_tool_result(self):
        items = _messages_to_input_items(
            [{"role": "tool", "content": '{"temp": 72}', "tool_call_id": "call_abc"}]
        )
        assert items == [
            {"type": "function_call_output", "call_id": "call_abc", "output": '{"temp": 72}'}
        ]

    def test_multi_message_conversation(self):
        messages = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "bye"},
        ]
        items = _messages_to_input_items(messages)
        assert len(items) == 4
        assert items[0]["role"] == "developer"
        assert items[1]["role"] == "user"
        assert items[2]["role"] == "assistant"
        assert items[3]["role"] == "user"


# ---------------------------------------------------------------------------
# Tests: Instantiation
# ---------------------------------------------------------------------------


class TestOpenAIResponsesInstantiation:
    """Tests for OpenAIResponsesClient construction."""

    def test_provider_name(self):
        client = _make_client()
        assert client._provider_name == "openai-responses"

    def test_default_endpoint(self):
        client = _make_client()
        assert client._default_endpoint() == "/v1/responses"

    def test_default_base_url(self):
        client = _make_client()
        assert client._base_url == "https://api.openai.com"

    def test_auth_error_when_no_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("KAOS_LLM_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = OpenAIResponsesClient(model="gpt-5")
        with pytest.raises(KaosLLMAuthError, match="OpenAI API key is not configured"):
            client._get_api_key_from_settings()


# ---------------------------------------------------------------------------
# Tests: _build_request
# ---------------------------------------------------------------------------


class TestOpenAIResponsesBuildRequest:
    """Tests for OpenAIResponsesClient._build_request()."""

    def test_build_request_basic(self):
        client = _make_client()
        messages = [{"role": "user", "content": "hello"}]
        req = client._build_request(messages)

        assert req.body["model"] == "gpt-5"
        assert req.body["input"] == [{"type": "message", "role": "user", "content": "hello"}]
        assert req.endpoint == "/v1/responses"
        assert req.provider == "openai-responses"

    def test_build_request_with_builtin_tools(self):
        client = _make_client()
        messages = [{"role": "user", "content": "search the web"}]
        req = client._build_request(
            messages,
            builtin_tools=[{"type": "web_search"}],
        )

        assert req.body["tools"] == [{"type": "web_search"}]

    def test_build_request_with_function_tools(self):
        client = _make_client()
        tool = ToolDefinition(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        req = client._build_request(
            [{"role": "user", "content": "weather?"}],
            tools=[tool],
        )

        assert len(req.body["tools"]) == 1
        t = req.body["tools"][0]
        assert t["type"] == "function"
        assert t["name"] == "get_weather"

    def test_build_request_mixed_tools(self):
        client = _make_client()
        tool = ToolDefinition(
            name="calc",
            description="Calculate",
            parameters={"type": "object", "properties": {}},
        )
        req = client._build_request(
            [{"role": "user", "content": "hi"}],
            tools=[tool],
            builtin_tools=[{"type": "code_interpreter"}],
        )

        assert len(req.body["tools"]) == 2

    def test_build_request_with_reasoning(self):
        client = _make_client()
        req = client._build_request(
            [{"role": "user", "content": "think"}],
            reasoning={"effort": "high", "summary": "auto"},
        )

        assert req.body["reasoning"] == {"effort": "high", "summary": "auto"}

    def test_build_request_reasoning_string(self):
        client = _make_client()
        req = client._build_request(
            [{"role": "user", "content": "think"}],
            reasoning="low",
        )

        assert req.body["reasoning"] == {"effort": "low"}

    def test_build_request_with_previous_response_id(self):
        client = _make_client()
        req = client._build_request(
            [{"role": "user", "content": "continue"}],
            previous_response_id="resp-abc123",
        )

        assert req.body["previous_response_id"] == "resp-abc123"

    def test_build_request_streaming(self):
        client = _make_client()
        req = client._build_request(
            [{"role": "user", "content": "hi"}],
            stream=True,
        )

        assert req.body["stream"] is True
        assert req.stream is True


# ---------------------------------------------------------------------------
# Tests: _parse_response
# ---------------------------------------------------------------------------


class TestOpenAIResponsesParseResponse:
    """Tests for OpenAIResponsesClient._parse_response()."""

    def test_parse_text_response(self):
        client = _make_client()
        request = _make_request()
        resp = client._parse_response(RESPONSES_TEXT_RESPONSE, request)

        assert isinstance(resp, ProviderResponse)
        assert resp.text == "Hello! How can I help?"
        assert resp.response_id == "resp-abc123"
        assert resp.stop_reason == "completed"
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 8
        assert resp.usage.total_tokens == 18

    def test_parse_tool_call_response(self):
        client = _make_client()
        request = _make_request()
        resp = client._parse_response(RESPONSES_TOOL_CALL_RESPONSE, request)

        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.id == "call_xyz"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "NYC"}

    def test_parse_reasoning_response(self):
        client = _make_client()
        request = _make_request()
        resp = client._parse_response(RESPONSES_REASONING_RESPONSE, request)

        assert resp.thinking == "Let me think about this..."
        assert resp.text == "The answer is 42."
        assert resp.usage.total_tokens == 50


# ---------------------------------------------------------------------------
# Tests: _parse_stream_chunk
# ---------------------------------------------------------------------------


class TestOpenAIResponsesParseStream:
    """Tests for OpenAIResponsesClient._parse_stream_chunk()."""

    def test_text_delta(self):
        client = _make_client()
        chunk = _single(
            client._parse_stream_chunk(
                {
                    "type": "response.output_text.delta",
                    "delta": "Hello",
                }
            )
        )
        assert chunk.type == "text_delta"
        assert chunk.text == "Hello"

    def test_output_item_added_function_call(self):
        """response.output_item.added carries call_id and name."""
        client = _make_client()
        chunk = _single(
            client._parse_stream_chunk(
                {
                    "type": "response.output_item.added",
                    "item": {
                        "type": "function_call",
                        "call_id": "call_abc",
                        "name": "get_weather",
                    },
                }
            )
        )
        assert chunk.type == "tool_call_delta"
        assert chunk.tool_call_delta is not None
        assert chunk.tool_call_delta["id"] == "call_abc"
        assert chunk.tool_call_delta["name"] == "get_weather"

    def test_tool_call_delta(self):
        """response.function_call_arguments.delta carries argument fragments."""
        client = _make_client()
        chunk = _single(
            client._parse_stream_chunk(
                {
                    "type": "response.function_call_arguments.delta",
                    "delta": '{"city":',
                    "item_id": "item_123",
                }
            )
        )
        assert chunk.type == "tool_call_delta"
        assert chunk.tool_call_delta is not None
        assert chunk.tool_call_delta["arguments"] == '{"city":'

    def test_completed_with_usage(self):
        client = _make_client()
        chunk = _single(
            client._parse_stream_chunk(
                {
                    "type": "response.completed",
                    "response": {
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        }
                    },
                }
            )
        )
        assert chunk.type == "usage"
        assert chunk.usage is not None
        assert chunk.usage.input_tokens == 10
        assert chunk.usage.output_tokens == 5

    def test_unknown_event_type(self):
        client = _make_client()
        chunk = _single(client._parse_stream_chunk({"type": "response.created"}))
        assert chunk.type == "text_delta"
        assert chunk.text == ""


# ---------------------------------------------------------------------------
# Tests: Round-trip with MockTransport
# ---------------------------------------------------------------------------


class TestOpenAIResponsesRoundtrip:
    """Full pipeline round-trips through the Responses client."""

    def test_chat_roundtrip(self) -> None:
        """Sync chat() through the full pipeline."""
        client = _make_client()
        _inject_mock_transport(client, RESPONSES_TEXT_RESPONSE)

        result = client.chat([{"role": "user", "content": "Hi"}])

        assert result.text == "Hello! How can I help?"
        assert result.provider == "openai-responses"
        assert result.model == "gpt-5"
        assert result.response_id == "resp-abc123"
        assert result.usage.input_tokens == 10

    async def test_chat_async_roundtrip(self) -> None:
        """Async chat_async() through the full pipeline."""
        client = _make_client()
        _inject_mock_transport(client, RESPONSES_TEXT_RESPONSE)

        result = await client.chat_async([{"role": "user", "content": "Hi"}])

        assert result.text == "Hello! How can I help?"
        assert result.response_id == "resp-abc123"

    def test_tool_call_roundtrip(self) -> None:
        """Tool call round-trip."""
        client = _make_client()
        _inject_mock_transport(client, RESPONSES_TOOL_CALL_RESPONSE)

        tool = ToolDefinition(
            name="get_weather",
            description="Get weather",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        )

        result = client.chat(
            [{"role": "user", "content": "Weather in NYC?"}],
            tools=[tool],
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "NYC"}
