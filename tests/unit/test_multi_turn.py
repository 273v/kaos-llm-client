"""Multi-turn tool-use continuation tests.

Verifies the FULL tool-use round-trip: call -> tool_calls -> AssistantMessage.from_response()
-> ToolResultMessage -> second call. This is the continuation flow that requires
provider-aware message formatting (Anthropic uses content blocks, OpenAI uses tool_calls).

Uses httpx.MockTransport to test without hitting real APIs. Each test exercises the
complete code path through chat() including request building, response parsing, and
message conversion for the continuation turn.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from kaos_llm_client.messages import AssistantMessage, ToolResultMessage, UserMessage
from kaos_llm_client.providers.anthropic import AnthropicClient
from kaos_llm_client.providers.google import GoogleClient
from kaos_llm_client.providers.openai import OpenAIClient
from kaos_llm_client.types import (
    ProviderRequest,
    StreamAccumulator,
    StreamChunk,
    ToolChoice,
    ToolDefinition,
    UsageInfo,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

WEATHER_TOOL = ToolDefinition(
    name="get_temperature",
    description="Get the current temperature for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)

# --- OpenAI canned responses ---

OPENAI_TOOL_CALL_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-tc1",
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
                        "id": "call_temp_001",
                        "type": "function",
                        "function": {
                            "name": "get_temperature",
                            "arguments": '{"city": "Tokyo"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 12, "total_tokens": 32},
}

OPENAI_TEXT_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-tc2",
    "object": "chat.completion",
    "model": "gpt-5",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The temperature in Tokyo is 22 degrees Celsius.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 40, "completion_tokens": 15, "total_tokens": 55},
}

# --- Anthropic canned responses ---

ANTHROPIC_TOOL_USE_RESPONSE: dict[str, Any] = {
    "id": "msg-tc1",
    "type": "message",
    "model": "claude-sonnet-4-6",
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_temp_001",
            "name": "get_temperature",
            "input": {"city": "Tokyo"},
        }
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 20, "output_tokens": 10},
}

ANTHROPIC_TEXT_RESPONSE: dict[str, Any] = {
    "id": "msg-tc2",
    "type": "message",
    "model": "claude-sonnet-4-6",
    "content": [{"type": "text", "text": "The temperature in Tokyo is 22 degrees Celsius."}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 35, "output_tokens": 15},
}

ANTHROPIC_THINKING_TOOL_RESPONSE: dict[str, Any] = {
    "id": "msg-tc3",
    "type": "message",
    "model": "claude-sonnet-4-6",
    "content": [
        {
            "type": "thinking",
            "thinking": "The user wants the temperature. I should call get_temperature.",
            "signature": "sig_abc123",
        },
        {
            "type": "tool_use",
            "id": "toolu_think_001",
            "name": "get_temperature",
            "input": {"city": "Berlin"},
        },
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 25, "output_tokens": 30},
}

# --- Google canned responses ---

GOOGLE_TOOL_CALL_RESPONSE: dict[str, Any] = {
    "candidates": [
        {
            "content": {
                "parts": [{"functionCall": {"name": "get_temperature", "args": {"city": "Tokyo"}}}],
                "role": "model",
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 20,
        "candidatesTokenCount": 10,
        "totalTokenCount": 30,
    },
}

GOOGLE_TEXT_RESPONSE: dict[str, Any] = {
    "candidates": [
        {
            "content": {
                "parts": [{"text": "The temperature in Tokyo is 22 degrees Celsius."}],
                "role": "model",
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 40,
        "candidatesTokenCount": 15,
        "totalTokenCount": 55,
    },
}


# ---------------------------------------------------------------------------
# Mock transport helpers
# ---------------------------------------------------------------------------


def _make_sequential_handler(
    payloads: list[dict[str, Any]],
    captured_requests: list[dict[str, Any]] | None = None,
) -> Any:
    """Return async + sync handlers that return payloads in order, optionally capturing requests."""
    call_index = {"n": 0}

    def _get_response(request_body: bytes) -> dict[str, Any]:
        if captured_requests is not None:
            captured_requests.append(json.loads(request_body))
        idx = min(call_index["n"], len(payloads) - 1)
        call_index["n"] += 1
        return payloads[idx]

    async def async_handler(request: httpx.Request) -> httpx.Response:
        body = request.content if isinstance(request.content, bytes) else b""
        return httpx.Response(200, json=_get_response(body))

    def sync_handler(request: httpx.Request) -> httpx.Response:
        body = request.content if isinstance(request.content, bytes) else b""
        return httpx.Response(200, json=_get_response(body))

    return async_handler, sync_handler


def _inject_sequential_mock(
    client: Any,
    payloads: list[dict[str, Any]],
    captured_requests: list[dict[str, Any]] | None = None,
) -> None:
    """Replace client transports with sequential mock handlers."""
    async_handler, sync_handler = _make_sequential_handler(payloads, captured_requests)
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
# TestOpenAIMultiTurnToolUse
# ---------------------------------------------------------------------------


class TestOpenAIMultiTurnToolUse:
    """OpenAI: full tool-use continuation flow with request body verification."""

    def test_tool_use_continuation(self) -> None:
        """First call returns tool_calls, build continuation, verify second request body."""
        captured: list[dict[str, Any]] = []
        client = OpenAIClient(model="gpt-5", api_key="test-key")
        _inject_sequential_mock(
            client,
            [OPENAI_TOOL_CALL_RESPONSE, OPENAI_TEXT_RESPONSE],
            captured,
        )

        # Turn 1: send user message with tool, get tool_calls back
        messages: list[dict[str, Any]] = [
            UserMessage("What is the temperature in Tokyo?"),
        ]
        response1 = client.chat(messages, tools=[WEATHER_TOOL])

        assert len(response1.tool_calls) == 1
        tc = response1.tool_calls[0]
        assert tc.name == "get_temperature"
        assert tc.arguments == {"city": "Tokyo"}
        assert tc.id == "call_temp_001"

        # Build continuation: assistant message + tool result
        assistant_msg = AssistantMessage.from_response(response1)
        tool_result = ToolResultMessage(tc.id, '{"celsius": 22}', name=tc.name)

        # Verify AssistantMessage has OpenAI format (tool_calls array, not content blocks)
        assert "tool_calls" in assistant_msg
        assert assistant_msg["tool_calls"][0]["id"] == "call_temp_001"
        assert assistant_msg["tool_calls"][0]["type"] == "function"
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "get_temperature"

        # Turn 2: send continuation
        messages.append(assistant_msg)
        messages.append(tool_result)
        response2 = client.chat(messages, tools=[WEATHER_TOOL])

        assert response2.text == "The temperature in Tokyo is 22 degrees Celsius."
        assert response2.stop_reason == "stop"

        # Verify the second request body contains the tool result message
        assert len(captured) == 2
        second_body = captured[1]
        body_messages = second_body["messages"]

        # Should have: user, assistant (with tool_calls), tool result
        assert len(body_messages) == 3
        assert body_messages[0]["role"] == "user"
        assert body_messages[1]["role"] == "assistant"
        assert "tool_calls" in body_messages[1]
        assert body_messages[2]["role"] == "tool"
        assert body_messages[2]["tool_call_id"] == "call_temp_001"
        assert body_messages[2]["content"] == '{"celsius": 22}'


# ---------------------------------------------------------------------------
# TestAnthropicMultiTurnToolUse
# ---------------------------------------------------------------------------


class TestAnthropicMultiTurnToolUse:
    """Anthropic: tool-use continuation with content block format verification."""

    def test_tool_use_continuation(self) -> None:
        """Verify Anthropic format: assistant has content blocks, tool result is user message."""
        captured: list[dict[str, Any]] = []
        client = AnthropicClient(model="claude-sonnet-4-6", api_key="test-key")
        _inject_sequential_mock(
            client,
            [ANTHROPIC_TOOL_USE_RESPONSE, ANTHROPIC_TEXT_RESPONSE],
            captured,
        )

        # Turn 1: tool call
        messages: list[dict[str, Any]] = [
            UserMessage("What is the temperature in Tokyo?"),
        ]
        response1 = client.chat(messages, tools=[WEATHER_TOOL])

        assert len(response1.tool_calls) == 1
        tc = response1.tool_calls[0]
        assert tc.name == "get_temperature"
        assert tc.id == "toolu_temp_001"

        # Build continuation
        assistant_msg = AssistantMessage.from_response(response1)
        tool_result = ToolResultMessage(tc.id, '{"celsius": 22}', name=tc.name)

        # Verify AssistantMessage has Anthropic format (content blocks, NOT tool_calls)
        assert "tool_calls" not in assistant_msg
        assert "content" in assistant_msg
        content_blocks = assistant_msg["content"]
        assert isinstance(content_blocks, list)
        assert len(content_blocks) == 1
        assert content_blocks[0]["type"] == "tool_use"
        assert content_blocks[0]["id"] == "toolu_temp_001"
        assert content_blocks[0]["name"] == "get_temperature"
        assert content_blocks[0]["input"] == {"city": "Tokyo"}

        # Turn 2: continuation
        messages.append(assistant_msg)
        messages.append(tool_result)
        response2 = client.chat(messages, tools=[WEATHER_TOOL])

        assert response2.text == "The temperature in Tokyo is 22 degrees Celsius."

        # Verify the second request body has Anthropic-format messages
        assert len(captured) == 2
        second_body = captured[1]
        body_messages = second_body["messages"]

        # Anthropic: user, assistant (content blocks), user (tool_result block)
        # The tool result is rewritten to a user message with tool_result content
        assert body_messages[0]["role"] == "user"
        assert body_messages[1]["role"] == "assistant"
        # Assistant has content blocks (not tool_calls)
        assert isinstance(body_messages[1]["content"], list)
        assert body_messages[1]["content"][0]["type"] == "tool_use"

        # Tool result is converted to a user message with tool_result block
        tool_result_msg = body_messages[2]
        assert tool_result_msg["role"] == "user"
        assert isinstance(tool_result_msg["content"], list)
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "toolu_temp_001"

    def test_thinking_preserved_in_continuation(self) -> None:
        """Thinking + tool_use blocks are preserved in AssistantMessage for Anthropic replay."""
        captured: list[dict[str, Any]] = []
        client = AnthropicClient(model="claude-sonnet-4-6", api_key="test-key")
        _inject_sequential_mock(
            client,
            [ANTHROPIC_THINKING_TOOL_RESPONSE, ANTHROPIC_TEXT_RESPONSE],
            captured,
        )

        messages: list[dict[str, Any]] = [
            UserMessage("What is the temperature in Berlin?"),
        ]
        response1 = client.chat(messages, tools=[WEATHER_TOOL])

        # Verify thinking was parsed
        assert response1.thinking is not None
        assert "temperature" in response1.thinking.lower()

        # Build continuation
        assistant_msg = AssistantMessage.from_response(response1)

        # Verify thinking blocks are preserved in the assistant message content
        content_blocks = assistant_msg["content"]
        assert isinstance(content_blocks, list)
        assert len(content_blocks) == 2  # thinking + tool_use

        # First block: thinking with signature (required for Anthropic replay)
        thinking_block = content_blocks[0]
        assert thinking_block["type"] == "thinking"
        assert "temperature" in thinking_block["thinking"].lower()
        assert thinking_block["signature"] == "sig_abc123"

        # Second block: tool_use
        tool_use_block = content_blocks[1]
        assert tool_use_block["type"] == "tool_use"
        assert tool_use_block["id"] == "toolu_think_001"
        assert tool_use_block["name"] == "get_temperature"
        assert tool_use_block["input"] == {"city": "Berlin"}

        # Complete the continuation and verify it goes through
        tc = response1.tool_calls[0]
        tool_result = ToolResultMessage(tc.id, '{"celsius": 18}', name=tc.name)
        messages.append(assistant_msg)
        messages.append(tool_result)
        response2 = client.chat(messages, tools=[WEATHER_TOOL])
        assert response2.text == "The temperature in Tokyo is 22 degrees Celsius."

        # Verify the second request preserves thinking in the body
        second_body = captured[1]
        body_messages = second_body["messages"]
        assistant_body = body_messages[1]
        assert assistant_body["role"] == "assistant"
        assert isinstance(assistant_body["content"], list)
        # Thinking block must be first (Anthropic requires this ordering)
        assert assistant_body["content"][0]["type"] == "thinking"
        assert assistant_body["content"][1]["type"] == "tool_use"


# ---------------------------------------------------------------------------
# TestGoogleMultiTurnToolUse
# ---------------------------------------------------------------------------


class TestGoogleMultiTurnToolUse:
    """Google: tool-use continuation flow with functionResponse format."""

    def test_tool_use_continuation(self) -> None:
        """Verify Google format: functionCall parts and functionResponse in continuation."""
        captured: list[dict[str, Any]] = []
        client = GoogleClient(model="gemini-2.5-flash", api_key="test-key")
        _inject_sequential_mock(
            client,
            [GOOGLE_TOOL_CALL_RESPONSE, GOOGLE_TEXT_RESPONSE],
            captured,
        )

        messages: list[dict[str, Any]] = [
            UserMessage("What is the temperature in Tokyo?"),
        ]
        response1 = client.chat(messages, tools=[WEATHER_TOOL])

        assert len(response1.tool_calls) == 1
        tc = response1.tool_calls[0]
        assert tc.name == "get_temperature"
        assert tc.arguments == {"city": "Tokyo"}

        # Build continuation -- Google requires name= on ToolResultMessage
        assistant_msg = AssistantMessage.from_response(response1)
        tool_result = ToolResultMessage(tc.id, '{"celsius": 22}', name=tc.name)

        messages.append(assistant_msg)
        messages.append(tool_result)
        response2 = client.chat(messages, tools=[WEATHER_TOOL])

        assert response2.text == "The temperature in Tokyo is 22 degrees Celsius."

        # Verify the second request body has Google format
        assert len(captured) == 2
        second_body = captured[1]
        contents = second_body["contents"]

        # Google: user, model (with functionCall), function (with functionResponse)
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"
        # Model message has functionCall part
        model_parts = contents[1]["parts"]
        has_function_call = any("functionCall" in p for p in model_parts)
        assert has_function_call

        # Function response message
        func_msg = contents[2]
        assert func_msg["role"] == "function"
        assert func_msg["parts"][0]["functionResponse"]["name"] == "get_temperature"


# ---------------------------------------------------------------------------
# TestParallelToolCallAccumulation
# ---------------------------------------------------------------------------


class TestParallelToolCallAccumulation:
    """StreamAccumulator: interleaved index-based deltas for parallel tool calls."""

    def test_interleaved_index_based_deltas(self) -> None:
        """Feed interleaved deltas at index 0 and 1, verify separate tool calls."""
        acc = StreamAccumulator(provider="openai", model="gpt-5", request_id="test-123")

        # Tool call 0 starts
        acc.feed(
            StreamChunk(
                type="tool_call_delta",
                tool_call_delta={
                    "index": 0,
                    "id": "call_a",
                    "name": "get_temperature",
                    "arguments": '{"ci',
                },
            )
        )

        # Tool call 1 starts (interleaved)
        acc.feed(
            StreamChunk(
                type="tool_call_delta",
                tool_call_delta={
                    "index": 1,
                    "id": "call_b",
                    "name": "get_humidity",
                    "arguments": '{"ci',
                },
            )
        )

        # Tool call 0 continues
        acc.feed(
            StreamChunk(
                type="tool_call_delta",
                tool_call_delta={
                    "index": 0,
                    "arguments": 'ty": "Tokyo"}',
                },
            )
        )

        # Tool call 1 continues
        acc.feed(
            StreamChunk(
                type="tool_call_delta",
                tool_call_delta={
                    "index": 1,
                    "arguments": 'ty": "Berlin"}',
                },
            )
        )

        # Done
        acc.feed(StreamChunk(type="done"))

        response = acc.accumulated
        assert len(response.tool_calls) == 2

        # Index 0
        tc0 = response.tool_calls[0]
        assert tc0.id == "call_a"
        assert tc0.name == "get_temperature"
        assert tc0.arguments == {"city": "Tokyo"}

        # Index 1
        tc1 = response.tool_calls[1]
        assert tc1.id == "call_b"
        assert tc1.name == "get_humidity"
        assert tc1.arguments == {"city": "Berlin"}


# ---------------------------------------------------------------------------
# TestAnthropicToolChoiceNone
# ---------------------------------------------------------------------------


class TestGeminiThinkingParsing:
    """Verify Gemini thinking parts and thoughtSignature preservation."""

    def test_thought_true_parsed_as_thinking(self) -> None:
        """Parts with thought=True become ContentPart(type='thinking'), not text."""
        from kaos_llm_client.providers.google import GoogleClient

        client = GoogleClient(model="gemini-2.5-flash", api_key="test-key")
        raw = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Let me reason...", "thought": True},
                            {"text": "The answer is 42."},
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 10,
                "totalTokenCount": 15,
            },
        }
        request = ProviderRequest(
            provider="google", model="gemini-2.5-flash", endpoint="/test", body={}
        )
        response = client._parse_response(raw, request)

        assert response.thinking == "Let me reason..."
        assert response.text == "The answer is 42."

    def test_thought_signature_in_raw(self) -> None:
        """thoughtSignature is preserved in ContentPart.raw for replay."""
        from kaos_llm_client.providers.google import GoogleClient

        client = GoogleClient(model="gemini-2.5-flash", api_key="test-key")
        raw = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Final answer", "thoughtSignature": "abc123sig"},
                        ],
                        "role": "model",
                    },
                }
            ],
            "usageMetadata": {},
        }
        request = ProviderRequest(
            provider="google", model="gemini-2.5-flash", endpoint="/test", body={}
        )
        response = client._parse_response(raw, request)

        text_part = next(p for p in response.parts if p.type == "text")
        assert text_part.raw is not None
        assert text_part.raw.get("thoughtSignature") == "abc123sig"


class TestStreamedRawPayload:
    """Verify StreamAccumulator.accumulated.raw is the last chunk, not a stub."""

    def test_accumulated_raw_is_last_chunk(self) -> None:
        """raw should be the last SSE chunk (contains usage/finish), not {'streamed_chunks': N}."""
        acc = StreamAccumulator(provider="openai", model="gpt-5", request_id="r1")
        acc.feed(
            StreamChunk(
                type="text_delta",
                text="Hello",
                raw={"id": "c1", "choices": [{"delta": {"content": "Hello"}}]},
            )
        )
        acc.feed(
            StreamChunk(
                type="text_delta",
                text=" world",
                raw={"id": "c1", "choices": [{"delta": {"content": " world"}}]},
            )
        )
        last_raw = {
            "id": "c1",
            "choices": [{"finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        acc.feed(
            StreamChunk(
                type="usage",
                usage=UsageInfo(input_tokens=5, output_tokens=2, total_tokens=7),
                raw=last_raw,
            )
        )
        acc.feed(StreamChunk(type="done"))

        result = acc.accumulated
        assert result.raw == last_raw
        assert "streamed_chunks" not in result.raw

    def test_accumulated_raw_empty_fallback(self) -> None:
        """If no raw chunks at all, fallback to stub."""
        acc = StreamAccumulator(provider="test", model="test", request_id="r1")
        acc.feed(StreamChunk(type="text_delta", text="Hi"))
        acc.feed(StreamChunk(type="done"))

        result = acc.accumulated
        assert result.raw == {"streamed_chunks": 0}


class TestAnthropicToolChoiceNone:
    """Verify ToolChoice(type='none') produces correct Anthropic wire format."""

    def test_none_maps_to_none(self) -> None:
        """ToolChoice(type='none') must produce {'type': 'none'}, not {'type': 'auto'}."""
        captured: list[dict[str, Any]] = []
        client = AnthropicClient(model="claude-sonnet-4-6", api_key="test-key")
        _inject_sequential_mock(
            client,
            [ANTHROPIC_TEXT_RESPONSE],
            captured,
        )

        client.chat(
            [UserMessage("Hello")],
            tools=[WEATHER_TOOL],
            tool_choice=ToolChoice(type="none"),
        )

        assert len(captured) == 1
        body = captured[0]
        assert body["tool_choice"] == {"type": "none"}
