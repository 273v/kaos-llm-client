"""Tests for kaos_llm_client.messages — typed message models."""

from __future__ import annotations

import json
from typing import Any

import pytest

from kaos_llm_client.messages import (
    AssistantMessage,
    CachePoint,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from kaos_llm_client.profiles import ModelProfile
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart, ProviderResponse, ToolCall, UsageInfo

# ---------------------------------------------------------------------------
# TestSystemMessage
# ---------------------------------------------------------------------------


class TestSystemMessage:
    def test_basic(self) -> None:
        msg = SystemMessage("Be concise.")
        assert msg == {"role": "system", "content": "Be concise."}

    def test_with_name(self) -> None:
        msg = SystemMessage("You are a helper.", name="helper")
        assert msg["role"] == "system"
        assert msg["content"] == "You are a helper."
        assert msg["name"] == "helper"

    def test_is_dict(self) -> None:
        msg = SystemMessage("x")
        assert isinstance(msg, dict)

    def test_role_key(self) -> None:
        msg = SystemMessage("anything")
        assert msg["role"] == "system"


# ---------------------------------------------------------------------------
# TestUserMessage
# ---------------------------------------------------------------------------


class TestUserMessage:
    def test_text_only(self) -> None:
        msg = UserMessage("Hello")
        assert msg == {"role": "user", "content": "Hello"}

    def test_multimodal_list(self) -> None:
        """Strings in a list are normalized to {"type": "text", "text": ...} dicts."""
        image_part = {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
        msg = UserMessage(["Describe this:", image_part])
        content = msg["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "Describe this:"}
        assert content[1] == image_part

    def test_with_name(self) -> None:
        msg = UserMessage("Hi there", name="alice")
        assert msg["name"] == "alice"
        assert msg["role"] == "user"
        assert msg["content"] == "Hi there"

    def test_content_is_list_for_multimodal(self) -> None:
        msg = UserMessage([{"type": "text", "text": "hello"}])
        assert isinstance(msg["content"], list)

    def test_invalid_content_item_raises(self) -> None:
        from collections.abc import Sequence
        from typing import cast

        bad: Sequence[str | dict[str, Any]] = cast(Sequence[str | dict[str, Any]], [123])
        with pytest.raises(TypeError, match="Content item must be str or dict"):
            UserMessage(bad)


# ---------------------------------------------------------------------------
# TestAssistantMessage
# ---------------------------------------------------------------------------


class TestAssistantMessage:
    def test_text_only(self) -> None:
        msg = AssistantMessage("Hi")
        assert msg == {"role": "assistant", "content": "Hi"}

    def test_with_tool_calls(self) -> None:
        tool_calls = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "test"}'},
            }
        ]
        msg = AssistantMessage("Let me search.", tool_calls=tool_calls)
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me search."
        assert msg["tool_calls"] == tool_calls

    def test_no_content(self) -> None:
        """When only tool_calls are provided, content key is omitted."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "fn", "arguments": "{}"},
            }
        ]
        msg = AssistantMessage(tool_calls=tool_calls)
        assert msg["role"] == "assistant"
        assert "content" not in msg
        assert msg["tool_calls"] == tool_calls

    def test_from_response(self) -> None:
        """Create AssistantMessage from a mock ProviderResponse with text + tool_calls."""
        tc = ToolCall(id="call_42", name="lookup", arguments={"id": 7})
        response = ProviderResponse(
            provider="test",
            model="test-model",
            raw={},
            parts=[
                ContentPart(type="text", text="I found it."),
                ContentPart(type="tool_use", tool_call=tc),
            ],
            usage=UsageInfo(input_tokens=10, output_tokens=20, total_tokens=30),
        )
        msg = AssistantMessage.from_response(response)

        assert msg["role"] == "assistant"
        assert msg["content"] == "I found it."
        assert len(msg["tool_calls"]) == 1

        fc = msg["tool_calls"][0]
        assert fc["id"] == "call_42"
        assert fc["type"] == "function"
        assert fc["function"]["name"] == "lookup"
        assert json.loads(fc["function"]["arguments"]) == {"id": 7}


# ---------------------------------------------------------------------------
# TestToolResultMessage
# ---------------------------------------------------------------------------


class TestToolResultMessage:
    def test_basic(self) -> None:
        msg = ToolResultMessage("call_1", "result")
        assert msg == {"role": "tool", "tool_call_id": "call_1", "content": "result"}

    def test_dict_content_serialized(self) -> None:
        """Dict content is serialized to a JSON string."""
        msg = ToolResultMessage("id_1", {"key": "val"})
        assert msg["content"] == json.dumps({"key": "val"})
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "id_1"

    def test_list_content_serialized(self) -> None:
        """List content is serialized to a JSON string."""
        msg = ToolResultMessage("id_2", [1, 2, 3])
        assert msg["content"] == json.dumps([1, 2, 3])

    def test_with_name(self) -> None:
        msg = ToolResultMessage("call_x", "done", name="search")
        assert msg["name"] == "search"
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_x"
        assert msg["content"] == "done"


# ---------------------------------------------------------------------------
# TestCachePoint
# ---------------------------------------------------------------------------


class TestCachePoint:
    def test_basic(self) -> None:
        msg = CachePoint()
        assert msg == {"role": "cache_point"}

    def test_is_dict(self) -> None:
        assert isinstance(CachePoint(), dict)


# ---------------------------------------------------------------------------
# TestMixedMessages
# ---------------------------------------------------------------------------


class TestMixedMessages:
    def test_typed_and_raw_mixed(self) -> None:
        """Typed messages and raw dicts can coexist in the same list since both are dicts."""
        messages: list[dict[str, Any]] = [
            SystemMessage("Be brief."),
            {"role": "user", "content": "What is 2+2?"},
            UserMessage("Follow up question."),
        ]
        # All elements are dicts
        for msg in messages:
            assert isinstance(msg, dict)
        # Check roles in order
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "user"

    def test_round_trip_with_function_client(self) -> None:
        """Typed messages pass through FunctionClient without conversion issues."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            # Echo back the number of messages received
            count = len(messages)
            return ProviderResponse(
                provider="function",
                model="test",
                raw={},
                parts=[ContentPart(type="text", text=f"Got {count} messages")],
                usage=UsageInfo(input_tokens=count, output_tokens=1, total_tokens=count + 1),
            )

        client = FunctionClient(function=handler)

        messages: list[dict[str, Any]] = [
            SystemMessage("You are concise."),
            CachePoint(),
            UserMessage("Hello!"),
        ]
        response = client.chat(messages)
        assert response.text == "Got 3 messages"

        # Verify the messages were recorded in call history
        recorded_msgs, _ = client.call_history[0]
        assert len(recorded_msgs) == 3
        assert recorded_msgs[0]["role"] == "system"
        assert recorded_msgs[1]["role"] == "cache_point"
        assert recorded_msgs[2]["role"] == "user"
