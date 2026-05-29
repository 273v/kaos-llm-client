"""Tests for kaos_llm_client.types — response/request type construction."""

from __future__ import annotations

from typing import Any

from kaos_llm_client.types import (
    CachePolicy,
    ContentPart,
    ProviderRequest,
    ProviderResponse,
    RequestOptions,
    StreamAccumulator,
    StreamChunk,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    UsageInfo,
)


class TestCachePolicy:
    def test_enum_values(self):
        assert CachePolicy.DEFAULT == "default"
        assert CachePolicy.SKIP == "skip"
        assert CachePolicy.FORCE == "force"


class TestRequestOptions:
    def test_defaults(self):
        opts = RequestOptions()
        assert opts.timeout is None
        assert opts.max_retries is None
        assert opts.cache_policy == CachePolicy.DEFAULT
        assert opts.extra_headers is None

    def test_custom_values(self):
        opts = RequestOptions(timeout=30.0, max_retries=5, cache_policy=CachePolicy.SKIP)
        assert opts.timeout == 30.0
        assert opts.max_retries == 5
        assert opts.cache_policy == CachePolicy.SKIP


class TestProviderRequest:
    def test_construction(self):
        req = ProviderRequest(
            provider="openai",
            model="gpt-5",
            endpoint="/v1/chat/completions",
            body={"messages": [{"role": "user", "content": "hello"}]},
        )
        assert req.provider == "openai"
        assert req.model == "gpt-5"
        assert req.stream is False
        assert req.request_id  # auto-generated UUID

    def test_auto_request_id(self):
        req1 = ProviderRequest(provider="x", model="y", endpoint="/", body={})
        req2 = ProviderRequest(provider="x", model="y", endpoint="/", body={})
        assert req1.request_id != req2.request_id


class TestUsageInfo:
    def test_defaults(self):
        usage = UsageInfo()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.reasoning_tokens is None
        assert usage.cache_read_tokens is None

    def test_full_usage(self):
        usage = UsageInfo(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            reasoning_tokens=20,
        )
        assert usage.total_tokens == 150
        assert usage.reasoning_tokens == 20


class TestToolCall:
    def test_construction(self):
        tc = ToolCall(
            id="call_123",
            name="search",
            arguments={"query": "test"},
        )
        assert tc.id == "call_123"
        assert tc.name == "search"
        assert tc.arguments == {"query": "test"}

    def test_with_raw(self):
        tc = ToolCall(
            id="call_123",
            name="search",
            arguments={"query": "test"},
            raw={"original": True},
        )
        assert tc.raw == {"original": True}


class TestContentPart:
    def test_text_part(self):
        part = ContentPart(type="text", text="Hello world")
        assert part.type == "text"
        assert part.text == "Hello world"
        assert part.thinking is None
        assert part.tool_call is None

    def test_thinking_part(self):
        part = ContentPart(type="thinking", thinking="Let me think...")
        assert part.type == "thinking"
        assert part.thinking == "Let me think..."

    def test_tool_use_part(self):
        tc = ToolCall(id="c1", name="fn", arguments={})
        part = ContentPart(type="tool_use", tool_call=tc)
        assert part.type == "tool_use"
        assert part.tool_call is not None
        assert part.tool_call.name == "fn"


class TestProviderResponse:
    def _make_response(self, **overrides: Any) -> ProviderResponse:  # type: ignore[no-any-explicit]
        defaults: dict[str, Any] = {
            "provider": "test",
            "model": "test-model",
            "raw": {},
        }
        defaults.update(overrides)
        return ProviderResponse(**defaults)

    def test_empty_response(self):
        resp = self._make_response()
        assert resp.text == ""
        assert resp.thinking is None
        assert resp.tool_calls == []
        assert resp.output_json is None

    def test_text_property(self):
        resp = self._make_response(
            parts=[
                ContentPart(type="text", text="Hello "),
                ContentPart(type="text", text="world"),
            ]
        )
        assert resp.text == "Hello world"

    def test_thinking_property(self):
        resp = self._make_response(
            parts=[
                ContentPart(type="thinking", thinking="Step 1. "),
                ContentPart(type="thinking", thinking="Step 2."),
                ContentPart(type="text", text="Answer"),
            ]
        )
        assert resp.thinking == "Step 1. Step 2."
        assert resp.text == "Answer"

    def test_tool_calls_property(self):
        tc1 = ToolCall(id="c1", name="fn1", arguments={"a": 1})
        tc2 = ToolCall(id="c2", name="fn2", arguments={"b": 2})
        resp = self._make_response(
            parts=[
                ContentPart(type="tool_use", tool_call=tc1),
                ContentPart(type="text", text="interim"),
                ContentPart(type="tool_use", tool_call=tc2),
            ]
        )
        calls = resp.tool_calls
        assert len(calls) == 2
        assert calls[0].name == "fn1"
        assert calls[1].name == "fn2"

    def test_output_json_valid(self):
        resp = self._make_response(parts=[ContentPart(type="text", text='{"key": "value"}')])
        assert resp.output_json == {"key": "value"}

    def test_output_json_with_code_fence(self):
        resp = self._make_response(
            parts=[ContentPart(type="text", text='```json\n{"key": "value"}\n```')]
        )
        assert resp.output_json == {"key": "value"}

    def test_output_json_invalid(self):
        resp = self._make_response(parts=[ContentPart(type="text", text="not json at all")])
        assert resp.output_json is None

    # --- Inline-unescaped-quote regression (silent truncation) ---

    _BAD = (
        '{"memo": "The clause says "shall remain in full force" and effect.", '
        '"score": 5, "needs_more_extraction": true}'
    )

    def test_output_json_complete_inline_quote_keeps_all_fields(self):
        # A complete object (stop_reason=end_turn) whose string field has an
        # inline unescaped quote must NOT be truncated to its first field.
        resp = self._make_response(
            parts=[ContentPart(type="text", text=self._BAD)],
            stop_reason="end_turn",
        )
        result = resp.output_json
        assert isinstance(result, dict)
        assert set(result) == {"memo", "score", "needs_more_extraction"}
        assert result["needs_more_extraction"] is True

    def test_output_json_no_partial_recovery_on_clean_stop(self):
        # Genuinely-unrepairable garbage with a clean stop reason returns None
        # (fail loud) rather than a silently-truncated fragment.
        truncated = '{"memo": "cut off mid stream with no close'
        resp = self._make_response(
            parts=[ContentPart(type="text", text=truncated)],
            stop_reason="end_turn",
        )
        assert resp.output_json is None

    def test_output_json_partial_recovery_on_max_tokens(self):
        # When the stream WAS truncated (max_tokens), partial recovery still
        # salvages a usable prefix.
        truncated = '{"memo": "cut off mid stream with no close'
        resp = self._make_response(
            parts=[ContentPart(type="text", text=truncated)],
            stop_reason="max_tokens",
        )
        result = resp.output_json
        assert isinstance(result, dict)
        assert "memo" in result

    def test_transport_metadata(self):
        resp = self._make_response(
            status_code=200,
            latency_ms=1234.5,
            request_id="req-123",
        )
        assert resp.status_code == 200
        assert resp.latency_ms == 1234.5
        assert resp.request_id == "req-123"


class TestToolDefinition:
    def test_construction(self):
        td = ToolDefinition(
            name="search",
            description="Search for things",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        assert td.name == "search"
        assert td.strict is None


class TestToolChoice:
    def test_default(self):
        tc = ToolChoice()
        assert tc.type == "auto"
        assert tc.name is None

    def test_specific(self):
        tc = ToolChoice(type="specific", name="my_tool")
        assert tc.type == "specific"
        assert tc.name == "my_tool"


class TestStreamChunk:
    def test_text_delta(self):
        chunk = StreamChunk(type="text_delta", text="Hello")
        assert chunk.type == "text_delta"
        assert chunk.text == "Hello"

    def test_thinking_delta(self):
        chunk = StreamChunk(type="thinking_delta", thinking="Hmm...")
        assert chunk.thinking == "Hmm..."

    def test_done(self):
        chunk = StreamChunk(type="done")
        assert chunk.type == "done"


class TestStreamAccumulator:
    def test_text_accumulation(self):
        acc = StreamAccumulator(provider="test", model="test", request_id="r1")
        acc.feed(StreamChunk(type="text_delta", text="Hello "))
        acc.feed(StreamChunk(type="text_delta", text="world"))
        acc.feed(StreamChunk(type="done"))

        assert acc.text_so_far == "Hello world"
        resp = acc.accumulated
        assert resp.text == "Hello world"
        assert resp.provider == "test"
        assert resp.model == "test"

    def test_thinking_accumulation(self):
        acc = StreamAccumulator(provider="test", model="test", request_id="r1")
        acc.feed(StreamChunk(type="thinking_delta", thinking="Step 1. "))
        acc.feed(StreamChunk(type="thinking_delta", thinking="Step 2."))
        acc.feed(StreamChunk(type="text_delta", text="Answer"))
        acc.feed(StreamChunk(type="done"))

        resp = acc.accumulated
        assert resp.thinking == "Step 1. Step 2."
        assert resp.text == "Answer"

    def test_tool_call_accumulation(self):
        acc = StreamAccumulator(provider="test", model="test", request_id="r1")
        # Start tool call
        acc.feed(
            StreamChunk(
                type="tool_call_delta",
                tool_call_delta={"id": "c1", "name": "search", "arguments": '{"q": '},
            )
        )
        # Continue arguments
        acc.feed(
            StreamChunk(
                type="tool_call_delta",
                tool_call_delta={"arguments": '"test"}'},
            )
        )
        acc.feed(StreamChunk(type="done"))

        resp = acc.accumulated
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search"
        assert resp.tool_calls[0].arguments == {"q": "test"}

    def test_usage_accumulation(self):
        acc = StreamAccumulator(provider="test", model="test", request_id="r1")
        acc.feed(StreamChunk(type="text_delta", text="Hi"))
        acc.feed(
            StreamChunk(
                type="usage",
                usage=UsageInfo(input_tokens=10, output_tokens=1, total_tokens=11),
            )
        )
        acc.feed(StreamChunk(type="done"))

        resp = acc.accumulated
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 1
