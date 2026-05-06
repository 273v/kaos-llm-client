"""Provider contract tests for the Anthropic client.

Tests ``_build_request()`` and ``_parse_response()`` directly — no HTTP calls.
"""

from __future__ import annotations

from kaos_llm_client.providers.anthropic import AnthropicClient
from kaos_llm_client.types import (
    ProviderRequest,
    ProviderResponse,
    ToolChoice,
    ToolDefinition,
)


def _make_client(model: str = "claude-sonnet-4-6") -> AnthropicClient:
    """Create an Anthropic client with a test key (no settings resolution)."""
    return AnthropicClient(model=model, api_key="test-key")


def _make_request(request_id: str = "req-test") -> ProviderRequest:
    """Create a minimal ProviderRequest for parse tests."""
    return ProviderRequest(
        provider="anthropic",
        model="claude-sonnet-4-6",
        endpoint="/v1/messages",
        body={},
        request_id=request_id,
    )


class TestAnthropicBuildRequest:
    """Tests for AnthropicClient._build_request()."""

    def test_build_request_basic(self):
        client = _make_client()
        messages = [{"role": "user", "content": "hello"}]
        req = client._build_request(messages)

        assert req.body["model"] == "claude-sonnet-4-6"
        assert req.body["messages"] == [{"role": "user", "content": "hello"}]
        assert "max_tokens" in req.body
        assert req.provider == "anthropic"
        assert req.endpoint == "/v1/messages"

    def test_system_prompt_extraction(self):
        client = _make_client()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]
        req = client._build_request(messages)

        # System message extracted to top-level "system" field
        assert req.body["system"] == "You are helpful."
        # User message remains in messages array
        assert len(req.body["messages"]) == 1
        assert req.body["messages"][0]["role"] == "user"
        # No system message left in messages
        assert all(m["role"] != "system" for m in req.body["messages"])

    def test_max_tokens_required(self):
        client = _make_client()
        messages = [{"role": "user", "content": "hi"}]

        # max_tokens always present, uses profile default when not specified
        req = client._build_request(messages)
        assert "max_tokens" in req.body
        assert req.body["max_tokens"] == client.profile.default_max_tokens

        # Explicit override
        req2 = client._build_request(messages, max_tokens=8192)
        assert req2.body["max_tokens"] == 8192

    def test_thinking_parameter_bool(self):
        client = _make_client()
        messages = [{"role": "user", "content": "think about this"}]
        req = client._build_request(messages, thinking=True)

        assert req.body["thinking"] == {"type": "enabled", "budget_tokens": 4096}

    def test_thinking_parameter_dict(self):
        client = _make_client()
        messages = [{"role": "user", "content": "think hard"}]
        thinking_config = {"type": "enabled", "budget_tokens": 8192}
        req = client._build_request(messages, thinking=thinking_config)

        assert req.body["thinking"] == thinking_config

    def test_tools_format(self):
        client = _make_client()
        tool = ToolDefinition(
            name="get_weather",
            description="Get weather for a city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        req = client._build_request([{"role": "user", "content": "weather?"}], tools=[tool])

        assert "tools" in req.body
        assert len(req.body["tools"]) == 1
        t = req.body["tools"][0]
        assert t["name"] == "get_weather"
        assert t["description"] == "Get weather for a city"
        # Anthropic uses "input_schema" instead of "parameters"
        assert t["input_schema"] == tool.parameters

    def test_tool_choice_auto(self):
        client = _make_client()
        choice = ToolChoice(type="auto")
        req = client._build_request([{"role": "user", "content": "hi"}], tool_choice=choice)
        assert req.body["tool_choice"] == {"type": "auto"}

    def test_tool_choice_specific(self):
        client = _make_client()
        choice = ToolChoice(type="specific", name="get_weather")
        req = client._build_request([{"role": "user", "content": "hi"}], tool_choice=choice)
        assert req.body["tool_choice"] == {"type": "tool", "name": "get_weather"}

    def test_tool_choice_required(self):
        client = _make_client()
        choice = ToolChoice(type="required")
        req = client._build_request([{"role": "user", "content": "hi"}], tool_choice=choice)
        assert req.body["tool_choice"] == {"type": "any"}


class TestAnthropicNativeStructuredOutput:
    """WS-TR.PR-1: native ``output_config.format`` wire.

    Anthropic's structured-outputs GA wire: ``output_config.format = {
    type: "json_schema", schema: <transformed_schema>}``. Mutex: request 400s
    if combined with document-block citations — we enforce client-side.
    """

    def test_apply_native_json_mode_with_schema(self):
        client = _make_client()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "age": {"type": "integer", "minimum": 0},
            },
        }
        kwargs = client._apply_native_json_mode({}, schema)

        assert "output_config" in kwargs
        fmt = kwargs["output_config"]["format"]
        assert fmt["type"] == "json_schema"
        # Transformer applied: strip list removed rejected keywords.
        transformed = fmt["schema"]
        assert "minLength" not in transformed["properties"]["name"]
        assert "minimum" not in transformed["properties"]["age"]
        # Strict-mode invariants applied.
        assert transformed["additionalProperties"] is False
        assert set(transformed["required"]) == {"name", "age"}
        # Canonicalized key order.
        assert list(transformed.keys()) == sorted(transformed.keys())

    def test_apply_native_json_mode_without_schema(self):
        """No schema = no output_config set; caller routes to prompted."""
        client = _make_client()
        kwargs = client._apply_native_json_mode({}, None)
        assert "output_config" not in kwargs

    def test_build_request_preserves_output_config(self):
        client = _make_client()
        messages = [{"role": "user", "content": "hi"}]
        req = client._build_request(
            messages,
            output_config={
                "format": {"type": "json_schema", "schema": {"type": "object"}},
            },
        )
        assert req.body["output_config"] == {
            "format": {"type": "json_schema", "schema": {"type": "object"}},
        }


class TestAnthropicCitationMutex:
    """output_config.format + document-block citations.enabled → 400.

    We enforce client-side via KaosLLMValidationError with a diagnostic fix.
    """

    def test_structured_plus_citations_raises(self):
        from kaos_llm_client.errors import KaosLLMValidationError

        client = _make_client()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize."},
                    {
                        "type": "document",
                        "source": {"type": "file", "file_id": "file_x"},
                        "citations": {"enabled": True},
                    },
                ],
            }
        ]
        import pytest

        with pytest.raises(KaosLLMValidationError, match="citations"):
            client._build_request(
                messages,
                output_config={
                    "format": {"type": "json_schema", "schema": {"type": "object"}},
                },
            )

    def test_structured_without_citations_ok(self):
        """Structured outputs alone works fine."""
        client = _make_client()
        messages = [{"role": "user", "content": "extract fields"}]
        req = client._build_request(
            messages,
            output_config={
                "format": {"type": "json_schema", "schema": {"type": "object"}},
            },
        )
        assert "output_config" in req.body

    def test_citations_without_structured_ok(self):
        """Citations alone works fine."""
        client = _make_client()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "cite please"},
                    {
                        "type": "document",
                        "source": {"type": "file", "file_id": "file_x"},
                        "citations": {"enabled": True},
                    },
                ],
            }
        ]
        req = client._build_request(messages)
        # No raise — request built normally.
        assert req.body["messages"] == messages

    def test_citations_disabled_does_not_trip_mutex(self):
        """Explicitly disabled citations don't trip the mutex."""
        client = _make_client()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {"type": "file", "file_id": "x"},
                        "citations": {"enabled": False},
                    },
                ],
            }
        ]
        req = client._build_request(
            messages,
            output_config={
                "format": {"type": "json_schema", "schema": {"type": "object"}},
            },
        )
        assert "output_config" in req.body


class TestAnthropicParseResponse:
    """Tests for AnthropicClient._parse_response()."""

    def test_parse_response_text(self):
        client = _make_client()
        raw = {
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 5, "output_tokens": 1},
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "id": "msg-1",
        }
        request = _make_request()
        resp = client._parse_response(raw, request)

        assert isinstance(resp, ProviderResponse)
        assert resp.text == "hello"
        assert resp.stop_reason == "end_turn"
        assert resp.response_id == "msg-1"
        assert resp.usage.input_tokens == 5
        assert resp.usage.output_tokens == 1
        assert resp.usage.total_tokens == 6
        assert resp.request_id == "req-test"

    def test_parse_response_thinking(self):
        client = _make_client()
        raw = {
            "content": [
                {"type": "thinking", "thinking": "Let me consider..."},
                {"type": "text", "text": "Here is the answer."},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20},
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "id": "msg-2",
        }
        request = _make_request()
        resp = client._parse_response(raw, request)

        assert resp.thinking == "Let me consider..."
        assert resp.text == "Here is the answer."
        assert len(resp.parts) == 2
        assert resp.parts[0].type == "thinking"
        assert resp.parts[1].type == "text"

    def test_parse_response_tool_use(self):
        client = _make_client()
        raw = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "get_weather",
                    "input": {"city": "NYC"},
                }
            ],
            "usage": {"input_tokens": 15, "output_tokens": 10},
            "model": "claude-sonnet-4-6",
            "stop_reason": "tool_use",
            "id": "msg-3",
        }
        request = _make_request()
        resp = client._parse_response(raw, request)

        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.id == "toolu_abc"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "NYC"}
        assert resp.stop_reason == "tool_use"


class TestAnthropicParseStream:
    """Tests for AnthropicClient._parse_stream_chunk()."""

    def test_parse_stream_chunk_text_delta(self):
        client = _make_client()
        data = {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "hi"},
        }
        chunk = client._parse_stream_chunk(data)

        assert chunk.type == "text_delta"
        assert chunk.text == "hi"

    def test_parse_stream_chunk_thinking_delta(self):
        client = _make_client()
        data = {
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "hmm..."},
        }
        chunk = client._parse_stream_chunk(data)

        assert chunk.type == "thinking_delta"
        assert chunk.thinking == "hmm..."


class TestAnthropicHeaders:
    """Tests for AnthropicClient._build_headers()."""

    def test_headers_x_api_key(self):
        client = _make_client()
        headers = client._build_headers()

        assert headers["x-api-key"] == "test-key"
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["content-type"] == "application/json"
        # Anthropic does NOT use Authorization Bearer
        assert "Authorization" not in headers
