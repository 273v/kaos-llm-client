"""Provider contract tests for OpenAI-compatible and OpenAI clients.

Tests ``_build_request()`` and ``_parse_response()`` directly — no HTTP calls.
"""

from __future__ import annotations

from typing import Any

from kaos_llm_client.providers.openai import OpenAIClient
from kaos_llm_client.providers.openai_compat import OpenAICompatibleClient
from kaos_llm_client.types import (
    ProviderRequest,
    ProviderResponse,
    ToolChoice,
    ToolDefinition,
)


def _make_compat_client(model: str = "gpt-5") -> OpenAICompatibleClient:
    """Create an OpenAI-compatible client with a test key (no settings resolution)."""
    return OpenAICompatibleClient(model=model, api_key="test-key")


def _make_openai_client(model: str = "gpt-5") -> OpenAIClient:
    """Create an OpenAI client with a test key (no settings resolution)."""
    return OpenAIClient(model=model, api_key="test-key")


def _make_request(request_id: str = "req-test") -> ProviderRequest:
    """Create a minimal ProviderRequest for parse tests."""
    return ProviderRequest(
        provider="openai-compatible",
        model="gpt-5",
        endpoint="/v1/chat/completions",
        body={},
        request_id=request_id,
    )


class TestOpenAICompatBuildRequest:
    """Tests for OpenAICompatibleClient._build_request()."""

    def test_build_request_basic(self):
        client = _make_compat_client()
        messages = [{"role": "user", "content": "hello"}]
        req = client._build_request(messages)

        assert req.body["model"] == "gpt-5"
        assert req.body["messages"] == messages
        assert req.provider == "openai-compatible"
        assert req.endpoint == "/v1/chat/completions"
        assert req.stream is False

    def test_build_request_with_system_message(self):
        client = _make_compat_client()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]
        req = client._build_request(messages)

        # System message stays in messages array for OpenAI-compatible
        assert req.body["messages"] == messages
        assert req.body["messages"][0]["role"] == "system"

    def test_build_request_with_max_tokens(self):
        client = _make_compat_client()
        messages = [{"role": "user", "content": "hi"}]

        # OpenAI-compatible does NOT require max_tokens — omitted by default
        field = client.profile.max_tokens_field
        req = client._build_request(messages)
        assert field not in req.body

        # Explicit override via kwargs
        req2 = client._build_request(messages, max_tokens=1024)
        assert req2.body[field] == 1024

    def test_build_request_with_tools(self):
        client = _make_compat_client()
        tool = ToolDefinition(
            name="get_weather",
            description="Get weather for a city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        req = client._build_request([{"role": "user", "content": "weather?"}], tools=[tool])

        assert "tools" in req.body
        assert len(req.body["tools"]) == 1
        t = req.body["tools"][0]
        assert t["type"] == "function"
        assert t["function"]["name"] == "get_weather"
        assert t["function"]["description"] == "Get weather for a city"
        assert t["function"]["parameters"] == tool.parameters

    def test_build_request_with_tool_choice_auto(self):
        client = _make_compat_client()
        choice = ToolChoice(type="auto")
        req = client._build_request([{"role": "user", "content": "hi"}], tool_choice=choice)
        assert req.body["tool_choice"] == "auto"

    def test_build_request_with_tool_choice_specific(self):
        client = _make_compat_client()
        choice = ToolChoice(type="specific", name="fn")
        req = client._build_request([{"role": "user", "content": "hi"}], tool_choice=choice)
        assert req.body["tool_choice"] == {
            "type": "function",
            "function": {"name": "fn"},
        }

    def test_build_request_kwargs_merge(self):
        client = _make_compat_client()
        req = client._build_request(
            [{"role": "user", "content": "hi"}],
            temperature=0.7,
            top_p=0.9,
        )
        assert req.body["temperature"] == 0.7
        assert req.body["top_p"] == 0.9

    def test_build_request_stream(self):
        client = _make_compat_client()
        req = client._build_request([{"role": "user", "content": "hi"}], stream=True)
        assert req.body["stream"] is True
        assert req.stream is True
        assert req.body["stream_options"] == {"include_usage": True}


class TestOpenAICompatParseResponse:
    """Tests for OpenAICompatibleClient._parse_response()."""

    def test_parse_response_text(self):
        client = _make_compat_client()
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
            "model": "gpt-5",
        }
        request = _make_request()
        resp = client._parse_response(raw, request)

        assert isinstance(resp, ProviderResponse)
        assert resp.text == "hello"
        assert resp.stop_reason == "stop"
        assert resp.response_id == "resp-1"
        assert resp.usage.input_tokens == 5
        assert resp.usage.output_tokens == 1
        assert resp.usage.total_tokens == 6
        assert resp.request_id == "req-test"

    def test_parse_response_tool_calls(self):
        client = _make_compat_client()
        raw = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "NYC"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "id": "resp-2",
            "model": "gpt-5",
        }
        request = _make_request()
        resp = client._parse_response(raw, request)

        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.id == "call_abc"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "NYC"}
        assert resp.stop_reason == "tool_calls"


class TestOpenAICompatParseStream:
    """Tests for OpenAICompatibleClient._parse_stream_chunk()."""

    @staticmethod
    def _single(result: Any) -> Any:
        """Extract a single StreamChunk from parse result (may be list)."""
        return result[0] if isinstance(result, list) else result

    def test_parse_stream_chunk_text(self):
        client = _make_compat_client()
        data = {"choices": [{"delta": {"content": "hi"}}]}
        chunk = self._single(client._parse_stream_chunk(data))

        assert chunk.type == "text_delta"
        assert chunk.text == "hi"

    def test_parse_stream_chunk_tool_call(self):
        client = _make_compat_client()
        data = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q":',
                                },
                            }
                        ]
                    }
                }
            ]
        }
        chunk = self._single(client._parse_stream_chunk(data))

        assert chunk.type == "tool_call_delta"
        assert chunk.tool_call_delta is not None
        assert chunk.tool_call_delta["id"] == "call_1"
        assert chunk.tool_call_delta["name"] == "search"
        assert chunk.tool_call_delta["arguments"] == '{"q":'

    def test_parse_stream_chunk_done(self):
        client = _make_compat_client()
        data = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        chunk = self._single(client._parse_stream_chunk(data))

        # Finish with no content returns a text_delta with empty text
        assert chunk.type == "text_delta"
        assert chunk.text == ""


class TestOpenAICompatHeaders:
    """Tests for OpenAICompatibleClient._build_headers()."""

    def test_headers_bearer_token(self):
        client = _make_compat_client()
        headers = client._build_headers()

        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"


class TestOpenAIClientNativeJson:
    """Tests for OpenAI-specific structured output features."""

    def test_openai_native_json_mode_with_schema(self):
        client = _make_openai_client()
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        kwargs = client._apply_native_json_mode({}, schema)

        assert "response_format" in kwargs
        rf = kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "output"
        assert rf["json_schema"]["strict"] is True
        # Schema is transformed by OpenAIJsonSchemaTransformer
        transformed = rf["json_schema"]["schema"]
        assert transformed["additionalProperties"] is False

    def test_openai_native_json_mode_no_schema(self):
        client = _make_openai_client()
        kwargs = client._apply_native_json_mode({}, None)

        assert kwargs["response_format"] == {"type": "json_object"}

    def test_openai_reasoning_parameter(self):
        client = _make_openai_client(model="o3-mini")
        messages = [{"role": "user", "content": "think hard"}]

        # Dict form: reasoning={"effort": "high"} → reasoning_effort="high"
        req = client._build_request(messages, reasoning={"effort": "high"})
        assert req.body["reasoning_effort"] == "high"

        # String form: reasoning="low" → reasoning_effort="low"
        req2 = client._build_request(messages, reasoning="low")
        assert req2.body["reasoning_effort"] == "low"
