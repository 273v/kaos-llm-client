"""Comprehensive tests for provider-specific features, edge cases, and gaps.

Covers multimodal inputs, tool result rewriting, multi-turn conversations,
Google tool calling, streaming edge cases, and Anthropic stream tool calls.
No HTTP calls — all tests use ``_build_request()`` and ``_parse_stream_chunk()`` directly.
"""

from __future__ import annotations

from kaos_llm_client.providers.anthropic import AnthropicClient
from kaos_llm_client.providers.google import GoogleClient
from kaos_llm_client.providers.openai_compat import OpenAICompatibleClient
from kaos_llm_client.types import (
    ProviderRequest,
    StreamAccumulator,
    StreamChunk,
    ToolDefinition,
    UsageInfo,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _anthropic(model: str = "claude-sonnet-4-6") -> AnthropicClient:
    return AnthropicClient(model=model, api_key="test-key")


def _google(model: str = "gemini-2.5-pro") -> GoogleClient:
    return GoogleClient(model=model, api_key="test-key")


def _openai(model: str = "gpt-5") -> OpenAICompatibleClient:
    return OpenAICompatibleClient(model=model, api_key="test-key")


def _anthropic_request(request_id: str = "req-test") -> ProviderRequest:
    return ProviderRequest(
        provider="anthropic",
        model="claude-sonnet-4-6",
        endpoint="/v1/messages",
        body={},
        request_id=request_id,
    )


def _google_request(request_id: str = "req-test") -> ProviderRequest:
    return ProviderRequest(
        provider="google",
        model="gemini-2.5-pro",
        endpoint="/v1beta/models/gemini-2.5-pro:generateContent",
        body={},
        request_id=request_id,
    )


# ===========================================================================
# TestMultimodalInputs
# ===========================================================================


class TestMultimodalInputs:
    """Multimodal image content conversion across providers."""

    def test_anthropic_image_url_data_uri(self):
        """Data URI image_url is converted to Anthropic base64 source format."""
        client = _anthropic()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANS"},
                    },
                ],
            }
        ]
        req = client._build_request(messages)

        parts = req.body["messages"][0]["content"]
        assert len(parts) == 2

        # Text part passes through
        assert parts[0] == {"type": "text", "text": "What is in this image?"}

        # Image part converted to Anthropic base64 source
        img = parts[1]
        assert img["type"] == "image"
        assert img["source"]["type"] == "base64"
        assert img["source"]["media_type"] == "image/png"
        assert img["source"]["data"] == "iVBORw0KGgoAAAANS"

    def test_anthropic_image_url_http(self):
        """HTTP URL image_url is converted to Anthropic URL source format."""
        client = _anthropic()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.jpg"},
                    },
                ],
            }
        ]
        req = client._build_request(messages)

        parts = req.body["messages"][0]["content"]
        assert len(parts) == 1
        img = parts[0]
        assert img["type"] == "image"
        assert img["source"]["type"] == "url"
        assert img["source"]["url"] == "https://example.com/cat.jpg"

    def test_google_image_url_data_uri(self):
        """Data URI image_url is converted to Google inline_data format."""
        client = _google()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"},
                    },
                ],
            }
        ]
        req = client._build_request(messages)

        parts = req.body["contents"][0]["parts"]
        assert len(parts) == 2

        # Text part
        assert parts[0] == {"text": "Describe this."}

        # Image part converted to Google inline_data
        img = parts[1]
        assert "inline_data" in img
        assert img["inline_data"]["mime_type"] == "image/jpeg"
        assert img["inline_data"]["data"] == "/9j/4AAQ"


# ===========================================================================
# TestToolResultMessages
# ===========================================================================


class TestToolResultMessages:
    """Tool result message rewriting across providers."""

    def test_anthropic_tool_result_rewriting(self):
        """role:'tool' messages are rewritten to role:'user' with tool_result blocks."""
        client = _anthropic()
        messages = [
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_abc",
                        "name": "get_weather",
                        "input": {"city": "NYC"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_abc",
                "content": "Sunny, 72F",
            },
        ]
        req = client._build_request(messages)

        api_msgs = req.body["messages"]
        assert len(api_msgs) == 3

        # The tool result becomes a user message with tool_result block
        tool_result_msg = api_msgs[2]
        assert tool_result_msg["role"] == "user"
        assert isinstance(tool_result_msg["content"], list)
        assert len(tool_result_msg["content"]) == 1

        block = tool_result_msg["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "toolu_abc"
        assert block["content"] == "Sunny, 72F"

    def test_anthropic_consecutive_tool_results_merged(self):
        """Two consecutive tool results are merged into one user message."""
        client = _anthropic()
        messages = [
            {"role": "user", "content": "Get weather for both cities"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "NYC"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "get_weather",
                        "input": {"city": "LA"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_1",
                "content": "Sunny, 72F",
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_2",
                "content": "Cloudy, 65F",
            },
        ]
        req = client._build_request(messages)

        api_msgs = req.body["messages"]
        # user + assistant + merged_user (not user + assistant + user + user)
        assert len(api_msgs) == 3

        merged = api_msgs[2]
        assert merged["role"] == "user"
        assert isinstance(merged["content"], list)
        assert len(merged["content"]) == 2
        assert merged["content"][0]["type"] == "tool_result"
        assert merged["content"][0]["tool_use_id"] == "toolu_1"
        assert merged["content"][1]["type"] == "tool_result"
        assert merged["content"][1]["tool_use_id"] == "toolu_2"

    def test_google_tool_result_as_function_response(self):
        """role:'tool' message becomes a Google functionResponse."""
        client = _google()
        messages = [
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "name": "get_weather",
                "content": '{"temp": 72, "condition": "sunny"}',
            },
        ]
        req = client._build_request(messages)

        contents = req.body["contents"]
        assert len(contents) == 3

        # Third message should be a function role with functionResponse
        tool_msg = contents[2]
        assert tool_msg["role"] == "function"
        assert len(tool_msg["parts"]) == 1

        fr = tool_msg["parts"][0]["functionResponse"]
        assert fr["name"] == "get_weather"
        # JSON string content is parsed into a dict
        assert fr["response"] == {"temp": 72, "condition": "sunny"}


# ===========================================================================
# TestMultiTurnConversation
# ===========================================================================


class TestMultiTurnConversation:
    """Multi-turn conversation structure across providers."""

    def test_openai_multi_turn_roundtrip(self):
        """System + user + assistant + user messages pass through in OpenAI format."""
        client = _openai()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "What is 2+2?"},
        ]
        req = client._build_request(messages)

        # OpenAI keeps all messages including system in the messages array
        assert req.body["messages"] == messages
        assert len(req.body["messages"]) == 4
        assert req.body["messages"][0]["role"] == "system"
        assert req.body["messages"][1]["role"] == "user"
        assert req.body["messages"][2]["role"] == "assistant"
        assert req.body["messages"][3]["role"] == "user"

    def test_anthropic_multi_turn_roundtrip(self):
        """System is extracted; user + assistant + user remain in messages."""
        client = _anthropic()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "What is 2+2?"},
        ]
        req = client._build_request(messages)

        # System extracted to top-level
        assert req.body["system"] == "You are helpful."

        # Only non-system messages remain
        api_msgs = req.body["messages"]
        assert len(api_msgs) == 3
        assert api_msgs[0]["role"] == "user"
        assert api_msgs[0]["content"] == "Hello"
        assert api_msgs[1]["role"] == "assistant"
        assert api_msgs[1]["content"] == "Hi! How can I help?"
        assert api_msgs[2]["role"] == "user"
        assert api_msgs[2]["content"] == "What is 2+2?"

    def test_anthropic_tool_use_multi_turn(self):
        """Full tool-use conversation.

        user -> assistant(tool_use) -> tool(result) -> assistant(text)
        """
        client = _anthropic()
        messages = [
            {"role": "user", "content": "What is the weather in NYC?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_abc",
                        "name": "get_weather",
                        "input": {"city": "NYC"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_abc",
                "content": "Sunny, 72F",
            },
            {
                "role": "assistant",
                "content": "The weather in NYC is sunny and 72F.",
            },
        ]
        req = client._build_request(messages)

        api_msgs = req.body["messages"]
        assert len(api_msgs) == 4

        # Message 0: user
        assert api_msgs[0]["role"] == "user"
        assert api_msgs[0]["content"] == "What is the weather in NYC?"

        # Message 1: assistant with tool_use (passed through)
        assert api_msgs[1]["role"] == "assistant"
        assert isinstance(api_msgs[1]["content"], list)
        assert api_msgs[1]["content"][0]["type"] == "tool_use"

        # Message 2: tool result rewritten as user
        assert api_msgs[2]["role"] == "user"
        assert api_msgs[2]["content"][0]["type"] == "tool_result"
        assert api_msgs[2]["content"][0]["tool_use_id"] == "toolu_abc"

        # Message 3: final assistant text
        assert api_msgs[3]["role"] == "assistant"
        assert api_msgs[3]["content"] == "The weather in NYC is sunny and 72F."


# ===========================================================================
# TestGoogleToolCalling
# ===========================================================================


class TestGoogleToolCalling:
    """Google Gemini tool calling: request format, response parsing, ID uniqueness."""

    def test_google_build_request_with_tools(self):
        """Tool definitions are converted to Google functionDeclarations format."""
        client = _google()
        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                },
            ),
            ToolDefinition(
                name="search",
                description="Search the web",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                },
            ),
        ]
        req = client._build_request([{"role": "user", "content": "weather?"}], tools=tools)

        assert "tools" in req.body
        tool_list = req.body["tools"]
        assert len(tool_list) == 1  # Google wraps all in one tools entry

        declarations = tool_list[0]["functionDeclarations"]
        assert len(declarations) == 2
        assert declarations[0]["name"] == "get_weather"
        assert declarations[0]["description"] == "Get weather for a location"
        assert declarations[0]["parameters"] == tools[0].parameters
        assert declarations[1]["name"] == "search"

    def test_google_parse_response_function_call(self):
        """functionCall in response is parsed into tool_calls."""
        client = _google()
        raw = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"city": "NYC"},
                                }
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }
        request = _google_request()
        resp = client._parse_response(raw, request)

        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "NYC"}
        assert tc.id.startswith("google_tc_")

    def test_google_tool_call_unique_ids(self):
        """Two function calls to the same function get different IDs."""
        client = _google()
        raw1 = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"city": "NYC"},
                                }
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 3,
                "totalTokenCount": 8,
            },
        }
        raw2 = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"city": "LA"},
                                }
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 3,
                "totalTokenCount": 8,
            },
        }
        request = _google_request()
        resp1 = client._parse_response(raw1, request)
        resp2 = client._parse_response(raw2, request)

        id1 = resp1.tool_calls[0].id
        id2 = resp2.tool_calls[0].id
        assert id1 != id2
        assert id1.startswith("google_tc_")
        assert id2.startswith("google_tc_")


# ===========================================================================
# TestStreamingEdgeCases
# ===========================================================================


class TestStreamingEdgeCases:
    """Streaming parse edge cases: parallel tool calls, usage merging, accumulation."""

    def test_openai_parallel_tool_calls_stream(self):
        """SSE chunk with multiple tool_calls entries returns a list of chunks."""
        client = _openai()
        data = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":',
                                },
                            },
                            {
                                "index": 1,
                                "id": "call_2",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q":',
                                },
                            },
                        ]
                    }
                }
            ]
        }
        result = client._parse_stream_chunk(data)

        # Multiple tool calls in one SSE event should return a list
        assert isinstance(result, list)
        assert len(result) == 2

        assert result[0].type == "tool_call_delta"
        tc0 = result[0].tool_call_delta
        assert tc0 is not None
        assert tc0["id"] == "call_1"
        assert tc0["name"] == "get_weather"

        assert result[1].type == "tool_call_delta"
        tc1 = result[1].tool_call_delta
        assert tc1 is not None
        assert tc1["id"] == "call_2"
        assert tc1["name"] == "search"

    def test_anthropic_usage_merging(self):
        """message_start and message_delta usage are both preserved."""
        client = _anthropic()

        # message_start carries input_tokens
        start_data = {
            "type": "message_start",
            "message": {
                "usage": {"input_tokens": 100},
            },
        }
        start_chunk = client._parse_stream_chunk(start_data)
        assert start_chunk.type == "usage"
        assert start_chunk.usage is not None
        assert start_chunk.usage.input_tokens == 100

        # message_delta carries output_tokens
        delta_data = {
            "type": "message_delta",
            "usage": {"output_tokens": 50},
        }
        delta_chunk = client._parse_stream_chunk(delta_data)
        assert delta_chunk.type == "usage"
        assert delta_chunk.usage is not None
        assert delta_chunk.usage.output_tokens == 50

        # Feed both into accumulator to verify merging
        acc = StreamAccumulator(provider="anthropic", model="claude-sonnet-4-6", request_id="r1")
        acc.feed(start_chunk)
        acc.feed(delta_chunk)

        # Both values should be preserved via _merge_usage
        assert acc._usage is not None
        assert acc._usage.input_tokens == 100
        assert acc._usage.output_tokens == 50

    def test_stream_accumulator_multiple_tool_calls(self):
        """Accumulate two interleaved tool calls with different IDs."""
        acc = StreamAccumulator(provider="openai", model="gpt-5", request_id="r1")

        # First tool call starts
        acc.feed(
            StreamChunk(
                type="tool_call_delta",
                tool_call_delta={"id": "call_A", "name": "get_weather", "arguments": ""},
            )
        )
        acc.feed(
            StreamChunk(
                type="tool_call_delta",
                tool_call_delta={"arguments": '{"city":'},
            )
        )
        acc.feed(
            StreamChunk(
                type="tool_call_delta",
                tool_call_delta={"arguments": '"NYC"}'},
            )
        )

        # Second tool call starts (has an id, so first is finalized)
        acc.feed(
            StreamChunk(
                type="tool_call_delta",
                tool_call_delta={"id": "call_B", "name": "search", "arguments": ""},
            )
        )
        acc.feed(
            StreamChunk(
                type="tool_call_delta",
                tool_call_delta={"arguments": '{"q":"test"}'},
            )
        )

        # Done finalizes the second
        acc.feed(StreamChunk(type="done"))

        response = acc.accumulated
        assert len(response.tool_calls) == 2

        tc_a = response.tool_calls[0]
        assert tc_a.id == "call_A"
        assert tc_a.name == "get_weather"
        assert tc_a.arguments == {"city": "NYC"}

        tc_b = response.tool_calls[1]
        assert tc_b.id == "call_B"
        assert tc_b.name == "search"
        assert tc_b.arguments == {"q": "test"}


# ===========================================================================
# TestAnthropicStreamToolCalls
# ===========================================================================


class TestAnthropicStreamToolCalls:
    """Anthropic SSE events for tool use streaming."""

    def test_anthropic_content_block_start_tool_use(self):
        """content_block_start with tool_use type emits tool_call_delta with id and name."""
        client = _anthropic()
        data = {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_xyz",
                "name": "get_weather",
            },
        }
        chunk = client._parse_stream_chunk(data)

        assert chunk.type == "tool_call_delta"
        assert chunk.tool_call_delta is not None
        assert chunk.tool_call_delta["id"] == "toolu_xyz"
        assert chunk.tool_call_delta["name"] == "get_weather"
        assert chunk.tool_call_delta["arguments"] == ""

    def test_anthropic_input_json_delta(self):
        """content_block_delta with input_json_delta emits a tool_call_delta."""
        client = _anthropic()
        data = {
            "type": "content_block_delta",
            "index": 1,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"city": "NYC"',
            },
        }
        chunk = client._parse_stream_chunk(data)

        assert chunk.type == "tool_call_delta"
        assert chunk.tool_call_delta is not None
        assert chunk.tool_call_delta["arguments"] == '{"city": "NYC"'

    def test_anthropic_full_tool_use_stream_roundtrip(self):
        """Full Anthropic tool-use stream: block_start -> json deltas -> block_stop -> done.

        Verifies that the accumulator produces the correct final tool call.
        """
        client = _anthropic()
        acc = StreamAccumulator(provider="anthropic", model="claude-sonnet-4-6", request_id="r1")

        # content_block_start for tool_use
        chunk1 = client._parse_stream_chunk(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "search",
                },
            }
        )
        acc.feed(chunk1)

        # input_json_delta chunks
        chunk2 = client._parse_stream_chunk(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"query"',
                },
            }
        )
        acc.feed(chunk2)

        chunk3 = client._parse_stream_chunk(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": ': "hello"}',
                },
            }
        )
        acc.feed(chunk3)

        # content_block_stop (no-op)
        chunk4 = client._parse_stream_chunk(
            {
                "type": "content_block_stop",
                "index": 0,
            }
        )
        acc.feed(chunk4)

        # done
        acc.feed(StreamChunk(type="done"))

        response = acc.accumulated
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert tc.id == "toolu_123"
        assert tc.name == "search"
        assert tc.arguments == {"query": "hello"}


# ===========================================================================
# TestUsageMerge
# ===========================================================================


class TestUsageMerge:
    """StreamAccumulator._merge_usage edge cases."""

    def test_merge_usage_first_wins_input_second_wins_output(self):
        """First usage has input_tokens, second has output_tokens. Both preserved."""
        acc = StreamAccumulator(provider="test", model="m", request_id="r1")

        acc.feed(
            StreamChunk(
                type="usage",
                usage=UsageInfo(input_tokens=100, output_tokens=0),
            )
        )
        acc.feed(
            StreamChunk(
                type="usage",
                usage=UsageInfo(input_tokens=0, output_tokens=50),
            )
        )

        assert acc._usage is not None
        assert acc._usage.input_tokens == 100
        assert acc._usage.output_tokens == 50

    def test_merge_usage_preserves_cache_tokens(self):
        """Cache tokens from first usage are preserved when second has none."""
        acc = StreamAccumulator(provider="test", model="m", request_id="r1")

        acc.feed(
            StreamChunk(
                type="usage",
                usage=UsageInfo(
                    input_tokens=100,
                    cache_read_tokens=20,
                    cache_creation_tokens=10,
                ),
            )
        )
        acc.feed(
            StreamChunk(
                type="usage",
                usage=UsageInfo(output_tokens=50),
            )
        )

        assert acc._usage is not None
        assert acc._usage.cache_read_tokens == 20
        assert acc._usage.cache_creation_tokens == 10
        assert acc._usage.output_tokens == 50

    def test_merge_usage_takes_max(self):
        """When both updates have the same field, max is taken."""
        acc = StreamAccumulator(provider="test", model="m", request_id="r1")

        acc.feed(
            StreamChunk(
                type="usage",
                usage=UsageInfo(input_tokens=50, output_tokens=10),
            )
        )
        acc.feed(
            StreamChunk(
                type="usage",
                usage=UsageInfo(input_tokens=100, output_tokens=5),
            )
        )

        assert acc._usage is not None
        # max(50, 100) = 100, max(10, 5) = 10
        assert acc._usage.input_tokens == 100
        assert acc._usage.output_tokens == 10
