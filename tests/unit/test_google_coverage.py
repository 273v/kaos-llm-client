"""Extended coverage tests for kaos_llm_client.providers.google.

Targets uncovered lines from google.py to raise coverage from ~56% to 90%+.
Tests exercise _convert_messages, _convert_content_to_parts, _parse_response,
_parse_stream_chunk, _build_request, _apply_native_json_mode, and Vertex AI
endpoint/header logic.
"""

from __future__ import annotations

import json

import pytest

from kaos_llm_client.errors import KaosLLMAuthError
from kaos_llm_client.profiles import GoogleJsonSchemaTransformer
from kaos_llm_client.providers.google import (
    GoogleClient,
    _convert_content_to_parts,
    _convert_messages,
)
from kaos_llm_client.settings import KaosLLMSettings
from kaos_llm_client.types import (
    ProviderRequest,
    ToolDefinition,
)

_VERTEX_BASE_URL = "https://us-central1-aiplatform.googleapis.com"


def _make_client(model: str = "gemini-2.5-pro") -> GoogleClient:
    return GoogleClient(model=model, api_key="test-key")


def _make_vertex_client(
    model: str = "gemini-2.5-pro",
    *,
    project: str = "my-project",
    location: str = "us-central1",
) -> GoogleClient:
    settings = KaosLLMSettings(google_project=project, google_location=location)
    return GoogleClient(
        model=model,
        api_key="ya29.test-token",
        base_url=_VERTEX_BASE_URL,
        settings=settings,
    )


def _make_request(request_id: str = "req-test") -> ProviderRequest:
    return ProviderRequest(
        provider="google",
        model="gemini-2.5-pro",
        endpoint="/v1beta/models/gemini-2.5-pro:generateContent",
        body={},
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# _convert_messages tests
# ---------------------------------------------------------------------------


class TestConvertMessages:
    """Tests for _convert_messages helper."""

    def test_system_message_extracted(self):
        """System message -> systemInstruction, not in contents."""
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "hi"},
        ]
        sys_inst, contents = _convert_messages(messages)

        assert sys_inst is not None
        assert sys_inst["parts"] == [{"text": "Be helpful."}]
        assert len(contents) == 1
        assert contents[0]["role"] == "user"

    def test_system_message_list_content(self):
        """System message with list content is joined into a single text."""
        messages = [
            {"role": "system", "content": [{"text": "Part A"}, {"text": "Part B"}]},
            {"role": "user", "content": "hi"},
        ]
        sys_inst, _contents = _convert_messages(messages)

        assert sys_inst is not None
        assert sys_inst["parts"][0]["text"] == "Part A Part B"

    def test_multiple_system_messages_accumulated(self):
        """Multiple system messages are accumulated into systemInstruction parts."""
        messages = [
            {"role": "system", "content": "First."},
            {"role": "system", "content": "Second."},
            {"role": "user", "content": "go"},
        ]
        sys_inst, _contents = _convert_messages(messages)

        assert sys_inst is not None
        assert len(sys_inst["parts"]) == 2
        assert sys_inst["parts"][0]["text"] == "First."
        assert sys_inst["parts"][1]["text"] == "Second."

    def test_tool_result_with_tool_call_id(self):
        """Tool result echoes tool_call_id as functionResponse.id."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "name": "get_weather",
                "content": '{"temp": 72}',
            }
        ]
        sys_inst, contents = _convert_messages(messages)

        assert sys_inst is None
        assert len(contents) == 1
        fr = contents[0]["parts"][0]["functionResponse"]
        assert fr["id"] == "call_abc123"
        assert fr["name"] == "get_weather"
        assert fr["response"] == {"temp": 72}

    def test_tool_result_name_fallback_to_tool_call_id(self):
        """When name is missing, falls back to tool_call_id for functionResponse.name."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_xyz",
                "content": "some text result",
            }
        ]
        _, contents = _convert_messages(messages)

        fr = contents[0]["parts"][0]["functionResponse"]
        assert fr["name"] == "call_xyz"
        assert fr["id"] == "call_xyz"
        assert fr["response"] == {"result": "some text result"}

    def test_assistant_message_with_tool_calls(self):
        """Assistant message with tool_calls produces functionCall parts."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "weather"}',
                        }
                    }
                ],
            }
        ]
        _, contents = _convert_messages(messages)

        assert len(contents) == 1
        assert contents[0]["role"] == "model"
        parts = contents[0]["parts"]
        # First part is text (empty), second is functionCall
        assert any("functionCall" in p for p in parts)
        fc_part = next(p for p in parts if "functionCall" in p)
        assert fc_part["functionCall"]["name"] == "search"
        assert fc_part["functionCall"]["args"] == {"query": "weather"}

    def test_assistant_tool_calls_invalid_json_args(self):
        """Invalid JSON in tool_calls arguments produces empty dict."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "broken",
                            "arguments": "not-json{",
                        }
                    }
                ],
            }
        ]
        _, contents = _convert_messages(messages)

        fc_part = next(p for p in contents[0]["parts"] if "functionCall" in p)
        assert fc_part["functionCall"]["args"] == {}

    def test_assistant_tool_calls_dict_args(self):
        """tool_call arguments already as dict are passed through."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "fn",
                            "arguments": {"key": "val"},
                        }
                    }
                ],
            }
        ]
        _, contents = _convert_messages(messages)

        fc_part = next(p for p in contents[0]["parts"] if "functionCall" in p)
        assert fc_part["functionCall"]["args"] == {"key": "val"}


# ---------------------------------------------------------------------------
# _convert_content_to_parts tests
# ---------------------------------------------------------------------------


class TestConvertContentToParts:
    """Tests for _convert_content_to_parts helper."""

    def test_plain_string(self):
        parts = _convert_content_to_parts("hello world")
        assert parts == [{"text": "hello world"}]

    def test_non_list_non_string(self):
        """Non-string, non-list content is stringified."""
        parts = _convert_content_to_parts(42)
        assert parts == [{"text": "42"}]

    def test_list_text_parts(self):
        content = [{"type": "text", "text": "aaa"}, {"type": "text", "text": "bbb"}]
        parts = _convert_content_to_parts(content)
        assert parts == [{"text": "aaa"}, {"text": "bbb"}]

    def test_list_string_items(self):
        """Plain strings in a list are converted to text parts."""
        parts = _convert_content_to_parts(["abc", "def"])
        assert parts == [{"text": "abc"}, {"text": "def"}]

    def test_image_data_uri(self):
        """data: URI image_url -> inline_data with correct mime and data."""
        content = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBOR=="},
            }
        ]
        parts = _convert_content_to_parts(content)

        assert len(parts) == 1
        assert "inline_data" in parts[0]
        assert parts[0]["inline_data"]["mime_type"] == "image/png"
        assert parts[0]["inline_data"]["data"] == "iVBOR=="

    def test_image_http_url_fallback(self):
        """HTTP image URL -> text fallback since Google requires inline_data."""
        content = [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/cat.png"},
            }
        ]
        parts = _convert_content_to_parts(content)

        assert len(parts) == 1
        assert parts[0] == {"text": "[Image: https://example.com/cat.png]"}

    def test_image_url_string_value(self):
        """image_url with a plain string value (not dict) is handled."""
        content = [
            {
                "type": "image_url",
                "image_url": "https://example.com/img.jpg",
            }
        ]
        parts = _convert_content_to_parts(content)
        assert parts[0] == {"text": "[Image: https://example.com/img.jpg]"}

    def test_document_base64(self):
        """Document part with base64 source -> inline_data."""
        content = [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "JVBER==",
                },
            }
        ]
        parts = _convert_content_to_parts(content)

        assert len(parts) == 1
        assert parts[0]["inline_data"]["mime_type"] == "application/pdf"
        assert parts[0]["inline_data"]["data"] == "JVBER=="

    def test_document_gcs_uri(self):
        """Document part with gs:// URL -> file_data."""
        content = [
            {
                "type": "document",
                "source": {
                    "type": "url",
                    "url": "gs://my-bucket/doc.pdf",
                    "media_type": "application/pdf",
                },
            }
        ]
        parts = _convert_content_to_parts(content)

        assert len(parts) == 1
        assert "file_data" in parts[0]
        assert parts[0]["file_data"]["file_uri"] == "gs://my-bucket/doc.pdf"
        assert parts[0]["file_data"]["mime_type"] == "application/pdf"

    def test_document_http_url_fallback(self):
        """Document part with HTTP URL -> text fallback."""
        content = [
            {
                "type": "document",
                "source": {
                    "type": "url",
                    "url": "https://example.com/doc.pdf",
                },
            }
        ]
        parts = _convert_content_to_parts(content)
        assert parts[0] == {"text": "[Document: https://example.com/doc.pdf]"}

    def test_document_unknown_source_type(self):
        """Document part with unknown source type -> text fallback."""
        content = [
            {
                "type": "document",
                "source": {"type": "unknown"},
            }
        ]
        parts = _convert_content_to_parts(content)
        assert len(parts) == 1
        assert parts[0]["text"].startswith("{")  # stringified dict

    def test_input_audio(self):
        """input_audio -> inline_data with correct MIME type."""
        content = [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": "AAAA==",
                    "format": "mp3",
                },
            }
        ]
        parts = _convert_content_to_parts(content)

        assert len(parts) == 1
        assert parts[0]["inline_data"]["mime_type"] == "audio/mpeg"
        assert parts[0]["inline_data"]["data"] == "AAAA=="

    def test_input_audio_wav(self):
        """WAV format uses audio/wav MIME type."""
        content = [
            {
                "type": "input_audio",
                "input_audio": {"data": "wav_data", "format": "wav"},
            }
        ]
        parts = _convert_content_to_parts(content)
        assert parts[0]["inline_data"]["mime_type"] == "audio/wav"

    def test_input_audio_unknown_format(self):
        """Unknown audio format falls back to audio/<format>."""
        content = [
            {
                "type": "input_audio",
                "input_audio": {"data": "data", "format": "aac"},
            }
        ]
        parts = _convert_content_to_parts(content)
        assert parts[0]["inline_data"]["mime_type"] == "audio/aac"

    def test_file_data_passthrough(self):
        """Pre-formatted file_data dict passes through as-is."""
        content = [
            {
                "file_data": {
                    "file_uri": "gs://bucket/file.txt",
                    "mime_type": "text/plain",
                }
            }
        ]
        parts = _convert_content_to_parts(content)

        assert len(parts) == 1
        assert parts[0]["file_data"]["file_uri"] == "gs://bucket/file.txt"

    def test_inline_data_passthrough(self):
        """Pre-formatted inline_data dict passes through."""
        content = [
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": "abc123",
                }
            }
        ]
        parts = _convert_content_to_parts(content)
        assert parts[0]["inline_data"]["mime_type"] == "image/jpeg"

    def test_unknown_dict_type_fallback(self):
        """Dict part with unknown type -> text fallback."""
        content = [{"type": "weird_thing", "data": "x"}]
        parts = _convert_content_to_parts(content)
        assert parts[0]["text"].startswith("{")

    def test_non_dict_non_string_in_list(self):
        """Non-dict, non-string items in list are stringified."""
        parts = _convert_content_to_parts([123, None])
        assert parts == [{"text": "123"}, {"text": "None"}]


# ---------------------------------------------------------------------------
# _parse_response tests
# ---------------------------------------------------------------------------


class TestParseResponseExtended:
    """Extended tests for GoogleClient._parse_response()."""

    def test_thought_true_becomes_thinking(self):
        """Part with thought=True produces a ContentPart(type='thinking')."""
        client = _make_client()
        raw = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Let me think...", "thought": True},
                            {"text": "The answer is 42."},
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
        }
        resp = client._parse_response(raw, _make_request())

        thinking_parts = [p for p in resp.parts if p.type == "thinking"]
        text_parts = [p for p in resp.parts if p.type == "text"]

        assert len(thinking_parts) == 1
        assert thinking_parts[0].thinking == "Let me think..."
        assert len(text_parts) == 1
        assert text_parts[0].text == "The answer is 42."

    def test_thought_signature_preserved_in_raw(self):
        """thoughtSignature is preserved in the raw field of the part."""
        client = _make_client()
        raw = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "thinking here",
                                "thought": True,
                                "thoughtSignature": "sig_abc123",
                            },
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
        }
        resp = client._parse_response(raw, _make_request())

        assert resp.parts[0].type == "thinking"
        assert resp.parts[0].raw is not None
        assert resp.parts[0].raw["thoughtSignature"] == "sig_abc123"

    def test_function_call_with_provider_id(self):
        """functionCall with id field preserved (Gemini 3+)."""
        client = _make_client()
        raw = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "id": "fc_server_001",
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
        }
        resp = client._parse_response(raw, _make_request())

        tc = resp.tool_calls
        assert len(tc) == 1
        assert tc[0].id == "fc_server_001"
        assert tc[0].name == "get_weather"
        assert tc[0].arguments == {"city": "NYC"}

    def test_function_call_no_id_fallback(self):
        """functionCall without id generates google_tc_* counter."""
        client = _make_client()
        client._tool_call_counter = 0

        raw = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "search",
                                    "args": {"q": "test"},
                                }
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
        }
        resp = client._parse_response(raw, _make_request())

        tc = resp.tool_calls
        assert len(tc) == 1
        assert tc[0].id == "google_tc_1"

    def test_function_call_counter_increments(self):
        """Counter increments across multiple calls without id."""
        client = _make_client()
        client._tool_call_counter = 5

        raw = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "a", "args": {}}},
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
        }
        client._parse_response(raw, _make_request())
        assert client._tool_call_counter == 6

    def test_inline_data_image(self):
        """inline_data with image/* MIME type -> ContentPart(type='image')."""
        client = _make_client()
        raw = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": "iVBOR==",
                                }
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
        }
        resp = client._parse_response(raw, _make_request())

        images = resp.images
        assert len(images) == 1
        assert images[0].media_type == "image/png"
        assert images[0].data == "iVBOR=="

    def test_inline_data_audio(self):
        """inline_data with audio/* MIME type -> ContentPart(type='audio')."""
        client = _make_client()
        raw = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "audio/wav",
                                    "data": "RIFF==",
                                }
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
        }
        resp = client._parse_response(raw, _make_request())

        assert resp.audio is not None
        assert resp.audio.media_type == "audio/wav"

    def test_empty_candidates(self):
        """Empty candidates list still returns a valid response."""
        client = _make_client()
        raw = {"candidates": [], "usageMetadata": {"promptTokenCount": 10}}
        resp = client._parse_response(raw, _make_request())

        assert len(resp.parts) == 0
        assert resp.stop_reason is None

    def test_usage_parsing(self):
        """usageMetadata is parsed into normalized UsageInfo."""
        client = _make_client()
        raw = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "ok"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }
        resp = client._parse_response(raw, _make_request())

        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5
        assert resp.usage.total_tokens == 15

    def test_model_version_from_response(self):
        """modelVersion in response is used for response.model."""
        client = _make_client()
        raw = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "hi"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "modelVersion": "gemini-2.5-pro-preview-05-06",
        }
        resp = client._parse_response(raw, _make_request())
        assert resp.model == "gemini-2.5-pro-preview-05-06"


# ---------------------------------------------------------------------------
# _parse_stream_chunk tests
# ---------------------------------------------------------------------------


class TestParseStreamChunk:
    """Tests for GoogleClient._parse_stream_chunk()."""

    def test_text_delta(self):
        client = _make_client()
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "hello "}],
                        "role": "model",
                    }
                }
            ]
        }
        chunk = client._parse_stream_chunk(data)

        assert chunk.type == "text_delta"
        assert chunk.text == "hello "

    def test_function_call_in_stream(self):
        """functionCall in stream produces tool_call_delta with counter id."""
        client = _make_client()
        client._tool_call_counter = 0
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_time",
                                    "args": {"tz": "UTC"},
                                }
                            }
                        ],
                        "role": "model",
                    }
                }
            ]
        }
        chunk = client._parse_stream_chunk(data)

        assert chunk.type == "tool_call_delta"
        assert chunk.tool_call_delta is not None
        assert chunk.tool_call_delta["id"] == "google_tc_1"
        assert chunk.tool_call_delta["name"] == "get_time"
        assert json.loads(chunk.tool_call_delta["arguments"]) == {"tz": "UTC"}

    def test_inline_data_image_in_stream(self):
        """inline_data image in stream -> text_delta with empty text."""
        client = _make_client()
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": "abc",
                                }
                            }
                        ],
                        "role": "model",
                    }
                }
            ]
        }
        chunk = client._parse_stream_chunk(data)
        assert chunk.type == "text_delta"
        assert chunk.text == ""

    def test_inline_data_audio_in_stream(self):
        """inline_data audio in stream -> text_delta with empty text."""
        client = _make_client()
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "audio/wav",
                                    "data": "riff",
                                }
                            }
                        ],
                        "role": "model",
                    }
                }
            ]
        }
        chunk = client._parse_stream_chunk(data)
        assert chunk.type == "text_delta"

    def test_usage_only_chunk(self):
        """Chunk with usageMetadata but no candidates -> usage chunk."""
        client = _make_client()
        data = {
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50,
                "totalTokenCount": 150,
            }
        }
        chunk = client._parse_stream_chunk(data)

        assert chunk.type == "usage"
        assert chunk.usage is not None
        assert chunk.usage.input_tokens == 100
        assert chunk.usage.output_tokens == 50

    def test_empty_candidates_no_usage(self):
        """Empty data with no candidates and no usage -> text_delta empty."""
        client = _make_client()
        chunk = client._parse_stream_chunk({})
        assert chunk.type == "text_delta"
        assert chunk.text == ""

    def test_usage_in_final_chunk_no_text(self):
        """Final chunk with usage and empty text parts -> usage type."""
        client = _make_client()
        data = {
            "candidates": [
                {
                    "content": {"parts": [], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }
        chunk = client._parse_stream_chunk(data)
        assert chunk.type == "usage"


# ---------------------------------------------------------------------------
# _build_request tests
# ---------------------------------------------------------------------------


class TestBuildRequestExtended:
    """Extended tests for GoogleClient._build_request()."""

    def test_with_tools(self):
        """Tools are formatted as functionDeclarations."""
        client = _make_client()
        tools = [
            ToolDefinition(
                name="search",
                description="Search the web",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            )
        ]
        req = client._build_request(
            [{"role": "user", "content": "find info"}],
            tools=tools,
        )

        assert "tools" in req.body
        declarations = req.body["tools"][0]["functionDeclarations"]
        assert len(declarations) == 1
        assert declarations[0]["name"] == "search"
        assert declarations[0]["description"] == "Search the web"

    def test_tool_without_description(self):
        """Tool without description omits description key."""
        client = _make_client()
        tools = [
            ToolDefinition(
                name="noop",
                parameters={"type": "object"},
            )
        ]
        req = client._build_request(
            [{"role": "user", "content": "test"}],
            tools=tools,
        )

        decl = req.body["tools"][0]["functionDeclarations"][0]
        assert "description" not in decl

    def test_generation_config_temperature(self):
        """Temperature is passed in generationConfig."""
        client = _make_client()
        req = client._build_request(
            [{"role": "user", "content": "hi"}],
            temperature=0.5,
        )

        gc = req.body["generationConfig"]
        assert gc["temperature"] == 0.5

    def test_generation_config_top_p_normalized(self):
        """top_p (snake_case) is normalized to topP (camelCase)."""
        client = _make_client()
        req = client._build_request(
            [{"role": "user", "content": "hi"}],
            top_p=0.9,
        )

        gc = req.body["generationConfig"]
        assert gc["topP"] == 0.9

    def test_generation_config_top_k_normalized(self):
        """top_k (snake_case) is normalized to topK."""
        client = _make_client()
        req = client._build_request(
            [{"role": "user", "content": "hi"}],
            top_k=40,
        )

        gc = req.body["generationConfig"]
        assert gc["topK"] == 40

    def test_stop_sequences(self):
        """stop kwarg maps to stopSequences."""
        client = _make_client()
        req = client._build_request(
            [{"role": "user", "content": "hi"}],
            stop=["END", "STOP"],
        )
        assert req.body["generationConfig"]["stopSequences"] == ["END", "STOP"]

    def test_stop_sequences_alt_key(self):
        """stop_sequences kwarg also maps to stopSequences."""
        client = _make_client()
        req = client._build_request(
            [{"role": "user", "content": "hi"}],
            stop_sequences=["FIN"],
        )
        assert req.body["generationConfig"]["stopSequences"] == ["FIN"]

    def test_streaming_endpoint(self):
        """stream=True uses streamGenerateContent endpoint."""
        client = _make_client()
        req = client._build_request(
            [{"role": "user", "content": "hi"}],
            stream=True,
        )
        assert "streamGenerateContent" in req.endpoint
        assert "alt=sse" in req.endpoint

    def test_max_tokens_via_field_name(self):
        """maxOutputTokens (profile field name) is respected."""
        client = _make_client()
        req = client._build_request(
            [{"role": "user", "content": "hi"}],
            maxOutputTokens=1000,
        )
        assert req.body["generationConfig"]["maxOutputTokens"] == 1000


# ---------------------------------------------------------------------------
# _apply_native_json_mode tests
# ---------------------------------------------------------------------------


class TestApplyNativeJsonMode:
    """Tests for GoogleClient._apply_native_json_mode()."""

    def test_merges_into_generation_config(self):
        """JSON mode config does not overwrite existing generationConfig."""
        client = _make_client()
        kwargs = client._apply_native_json_mode({}, None)

        assert "_google_json_config" in kwargs
        assert kwargs["_google_json_config"]["responseMimeType"] == "application/json"

    def test_with_schema(self):
        """Schema is included as responseSchema."""
        client = _make_client()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        kwargs = client._apply_native_json_mode({}, schema)

        config = kwargs["_google_json_config"]
        assert "responseSchema" in config

    def test_applies_transformer(self):
        """GoogleJsonSchemaTransformer is applied when profile has it."""
        client = _make_client()
        # Verify the profile has the transformer
        assert client.profile.json_schema_transformer is GoogleJsonSchemaTransformer

        # Schema with features Google doesn't support (title, const, default)
        schema = {
            "type": "object",
            "title": "MySchema",
            "properties": {
                "status": {"type": "string", "const": "active"},
            },
        }
        kwargs = client._apply_native_json_mode({}, schema)

        transformed = kwargs["_google_json_config"]["responseSchema"]
        # title should be stripped, const should become enum
        assert "title" not in transformed
        assert transformed["properties"]["status"]["enum"] == ["active"]

    def test_build_request_merges_json_config(self):
        """_build_request merges _google_json_config into generationConfig."""
        client = _make_client()
        req = client._build_request(
            [{"role": "user", "content": "give json"}],
            max_tokens=500,
            _google_json_config={"responseMimeType": "application/json"},
        )

        gc = req.body["generationConfig"]
        assert gc["maxOutputTokens"] == 500
        assert gc["responseMimeType"] == "application/json"

    def test_build_request_json_config_creates_gen_config(self):
        """_google_json_config creates generationConfig if none exists."""
        client = _make_client()
        # Build a request that would have empty generationConfig,
        # then add _google_json_config
        req = client._build_request(
            [{"role": "user", "content": "json"}],
            _google_json_config={"responseMimeType": "application/json"},
        )
        assert "generationConfig" in req.body
        assert req.body["generationConfig"]["responseMimeType"] == "application/json"


# ---------------------------------------------------------------------------
# Vertex AI endpoint and headers
# ---------------------------------------------------------------------------


class TestVertexEndpointExtended:
    """Extended Vertex AI tests for endpoint format."""

    def test_vertex_default_endpoint_format(self):
        client = _make_vertex_client(project="p1", location="us-central1")
        endpoint = client._default_endpoint()
        assert endpoint.startswith("/v1/projects/p1/locations/us-central1/")
        assert endpoint.endswith(":generateContent")

    def test_vertex_stream_endpoint_format(self):
        client = _make_vertex_client(project="p1", location="eu-west1")
        endpoint = client._stream_endpoint()
        assert "streamGenerateContent" in endpoint
        assert "alt=sse" in endpoint
        assert "eu-west1" in endpoint


class TestVertexHeadersExtended:
    """Extended Vertex AI header tests."""

    def test_vertex_uses_bearer(self):
        client = _make_vertex_client()
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer ya29.test-token"
        assert "x-goog-api-key" not in headers

    def test_ai_studio_uses_api_key(self):
        client = _make_client()
        headers = client._build_headers()
        assert headers["x-goog-api-key"] == "test-key"
        assert "Authorization" not in headers


# ---------------------------------------------------------------------------
# Auth error tests
# ---------------------------------------------------------------------------


class TestGoogleAuth:
    """Tests for API key validation in GoogleClient."""

    def test_missing_api_key_ai_studio(self, monkeypatch: pytest.MonkeyPatch):
        """Missing Google API key raises KaosLLMAuthError."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_GENERATIVE_AI_API_KEY", raising=False)
        monkeypatch.delenv("KAOS_LLM_GOOGLE_API_KEY", raising=False)
        settings = KaosLLMSettings(google_api_key=None)
        client = GoogleClient(model="gemini-2.5-pro", settings=settings)
        with pytest.raises(KaosLLMAuthError, match="not configured"):
            client._build_headers()

    def test_missing_api_key_vertex(self, monkeypatch: pytest.MonkeyPatch):
        """Missing Vertex access token raises KaosLLMAuthError."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_GENERATIVE_AI_API_KEY", raising=False)
        monkeypatch.delenv("KAOS_LLM_GOOGLE_API_KEY", raising=False)
        settings = KaosLLMSettings(google_api_key=None, google_project="proj")
        client = GoogleClient(model="gemini-2.5-pro", settings=settings, base_url=_VERTEX_BASE_URL)
        with pytest.raises(KaosLLMAuthError, match="access token"):
            client._build_headers()

    def test_empty_api_key(self, monkeypatch: pytest.MonkeyPatch):
        """Empty string API key raises KaosLLMAuthError."""
        from pydantic import SecretStr

        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_GENERATIVE_AI_API_KEY", raising=False)
        monkeypatch.delenv("KAOS_LLM_GOOGLE_API_KEY", raising=False)
        settings = KaosLLMSettings(google_api_key=SecretStr(""))
        client = GoogleClient(model="gemini-2.5-pro", settings=settings)
        with pytest.raises(KaosLLMAuthError, match="empty"):
            client._build_headers()

    def test_missing_vertex_project(self):
        """Vertex mode without project raises KaosLLMAuthError."""
        settings = KaosLLMSettings(google_project=None)
        client = GoogleClient(
            model="gemini-2.5-pro",
            api_key="tok",
            base_url=_VERTEX_BASE_URL,
            settings=settings,
        )
        with pytest.raises(KaosLLMAuthError, match="project ID"):
            client._default_endpoint()


# ---------------------------------------------------------------------------
# AI Studio stream endpoint
# ---------------------------------------------------------------------------


class TestAIStudioStreamEndpoint:
    """Test AI Studio stream endpoint."""

    def test_stream_endpoint_ai_studio(self):
        client = _make_client(model="gemini-2.0-flash")
        endpoint = client._stream_endpoint()
        assert endpoint == "/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse"
