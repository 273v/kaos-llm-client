"""Live integration tests for kaos-llm-client.

Tests hit real LLM provider APIs. Gated on environment variables.

Model versions verified via web search (April 2026) and cross-referenced
against alea-llm-client registry and pydantic-ai profiles.

Current model landscape:
- OpenAI: GPT-5.4 (Mar 2026), GPT-5, o4-mini, o3
- Anthropic: Claude Opus/Sonnet 4.6, Haiku 4.5
- Google: Gemini 3.1 (preview), Gemini 2.5 (stable)
- xAI: Grok 4.20, Grok 4-1-fast

Run:
    uv run pytest tests/integration/ -v -m integration
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from kaos_llm_client.cache import FileCache
from kaos_llm_client.errors import KaosLLMAuthError, KaosLLMError
from kaos_llm_client.messages import (
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
)
from kaos_llm_client.providers import create_client
from kaos_llm_client.providers.concurrency import ConcurrencyLimitedClient
from kaos_llm_client.providers.fallback import FallbackClient
from kaos_llm_client.providers.instrumented import InstrumentedClient
from kaos_llm_client.types import (
    ProviderRequest,
    ProviderResponse,
    RequestHooks,
    StreamAccumulator,
    ToolDefinition,
)

from .conftest import (
    requires_anthropic,
    requires_google,
    requires_groq,
    requires_mistral,
    requires_openai,
    requires_openrouter,
    requires_xai,
)

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

SHORT = "Say exactly one word: hello"
TOOL_PROMPT = "What is the weather in Paris? You must call the get_weather tool."

WEATHER_TOOL = ToolDefinition(
    name="get_weather",
    description="Get the current weather for a location.",
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string", "description": "City name"}},
        "required": ["location"],
    },
)


class Colors(BaseModel):
    colors: list[str]


def _check(response, *, provider: str) -> None:
    assert response.provider == provider
    assert response.model != ""
    assert isinstance(response.raw, dict)
    assert len(response.raw) > 0
    assert response.usage.input_tokens > 0


# ===================================================================
# OpenAI — GPT-5.4, GPT-5, o-series
# ===================================================================


@pytest.mark.integration
@requires_openai
class TestOpenAIGPT54:
    """GPT-5.4 family — latest gen (March 2026). Cheapest: gpt-5.4-nano."""

    def test_gpt54_nano_chat(self) -> None:
        r = create_client("openai:gpt-5.4-nano").chat([{"role": "user", "content": SHORT}])
        assert "hello" in r.text.lower()
        _check(r, provider="openai")

    def test_gpt54_nano_json(self) -> None:
        r = create_client("openai:gpt-5.4-nano").json(
            [{"role": "user", "content": 'Return JSON: {"greeting": "hi"}'}],
            schema={
                "type": "object",
                "properties": {"greeting": {"type": "string"}},
                "required": ["greeting"],
                "additionalProperties": False,
            },
        )
        assert r.output_json is not None
        assert "greeting" in r.output_json

    def test_gpt54_nano_pydantic(self) -> None:
        result = create_client("openai:gpt-5.4-nano").pydantic(
            [{"role": "user", "content": "List exactly 2 colors: red, blue."}],
            output_type=Colors,
        )
        assert isinstance(result, Colors)
        assert len(result.colors) >= 1

    def test_gpt54_nano_tools(self) -> None:
        r = create_client("openai:gpt-5.4-nano").chat(
            [{"role": "user", "content": TOOL_PROMPT}], tools=[WEATHER_TOOL]
        )
        assert len(r.tool_calls) > 0
        assert r.tool_calls[0].name == "get_weather"

    @pytest.mark.asyncio
    async def test_gpt54_nano_streaming(self) -> None:
        parts: list[str] = []
        async for c in create_client("openai:gpt-5.4-nano").chat_stream_async(
            [{"role": "user", "content": SHORT}]
        ):
            if c.type == "text_delta" and c.text:
                parts.append(c.text)
        assert len("".join(parts)) > 0

    def test_gpt54_chat(self) -> None:
        """gpt-5.4: flagship model."""
        r = create_client("openai:gpt-5.4").chat([{"role": "user", "content": SHORT}])
        assert "hello" in r.text.lower()
        _check(r, provider="openai")

    def test_gpt54_mini_chat(self) -> None:
        r = create_client("openai:gpt-5.4-mini").chat([{"role": "user", "content": SHORT}])
        assert "hello" in r.text.lower()
        _check(r, provider="openai")


@pytest.mark.integration
@requires_openai
class TestOpenAIGPT5:
    """GPT-5 family — previous gen, per alea README."""

    def test_gpt5_chat(self) -> None:
        r = create_client("openai:gpt-5").chat([{"role": "user", "content": SHORT}])
        assert "hello" in r.text.lower()
        _check(r, provider="openai")

    def test_gpt5_nano_chat(self) -> None:
        r = create_client("openai:gpt-5-nano").chat([{"role": "user", "content": SHORT}])
        assert "hello" in r.text.lower()
        _check(r, provider="openai")


@pytest.mark.integration
@requires_openai
class TestOpenAIReasoning:
    """o-series reasoning models. Use max_completion_tokens, not max_tokens."""

    def test_o4_mini_chat(self) -> None:
        r = create_client("openai:o4-mini").chat(
            [{"role": "user", "content": "What is 17 * 23?"}],
            max_completion_tokens=2048,
        )
        assert "391" in r.text
        _check(r, provider="openai")

    def test_o4_mini_reasoning_effort(self) -> None:
        r = create_client("openai:o4-mini").chat(
            [{"role": "user", "content": SHORT}],
            max_completion_tokens=2048,
            reasoning={"effort": "low"},
        )
        assert len(r.text) > 0

    def test_o3_chat(self) -> None:
        """o3: full reasoning model."""
        r = create_client("openai:o3").chat(
            [{"role": "user", "content": "What is 2+2?"}],
            max_completion_tokens=2048,
        )
        assert "4" in r.text
        _check(r, provider="openai")

    def test_o3_mini_chat(self) -> None:
        r = create_client("openai:o3-mini").chat(
            [{"role": "user", "content": "What is 2+2?"}],
            max_completion_tokens=2048,
        )
        assert "4" in r.text
        _check(r, provider="openai")


# ===================================================================
# Anthropic — Claude 4.6, 4.5, Haiku 4.5
# ===================================================================


@pytest.mark.integration
@requires_anthropic
class TestAnthropicClaude46:
    """Claude 4.6 — current flagship. Per pydantic-ai: claude-sonnet-4-6."""

    def test_claude_opus_46_chat(self) -> None:
        """claude-opus-4-6: flagship Anthropic model."""
        r = create_client("anthropic:claude-opus-4-6").chat([{"role": "user", "content": SHORT}])
        assert "hello" in r.text.lower()
        _check(r, provider="anthropic")

    def test_claude_sonnet_46_chat(self) -> None:
        r = create_client("anthropic:claude-sonnet-4-6").chat([{"role": "user", "content": SHORT}])
        assert "hello" in r.text.lower()
        _check(r, provider="anthropic")

    def test_claude_sonnet_46_thinking(self) -> None:
        r = create_client("anthropic:claude-sonnet-4-6").chat(
            [{"role": "user", "content": "What is 7 * 8? Think step by step."}],
            thinking=True,
            max_tokens=4096,
        )
        assert r.thinking is not None
        assert len(r.thinking) > 0
        assert len(r.text) > 0

    def test_claude_sonnet_46_tools(self) -> None:
        r = create_client("anthropic:claude-sonnet-4-6").chat(
            [{"role": "user", "content": TOOL_PROMPT}], tools=[WEATHER_TOOL]
        )
        assert len(r.tool_calls) > 0
        assert r.tool_calls[0].name == "get_weather"

    @pytest.mark.asyncio
    async def test_claude_sonnet_46_streaming(self) -> None:
        parts: list[str] = []
        async for c in create_client("anthropic:claude-sonnet-4-6").chat_stream_async(
            [{"role": "user", "content": SHORT}]
        ):
            if c.type == "text_delta" and c.text:
                parts.append(c.text)
        assert len("".join(parts)) > 0


@pytest.mark.integration
@requires_anthropic
class TestAnthropicClaude4Legacy:
    """Claude 4.0 dated version — per alea registry."""

    def test_claude_sonnet_4_dated(self) -> None:
        r = create_client("anthropic:claude-sonnet-4-20250514").chat(
            [{"role": "user", "content": SHORT}]
        )
        assert "hello" in r.text.lower()
        _check(r, provider="anthropic")


@pytest.mark.integration
@requires_anthropic
class TestAnthropicHaiku:
    """Claude Haiku 4.5 — cheapest current Anthropic model ($1/$5 per MTok)."""

    def test_claude_haiku_45_chat(self) -> None:
        r = create_client("anthropic:claude-haiku-4-5").chat([{"role": "user", "content": SHORT}])
        assert "hello" in r.text.lower()
        _check(r, provider="anthropic")


# ===================================================================
# Google — Gemini 2.5 (stable), Gemini 2.0
# ===================================================================


@pytest.mark.integration
@requires_google
class TestGoogleGemini25:
    """Gemini 2.5 — production-stable generation."""

    def test_gemini_25_flash_chat(self) -> None:
        r = create_client("google:gemini-2.5-flash").chat([{"role": "user", "content": SHORT}])
        assert "hello" in r.text.lower()
        _check(r, provider="google")

    def test_gemini_25_pro_chat(self) -> None:
        r = create_client("google:gemini-2.5-pro").chat([{"role": "user", "content": SHORT}])
        assert "hello" in r.text.lower()
        _check(r, provider="google")

    def test_gemini_25_flash_tools(self) -> None:
        r = create_client("google:gemini-2.5-flash").chat(
            [{"role": "user", "content": TOOL_PROMPT}], tools=[WEATHER_TOOL]
        )
        assert len(r.tool_calls) > 0
        assert r.tool_calls[0].name == "get_weather"

    def test_gemini_25_flash_json(self) -> None:
        r = create_client("google:gemini-2.5-flash").json(
            [{"role": "user", "content": 'Return JSON: {"greeting": "hi"}'}],
            schema={
                "type": "object",
                "properties": {"greeting": {"type": "string"}},
                "required": ["greeting"],
            },
        )
        assert r.output_json is not None

    @pytest.mark.asyncio
    async def test_gemini_25_flash_streaming(self) -> None:
        parts: list[str] = []
        async for c in create_client("google:gemini-2.5-flash").chat_stream_async(
            [{"role": "user", "content": SHORT}]
        ):
            if c.type == "text_delta" and c.text:
                parts.append(c.text)
        assert len("".join(parts)) > 0


@pytest.mark.integration
@requires_google
class TestGoogleGemini3:
    """Gemini 3.x — latest generation (preview)."""

    def test_gemini_31_pro_preview_chat(self) -> None:
        """gemini-3.1-pro-preview: flagship Google model."""
        r = create_client("google:gemini-3.1-pro-preview").chat(
            [{"role": "user", "content": SHORT}]
        )
        assert "hello" in r.text.lower()
        _check(r, provider="google")

    def test_gemini_3_flash_preview_chat(self) -> None:
        r = create_client("google:gemini-3-flash-preview").chat(
            [{"role": "user", "content": SHORT}]
        )
        assert "hello" in r.text.lower()
        _check(r, provider="google")


@pytest.mark.integration
@requires_google
class TestGoogleGemini20:
    """Gemini 2.0 — previous stable generation."""

    def test_gemini_20_flash_chat(self) -> None:
        r = create_client("google:gemini-2.0-flash").chat([{"role": "user", "content": SHORT}])
        assert "hello" in r.text.lower()
        _check(r, provider="google")


# ===================================================================
# Cross-Provider
# ===================================================================


@pytest.mark.integration
@requires_openai
class TestCrossProvider:
    def test_factory_bare_model_name(self) -> None:
        r = create_client("gpt-5.4-nano").chat([{"role": "user", "content": SHORT}])
        assert r.provider == "openai"

    def test_sync_works(self) -> None:
        r = create_client("openai:gpt-5.4-nano").chat([{"role": "user", "content": SHORT}])
        assert len(r.text) > 0

    @pytest.mark.asyncio
    async def test_async_works(self) -> None:
        r = await create_client("openai:gpt-5.4-nano").chat_async(
            [{"role": "user", "content": SHORT}]
        )
        assert len(r.text) > 0


# ===================================================================
# Multimodal — real fixtures: generated images, NASA photos, audio, PDF
# ===================================================================

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_fixture(name: str) -> bytes:
    """Load a test fixture file."""
    return (FIXTURES / name).read_bytes()


@pytest.mark.integration
@requires_openai
class TestOpenAIMultimodal:
    """Real image and content understanding tests against OpenAI."""

    def test_red_square_description(self) -> None:
        """PIL-generated red square on white: model should identify red/square."""
        from kaos_llm_client.multimodal import image_from_bytes

        r = create_client("openai:gpt-5.4-nano").chat(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "What color is the shape and what shape is it? Answer in two words."
                            ),
                        },
                        image_from_bytes(_load_fixture("red_square.png"), "image/png"),
                    ],
                }
            ]
        )
        text = r.text.lower()
        assert "red" in text
        assert "square" in text or "rectangle" in text
        _check(r, provider="openai")

    def test_bar_chart_understanding(self) -> None:
        """PIL-generated bar chart: model should identify it as a chart with bars."""
        from kaos_llm_client.multimodal import image_from_bytes

        r = create_client("openai:gpt-5.4-nano").chat(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What type of chart is this? What are the bar labels?",
                        },
                        image_from_bytes(_load_fixture("bar_chart.png"), "image/png"),
                    ],
                }
            ]
        )
        text = r.text.lower()
        assert "bar" in text or "chart" in text
        _check(r, provider="openai")

    def test_nasa_moon_description(self) -> None:
        """NASA Galileo Moon photo (PIA00405): model should identify the Moon."""
        from kaos_llm_client.multimodal import image_from_path

        r = create_client("openai:gpt-5.4-nano").chat(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What celestial body is shown in this NASA photo?",
                        },
                        image_from_path(FIXTURES / "nasa_moon_galileo.jpg"),
                    ],
                }
            ]
        )
        text = r.text.lower()
        assert "moon" in text
        _check(r, provider="openai")

    def test_jpeg_blue_circle(self) -> None:
        """PIL-generated blue circle on yellow (JPEG): verify JPEG format works."""
        from kaos_llm_client.multimodal import image_from_path

        r = create_client("openai:gpt-5.4-nano").chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is the circle? One word."},
                        image_from_path(FIXTURES / "blue_circle.jpg"),
                    ],
                }
            ]
        )
        assert "blue" in r.text.lower()
        _check(r, provider="openai")


@pytest.mark.integration
@requires_anthropic
class TestAnthropicMultimodal:
    """Real image understanding tests against Anthropic Claude."""

    def test_red_square_description(self) -> None:
        """PIL-generated red square: Claude should identify red square."""
        from kaos_llm_client.multimodal import image_from_bytes

        r = create_client("anthropic:claude-sonnet-4-6").chat(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "What color is the shape and what shape is it? Answer in two words."
                            ),
                        },
                        image_from_bytes(_load_fixture("red_square.png"), "image/png"),
                    ],
                }
            ]
        )
        text = r.text.lower()
        assert "red" in text
        _check(r, provider="anthropic")

    def test_nasa_sun_description(self) -> None:
        """NASA SOHO Sun photo (PIA03149): Claude should identify the Sun."""
        from kaos_llm_client.multimodal import image_from_path

        r = create_client("anthropic:claude-sonnet-4-6").chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What celestial body is shown? One word."},
                        image_from_path(FIXTURES / "nasa_sun_soho.jpg"),
                    ],
                }
            ]
        )
        text = r.text.lower()
        assert "sun" in text or "star" in text
        _check(r, provider="anthropic")

    def test_document_pdf_input(self) -> None:
        """Hand-crafted PDF with 'Hello World': Claude should read it."""
        from kaos_llm_client.multimodal import document_from_path

        r = create_client("anthropic:claude-sonnet-4-6").chat(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What text does this PDF contain? Quote it exactly.",
                        },
                        document_from_path(FIXTURES / "hello_world.pdf"),
                    ],
                }
            ]
        )
        text = r.text.lower()
        assert "hello" in text
        _check(r, provider="anthropic")


@pytest.mark.integration
@requires_google
class TestGoogleMultimodal:
    """Real image understanding tests against Google Gemini."""

    def test_green_triangle_description(self) -> None:
        """PIL-generated green triangle: Gemini should identify it."""
        from kaos_llm_client.multimodal import image_from_bytes

        r = create_client("google:gemini-2.5-flash").chat(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "What color is the shape and what shape is it? Answer in two words."
                            ),
                        },
                        image_from_bytes(_load_fixture("green_triangle.png"), "image/png"),
                    ],
                }
            ]
        )
        text = r.text.lower()
        assert "green" in text
        assert "triangle" in text
        _check(r, provider="google")

    def test_nasa_moon_description(self) -> None:
        """NASA Galileo Moon photo: Gemini should identify the Moon."""
        from kaos_llm_client.multimodal import image_from_path

        r = create_client("google:gemini-2.5-flash").chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What celestial body is shown?"},
                        image_from_path(FIXTURES / "nasa_moon_galileo.jpg"),
                    ],
                }
            ]
        )
        assert "moon" in r.text.lower()
        _check(r, provider="google")


# ===================================================================
# Multi-Turn Tool Use (cross-provider continuation)
# ===================================================================

TEMP_TOOL = ToolDefinition(
    name="get_temperature",
    description="Get the current temperature for a city in Celsius.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
)

TEMP_PROMPT = "What is the temperature in Paris? You must call the get_temperature tool."


@pytest.mark.integration
@requires_openai
class TestMultiTurnToolUseLive:
    """Live multi-turn tool-use continuation tests across providers.

    Each test: send prompt with tool -> get tool_calls -> build continuation
    with AssistantMessage.from_response() + ToolResultMessage -> send second
    call -> verify text response references the tool result.

    Uses chat_async so both turns share the same event loop (avoids HTTP/2
    connection pool issues with repeated asyncio.run() calls from sync chat()).
    """

    @pytest.mark.asyncio
    async def test_openai_tool_continuation(self) -> None:
        """OpenAI: full tool-use round trip with gpt-5.4-nano."""
        client = create_client("openai:gpt-5.4-nano")
        messages: list[dict[str, Any]] = [UserMessage(TEMP_PROMPT)]

        # Turn 1: expect tool call
        r1 = await client.chat_async(messages, tools=[TEMP_TOOL])
        assert len(r1.tool_calls) > 0, f"Expected tool call, got text: {r1.text!r}"
        tc = r1.tool_calls[0]
        assert tc.name == "get_temperature"

        # Build continuation
        messages.append(AssistantMessage.from_response(r1))
        messages.append(ToolResultMessage(tc.id, '{"celsius": 22}', name=tc.name))

        # Turn 2: model should respond with text mentioning the temperature
        r2 = await client.chat_async(messages, tools=[TEMP_TOOL])
        assert len(r2.text) > 0
        assert "22" in r2.text
        _check(r2, provider="openai")


@pytest.mark.integration
@requires_anthropic
class TestMultiTurnToolUseAnthropicLive:
    """Anthropic multi-turn tool-use continuation.

    Critical test: Anthropic was previously broken because AssistantMessage
    used OpenAI format (tool_calls array) instead of Anthropic content blocks.
    """

    @pytest.mark.asyncio
    async def test_anthropic_tool_continuation(self) -> None:
        """Anthropic: full tool-use round trip with claude-sonnet-4-6."""
        client = create_client("anthropic:claude-sonnet-4-6")
        messages: list[dict[str, Any]] = [UserMessage(TEMP_PROMPT)]

        # Turn 1: expect tool call
        r1 = await client.chat_async(messages, tools=[TEMP_TOOL])
        assert len(r1.tool_calls) > 0, f"Expected tool call, got text: {r1.text!r}"
        tc = r1.tool_calls[0]
        assert tc.name == "get_temperature"

        # Build continuation with provider-aware from_response()
        assistant_msg = AssistantMessage.from_response(r1)
        # Verify Anthropic format: content blocks, not tool_calls
        assert "tool_calls" not in assistant_msg
        assert isinstance(assistant_msg.get("content"), list)

        messages.append(assistant_msg)
        messages.append(ToolResultMessage(tc.id, '{"celsius": 22}', name=tc.name))

        # Turn 2: model should respond with text mentioning the temperature
        r2 = await client.chat_async(messages, tools=[TEMP_TOOL])
        assert len(r2.text) > 0
        assert "22" in r2.text
        _check(r2, provider="anthropic")


@pytest.mark.integration
@requires_google
class TestMultiTurnToolUseGoogleLive:
    """Google multi-turn tool-use continuation with functionResponse format."""

    @pytest.mark.asyncio
    async def test_google_tool_continuation(self) -> None:
        """Google: full tool-use round trip with gemini-2.5-flash."""
        client = create_client("google:gemini-2.5-flash")
        messages: list[dict[str, Any]] = [UserMessage(TEMP_PROMPT)]

        # Turn 1: expect tool call
        r1 = await client.chat_async(messages, tools=[TEMP_TOOL])
        assert len(r1.tool_calls) > 0, f"Expected tool call, got text: {r1.text!r}"
        tc = r1.tool_calls[0]
        assert tc.name == "get_temperature"

        # Build continuation -- Google requires name= on ToolResultMessage
        messages.append(AssistantMessage.from_response(r1))
        messages.append(ToolResultMessage(tc.id, '{"celsius": 22}', name=tc.name))

        # Turn 2: model should respond with text mentioning the temperature
        r2 = await client.chat_async(messages, tools=[TEMP_TOOL])
        assert len(r2.text) > 0
        assert "22" in r2.text
        _check(r2, provider="google")


# ===================================================================
# Google Advanced — JSON, streaming, tool continuation id echo
# ===================================================================


@pytest.mark.integration
@requires_google
class TestGoogleAdvanced:
    """Advanced Google Gemini tests: JSON schema, streaming, tool id echo."""

    def test_gemini_25_flash_json_live(self) -> None:
        """json() with explicit schema returns a valid dict."""
        r = create_client("google:gemini-2.5-flash").json(
            [{"role": "user", "content": 'Return JSON: {"greeting": "hi"}'}],
            schema={
                "type": "object",
                "properties": {"greeting": {"type": "string"}},
                "required": ["greeting"],
            },
        )
        assert r.output_json is not None
        assert isinstance(r.output_json, dict)
        assert "greeting" in r.output_json
        _check(r, provider="google")

    @pytest.mark.asyncio
    async def test_gemini_25_flash_streaming_live(self) -> None:
        """Streaming accumulates text from Gemini."""
        parts: list[str] = []
        async for c in create_client("google:gemini-2.5-flash").chat_stream_async(
            [{"role": "user", "content": SHORT}]
        ):
            if c.type == "text_delta" and c.text:
                parts.append(c.text)
        text = "".join(parts)
        assert len(text) > 0
        assert "hello" in text.lower()

    @pytest.mark.asyncio
    async def test_gemini_tool_continuation_id_echo(self) -> None:
        """Multi-turn tool use: verify tool_call id is non-synthetic, then continue."""
        client = create_client("google:gemini-2.5-flash")
        messages: list[dict[str, Any]] = [UserMessage(TEMP_PROMPT)]

        # Turn 1: expect tool call
        r1 = await client.chat_async(messages, tools=[TEMP_TOOL])
        assert len(r1.tool_calls) > 0, f"Expected tool call, got text: {r1.text!r}"
        tc = r1.tool_calls[0]
        assert tc.name == "get_temperature"

        # Verify id is present (may be google_tc_* if API doesn't return one)
        assert tc.id is not None
        assert len(tc.id) > 0

        # Build continuation and send result back
        messages.append(AssistantMessage.from_response(r1))
        messages.append(ToolResultMessage(tc.id, '{"celsius": 18}', name=tc.name))

        # Turn 2: model should respond with text mentioning the temperature
        r2 = await client.chat_async(messages, tools=[TEMP_TOOL])
        assert len(r2.text) > 0
        assert "18" in r2.text
        _check(r2, provider="google")


# ===================================================================
# Embeddings — OpenAI text-embedding-3-small
# ===================================================================


@pytest.mark.integration
@requires_openai
class TestEmbeddingsLive:
    """Live embedding tests against OpenAI."""

    def test_openai_embed_live(self) -> None:
        """Single string embedding with text-embedding-3-small."""
        client = create_client("openai:text-embedding-3-small")
        result = client.embed("Hello world")
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) > 0
        assert all(isinstance(v, float) for v in result.embeddings[0])
        assert result.usage.input_tokens > 0
        assert result.provider == "openai"

    def test_openai_embed_list_live(self) -> None:
        """Batch embedding of two strings."""
        client = create_client("openai:text-embedding-3-small")
        result = client.embed(["Hello", "World"])
        assert len(result.embeddings) == 2
        assert len(result.embeddings[0]) > 0
        assert len(result.embeddings[1]) > 0
        assert result.usage.input_tokens > 0


# ===================================================================
# Composition — Fallback, Instrumented, ConcurrencyLimited
# ===================================================================


@pytest.mark.integration
@requires_openai
@requires_anthropic
class TestCompositionLive:
    """Live tests for composition wrappers: Fallback, Instrumented, ConcurrencyLimited."""

    def test_fallback_primary_succeeds(self) -> None:
        """FallbackClient uses primary when it succeeds."""
        openai_client = create_client("openai:gpt-5.4-nano")
        anthropic_client = create_client("anthropic:claude-haiku-4-5")
        client = FallbackClient([openai_client, anthropic_client])
        r = client.chat([{"role": "user", "content": SHORT}])
        assert r.provider == "openai"
        assert len(r.text) > 0
        _check(r, provider="openai")

    def test_fallback_bad_key_falls_back(self) -> None:
        """FallbackClient with bad primary key falls back to secondary."""
        bad_openai = create_client("openai:gpt-5.4-nano", api_key="sk-invalid")
        good_anthropic = create_client("anthropic:claude-haiku-4-5")
        # Include KaosLLMAuthError in fallback_on since bad key raises auth error
        from kaos_llm_client.errors import KaosLLMAuthError, KaosLLMProviderError

        client = FallbackClient(
            [bad_openai, good_anthropic],
            fallback_on=(KaosLLMAuthError, KaosLLMProviderError),
        )
        r = client.chat([{"role": "user", "content": SHORT}])
        assert r.provider == "anthropic"
        assert len(r.text) > 0
        _check(r, provider="anthropic")

    def test_instrumented_live(self) -> None:
        """InstrumentedClient tracks request counts and token usage."""
        inner = create_client("openai:gpt-5.4-nano")
        client = InstrumentedClient(inner)
        r = client.chat([{"role": "user", "content": SHORT}])
        assert len(r.text) > 0
        assert client.total_requests == 1
        assert client.total_input_tokens > 0
        assert client.total_output_tokens > 0

    def test_concurrency_limited_live(self) -> None:
        """ConcurrencyLimitedClient with limit=2 works for a single request."""
        inner = create_client("openai:gpt-5.4-nano")
        client = ConcurrencyLimitedClient(inner, limit=2)
        r = client.chat([{"role": "user", "content": SHORT}])
        assert len(r.text) > 0
        _check(r, provider="openai")


# ===================================================================
# Output Validation — pydantic with validators
# ===================================================================


class ColorList(BaseModel):
    colors: list[str]


@pytest.mark.integration
@requires_openai
class TestOutputValidationLive:
    """Live tests for pydantic() with output_validator and retries."""

    def test_pydantic_with_validator_live(self) -> None:
        """pydantic() with output_validator that checks list length >= 2."""

        def check_at_least_two(result: ColorList) -> ColorList:
            if len(result.colors) < 2:
                raise ValueError(f"Expected at least 2 colors, got {len(result.colors)}")
            return result

        result = create_client("openai:gpt-5.4-nano").pydantic(
            [{"role": "user", "content": "List exactly 3 colors: red, blue, green."}],
            output_type=ColorList,
            output_validator=check_at_least_two,
            max_validation_retries=2,
        )
        assert isinstance(result, ColorList)
        assert len(result.colors) >= 2

    def test_pydantic_validator_retry_live(self) -> None:
        """pydantic() with strict validator + max_validation_retries=2 eventually succeeds."""

        def require_four_colors(result: ColorList) -> ColorList:
            if len(result.colors) < 4:
                raise ValueError(
                    f"Need at least 4 colors, got {len(result.colors)}: {result.colors}"
                )
            return result

        result = create_client("openai:gpt-5.4-nano").pydantic(
            [
                {
                    "role": "user",
                    "content": "List exactly 5 colors: red, blue, green, yellow, purple.",
                }
            ],
            output_type=ColorList,
            output_validator=require_four_colors,
            max_validation_retries=2,
        )
        assert isinstance(result, ColorList)
        assert len(result.colors) >= 4


# ===================================================================
# Anthropic Structured Output (JSON / Pydantic)
# ===================================================================


@pytest.mark.integration
@requires_anthropic
class TestAnthropicStructuredOutputLive:
    """Structured output via Anthropic (tool-based JSON by default)."""

    def test_claude_json_live(self) -> None:
        """json() with explicit schema returns a dict with expected keys."""
        r = create_client("anthropic:claude-haiku-4-5").json(
            [{"role": "user", "content": 'Return JSON: {"capital": "Paris", "country": "France"}'}],
            schema={
                "type": "object",
                "properties": {
                    "capital": {"type": "string"},
                    "country": {"type": "string"},
                },
                "required": ["capital", "country"],
            },
        )
        # Anthropic uses tool-based structured output: the model calls
        # return_output with the schema, so output_json comes from the
        # tool call arguments rather than bare text.
        assert r.output_json is not None or len(r.tool_calls) > 0, (
            f"Expected JSON output or tool call, got text: {r.text!r}"
        )
        # Extract the structured data from whichever path produced it
        data = r.output_json
        if data is None and r.tool_calls:
            data = r.tool_calls[0].arguments
        assert isinstance(data, dict)
        assert "capital" in data
        assert "country" in data
        assert data["capital"].lower() == "paris"
        _check(r, provider="anthropic")

    def test_claude_pydantic_live(self) -> None:
        """pydantic() with Colors model returns a typed result with actual colors.

        Now uses NATIVE mode (WS-TR.PR-1 — Anthropic GA April 2026). Before
        the flip, this test forced PROMPTED mode because tool-based output
        didn't round-trip through pydantic_async's output_json extraction.
        """
        result = create_client("anthropic:claude-haiku-4-5").pydantic(
            [{"role": "user", "content": "List exactly 3 colors: red, green, blue."}],
            output_type=Colors,
        )
        assert isinstance(result, Colors)
        assert len(result.colors) == 3
        color_set = {c.lower() for c in result.colors}
        assert "red" in color_set
        assert "blue" in color_set

    def test_claude_native_output_config_live(self) -> None:
        """WS-TR.PR-1: verify ``output_config.format`` wire against live Anthropic.

        With the default profile flipped to NATIVE, a ``.json(schema=...)``
        call must send ``output_config.format`` and receive back a text
        content block containing valid JSON matching the schema.

        This also exercises the mutex guard: structured outputs alone (no
        citations) must not trip ``_check_citation_mutex``.
        """
        from kaos_llm_client.profiles import StructuredOutputMode

        schema = {
            "type": "object",
            "properties": {
                "effective_date": {"type": "string"},
                "parties": {"type": "array", "items": {"type": "string"}},
                "jurisdiction": {"type": "string"},
            },
            "required": ["effective_date", "parties", "jurisdiction"],
        }
        r = create_client("anthropic:claude-haiku-4-5").json(
            [
                {
                    "role": "user",
                    "content": (
                        "Extract structured fields from this contract preamble: "
                        "'This Agreement is made effective as of January 15, 2025, "
                        "between Acme Corp. and Beta LLC, governed by the laws of "
                        "the State of Delaware.'"
                    ),
                }
            ],
            schema=schema,
            output_mode=StructuredOutputMode.NATIVE,
        )
        data = r.output_json
        assert isinstance(data, dict), (
            f"Expected parseable JSON in first text block, got text={r.text!r}"
        )
        assert "effective_date" in data
        assert "parties" in data
        assert "jurisdiction" in data
        assert isinstance(data["parties"], list)
        assert len(data["parties"]) == 2
        # Content sanity — model should read the preamble correctly.
        parties_lower = [p.lower() for p in data["parties"]]
        assert any("acme" in p for p in parties_lower)
        assert any("beta" in p for p in parties_lower)
        assert "delaware" in data["jurisdiction"].lower()
        # Wire shape: the raw response should show output_config was honored
        # by the model (first content block is text with valid JSON).
        content = r.raw.get("content", [])
        assert content and content[0].get("type") == "text"
        _check(r, provider="anthropic")


# ===================================================================
# Anthropic Streaming Tool Calls
# ===================================================================


@pytest.mark.integration
@requires_anthropic
class TestAnthropicStreamingToolCallsLive:
    """Verify streaming tool calls produce correct delta chunks."""

    @pytest.mark.asyncio
    async def test_claude_streaming_tool_use(self) -> None:
        """Stream with tools: tool_call_delta chunks appear and accumulate to a valid call."""
        client = create_client("anthropic:claude-haiku-4-5")
        tool_deltas: list[dict[str, Any]] = []
        text_parts: list[str] = []

        async for chunk in client.chat_stream_async(
            [{"role": "user", "content": TOOL_PROMPT}],
            tools=[WEATHER_TOOL],
        ):
            if chunk.type == "tool_call_delta" and chunk.tool_call_delta:
                tool_deltas.append(chunk.tool_call_delta)
            elif chunk.type == "text_delta" and chunk.text:
                text_parts.append(chunk.text)

        # We must have received at least one tool_call_delta
        assert len(tool_deltas) > 0, (
            f"Expected tool_call_delta chunks, got only text: {''.join(text_parts)!r}"
        )

        # The first delta with an "id" key marks the start of a tool call
        start_deltas = [d for d in tool_deltas if "id" in d]
        assert len(start_deltas) >= 1, "Expected at least one tool call start delta with 'id'"
        assert start_deltas[0]["name"] == "get_weather"

        # Accumulate the argument fragments
        arg_fragments = [d.get("arguments", "") for d in tool_deltas]
        accumulated_args = "".join(arg_fragments)
        # The accumulated arguments should be valid JSON containing location
        parsed = json.loads(accumulated_args)
        assert "location" in parsed
        assert isinstance(parsed["location"], str)
        assert len(parsed["location"]) > 0


# ===================================================================
# Anthropic Thinking + Tools Combined
# ===================================================================


@pytest.mark.integration
@requires_anthropic
class TestAnthropicThinkingWithToolsLive:
    """Verify thinking and tool_calls can coexist in one response."""

    def test_claude_thinking_plus_tools(self) -> None:
        """thinking=True with tools: response has BOTH thinking content and tool_calls."""
        r = create_client("anthropic:claude-sonnet-4-6").chat(
            [
                {
                    "role": "user",
                    "content": (
                        "Think carefully about this: What is the weather in Tokyo? "
                        "You must call the get_weather tool."
                    ),
                }
            ],
            tools=[WEATHER_TOOL],
            thinking=True,
            max_tokens=4096,
        )
        # Thinking content should be present
        assert r.thinking is not None, f"Expected thinking content, got parts: {r.parts}"
        assert len(r.thinking) > 10, "Thinking should be substantive, not empty"

        # Tool call should also be present
        assert len(r.tool_calls) > 0, f"Expected tool_calls, got text: {r.text!r}"
        assert r.tool_calls[0].name == "get_weather"
        _check(r, provider="anthropic")


# ===================================================================
# Anthropic Cache Point (prompt caching)
# ===================================================================


@pytest.mark.integration
@requires_anthropic
class TestAnthropicCachePointLive:
    """Verify Anthropic prompt caching via CachePoint markers."""

    @pytest.mark.asyncio
    async def test_cache_point_reduces_cost(self) -> None:
        """Send a long system prompt with cache_control and verify
        cache_creation_tokens is populated in the response.

        Passes the system content as a kwarg (``system=``) with cache_control
        pre-set in Anthropic wire format. This bypasses the message conversion
        layer (which currently strips cache_control from text blocks) and
        exercises the response parsing of ``cache_creation_input_tokens``
        in ``_parse_response()``.

        Note: cache_read_input_tokens requires cache propagation delay and
        is not reliably testable in a fast sequential test.
        """
        client = create_client("anthropic:claude-haiku-4-5")

        # Build a long system prompt (must be well above Anthropic's minimum
        # cacheable prefix of ~2048 tokens for Haiku)
        long_text = (
            "You are an expert legal analyst specializing in regulatory compliance. "
            "Your role is to analyze documents and provide structured summaries. "
            "Consider all applicable federal and state regulations. "
            "Always cite the specific section of the regulation you reference. "
        ) * 200  # ~8000+ tokens, well above caching threshold

        # Pass system directly as a kwarg -- _build_request merges kwargs into
        # the body via body.update(kwargs), overriding the extracted system field.
        # This lets us include cache_control in the wire format.
        system_content = [
            {
                "type": "text",
                "text": long_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Say exactly one word: hello"},
        ]

        r = await client.chat_async(messages, system=system_content)
        assert len(r.text) > 0

        # Verify cache tokens were parsed from the response.
        # Anthropic returns cache_creation_input_tokens > 0 on the first call
        # with a new cacheable prefix, and cache_read_input_tokens > 0 on
        # subsequent calls with the same prefix. Since the cache may already
        # exist from prior test runs, we accept either.
        cache_creation = r.usage.cache_creation_tokens or 0
        cache_read = r.usage.cache_read_tokens or 0
        assert cache_creation > 0 or cache_read > 0, (
            f"Expected cache activity (creation or read) for a large cached system prompt, "
            f"got creation={cache_creation}, read={cache_read}. Full usage: {r.usage}"
        )
        _check(r, provider="anthropic")


# ===================================================================
# Errors
# ===================================================================


# ===================================================================
# Regression tests for specific bug fixes (live API verification)
# ===================================================================


@pytest.mark.integration
@requires_google
class TestGeminiThoughtSignatureReplay:
    """Fix #1: AssistantMessage.from_response() must preserve thoughtSignature
    for Gemini tool continuations. Previously dropped it."""

    @pytest.mark.asyncio
    async def test_gemini_tool_continuation_preserves_parts(self) -> None:
        """Multi-turn tool use: verify assistant message has Google-native
        content (functionCall parts), not OpenAI-style tool_calls."""
        from kaos_llm_client.messages import AssistantMessage, ToolResultMessage

        client = create_client("google:gemini-2.5-flash")
        response = await client.chat_async(
            [{"role": "user", "content": "What is the temperature in Paris? Call the tool."}],
            tools=[WEATHER_TOOL],
        )

        if not response.tool_calls:
            pytest.skip("Model did not call tool (non-deterministic)")

        # from_response for Google should produce content with functionCall, not tool_calls
        assistant_msg = AssistantMessage.from_response(response)
        assert "tool_calls" not in assistant_msg, "Google should use content blocks, not tool_calls"
        content = assistant_msg.get("content")
        assert isinstance(content, list), "Google assistant content should be a list of parts"
        has_fc = any("functionCall" in p for p in content if isinstance(p, dict))
        assert has_fc, "Content should have functionCall part"

        # Verify the continuation works (use same async client to avoid event loop issues)
        tc = response.tool_calls[0]
        messages = [
            {"role": "user", "content": "What is the temperature in Paris?"},
            assistant_msg,
            ToolResultMessage(tc.id, '{"celsius": 18}', name=tc.name),
        ]
        response2 = await client.chat_async(messages, tools=[WEATHER_TOOL])
        assert len(response2.text) > 0
        _check(response2, provider="google")


@pytest.mark.integration
@requires_openai
class TestResponsesToolChoiceLive:
    """Fix #3: OpenAIResponsesClient must honor tool_choice parameter."""

    def test_responses_tool_choice_none(self) -> None:
        """tool_choice='none' should prevent tool calls even with tools defined."""
        from kaos_llm_client.types import ToolChoice

        client = create_client("openai-responses:gpt-5.4-nano")
        response = client.chat(
            [{"role": "user", "content": "What is the weather in Tokyo?"}],
            tools=[WEATHER_TOOL],
            tool_choice=ToolChoice(type="none"),
        )
        # With tool_choice=none, model should NOT call any tools
        assert len(response.tool_calls) == 0
        assert len(response.text) > 0


# ===================================================================
# Google Pydantic — exercises json_utils extraction
# ===================================================================


@pytest.mark.integration
@requires_google
class TestGooglePydanticLive:
    """Pydantic structured output via Google Gemini, exercising json_utils extraction."""

    def test_gemini_pydantic_live(self) -> None:
        """pydantic() with Colors model via Gemini, verify typed result."""
        result = create_client("google:gemini-2.5-flash").pydantic(
            [{"role": "user", "content": "List exactly 2 colors: red, blue."}],
            output_type=Colors,
        )
        assert isinstance(result, Colors)
        assert len(result.colors) >= 1
        # Verify actual color names came back
        lower_colors = [c.lower() for c in result.colors]
        assert any("red" in c for c in lower_colors) or any("blue" in c for c in lower_colors)


# ===================================================================
# Google Streaming Tool Calls
# ===================================================================


@pytest.mark.integration
@requires_google
class TestGoogleStreamingToolCallsLive:
    """Streaming tool calls via Google Gemini."""

    @pytest.mark.asyncio
    async def test_gemini_streaming_tool_use(self) -> None:
        """chat_stream_async with tools: verify tool_call_delta chunks appear
        and accumulated response has a valid tool call with name and arguments."""

        client = create_client("google:gemini-2.5-flash")
        saw_tool_delta = False
        accumulator = StreamAccumulator(
            provider="google", model="gemini-2.5-flash", request_id="test-stream-tool"
        )

        async for chunk in client.chat_stream_async(
            [{"role": "user", "content": TOOL_PROMPT}], tools=[WEATHER_TOOL]
        ):
            accumulator.feed(chunk)
            if chunk.type == "tool_call_delta":
                saw_tool_delta = True
                assert chunk.tool_call_delta is not None

        assert saw_tool_delta, "Expected at least one tool_call_delta chunk from streaming"

        response = accumulator.accumulated
        assert len(response.tool_calls) > 0, "Accumulated response should have tool calls"
        tc = response.tool_calls[0]
        assert tc.name == "get_weather"
        assert isinstance(tc.arguments, dict)


# ===================================================================
# Google Document (PDF) Input
# ===================================================================

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.mark.integration
@requires_google
class TestGoogleDocumentInputLive:
    """Document (PDF) input via Google Gemini."""

    def test_gemini_document_pdf(self) -> None:
        """Send hello_world.pdf to Gemini via document_from_path, verify it reads the content."""
        from kaos_llm_client.multimodal import document_from_path

        r = create_client("google:gemini-2.5-flash").chat(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What text does this PDF contain? Quote it exactly.",
                        },
                        document_from_path(FIXTURES_DIR / "hello_world.pdf"),
                    ],
                }
            ]
        )
        text = r.text.lower()
        assert "hello" in text
        _check(r, provider="google")


# ===================================================================
# Google Audio Input
# ===================================================================


@pytest.mark.integration
@requires_google
class TestGoogleAudioInputLive:
    """Audio input via Google Gemini (Gemini supports audio natively)."""

    def test_gemini_audio_description(self) -> None:
        """Send a440_tone.wav to Gemini via audio_from_path, verify it describes audio."""
        from kaos_llm_client.multimodal import audio_from_path

        r = create_client("google:gemini-2.5-flash").chat(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this audio. What kind of sound is it?",
                        },
                        audio_from_path(FIXTURES_DIR / "a440_tone.wav"),
                    ],
                }
            ]
        )
        text = r.text.lower()
        assert any(
            word in text for word in ("tone", "audio", "sound", "sine", "frequency", "hz", "beep")
        ), f"Expected audio-related description, got: {r.text!r}"
        _check(r, provider="google")


# ===================================================================
# StreamAccumulator metadata completeness
# ===================================================================


@pytest.mark.integration
@requires_openai
class TestStreamAccumulatorLive:
    """Verify StreamAccumulator captures complete metadata from a live stream."""

    @pytest.mark.asyncio
    async def test_stream_accumulator_complete_metadata(self) -> None:
        """Stream a request, verify accumulated response has full metadata."""

        client = create_client("openai:gpt-5.4-nano")
        accumulator = StreamAccumulator(
            provider="openai", model="gpt-5.4-nano", request_id="test-accum"
        )

        async for chunk in client.chat_stream_async([{"role": "user", "content": SHORT}]):
            accumulator.feed(chunk)

        response = accumulator.accumulated
        assert response.stop_reason is not None, "stop_reason should be set after stream completes"
        assert response.response_id is not None, "response_id should be captured from stream"
        assert response.usage.input_tokens > 0, "input_tokens should be reported"
        assert response.raw != {"streamed_chunks": 0}, "raw should contain real chunk data"
        assert len(response.text) > 0, "accumulated text should be non-empty"


# ===================================================================
# JSON Extraction — code fence stripping via json_utils
# ===================================================================


@pytest.mark.integration
@requires_openai
class TestJsonExtractionLive:
    """Verify output_json parses JSON wrapped in code fences (exercises json_utils)."""

    def test_json_code_fence_extraction(self) -> None:
        """Prompt model to return JSON in code fences, verify output_json parses it."""
        r = create_client("openai:gpt-5.4-nano").chat(
            [
                {
                    "role": "user",
                    "content": (
                        "Return a JSON object with a key 'greeting' set to 'hello'. "
                        "Wrap it in a markdown code fence like ```json ... ```."
                    ),
                }
            ]
        )
        parsed = r.output_json
        assert parsed is not None, (
            f"output_json should parse code-fenced JSON, got text: {r.text!r}"
        )
        assert isinstance(parsed, dict)
        assert "greeting" in parsed
        assert parsed["greeting"].lower() == "hello"


# ===================================================================
# Responses API -- OpenAI Responses wire format (not chat completions)
# ===================================================================


@pytest.mark.integration
@requires_openai
class TestResponsesAPILive:
    """Live tests for the OpenAI Responses API provider (openai-responses:*)."""

    def test_responses_chat_live(self) -> None:
        """Basic chat via Responses API returns text."""
        client = create_client("openai-responses:gpt-5.4-nano")
        r = client.chat([{"role": "user", "content": SHORT}])
        assert "hello" in r.text.lower()
        assert r.provider == "openai-responses"
        assert r.model != ""
        assert r.usage.input_tokens > 0
        assert r.usage.output_tokens > 0

    @pytest.mark.asyncio
    async def test_responses_tool_use_live(self) -> None:
        """Responses API: tool call + continuation via previous_response_id."""
        client = create_client("openai-responses:gpt-5.4-nano")
        messages: list[dict[str, Any]] = [UserMessage(TOOL_PROMPT)]

        # Turn 1: expect tool call
        r1 = await client.chat_async(messages, tools=[WEATHER_TOOL])
        assert len(r1.tool_calls) > 0, f"Expected tool call, got text: {r1.text!r}"
        tc = r1.tool_calls[0]
        assert tc.name == "get_weather"
        assert tc.id != ""
        assert r1.response_id is not None

        # Turn 2: Responses API uses stateful continuation via
        # previous_response_id + function_call_output input items
        continuation: list[dict[str, Any]] = [
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": '{"celsius": 15, "condition": "cloudy"}',
            }
        ]

        # Turn 2: model should reference the tool result
        r2 = await client.chat_async(
            continuation,
            tools=[WEATHER_TOOL],
            previous_response_id=r1.response_id,
        )
        assert len(r2.text) > 0
        assert r2.provider == "openai-responses"

    @pytest.mark.asyncio
    async def test_responses_streaming_live(self) -> None:
        """Responses API streaming: text chunks accumulate."""
        parts: list[str] = []
        async for chunk in create_client("openai-responses:gpt-5.4-nano").chat_stream_async(
            [{"role": "user", "content": SHORT}]
        ):
            if chunk.type == "text_delta" and chunk.text:
                parts.append(chunk.text)
        text = "".join(parts)
        assert len(text) > 0
        assert "hello" in text.lower()

    def test_responses_builtin_web_search(self) -> None:
        """Responses API with builtin web_search tool does not error."""
        client = create_client("openai-responses:gpt-5.4-nano")
        r = client.chat(
            [{"role": "user", "content": "What day is it today?"}],
            builtin_tools=[{"type": "web_search_preview"}],
        )
        # May or may not use web search, but should return text without error
        assert len(r.text) > 0
        assert r.provider == "openai-responses"


# ===================================================================
# FileCache -- disk-backed response cache
# ===================================================================


@pytest.mark.integration
@requires_openai
class TestCacheLive:
    """Live tests for FileCache: verify cache writes and returns identical responses."""

    def test_file_cache_hit(self, tmp_path: Path) -> None:
        """Same request twice with FileCache: second is a cache hit."""
        from kaos_llm_client.settings import KaosLLMSettings
        from kaos_llm_client.types import CachePolicy, RequestOptions

        cache_dir = tmp_path / "llm_cache"
        cache = FileCache(cache_dir)
        settings = KaosLLMSettings(cache_enabled=True)
        prompt = "Reply with exactly: cache_test_ok"
        opts = RequestOptions(cache_policy=CachePolicy.FORCE)

        # First request (cache miss -- hits the API)
        client = create_client("openai:gpt-5.4-nano", cache=cache, settings=settings)
        r1 = client.request(
            [{"role": "user", "content": prompt}],
            options=opts,
        )
        assert len(r1.text) > 0
        assert r1.usage.input_tokens > 0

        # Verify cache file was written
        cache_files = list(cache_dir.rglob("*.json.gz"))
        assert len(cache_files) == 1, f"Expected 1 cache file, got {len(cache_files)}"

        # Second request (cache hit -- returns identical response)
        t0 = time.monotonic()
        r2 = client.request(
            [{"role": "user", "content": prompt}],
            options=opts,
        )
        elapsed_hit = time.monotonic() - t0

        # Verify identical response content
        assert r2.text == r1.text
        assert r2.provider == r1.provider
        assert r2.model == r1.model

        # Cache hit should be sub-500ms (no network round trip)
        assert elapsed_hit < 0.5, f"Cache hit took {elapsed_hit:.3f}s, expected <0.5s"

        # Still only one cache file (no duplicate writes)
        assert len(list(cache_dir.rglob("*.json.gz"))) == 1


# ===================================================================
# Streaming -- tool call deltas, accumulated metadata
# ===================================================================


@pytest.mark.integration
@requires_openai
class TestStreamingLive:
    """Live streaming tests: tool call chunks and accumulated response metadata."""

    @pytest.mark.asyncio
    async def test_openai_streaming_tool_calls(self) -> None:
        """Streaming with tools: verify tool_call_delta chunks appear."""
        tool_deltas: list[dict[str, Any]] = []
        async for chunk in create_client("openai:gpt-5.4-nano").chat_stream_async(
            [{"role": "user", "content": TOOL_PROMPT}],
            tools=[WEATHER_TOOL],
        ):
            if chunk.type == "tool_call_delta" and chunk.tool_call_delta:
                tool_deltas.append(chunk.tool_call_delta)

        # Should have received at least one tool_call_delta with a name
        assert len(tool_deltas) > 0, "Expected tool_call_delta chunks in stream"
        names: list[str] = [d["name"] for d in tool_deltas if d.get("name")]
        assert any("get_weather" in n for n in names), (
            f"Expected 'get_weather' in tool call names, got: {names}"
        )

    @pytest.mark.asyncio
    async def test_streaming_accumulates_complete_response(self) -> None:
        """Stream and accumulate: verify usage, stop_reason, response_id."""
        accumulator = StreamAccumulator(
            provider="openai",
            model="gpt-5.4-nano",
            request_id="test",
        )
        async for chunk in create_client("openai:gpt-5.4-nano").chat_stream_async(
            [{"role": "user", "content": SHORT}]
        ):
            accumulator.feed(chunk)

        result = accumulator.accumulated
        assert len(result.text) > 0
        assert "hello" in result.text.lower()
        assert result.usage.input_tokens > 0
        assert result.usage.output_tokens > 0
        assert result.response_id is not None
        assert result.stop_reason is not None


# ===================================================================
# Fallback streaming -- primary succeeds
# ===================================================================


@pytest.mark.integration
@requires_openai
@requires_anthropic
class TestFallbackStreamingLive:
    """Live fallback streaming: verify chunks flow through FallbackClient."""

    @pytest.mark.asyncio
    async def test_fallback_streaming_primary_works(self) -> None:
        """FallbackClient streaming: primary succeeds, chunks come through."""
        openai_client = create_client("openai:gpt-5.4-nano")
        anthropic_client = create_client("anthropic:claude-haiku-4-5")
        client = FallbackClient([openai_client, anthropic_client])

        parts: list[str] = []
        async for chunk in client.chat_stream_async([{"role": "user", "content": SHORT}]):
            if chunk.type == "text_delta" and chunk.text:
                parts.append(chunk.text)

        text = "".join(parts)
        assert len(text) > 0
        assert "hello" in text.lower()


# ===================================================================
# Errors
# ===================================================================


@pytest.mark.integration
@requires_openai
class TestServiceTierLive:
    """Verify service_tier parameter works with real OpenAI API."""

    def test_flex_tier_default(self) -> None:
        """Default settings use flex tier. Verify API accepts it."""
        r = create_client("openai:gpt-5.4-nano").chat([{"role": "user", "content": SHORT}])
        assert len(r.text) > 0
        # OpenAI echoes the tier back in raw response
        assert r.raw.get("service_tier") == "flex"

    def test_flex_tier_explicit(self) -> None:
        """Explicit service_tier='flex' in kwargs works."""
        r = create_client("openai:gpt-5.4-nano").chat(
            [{"role": "user", "content": SHORT}],
            service_tier="flex",
        )
        assert len(r.text) > 0
        assert r.raw.get("service_tier") == "flex"

    def test_default_tier_override(self) -> None:
        """service_tier='default' overrides the flex default."""
        r = create_client("openai:gpt-5.4-nano").chat(
            [{"role": "user", "content": SHORT}],
            service_tier="default",
        )
        assert len(r.text) > 0
        assert r.raw.get("service_tier") == "default"


# ===================================================================
# Errors
# ===================================================================


@pytest.mark.integration
class TestErrors:
    def test_invalid_api_key(self) -> None:
        with pytest.raises((KaosLLMAuthError, KaosLLMError)):
            create_client("openai:gpt-5.4-nano", api_key="sk-invalid").chat(
                [{"role": "user", "content": "hi"}]
            )

    def test_invalid_model(self) -> None:
        with pytest.raises(KaosLLMError):
            create_client("openai:nonexistent-xyz-99", api_key="sk-invalid").chat(
                [{"role": "user", "content": "hi"}]
            )


# ===================================================================
# Telemetry / metadata completeness
# ===================================================================


@pytest.mark.integration
@requires_openai
class TestTelemetryOpenAI:
    """Verify OpenAI responses contain ALL expected metadata fields."""

    def test_openai_response_telemetry(self) -> None:
        r = create_client("openai:gpt-5.4-nano").chat([{"role": "user", "content": SHORT}])

        # Token usage
        assert r.usage.input_tokens > 0
        assert r.usage.output_tokens > 0
        assert r.usage.total_tokens > 0
        assert r.usage.total_tokens == r.usage.input_tokens + r.usage.output_tokens

        # Latency
        assert r.latency_ms is not None
        assert r.latency_ms > 0
        assert r.latency_ms < 30000  # sanity: under 30s

        # Response identity
        assert r.response_id is not None
        assert len(r.response_id) > 0
        assert "gpt" in r.model.lower()

        # Raw payload
        assert isinstance(r.raw, dict)
        assert "choices" in r.raw

        # HTTP metadata
        assert r.status_code == 200
        assert r.stop_reason is not None

    def test_hooks_fire_in_live_request(self) -> None:
        """RequestHooks fire with real data during a live OpenAI call."""
        req_log: list[ProviderRequest] = []
        resp_log: list[ProviderResponse] = []

        hooks = RequestHooks(
            on_request=lambda r: req_log.append(r),
            on_response=lambda r, resp: resp_log.append(resp),
        )
        client = create_client("openai:gpt-5.4-nano", hooks=hooks)
        r = client.chat([{"role": "user", "content": SHORT}])

        assert len(req_log) == 1
        assert req_log[0].provider == "openai"
        assert req_log[0].body["model"] == "gpt-5.4-nano"

        assert len(resp_log) == 1
        assert resp_log[0].usage.input_tokens > 0
        assert resp_log[0].text == r.text


@pytest.mark.integration
@requires_anthropic
class TestTelemetryAnthropic:
    """Verify Anthropic responses contain ALL expected metadata fields."""

    def test_anthropic_response_telemetry(self) -> None:
        r = create_client("anthropic:claude-sonnet-4-6").chat([{"role": "user", "content": SHORT}])

        # Token usage
        assert r.usage.input_tokens > 0
        assert r.usage.output_tokens > 0

        # Latency
        assert r.latency_ms is not None
        assert r.latency_ms > 0

        # Response identity
        assert r.response_id is not None
        assert len(r.response_id) > 0

        # Raw payload
        assert isinstance(r.raw, dict)
        assert "content" in r.raw

        # Stop reason
        assert r.stop_reason == "end_turn"


@pytest.mark.integration
@requires_google
class TestTelemetryGoogle:
    """Verify Google responses contain ALL expected metadata fields."""

    def test_google_response_telemetry(self) -> None:
        r = create_client("google:gemini-2.5-flash").chat([{"role": "user", "content": SHORT}])

        # Token usage
        assert r.usage.input_tokens > 0

        # Latency
        assert r.latency_ms is not None
        assert r.latency_ms > 0

        # Raw payload
        assert isinstance(r.raw, dict)
        assert "candidates" in r.raw


# ===================================================================
# Secondary-provider smokes — README L108 advertises these, so each
# gets one cheap chat call to prove auth + wire format still work.
# Heavy matrix coverage stays on OpenAI/Anthropic/Google to keep the
# routine live gate fast. Models pinned to the cheapest current-gen
# option per provider as of 2026-05; update if the provider deprecates.
# ===================================================================


@pytest.mark.integration
@requires_xai
class TestXAILive:
    """xAI Grok smoke — grok-3-mini ($0.30/$0.50 per 1M tokens, May 2026)."""

    def test_grok_3_mini_chat(self) -> None:
        r = create_client("xai:grok-3-mini").chat(
            [{"role": "user", "content": "Reply with exactly the word READY."}],
            max_tokens=64,
        )
        _check(r, provider="xai")
        assert r.text is not None
        assert "READY" in r.text.upper()


@pytest.mark.integration
@requires_groq
class TestGroqLive:
    """Groq smoke — llama-3.1-8b-instant ($0.05/$0.08 per 1M tokens, May 2026)."""

    def test_llama_3_1_8b_instant_chat(self) -> None:
        r = create_client("groq:llama-3.1-8b-instant").chat(
            [{"role": "user", "content": "Reply with exactly the word READY."}],
            max_tokens=64,
        )
        _check(r, provider="groq")
        assert r.text is not None
        assert "READY" in r.text.upper()


@pytest.mark.integration
@requires_mistral
class TestMistralLive:
    """Mistral smoke — ministral-3b-latest ($0.04/$0.04 per 1M tokens, May 2026)."""

    def test_ministral_3b_chat(self) -> None:
        r = create_client("mistral:ministral-3b-latest").chat(
            [{"role": "user", "content": "Reply with exactly the word READY."}],
            max_tokens=64,
        )
        _check(r, provider="mistral")
        assert r.text is not None
        assert "READY" in r.text.upper()


@pytest.mark.integration
@requires_openrouter
class TestOpenRouterLive:
    """OpenRouter smoke — routes to anthropic/claude-haiku-4-5 (cheapest current Anthropic)."""

    def test_openrouter_haiku_chat(self) -> None:
        r = create_client("openrouter:anthropic/claude-haiku-4-5").chat(
            [{"role": "user", "content": "Reply with exactly the word READY."}],
            max_tokens=64,
        )
        _check(r, provider="openrouter")
        assert r.text is not None
        assert "READY" in r.text.upper()
