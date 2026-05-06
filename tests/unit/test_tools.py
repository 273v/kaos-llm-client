"""Unit tests for kaos-llm-client MCP tools.

Tests tool metadata validity, local-only tools (provider-check, cost-estimate),
and error handling. Does NOT test actual API calls -- those are integration tests.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any, ClassVar, cast

if TYPE_CHECKING:
    from kaos_core.registry.container import KaosRuntime
from unittest.mock import patch

import pytest

from kaos_llm_client.tools import (
    KaosLLMChatTool,
    KaosLLMCostEstimateTool,
    KaosLLMEmbedTool,
    KaosLLMProviderCheckTool,
    KaosLLMPydanticTool,
    KaosLLMStructuredOutputTool,
    KaosLLMToolCallTool,
    _estimate_tokens,
    _lookup_pricing,
    register_llm_tools,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Pattern from kaos-core: tool names must be kaos-{module}-{action}
_TOOL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+){2,}$")


def _run(coro: Any) -> Any:
    """Run an async coroutine synchronously.

    Uses ``asyncio.run`` rather than
    ``asyncio.get_event_loop().run_until_complete`` because the latter
    only works when there is a current event loop on the main thread.
    Under Python 3.12+, ``get_event_loop()`` raises ``RuntimeError`` if
    no current loop exists, and pytest-asyncio (auto mode) closes the
    loop after each ``async def test_*`` runs — so any sync test in
    this file that runs AFTER an async test elsewhere in the suite
    would trip on the missing loop. ``asyncio.run`` creates a fresh
    loop, runs the coroutine, and tears it down, working in any
    pytest-asyncio test ordering.
    """
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Metadata validity tests
# ---------------------------------------------------------------------------


class TestToolMetadata:
    """Test metadata for all 7 tools (3 existing + 4 new)."""

    ALL_TOOLS: ClassVar[list[Any]] = [
        KaosLLMChatTool(),
        KaosLLMStructuredOutputTool(),
        KaosLLMEmbedTool(),
        KaosLLMToolCallTool(),
        KaosLLMPydanticTool(),
        KaosLLMProviderCheckTool(),
        KaosLLMCostEstimateTool(),
    ]

    @pytest.mark.parametrize(
        "tool",
        ALL_TOOLS,
        ids=lambda t: t.metadata.name,
    )
    def test_tool_name_format(self, tool: Any) -> None:
        """Tool names must match kaos-{module}-{action} pattern."""
        assert _TOOL_NAME_PATTERN.match(tool.metadata.name), (
            f"Tool name '{tool.metadata.name}' does not match required pattern"
        )

    @pytest.mark.parametrize(
        "tool",
        ALL_TOOLS,
        ids=lambda t: t.metadata.name,
    )
    def test_annotations_not_none(self, tool: Any) -> None:
        """Annotations must be set on every tool (never None)."""
        assert tool.metadata.annotations is not None

    @pytest.mark.parametrize(
        "tool",
        ALL_TOOLS,
        ids=lambda t: t.metadata.name,
    )
    def test_module_and_version(self, tool: Any) -> None:
        """Module name and version must be set."""
        assert tool.metadata.module_name == "kaos-llm"
        assert tool.metadata.version == "0.1.0"

    @pytest.mark.parametrize(
        "tool",
        ALL_TOOLS,
        ids=lambda t: t.metadata.name,
    )
    def test_has_description(self, tool: Any) -> None:
        """Tools must have non-empty descriptions."""
        assert tool.metadata.description
        assert len(tool.metadata.description) > 20

    @pytest.mark.parametrize(
        "tool",
        ALL_TOOLS,
        ids=lambda t: t.metadata.name,
    )
    def test_read_only_hint(self, tool: Any) -> None:
        """All LLM tools should be read-only."""
        assert tool.metadata.annotations.readOnlyHint is True

    def test_new_tool_names(self) -> None:
        """Verify the 4 new tools have correct names."""
        new_tools = [
            KaosLLMToolCallTool(),
            KaosLLMPydanticTool(),
            KaosLLMProviderCheckTool(),
            KaosLLMCostEstimateTool(),
        ]
        expected_names = {
            "kaos-llm-tools",
            "kaos-llm-pydantic",
            "kaos-llm-provider-check",
            "kaos-llm-cost-estimate",
        }
        actual_names = {t.metadata.name for t in new_tools}
        assert actual_names == expected_names

    def test_open_world_hints(self) -> None:
        """API-calling tools = openWorld=True; local-only = False."""
        # ToolMetadata.annotations is typed Optional; assert non-None per
        # the platform rule that every tool MUST set ToolAnnotations.
        tool_call_ann = KaosLLMToolCallTool().metadata.annotations
        pydantic_ann = KaosLLMPydanticTool().metadata.annotations
        provider_check_ann = KaosLLMProviderCheckTool().metadata.annotations
        cost_estimate_ann = KaosLLMCostEstimateTool().metadata.annotations
        assert tool_call_ann is not None
        assert pydantic_ann is not None
        assert provider_check_ann is not None
        assert cost_estimate_ann is not None

        # API-calling tools
        assert tool_call_ann.openWorldHint is True
        assert pydantic_ann.openWorldHint is True

        # Local-only tools
        assert provider_check_ann.openWorldHint is False
        assert cost_estimate_ann.openWorldHint is False


# ---------------------------------------------------------------------------
# ToolCallTool tests
# ---------------------------------------------------------------------------


class TestToolCallTool:
    """Test KaosLLMToolCallTool error handling."""

    def test_missing_model(self) -> None:
        tool = KaosLLMToolCallTool()
        result = _run(tool.execute({}))
        assert result.isError
        assert "model" in result.require_text().lower()

    def test_missing_messages(self) -> None:
        tool = KaosLLMToolCallTool()
        result = _run(tool.execute({"model": "openai:gpt-5"}))
        assert result.isError
        assert "messages" in result.require_text().lower()

    def test_missing_tools(self) -> None:
        tool = KaosLLMToolCallTool()
        result = _run(
            tool.execute(
                {
                    "model": "openai:gpt-5",
                    "messages": [{"role": "user", "content": "hi"}],
                }
            )
        )
        assert result.isError
        assert "tools" in result.require_text().lower()

    def test_invalid_tool_definition(self) -> None:
        tool = KaosLLMToolCallTool()
        result = _run(
            tool.execute(
                {
                    "model": "openai:gpt-5",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": ["not-a-dict"],
                }
            )
        )
        assert result.isError
        assert "index 0" in result.require_text().lower()

    def test_tool_missing_name(self) -> None:
        tool = KaosLLMToolCallTool()
        result = _run(
            tool.execute(
                {
                    "model": "openai:gpt-5",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"description": "no name", "parameters": {}}],
                }
            )
        )
        assert result.isError
        assert "name" in result.require_text().lower()

    def test_default_model(self) -> None:
        tool = KaosLLMToolCallTool(default_model="openai:gpt-5")
        meta = tool.metadata
        # model should not be required when default is set
        model_param = next(p for p in meta.input_schema if p.name == "model")
        assert model_param.required is False


# ---------------------------------------------------------------------------
# PydanticTool tests
# ---------------------------------------------------------------------------


class TestPydanticTool:
    """Test KaosLLMPydanticTool error handling."""

    def test_missing_model(self) -> None:
        tool = KaosLLMPydanticTool()
        result = _run(tool.execute({}))
        assert result.isError
        assert "model" in result.require_text().lower()

    def test_missing_messages(self) -> None:
        tool = KaosLLMPydanticTool()
        result = _run(tool.execute({"model": "openai:gpt-5"}))
        assert result.isError
        assert "messages" in result.require_text().lower()

    def test_missing_schema(self) -> None:
        tool = KaosLLMPydanticTool()
        result = _run(
            tool.execute(
                {
                    "model": "openai:gpt-5",
                    "messages": [{"role": "user", "content": "hi"}],
                }
            )
        )
        assert result.isError
        assert "schema" in result.require_text().lower()

    def test_invalid_schema_type(self) -> None:
        tool = KaosLLMPydanticTool()
        result = _run(
            tool.execute(
                {
                    "model": "openai:gpt-5",
                    "messages": [{"role": "user", "content": "hi"}],
                    "schema": "not-a-dict",
                }
            )
        )
        assert result.isError
        assert "JSON object" in result.require_text()


# ---------------------------------------------------------------------------
# ProviderCheckTool tests
# ---------------------------------------------------------------------------


class TestProviderCheckTool:
    """Test KaosLLMProviderCheckTool."""

    def test_check_all_providers_no_keys(self) -> None:
        """With no keys configured, all providers show as not configured."""
        tool = KaosLLMProviderCheckTool()
        # Clear all env vars
        env_overrides = {
            "KAOS_LLM_OPENAI_API_KEY": "",
            "OPENAI_API_KEY": "",
            "KAOS_LLM_ANTHROPIC_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "KAOS_LLM_GOOGLE_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "GOOGLE_GENERATIVE_AI_API_KEY": "",
            "KAOS_LLM_XAI_API_KEY": "",
            "XAI_API_KEY": "",
            "KAOS_LLM_GROQ_API_KEY": "",
            "GROQ_API_KEY": "",
            "KAOS_LLM_MISTRAL_API_KEY": "",
            "MISTRAL_API_KEY": "",
            "KAOS_LLM_OPENROUTER_API_KEY": "",
            "OPENROUTER_API_KEY": "",
        }
        with patch.dict("os.environ", env_overrides, clear=False):
            result = _run(tool.execute({}))
        assert not result.isError
        output = result.require_structured()
        assert output["configured_count"] == 0
        assert len(output["providers"]) == 7

    def test_check_specific_providers(self) -> None:
        """Can check a subset of providers."""
        tool = KaosLLMProviderCheckTool()
        env_overrides = {
            "KAOS_LLM_OPENAI_API_KEY": "sk-test-key",
            "KAOS_LLM_ANTHROPIC_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
        }
        with patch.dict("os.environ", env_overrides, clear=False):
            result = _run(tool.execute({"providers": ["openai", "anthropic"]}))
        assert not result.isError
        output = result.require_structured()
        assert len(output["providers"]) == 2
        openai_result = next(p for p in output["providers"] if p["name"] == "openai")
        assert openai_result["configured"] is True

    def test_unknown_provider(self) -> None:
        """Unknown provider names produce an error."""
        tool = KaosLLMProviderCheckTool()
        result = _run(tool.execute({"providers": ["unknown_provider"]}))
        assert result.isError
        assert "unknown_provider" in result.require_text().lower()

    def test_check_with_legacy_env(self) -> None:
        """Legacy env vars (e.g. OPENAI_API_KEY) are picked up."""
        tool = KaosLLMProviderCheckTool()
        env_overrides = {
            "KAOS_LLM_OPENAI_API_KEY": "",
            "OPENAI_API_KEY": "sk-legacy-key",
        }
        with patch.dict("os.environ", env_overrides, clear=False):
            result = _run(tool.execute({"providers": ["openai"]}))
        assert not result.isError
        output = result.require_structured()
        openai_result = next(p for p in output["providers"] if p["name"] == "openai")
        assert openai_result["configured"] is True


# ---------------------------------------------------------------------------
# CostEstimateTool tests
# ---------------------------------------------------------------------------


class TestCostEstimateTool:
    """Test KaosLLMCostEstimateTool."""

    def test_missing_model(self) -> None:
        tool = KaosLLMCostEstimateTool()
        result = _run(tool.execute({}))
        assert result.isError
        assert "model" in result.require_text().lower()

    def test_missing_input_text(self) -> None:
        tool = KaosLLMCostEstimateTool()
        result = _run(tool.execute({"model": "gpt-5"}))
        assert result.isError
        assert "input_text" in result.require_text().lower()

    def test_unknown_model(self) -> None:
        tool = KaosLLMCostEstimateTool()
        result = _run(tool.execute({"model": "unknown-model-xyz", "input_text": "hello"}))
        assert result.isError
        assert "unknown" in result.require_text().lower()

    def test_known_model_estimate(self) -> None:
        """gpt-5 with known pricing produces a valid estimate."""
        tool = KaosLLMCostEstimateTool()
        text = "Hello, world! " * 100  # ~1400 chars -> ~350 tokens
        result = _run(
            tool.execute({"model": "gpt-5", "input_text": text, "max_output_tokens": 500})
        )
        assert not result.isError
        output = result.require_structured()
        assert output["model"] == "gpt-5"
        assert output["estimated_input_tokens"] > 0
        assert output["max_output_tokens"] == 500
        assert output["estimated_cost_usd"] > 0
        assert "input" in output["pricing_per_1m"]
        assert "output" in output["pricing_per_1m"]

    def test_provider_prefix_stripped(self) -> None:
        """Provider prefix (e.g. 'openai:gpt-5') is stripped for lookup."""
        tool = KaosLLMCostEstimateTool()
        result = _run(tool.execute({"model": "openai:gpt-5", "input_text": "test"}))
        assert not result.isError
        output = result.require_structured()
        assert output["model"] == "gpt-5"

    def test_default_max_output_tokens(self) -> None:
        """Default max_output_tokens is 1000."""
        tool = KaosLLMCostEstimateTool()
        result = _run(tool.execute({"model": "gpt-5", "input_text": "test"}))
        assert not result.isError
        output = result.require_structured()
        assert output["max_output_tokens"] == 1000

    def test_anthropic_model(self) -> None:
        """Anthropic model pricing works."""
        tool = KaosLLMCostEstimateTool()
        result = _run(tool.execute({"model": "claude-sonnet-4-6", "input_text": "hello"}))
        assert not result.isError
        output = result.require_structured()
        assert output["pricing_per_1m"]["input"] == 3.00
        assert output["pricing_per_1m"]["output"] == 15.00

    def test_prefix_match(self) -> None:
        """Versioned model names match via prefix (e.g. gpt-5-0125 -> gpt-5)."""
        tool = KaosLLMCostEstimateTool()
        result = _run(tool.execute({"model": "gpt-5-0125-preview", "input_text": "test"}))
        assert not result.isError
        output = result.require_structured()
        assert output["pricing_per_1m"]["input"] == 2.00


# ---------------------------------------------------------------------------
# Token estimation and pricing helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    """Test internal helper functions."""

    def test_estimate_tokens_short(self) -> None:
        assert _estimate_tokens("hi") >= 1

    def test_estimate_tokens_long(self) -> None:
        tokens = _estimate_tokens("a" * 400)
        assert tokens == 100  # 400 chars / 4

    def test_estimate_tokens_empty(self) -> None:
        assert _estimate_tokens("") == 1  # minimum 1

    def test_lookup_pricing_exact(self) -> None:
        pricing = _lookup_pricing("gpt-5")
        assert pricing is not None
        assert pricing["input"] > 0
        assert pricing["output"] > 0

    def test_lookup_pricing_prefix(self) -> None:
        pricing = _lookup_pricing("gpt-5-turbo")
        assert pricing is not None

    def test_lookup_pricing_unknown(self) -> None:
        pricing = _lookup_pricing("totally-unknown-model")
        assert pricing is None


# ---------------------------------------------------------------------------
# Registration test
# ---------------------------------------------------------------------------


class TestRegistration:
    """Test tool registration."""

    def test_register_count(self) -> None:
        """register_llm_tools should register 7 tools."""

        class FakeRegistry:
            def __init__(self) -> None:
                self.tools: list[Any] = []

            def register_tool(self, tool: Any) -> None:
                self.tools.append(tool)

        class FakeRuntime:
            def __init__(self) -> None:
                self.tools = FakeRegistry()
                self.module_settings: dict[str, Any] = {}

        runtime = FakeRuntime()
        # FakeRuntime is a duck-typed test fake; cast for ty.
        count = register_llm_tools(cast("KaosRuntime", runtime))
        assert count == 7
        assert len(runtime.tools.tools) == 7

    def test_register_names(self) -> None:
        """All 7 expected tool names are registered."""

        class FakeRegistry:
            def __init__(self) -> None:
                self.tools: list[Any] = []

            def register_tool(self, tool: Any) -> None:
                self.tools.append(tool)

        class FakeRuntime:
            def __init__(self) -> None:
                self.tools = FakeRegistry()
                self.module_settings: dict[str, Any] = {}

        runtime = FakeRuntime()
        register_llm_tools(cast("KaosRuntime", runtime))
        names = {t.metadata.name for t in runtime.tools.tools}
        expected = {
            "kaos-llm-chat",
            "kaos-llm-json",
            "kaos-llm-embed",
            "kaos-llm-tools",
            "kaos-llm-pydantic",
            "kaos-llm-provider-check",
            "kaos-llm-cost-estimate",
        }
        assert names == expected
