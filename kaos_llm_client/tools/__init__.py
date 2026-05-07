"""MCP tool definitions for LLM client operations.

KaosTool implementations registered with KaosRuntime and exposed via
kaos-mcp. Each tool provides LLM inference capabilities (chat, structured
output, embeddings, tool-calling, Pydantic-typed output, plus local
provider-info / cost-estimate utilities) across multiple providers.

The historical monolithic ``kaos_llm_client/tools.py`` was split into
this subpackage as part of audit-01 KLC-03 (file > 1500 lines + mixed
concerns). The public import path is unchanged — every name that used
to be reachable via ``from kaos_llm_client.tools import X`` still
resolves through the re-exports below.

Layout:

  - ``_common.py``      — shared helpers + constants
  - ``chat.py``         — KaosLLMChatTool
  - ``structured.py``   — KaosLLMStructuredOutputTool
  - ``embed.py``        — KaosLLMEmbedTool + _estimate_tokens
  - ``tool_call.py``    — KaosLLMToolCallTool
  - ``pydantic.py``     — KaosLLMPydanticTool
  - ``provider_check.py`` — KaosLLMProviderCheckTool
  - ``cost_estimate.py``  — KaosLLMCostEstimateTool
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_llm_client.cost import (
    lookup_pricing as _lookup_pricing,  # noqa: F401  (back-compat re-export)
)
from kaos_llm_client.tools._common import (
    _MODEL_PRICING as _MODEL_PRICING,
)
from kaos_llm_client.tools._common import (
    _format_llm_error as _format_llm_error,
)
from kaos_llm_client.tools._common import (
    _tool_log_extra as _tool_log_extra,
)
from kaos_llm_client.tools._common import logger as logger

# Public tool classes
from kaos_llm_client.tools.chat import KaosLLMChatTool
from kaos_llm_client.tools.cost_estimate import KaosLLMCostEstimateTool
from kaos_llm_client.tools.embed import KaosLLMEmbedTool
from kaos_llm_client.tools.embed import (
    _estimate_tokens as _estimate_tokens,
)
from kaos_llm_client.tools.provider_check import KaosLLMProviderCheckTool
from kaos_llm_client.tools.pydantic import KaosLLMPydanticTool
from kaos_llm_client.tools.structured import KaosLLMStructuredOutputTool
from kaos_llm_client.tools.tool_call import KaosLLMToolCallTool

if TYPE_CHECKING:
    from kaos_core import KaosRuntime, KaosTool


__all__ = [
    "KaosLLMChatTool",
    "KaosLLMCostEstimateTool",
    "KaosLLMEmbedTool",
    "KaosLLMProviderCheckTool",
    "KaosLLMPydanticTool",
    "KaosLLMStructuredOutputTool",
    "KaosLLMToolCallTool",
    "register_llm_tools",
]


def register_llm_tools(
    runtime: KaosRuntime,
    *,
    default_model: str | None = None,
) -> int:
    """Register all LLM tools with the runtime. Returns count.

    Args:
        runtime: The KaosRuntime to register tools with.
        default_model: Optional default model string for tools.
            When set, tools can be called without specifying a model.
    """
    from kaos_llm_client.settings import KaosLLMSettings

    runtime.module_settings["llm"] = KaosLLMSettings()

    tools: list[KaosTool] = [
        KaosLLMChatTool(default_model=default_model),
        KaosLLMStructuredOutputTool(default_model=default_model),
        KaosLLMEmbedTool(default_model=default_model),
        KaosLLMToolCallTool(default_model=default_model),
        KaosLLMPydanticTool(default_model=default_model),
        KaosLLMProviderCheckTool(),
        KaosLLMCostEstimateTool(),
    ]
    for tool in tools:
        runtime.tools.register_tool(tool)
    return len(tools)
