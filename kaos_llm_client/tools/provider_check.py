"""KaosLLMProviderCheckTool — extracted from tools.py per audit-01 KLC-03."""

from __future__ import annotations

from typing import Any

from kaos_core import KaosContext, KaosTool, ToolMetadata, ToolResult
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

from kaos_llm_client.tools._common import (
    _LOCAL_ANNOTATIONS,
    _MODULE,
    _VERSION,
    _tool_log_extra,
    logger,
)


class KaosLLMProviderCheckTool(KaosTool):
    """Check which LLM providers have API keys configured."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-llm-provider-check",
            display_name="LLM Provider Check",
            description=(
                "Check which LLM providers have valid API keys configured. "
                "Does NOT make API calls -- only checks if keys are set in the environment. "
                "Use this to discover available providers before making LLM calls. "
                "For credential verification via actual API call, use 'kaos-llm-chat' "
                "with a simple test message."
            ),
            category=ToolCategory.INTEGRATION,
            capability=ToolCapability.VALIDATE,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_LOCAL_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="providers",
                    type="array",
                    description=(
                        "Optional list of provider names to check "
                        "(e.g., ['openai', 'anthropic']). "
                        "If empty or omitted, checks all known providers."
                    ),
                    required=False,
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        from kaos_llm_client.settings import KaosLLMSettings

        # Use from_context so KaosContext._config overrides apply when
        # MCP clients pass per-request settings (KLC-01 fix).
        settings = KaosLLMSettings.from_context(context)

        # Map provider name -> (settings field name, base URL field name)
        provider_map: dict[str, tuple[str, str]] = {
            "openai": ("openai_api_key", "openai_base_url"),
            "anthropic": ("anthropic_api_key", "anthropic_base_url"),
            "google": ("google_api_key", "google_base_url"),
            "xai": ("xai_api_key", "xai_base_url"),
            "groq": ("groq_api_key", "groq_base_url"),
            "mistral": ("mistral_api_key", "mistral_base_url"),
            "openrouter": ("openrouter_api_key", "openrouter_base_url"),
        }

        requested = inputs.get("providers")
        if requested and isinstance(requested, list):
            # Validate requested providers
            unknown = [p for p in requested if p not in provider_map]
            if unknown:
                return ToolResult.create_error(
                    f"Unknown provider(s): {', '.join(unknown)}. "
                    f"Known providers: {', '.join(sorted(provider_map))}."
                )
            check_providers = {k: v for k, v in provider_map.items() if k in requested}
        else:
            check_providers = provider_map

        results: list[dict[str, Any]] = []
        configured_count = 0

        for name, (key_field, url_field) in sorted(check_providers.items()):
            key_value = getattr(settings, key_field, None)
            base_url = getattr(settings, url_field, None)
            is_configured = key_value is not None and key_value.get_secret_value() != ""
            if is_configured:
                configured_count += 1
            results.append(
                {
                    "name": name,
                    "configured": is_configured,
                    "base_url": base_url,
                }
            )

        result_data = {
            "providers": results,
            "configured_count": configured_count,
            "total_checked": len(results),
        }

        configured_names = [r["name"] for r in results if r["configured"]]
        if configured_names:
            summary = f"{configured_count} provider(s) configured: {', '.join(configured_names)}"
        else:
            summary = "No providers configured. Set API keys in environment variables."

        logger.debug(
            "Provider check: %d/%d configured",
            configured_count,
            len(results),
            extra=_tool_log_extra(
                context,
                tool_name="kaos-llm-provider-check",
                configured_count=configured_count,
                total_checked=len(results),
            ),
        )

        return ToolResult.create_success(output=result_data, summary=summary)
