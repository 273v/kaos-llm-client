"""Auto-extracted from the historical kaos_llm_client/tools.py per audit-01 KLC-03.

The tool class is unchanged in behaviour; only its module path moved.
Public API still resolves through ``kaos_llm_client.tools.<ClassName>``
via the re-exports in ``tools/__init__.py``.
"""

from __future__ import annotations

from typing import Any

from kaos_core import KaosContext, KaosTool, ToolMetadata, ToolResult
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

from kaos_llm_client.tools._common import (
    _KNOWN_PROVIDERS,
    _LLM_ANNOTATIONS,
    _MODULE,
    _VERSION,
    _format_llm_error,
    _store_artifact,
    _tool_log_extra,
    logger,
)


class KaosLLMChatTool(KaosTool):
    """Send a chat message to an LLM provider and get a response."""

    def __init__(self, *, default_model: str | None = None) -> None:
        super().__init__()
        self._default_model = default_model

    @property
    def metadata(self) -> ToolMetadata:
        model_desc = "Model string in 'provider:model' format (e.g., 'openai:gpt-5')."
        if self._default_model:
            model_desc += f" Defaults to '{self._default_model}' if omitted."

        return ToolMetadata(
            name="kaos-llm-chat",
            display_name="LLM Chat",
            description=(
                "Send a chat message to an LLM provider and get a response. "
                "Supports all major providers (OpenAI, Anthropic, Google, xAI, Groq, "
                "Mistral, OpenRouter). Use 'provider:model' format for the model string. "
                "For structured JSON output, use 'kaos-llm-json' instead. "
                "For embeddings, use 'kaos-llm-embed'."
            ),
            category=ToolCategory.INTEGRATION,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_LLM_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="model",
                    type="string",
                    description=model_desc,
                    required=self._default_model is None,
                ),
                ParameterSchema(
                    name="message",
                    type="string",
                    description="The user message to send to the model.",
                ),
                ParameterSchema(
                    name="system",
                    type="string",
                    description="Optional system prompt to set model behavior.",
                    required=False,
                ),
                ParameterSchema(
                    name="max_tokens",
                    type="integer",
                    description=(
                        "Maximum number of tokens in the response. "
                        "If omitted, uses the provider's default."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="temperature",
                    type="number",
                    description=(
                        "Sampling temperature (0.0-2.0). Lower = more deterministic, "
                        "higher = more creative. Default varies by provider."
                    ),
                    required=False,
                    constraints={"minimum": 0.0, "maximum": 2.0},
                ),
                ParameterSchema(
                    name="top_p",
                    type="number",
                    description=(
                        "Nucleus sampling: only consider tokens with cumulative "
                        "probability >= top_p (0.0-1.0). Alternative to temperature."
                    ),
                    required=False,
                    constraints={"minimum": 0.0, "maximum": 1.0},
                ),
                ParameterSchema(
                    name="reasoning_effort",
                    type="string",
                    description=(
                        "Reasoning effort for o-series models (o3, o4-mini). "
                        "Values: 'low', 'medium', 'high'. Ignored by non-reasoning models."
                    ),
                    required=False,
                    constraints={"enum": ["low", "medium", "high"]},
                ),
                ParameterSchema(
                    name="service_tier",
                    type="string",
                    description=(
                        "Processing tier for cost/latency tradeoff. "
                        "'flex' = ~50%% cost savings with variable latency (OpenAI, Google). "
                        "'priority' = premium pricing with lowest latency. "
                        "'default' = standard processing. "
                        "Supported by OpenAI (GPT-5, o3, o4-mini) and Google Gemini."
                    ),
                    required=False,
                    constraints={"enum": ["auto", "default", "flex", "priority"]},
                ),
                ParameterSchema(
                    name="provider",
                    type="string",
                    description=(
                        "Override the provider (e.g., 'openai', 'anthropic'). "
                        "Normally inferred from the model string prefix."
                    ),
                    required=False,
                    constraints={
                        "enum": [
                            "openai",
                            "anthropic",
                            "google",
                            "xai",
                            "groq",
                            "mistral",
                            "openrouter",
                            "openai-compatible",
                        ]
                    },
                ),
                ParameterSchema(
                    name="store_artifact",
                    type="boolean",
                    description=(
                        "Store the response as a session artifact. "
                        "When true, the full response is written to VFS and "
                        "discoverable via kaos://session/{session_id}/artifacts. "
                        "Useful for large responses or when you need to reference "
                        "the result later."
                    ),
                    required=False,
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        model = inputs.get("model") or self._default_model
        if not model:
            return ToolResult.create_error(
                "Missing required parameter 'model'. "
                "Provide a model string like 'openai:gpt-5' or 'anthropic:claude-sonnet-4-6'. "
                f"Supported providers: {_KNOWN_PROVIDERS}."
            )

        message = inputs.get("message")
        if not message:
            return ToolResult.create_error(
                "Missing required parameter 'message'. "
                "Provide the text message to send to the model."
            )

        # Build model string with optional provider override
        provider = inputs.get("provider")
        if provider and ":" not in model:
            model = f"{provider}:{model}"

        # Build messages
        messages: list[dict[str, Any]] = []
        system = inputs.get("system")
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        kwargs: dict[str, Any] = {}
        for param in ("max_tokens", "temperature", "top_p", "service_tier"):
            val = inputs.get(param)
            if val is not None:
                kwargs[param] = val

        try:
            from kaos_llm_client.providers import create_client

            # Pass `context=` (not `settings=`) so KaosLLMSettings.from_context
            # is invoked inside BaseProviderClient and any KaosContext._config
            # overrides win over env vars (KLC-01 fix). See providers/base.py.
            client = create_client(model, context=context)
            response = await client.chat_async(messages=messages, **kwargs)
        except ImportError as exc:
            return ToolResult.create_error(
                f"Missing provider dependency: {exc}. "
                "Install the required provider package (e.g., pip install openai)."
            )
        except Exception as exc:
            return _format_llm_error(exc, model)

        # Build structured result
        usage_dict = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        if response.usage.reasoning_tokens:
            usage_dict["reasoning_tokens"] = response.usage.reasoning_tokens
        if response.usage.cache_read_tokens:
            usage_dict["cache_read_tokens"] = response.usage.cache_read_tokens

        result_data: dict[str, Any] = {
            "model": response.model,
            "provider": response.provider,
            "text": response.text,
            "usage": usage_dict,
            "stop_reason": response.stop_reason,
            "latency_ms": response.latency_ms,
        }

        logger.debug(
            "LLM chat completed: model=%s, tokens=%d",
            response.model,
            response.usage.total_tokens,
            extra=_tool_log_extra(
                context,
                tool_name="kaos-llm-chat",
                provider=response.provider,
                model=response.model,
                request_id=(response.request_id or "")[:16] or None,
                response_id=response.response_id,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
                latency_ms=response.latency_ms,
            ),
        )

        # Optionally store as artifact
        if inputs.get("store_artifact") and context is not None:
            artifact_info = await _store_artifact(
                context, result_data, tool_name="kaos-llm-chat", model=model
            )
            if artifact_info:
                result_data["artifact"] = artifact_info

        return ToolResult.create_success(
            output=result_data,
            summary=response.text,
        )
