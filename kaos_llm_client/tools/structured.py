"""Auto-extracted from the historical kaos_llm_client/tools.py per audit-01 KLC-03.

The tool class is unchanged in behaviour; only its module path moved.
Public API still resolves through ``kaos_llm_client.tools.<ClassName>``
via the re-exports in ``tools/__init__.py``.
"""

from __future__ import annotations

import json
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


class KaosLLMStructuredOutputTool(KaosTool):
    """Get structured JSON output from an LLM matching a schema."""

    def __init__(self, *, default_model: str | None = None) -> None:
        super().__init__()
        self._default_model = default_model

    @property
    def metadata(self) -> ToolMetadata:
        model_desc = "Model string in 'provider:model' format (e.g., 'openai:gpt-5')."
        if self._default_model:
            model_desc += f" Defaults to '{self._default_model}' if omitted."

        return ToolMetadata(
            name="kaos-llm-json",
            display_name="LLM Structured JSON",
            description=(
                "Get structured JSON output from an LLM matching a JSON schema. "
                "Uses the best available strategy for each provider (native JSON mode, "
                "tool-based extraction, or prompted output). "
                "For free-form chat, use 'kaos-llm-chat' instead."
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
                    description="The user message describing what structured output to produce.",
                ),
                ParameterSchema(
                    name="schema",
                    type="object",
                    description=(
                        "JSON Schema that the output must conform to. "
                        'Example: {"type": "object", "properties": {"name": {"type": "string"}, '
                        '"age": {"type": "integer"}}, "required": ["name", "age"]}'
                    ),
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
                    description="Sampling temperature (0.0-2.0).",
                    required=False,
                    constraints={"minimum": 0.0, "maximum": 2.0},
                ),
                ParameterSchema(
                    name="store_artifact",
                    type="boolean",
                    description="Store the JSON response as a session artifact.",
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
                "Provide the text describing what structured output to produce."
            )

        schema = inputs.get("schema")
        if not schema:
            return ToolResult.create_error(
                "Missing required parameter 'schema'. "
                "Provide a JSON Schema dict that the output must conform to. "
                'Example: {"type": "object", "properties": {"name": {"type": "string"}}, '
                '"required": ["name"]}'
            )

        # Validate schema is a dict
        if not isinstance(schema, dict):
            return ToolResult.create_error(
                f"Parameter 'schema' must be a JSON object, got {type(schema).__name__}. "
                "Provide a valid JSON Schema as a dict."
            )

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
        reasoning_effort = inputs.get("reasoning_effort")
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}

        try:
            from kaos_llm_client.providers import create_client

            # Pass `context=` (not `settings=`) so KaosLLMSettings.from_context
            # is invoked inside BaseProviderClient and any KaosContext._config
            # overrides win over env vars (KLC-01 fix). See providers/base.py.
            client = create_client(model, context=context)
            response = await client.json_async(messages=messages, schema=schema, **kwargs)
        except ImportError as exc:
            return ToolResult.create_error(
                f"Missing provider dependency: {exc}. "
                "Install the required provider package (e.g., pip install openai)."
            )
        except Exception as exc:
            return _format_llm_error(exc, model)

        # Parse JSON from response
        output_json = response.output_json
        if output_json is None:
            return ToolResult.create_error(
                f"Model returned non-JSON output: {response.text[:200]}. "
                "The model may not support structured output for this request. "
                "Try a different model or simplify the schema. "
                "Alternative: use 'kaos-llm-chat' with explicit JSON instructions."
            )

        usage_dict = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        result_data = {
            "model": response.model,
            "provider": response.provider,
            "output": output_json,
            "usage": usage_dict,
            "stop_reason": response.stop_reason,
            "latency_ms": response.latency_ms,
        }

        logger.debug(
            "LLM JSON completed: model=%s, tokens=%d",
            response.model,
            response.usage.total_tokens,
            extra=_tool_log_extra(
                context,
                tool_name="kaos-llm-json",
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

        summary = json.dumps(output_json, indent=2)
        if len(summary) > 500:
            summary = summary[:497] + "..."

        if inputs.get("store_artifact") and context is not None:
            artifact_info = await _store_artifact(
                context, result_data, tool_name="kaos-llm-json", model=model
            )
            if artifact_info:
                result_data["artifact"] = artifact_info

        return ToolResult.create_success(
            output=result_data,
            summary=summary,
        )


# Local-only tools (no external API calls).
