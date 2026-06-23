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
    _tool_log_extra,
    logger,
)


class KaosLLMPydanticTool(KaosTool):
    """Get structured JSON output validated against a JSON schema."""

    def __init__(self, *, default_model: str | None = None) -> None:
        super().__init__()
        self._default_model = default_model

    @property
    def metadata(self) -> ToolMetadata:
        model_desc = "Model string in 'provider:model' format (e.g., 'openai:gpt-5')."
        if self._default_model:
            model_desc += f" Defaults to '{self._default_model}' if omitted."

        return ToolMetadata(
            name="kaos-llm-pydantic",
            display_name="LLM Pydantic Output",
            description=(
                "Get structured JSON output from an LLM, validated against a JSON schema. "
                "Uses the best available strategy for each provider (native JSON mode, "
                "tool-based extraction, or prompted output) with automatic retry on "
                "validation failure. Similar to 'kaos-llm-json' but with retry-based "
                "validation. For simple structured output without retries, use 'kaos-llm-json'."
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
                    name="messages",
                    type="array",
                    description=(
                        "Conversation messages as an array of {role, content} objects. "
                        'Example: [{"role": "user", "content": "Extract the person\'s name"}]'
                    ),
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
                    name="max_retries",
                    type="integer",
                    description=(
                        "Maximum number of validation retries if the model output "
                        "does not match the schema (default: 2). Each retry appends "
                        "the error to the conversation for self-correction."
                    ),
                    required=False,
                    constraints={"minimum": 0, "maximum": 5},
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

        messages = inputs.get("messages")
        if not messages or not isinstance(messages, list):
            return ToolResult.create_error(
                "Missing required parameter 'messages'. "
                "Provide an array of message objects with 'role' and 'content' fields. "
                'Example: [{"role": "user", "content": "Extract the data"}]'
            )

        schema = inputs.get("schema")
        if not schema:
            return ToolResult.create_error(
                "Missing required parameter 'schema'. "
                "Provide a JSON Schema dict that the output must conform to. "
                'Example: {"type": "object", "properties": {"name": {"type": "string"}}, '
                '"required": ["name"]}'
            )

        if not isinstance(schema, dict):
            return ToolResult.create_error(
                f"Parameter 'schema' must be a JSON object, got {type(schema).__name__}. "
                "Provide a valid JSON Schema as a dict."
            )

        # Build messages with optional system prompt
        conv_messages: list[dict[str, Any]] = []
        system = inputs.get("system")
        if system:
            conv_messages.append({"role": "system", "content": system})
        conv_messages.extend(messages)

        kwargs: dict[str, Any] = {}
        max_tokens = inputs.get("max_tokens")
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        max_retries = inputs.get("max_retries", 2)

        try:
            from kaos_llm_client.providers import create_client

            # Pass `context=` (not `settings=`) so KaosLLMSettings.from_context
            # is invoked inside BaseProviderClient and any KaosContext._config
            # overrides win over env vars (KLC-01 fix). See providers/base.py.
            client = create_client(model, context=context)

            # Use json_async with retry logic for schema validation
            last_response = None
            current_messages = list(conv_messages)

            for attempt in range(max_retries + 1):
                response = await client.json_async(
                    messages=current_messages, schema=schema, **kwargs
                )
                last_response = response

                output_json = response.output_json
                if output_json is not None:
                    # Successfully got valid JSON
                    break

                # Retry: append error feedback
                if attempt < max_retries:
                    current_messages = [
                        *list(conv_messages),
                        {"role": "assistant", "content": response.text},
                        {
                            "role": "user",
                            "content": (
                                "Your response was not valid JSON matching the schema. "
                                "Please return valid JSON only, no other text."
                            ),
                        },
                    ]
            else:
                # All retries exhausted
                if last_response is None or last_response.output_json is None:
                    preview = last_response.text[:200] if last_response else "(no response)"
                    return ToolResult.create_error(
                        f"Model failed to produce valid JSON after {max_retries + 1} attempts. "
                        f"Last output: {preview}. "
                        "Try a different model, simplify the schema, or add clearer instructions. "
                        "Alternative: use 'kaos-llm-json' for simpler schema extraction."
                    )

        except ImportError as exc:
            return ToolResult.create_error(
                f"Missing provider dependency: {exc}. "
                "Install the required provider package (e.g., pip install openai)."
            )
        except Exception as exc:
            return _format_llm_error(exc, model)

        # Explicit guard rather than ``assert`` — asserts are stripped under
        # ``python -O`` and the type narrowing is load-bearing here.
        if last_response is None:
            return ToolResult.create_error(
                "Internal error: json tool produced no response and no error. "
                "This is a bug in kaos-llm-client; please report it."
            )
        output_json = last_response.output_json
        if output_json is None:
            return ToolResult.create_error(
                f"Model returned non-JSON output: {last_response.text[:200]}. "
                "The model may not support structured output for this request. "
                "Try a different model or simplify the schema. "
                "Alternative: use 'kaos-llm-chat' with explicit JSON instructions."
            )

        usage_dict = {
            "input_tokens": last_response.usage.input_tokens,
            "output_tokens": last_response.usage.output_tokens,
            "total_tokens": last_response.usage.total_tokens,
        }

        result_data = {
            "model": last_response.model,
            "provider": last_response.provider,
            "output": output_json,
            "usage": usage_dict,
            "stop_reason": last_response.stop_reason,
            "latency_ms": last_response.latency_ms,
        }

        logger.debug(
            "LLM pydantic completed: model=%s, tokens=%d",
            last_response.model,
            last_response.usage.total_tokens,
            extra=_tool_log_extra(
                context,
                tool_name="kaos-llm-pydantic",
                provider=last_response.provider,
                model=last_response.model,
                request_id=(last_response.request_id or "")[:16] or None,
                response_id=last_response.response_id,
                input_tokens=last_response.usage.input_tokens,
                output_tokens=last_response.usage.output_tokens,
                total_tokens=last_response.usage.total_tokens,
                latency_ms=last_response.latency_ms,
            ),
        )

        summary = json.dumps(output_json, indent=2)
        if len(summary) > 500:
            summary = summary[:497] + "..."

        return ToolResult.create_success(output=result_data, summary=summary)
