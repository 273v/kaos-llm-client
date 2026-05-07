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
    _tool_log_extra,
    logger,
)


class KaosLLMToolCallTool(KaosTool):
    """Send a chat message with tool definitions and get tool calls back."""

    def __init__(self, *, default_model: str | None = None) -> None:
        super().__init__()
        self._default_model = default_model

    @property
    def metadata(self) -> ToolMetadata:
        model_desc = "Model string in 'provider:model' format (e.g., 'openai:gpt-5')."
        if self._default_model:
            model_desc += f" Defaults to '{self._default_model}' if omitted."

        return ToolMetadata(
            name="kaos-llm-tools",
            display_name="LLM Tool Call",
            description=(
                "Send messages to an LLM with tool definitions and receive tool calls. "
                "The model decides which tools to call based on the conversation. "
                "Returns the model's text response and any tool calls with their arguments. "
                "For simple chat without tools, use 'kaos-llm-chat'. "
                "For structured JSON output, use 'kaos-llm-json'."
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
                        'Example: [{"role": "user", "content": "What is the weather?"}]'
                    ),
                ),
                ParameterSchema(
                    name="tools",
                    type="array",
                    description=(
                        "Tool definitions the model can call. Each tool has 'name', "
                        "'description', and 'parameters' (JSON Schema). "
                        'Example: [{"name": "get_weather", "description": "Get weather", '
                        '"parameters": {"type": "object", "properties": '
                        '{"city": {"type": "string"}}}}]'
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
                    name="tool_choice",
                    type="string",
                    description=(
                        "Control tool calling behavior. "
                        "'auto' = model decides (default). "
                        "'required' = model must call a tool. "
                        "'none' = model must not call tools."
                    ),
                    required=False,
                    constraints={"enum": ["auto", "required", "none"]},
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
                'Example: [{"role": "user", "content": "Hello"}]'
            )

        raw_tools = inputs.get("tools")
        if not raw_tools or not isinstance(raw_tools, list):
            return ToolResult.create_error(
                "Missing required parameter 'tools'. "
                "Provide an array of tool definition objects with 'name', 'description', "
                "and 'parameters' fields. For chat without tools, use 'kaos-llm-chat' instead."
            )

        # Build tool definitions
        from kaos_llm_client.types import ToolChoice, ToolDefinition

        tool_defs: list[ToolDefinition] = []
        for i, raw_tool_item in enumerate(raw_tools):
            if not isinstance(raw_tool_item, dict):
                return ToolResult.create_error(
                    f"Tool at index {i} is not an object. "
                    "Each tool must be a dict with 'name' and 'parameters' fields."
                )
            td = dict(raw_tool_item)  # type: dict[str, Any]
            name = td.get("name")
            if not name:
                return ToolResult.create_error(
                    f"Tool at index {i} missing 'name'. "
                    "Every tool definition must have a 'name' field."
                )
            tool_defs.append(
                ToolDefinition(
                    name=name,
                    description=td.get("description"),
                    parameters=td.get("parameters", {}),
                )
            )

        # Build messages with optional system prompt
        conv_messages: list[dict[str, Any]] = []
        system = inputs.get("system")
        if system:
            conv_messages.append({"role": "system", "content": system})
        conv_messages.extend(messages)

        # Parse tool_choice
        tool_choice_input = inputs.get("tool_choice")
        tc: ToolChoice | None = None
        if tool_choice_input:
            tc = ToolChoice(type=tool_choice_input)

        kwargs: dict[str, Any] = {}
        max_tokens = inputs.get("max_tokens")
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            from kaos_llm_client.providers import create_client

            # Pass `context=` (not `settings=`) so KaosLLMSettings.from_context
            # is invoked inside BaseProviderClient and any KaosContext._config
            # overrides win over env vars (KLC-01 fix). See providers/base.py.
            client = create_client(model, context=context)
            response = await client.chat_async(
                messages=conv_messages,
                tools=tool_defs,
                tool_choice=tc,
                **kwargs,
            )
        except ImportError as exc:
            return ToolResult.create_error(
                f"Missing provider dependency: {exc}. "
                "Install the required provider package (e.g., pip install openai)."
            )
        except Exception as exc:
            return _format_llm_error(exc, model)

        # Build tool calls list
        tool_calls_data: list[dict[str, Any]] = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in response.tool_calls
        ]

        usage_dict = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        result_data: dict[str, Any] = {
            "model": response.model,
            "provider": response.provider,
            "text": response.text,
            "tool_calls": tool_calls_data,
            "usage": usage_dict,
            "stop_reason": response.stop_reason,
        }

        n_calls = len(tool_calls_data)
        logger.info(
            "LLM tool call completed: model=%s, tool_calls=%d, tokens=%d",
            response.model,
            n_calls,
            response.usage.total_tokens,
            extra=_tool_log_extra(
                context,
                tool_name="kaos-llm-tools",
                provider=response.provider,
                model=response.model,
                request_id=(response.request_id or "")[:16] or None,
                response_id=response.response_id,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
                latency_ms=response.latency_ms,
                tool_call_count=n_calls,
            ),
        )

        if n_calls > 0:
            tool_names = ", ".join(tc["name"] for tc in tool_calls_data)
            summary = f"Model called {n_calls} tool(s): {tool_names}"
        else:
            text_preview = response.text[:200] if response.text else "(no text)"
            summary = f"Model returned text without tool calls: {text_preview}"

        return ToolResult.create_success(output=result_data, summary=summary)
