"""OpenAI Responses API provider client.

The Responses API (``/v1/responses``) is a distinct wire format from chat
completions. It supports builtin tools (web_search, code_interpreter),
reasoning summaries, and stateful conversations via ``previous_response_id``.

NOT a subclass of ``OpenAICompatibleClient`` -- the request/response shapes
are fundamentally different.
"""

from __future__ import annotations

import json
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_client.errors import KaosLLMAuthError
from kaos_llm_client.providers.base import BaseProviderClient
from kaos_llm_client.types import (
    ContentPart,
    ProviderRequest,
    ProviderResponse,
    StreamChunk,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    UsageInfo,
)

logger = get_logger("kaos_llm_client.providers.openai_responses")


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------


def _messages_to_input_items(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert chat-style messages to Responses API input items.

    Mapping:
    - system → ``{"type": "message", "role": "developer", "content": ...}``
    - user/assistant → ``{"type": "message", "role": ..., "content": ...}``
    - tool → ``{"type": "function_call_output", "call_id": ..., "output": ...}``
    """
    items: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            items.append(
                {
                    "type": "message",
                    "role": "developer",
                    "content": content,
                }
            )
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": content if isinstance(content, str) else json.dumps(content),
                }
            )
        else:
            # user, assistant
            items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": content,
                }
            )
    return items


def _tool_def_to_responses(tool: ToolDefinition) -> dict[str, Any]:
    """Convert a ``ToolDefinition`` to Responses API function tool format."""
    func: dict[str, Any] = {
        "type": "function",
        "name": tool.name,
        "parameters": tool.parameters,
    }
    if tool.description is not None:
        func["description"] = tool.description
    if tool.strict is not None:
        func["strict"] = tool.strict
    return func


# ---------------------------------------------------------------------------
# OpenAI Responses API client
# ---------------------------------------------------------------------------


class OpenAIResponsesClient(BaseProviderClient):
    """Client for the OpenAI Responses API (``/v1/responses``).

    This is NOT an OpenAI-compatible (chat completions) client. The Responses
    API has a different wire format with support for:

    - **Builtin tools**: ``web_search``, ``code_interpreter``, ``file_search``
    - **Reasoning**: ``reasoning`` parameter with ``effort`` and ``summary``
    - **Stateful conversations**: ``previous_response_id`` for multi-turn
    - **Developer messages**: system prompts are sent as ``role: "developer"``
    """

    _provider_name: str = "openai-responses"

    # --- Abstract method implementations ---

    def _get_default_base_url(self) -> str:
        return self._settings.openai_base_url

    def _get_api_key_from_settings(self) -> str:
        key = self._settings.openai_api_key
        if key is None:
            raise KaosLLMAuthError(
                "OpenAI API key is not configured.",
                provider=self._provider_name,
                fix="Set KAOS_LLM_OPENAI_API_KEY environment variable or pass api_key= "
                "to the client constructor.",
            )
        secret = key.get_secret_value()
        if not secret:
            raise KaosLLMAuthError(
                "OpenAI API key is empty.",
                provider=self._provider_name,
                fix="Set KAOS_LLM_OPENAI_API_KEY to a valid API key.",
            )
        return secret

    def _build_headers(self) -> dict[str, str]:
        api_key = self._resolve_api_key()
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _default_endpoint(self) -> str:
        return "/v1/responses"

    # --- Request building ---

    def _build_request(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ProviderRequest:
        """Build an OpenAI Responses API request body."""
        input_items = _messages_to_input_items(messages)

        body: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
        }

        # Streaming
        if stream:
            body["stream"] = True

        # Build tools list: function tools + builtin tools
        all_tools: list[dict[str, Any]] = []
        if tools:
            all_tools.extend(_tool_def_to_responses(t) for t in tools)

        # Builtin tools (web_search, code_interpreter, file_search)
        builtin_tools = kwargs.pop("builtin_tools", None)
        if builtin_tools:
            all_tools.extend(builtin_tools)

        if all_tools:
            body["tools"] = all_tools

        # Tool choice — Responses API uses same values as Chat Completions
        if tool_choice is not None:
            if tool_choice.type == "auto":
                body["tool_choice"] = "auto"
            elif tool_choice.type == "none":
                body["tool_choice"] = "none"
            elif tool_choice.type == "required":
                body["tool_choice"] = "required"
            elif tool_choice.type == "specific" and tool_choice.name:
                body["tool_choice"] = {
                    "type": "function",
                    "name": tool_choice.name,
                }

        # Reasoning parameter
        reasoning = kwargs.pop("reasoning", None)
        if reasoning is not None:
            if isinstance(reasoning, dict):
                body["reasoning"] = reasoning
            else:
                body["reasoning"] = {"effort": str(reasoning)}

        # Previous response ID for multi-turn
        previous_response_id = kwargs.pop("previous_response_id", None)
        if previous_response_id is not None:
            body["previous_response_id"] = previous_response_id

        # Max tokens
        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is not None:
            body["max_output_tokens"] = max_tokens

        # Merge remaining kwargs
        body.update(kwargs)

        return ProviderRequest(
            provider=self._provider_name,
            model=self.model,
            endpoint=self._default_endpoint(),
            body=body,
            stream=stream,
        )

    # --- Response parsing ---

    def _parse_response(self, raw: dict[str, Any], request: ProviderRequest) -> ProviderResponse:
        """Parse an OpenAI Responses API response."""
        parts: list[ContentPart] = []
        stop_reason: str | None = raw.get("status")

        # Parse output items
        output = raw.get("output", [])
        for item in output:
            item_type = item.get("type", "")

            if item_type == "message":
                # Text message — extract text from content parts
                content_parts = item.get("content", [])
                for cp in content_parts:
                    if cp.get("type") == "output_text":
                        text = cp.get("text", "")
                        if text:
                            parts.append(ContentPart(type="text", text=text))

            elif item_type == "function_call":
                # Function/tool call
                args_str = item.get("arguments", "{}")
                try:
                    arguments = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, ValueError):
                    arguments = {}

                parts.append(
                    ContentPart(
                        type="tool_use",
                        tool_call=ToolCall(
                            id=item.get("call_id", item.get("id", "")),
                            name=item.get("name", ""),
                            arguments=arguments,
                            raw=item,
                        ),
                    )
                )

            elif item_type == "reasoning":
                # Reasoning/thinking block
                summary = item.get("summary", [])
                thinking_parts = []
                for s in summary:
                    if s.get("type") == "summary_text":
                        thinking_parts.append(s.get("text", ""))
                thinking_text = "\n".join(thinking_parts)
                if thinking_text:
                    parts.append(ContentPart(type="thinking", thinking=thinking_text))

        # Usage
        usage = self._parse_usage(raw.get("usage"))

        return ProviderResponse(
            provider=self._provider_name,
            model=raw.get("model", self.model),
            raw=raw,
            parts=parts,
            usage=usage,
            stop_reason=stop_reason,
            response_id=raw.get("id"),
            request_id=request.request_id,
        )

    def _parse_usage(self, usage_raw: dict[str, Any] | None) -> UsageInfo:
        """Parse Responses API usage into normalized ``UsageInfo``."""
        if not usage_raw:
            return UsageInfo()

        input_tokens = usage_raw.get("input_tokens", 0)
        output_tokens = usage_raw.get("output_tokens", 0)
        total_tokens = usage_raw.get("total_tokens", input_tokens + output_tokens)

        return UsageInfo(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    # --- Stream parsing ---

    def _parse_stream_chunk(self, data: dict[str, Any]) -> StreamChunk | list[StreamChunk]:
        """Parse one SSE chunk from the Responses API stream.

        Key event types:
        - ``response.output_text.delta`` → text_delta
        - ``response.function_call_arguments.delta`` → tool_call_delta
        - ``response.completed`` → usage
        """
        event_type = data.get("type", "")

        # Text delta
        if event_type == "response.output_text.delta":
            return StreamChunk(
                type="text_delta",
                text=data.get("delta", ""),
                raw=data,
            )

        # Output item added — carries function name and call_id for tool calls
        if event_type == "response.output_item.added":
            item = data.get("item", {})
            if item.get("type") == "function_call":
                return StreamChunk(
                    type="tool_call_delta",
                    tool_call_delta={
                        "id": item.get("call_id", ""),
                        "name": item.get("name", ""),
                        "arguments": "",
                    },
                    raw=data,
                )
            return StreamChunk(type="text_delta", text="", raw=data)

        # Tool call argument delta
        if event_type == "response.function_call_arguments.delta":
            tool_delta: dict[str, Any] = {
                "arguments": data.get("delta", ""),
            }
            # item_id references the output item, NOT the call_id
            if "item_id" in data:
                tool_delta["id"] = ""  # Don't override; id set by output_item.added
            return StreamChunk(
                type="tool_call_delta",
                tool_call_delta=tool_delta,
                raw=data,
            )

        # Tool call arguments finalized — no-op for accumulator since
        # output_item.added already set id+name and deltas carried arguments.
        # Emitting name here would double-append it in the accumulator.
        if event_type == "response.function_call_arguments.done":
            return StreamChunk(
                type="text_delta",
                text="",
                raw=data,
            )

        # Response completed — extract usage
        if event_type == "response.completed":
            response = data.get("response", {})
            usage = self._parse_usage(response.get("usage"))
            return StreamChunk(
                type="usage",
                usage=usage,
                raw=data,
            )

        # Default: no-op text delta
        return StreamChunk(type="text_delta", text="", raw=data)
