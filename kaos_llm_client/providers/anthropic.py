"""Anthropic Messages API provider client.

Direct HTTP client for the Anthropic Messages API — no SDK dependency.
Handles system message extraction, thinking, multimodal content, tools,
and Anthropic-specific SSE streaming.
"""

from __future__ import annotations

from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_client.errors import KaosLLMAuthError, KaosLLMValidationError
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

logger = get_logger("kaos_llm_client.providers.anthropic")

_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicClient(BaseProviderClient):
    """Client for the Anthropic Messages API.

    Wire format reference: https://docs.anthropic.com/en/api/messages

    Key differences from OpenAI-style APIs:
    - System prompt is a top-level field, not a message role
    - ``max_tokens`` is always required
    - Auth uses ``x-api-key`` header (not Bearer token)
    - Streaming uses typed SSE events (``content_block_start``, ``content_block_delta``, etc.)
    - Tool results are sent as ``role: "user"`` messages containing ``tool_result`` content blocks
    """

    _provider_name = "anthropic"

    # --- Abstract method implementations ---

    def _get_default_base_url(self) -> str:
        return self._settings.anthropic_base_url

    def _default_endpoint(self) -> str:
        return "/v1/messages"

    def _build_headers(self) -> dict[str, str]:
        api_key = self._resolve_api_key()
        return {
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def _get_api_key_from_settings(self) -> str:
        key = self._settings.anthropic_api_key
        if key is None:
            raise KaosLLMAuthError(
                "Anthropic API key not configured. "
                "Set KAOS_LLM_ANTHROPIC_API_KEY environment variable or pass api_key=.",
                provider="anthropic",
                fix="Export KAOS_LLM_ANTHROPIC_API_KEY=sk-ant-... or pass api_key= to the client.",
            )
        secret = key.get_secret_value()
        if not secret:
            raise KaosLLMAuthError(
                "Anthropic API key is empty.",
                provider="anthropic",
                fix="Set a valid KAOS_LLM_ANTHROPIC_API_KEY value.",
            )
        return secret

    # --- CachePoint preprocessing ---

    def _preprocess_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert CachePoint markers to Anthropic cache_control annotations.

        When a CachePoint appears between messages, we add ``cache_control``
        to the last content block of the preceding message. This tells
        Anthropic to cache all content up to that point, reducing cost on
        subsequent requests with the same prefix.
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "cache_point":
                # Add cache_control to the last content block of the previous message
                if result:
                    self._add_cache_control(result[-1])
                continue
            result.append(msg)
        return result

    def _add_cache_control(self, message: dict[str, Any]) -> None:
        """Add cache_control to the last content block of a message."""
        content = message.get("content")
        if isinstance(content, str):
            # Convert to content block format so we can add cache_control
            message["content"] = [
                {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
            ]
        elif isinstance(content, list) and content:
            # Add cache_control to the last block
            last_block = content[-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = {"type": "ephemeral"}

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
        """Build an Anthropic Messages API request.

        Handles:
        - System message extraction to top-level ``system`` field
        - ``max_tokens`` enforcement (required by Anthropic)
        - Thinking parameter conversion
        - Multimodal content conversion (image_url -> base64 source)
        - Tool definitions and tool_choice conversion
        - Tool result message rewriting (role "tool" -> user with tool_result blocks)
        """
        body: dict[str, Any] = {"model": self.model}

        # Extract system messages and convert remaining messages
        system_parts, api_messages = self._extract_system_messages(messages)
        if system_parts:
            body["system"] = system_parts
        body["messages"] = api_messages

        # max_tokens is always required
        max_tokens = kwargs.pop("max_tokens", None) or self.profile.default_max_tokens
        body["max_tokens"] = max_tokens

        # Thinking / extended thinking
        thinking = kwargs.pop("thinking", None)
        if thinking is not None:
            thinking_config = self._convert_thinking(thinking)
            body["thinking"] = thinking_config

            # Anthropic requires max_tokens > thinking.budget_tokens
            budget = thinking_config.get("budget_tokens", 0)
            if budget and max_tokens <= budget:
                body["max_tokens"] = budget + max_tokens

            # When thinking is enabled, Anthropic requires temperature to be absent
            kwargs.pop("temperature", None)

        # Streaming
        if stream:
            body["stream"] = True

        # Tools
        if tools:
            body["tools"] = [self._convert_tool_definition(t) for t in tools]

        # Tool choice
        if tool_choice is not None:
            body["tool_choice"] = self._convert_tool_choice(tool_choice)

        # Merge remaining kwargs into body
        body.update(kwargs)

        # Client-side precondition: structured outputs + citations → 400.
        self._check_citation_mutex(body)

        return ProviderRequest(
            provider=self._provider_name,
            model=self.model,
            endpoint=self._default_endpoint(),
            body=body,
            stream=stream,
        )

    def _extract_system_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]] | str, list[dict[str, Any]]]:
        """Separate system messages from conversation messages.

        Returns:
            Tuple of (system_content, non_system_messages).
            system_content is a string if there's a single text system message,
            or a list of content blocks if multiple or structured.
        """
        system_parts: list[dict[str, Any]] = []
        api_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_parts.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            system_parts.append({"type": "text", "text": part.get("text", "")})
                        elif isinstance(part, str):
                            system_parts.append({"type": "text", "text": part})
                continue

            # Convert tool result messages
            if msg.get("role") == "tool":
                api_messages = self._append_tool_result(api_messages, msg)
                continue

            # Convert message content for multimodal
            converted = self._convert_message(msg)
            api_messages.append(converted)

        # Simplify: single text system part -> plain string
        if len(system_parts) == 1:
            return system_parts[0]["text"], api_messages
        if not system_parts:
            return "", api_messages
        return system_parts, api_messages

    def _convert_message(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Convert a single message to Anthropic format, handling multimodal content."""
        content = msg.get("content")
        role = msg.get("role", "user")

        if content is None or isinstance(content, str):
            return {"role": role, "content": content or ""}

        if isinstance(content, list):
            converted_parts = []
            for part in content:
                if isinstance(part, str):
                    converted_parts.append({"type": "text", "text": part})
                elif isinstance(part, dict):
                    converted_parts.append(self._convert_content_part(part))
            return {"role": role, "content": converted_parts}

        return {"role": role, "content": str(content)}

    def _convert_content_part(self, part: dict[str, Any]) -> dict[str, Any]:
        """Convert a single content part to Anthropic format.

        Handles:
        - ``{"type": "text", "text": "..."}`` -> pass through
        - ``{"type": "image_url", "image_url": {"url": "data:..."}}`` -> base64 image source
        - ``{"type": "document", "source": {...}}`` -> Anthropic document source
        - ``{"type": "input_audio", ...}`` -> text fallback (not supported by Anthropic)
        - Other types -> pass through
        """
        part_type = part.get("type")

        if part_type == "text":
            return {"type": "text", "text": part.get("text", "")}

        if part_type == "image_url":
            return self._convert_image_url(part)

        if part_type == "document":
            return self._convert_document(part)

        if part_type == "input_audio":
            logger.warning(
                "Anthropic does not support audio input; skipping audio part",
                extra=self._log_extra(
                    provider=self._provider_name,
                    model=self.model,
                ),
            )
            return {"type": "text", "text": "[Audio content not supported by Anthropic]"}

        # Pass through any Anthropic-native content blocks
        return part

    def _convert_image_url(self, part: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI-style image_url to Anthropic base64 image source.

        Input format::

            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}

        Output format::

            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": "iVBOR..."
            }}
        """
        image_url = part.get("image_url", {})
        url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)

        if url.startswith("data:"):
            # Parse data URI: data:image/png;base64,iVBOR...
            header, _, data = url.partition(",")
            # header is "data:image/png;base64"
            media_type = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            }

        # URL-based image (Anthropic supports this too)
        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": url,
            },
        }

    def _convert_document(self, part: dict[str, Any]) -> dict[str, Any]:
        """Convert a document content part to Anthropic format.

        Input formats::

            {"type": "document", "source": {
                "type": "base64", "media_type": "application/pdf", "data": "..."
            }}
            {"type": "document", "source": {"type": "url", "url": "https://..."}}

        Also handles data URIs in the ``url`` field of URL-type sources.
        """
        source = part.get("source", {})
        source_type = source.get("type", "")

        if source_type == "base64":
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": source.get("media_type", "application/pdf"),
                    "data": source.get("data", ""),
                },
            }

        if source_type == "url":
            url = source.get("url", "")
            if url.startswith("data:"):
                # Parse data URI: data:application/pdf;base64,...
                header, _, data = url.partition(",")
                media_type = (
                    header.split(":")[1].split(";")[0] if ":" in header else "application/pdf"
                )
                return {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                }
            return {
                "type": "document",
                "source": {
                    "type": "url",
                    "url": url,
                },
            }

        # Unknown source type — pass through as-is
        return part

    def _append_tool_result(
        self, api_messages: list[dict[str, Any]], tool_msg: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Convert a tool result message to Anthropic format.

        OpenAI-style tool results come as
        ``{"role": "tool", "tool_call_id": "...", "content": "..."}``.
        Anthropic expects them as user messages with ``tool_result`` content blocks.

        If the previous message is already a user message with tool_result blocks,
        we append to it (Anthropic allows multiple tool_results per user message).
        """
        tool_result_block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_msg.get("tool_call_id", ""),
        }

        content = tool_msg.get("content", "")
        if isinstance(content, str | list):
            tool_result_block["content"] = content
        else:
            tool_result_block["content"] = str(content)

        # Check if we can merge with the previous user message
        if (
            api_messages
            and api_messages[-1].get("role") == "user"
            and isinstance(api_messages[-1].get("content"), list)
            and api_messages[-1]["content"]
            and isinstance(api_messages[-1]["content"][0], dict)
            and api_messages[-1]["content"][0].get("type") == "tool_result"
        ):
            api_messages[-1]["content"].append(tool_result_block)
        else:
            api_messages.append({"role": "user", "content": [tool_result_block]})

        return api_messages

    def _convert_thinking(self, thinking: bool | dict[str, Any]) -> dict[str, Any]:
        """Convert thinking parameter to Anthropic format.

        - ``True`` -> ``{"type": "enabled", "budget_tokens": 4096}``
        - ``dict`` -> pass through directly (e.g., ``{"type": "enabled", "budget_tokens": 10000}``)
        """
        if isinstance(thinking, dict):
            return thinking
        if thinking is True:
            return {"type": "enabled", "budget_tokens": 4096}
        return {"type": "disabled"}

    def _convert_tool_definition(self, tool: ToolDefinition) -> dict[str, Any]:
        """Convert a ToolDefinition to Anthropic's tool format."""
        result: dict[str, Any] = {
            "name": tool.name,
            "input_schema": tool.parameters,
        }
        if tool.description is not None:
            result["description"] = tool.description
        return result

    def _convert_tool_choice(self, tool_choice: ToolChoice) -> dict[str, Any]:
        """Convert a ToolChoice to Anthropic's tool_choice format.

        - ``auto`` -> ``{"type": "auto"}``
        - ``none`` -> ``{"type": "none"}`` (Anthropic does not have "none", but we pass it)
        - ``required`` -> ``{"type": "any"}``
        - ``specific`` -> ``{"type": "tool", "name": "..."}``
        """
        if tool_choice.type == "auto":
            return {"type": "auto"}
        if tool_choice.type == "required":
            return {"type": "any"}
        if tool_choice.type == "specific" and tool_choice.name:
            return {"type": "tool", "name": tool_choice.name}
        if tool_choice.type == "none":
            return {"type": "none"}
        return {"type": "auto"}

    # --- Native structured output (output_config.format) ---

    def _apply_native_json_mode(
        self, kwargs: dict[str, Any], schema: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Apply Anthropic native structured output via ``output_config.format``.

        Wire format (GA April 2026, no beta header required)::

            {
              "model": "claude-haiku-4-5",
              ...,
              "output_config": {
                "format": {
                  "type": "json_schema",
                  "schema": { ...transformed_schema... }
                }
              }
            }

        When ``schema`` is provided, the profile's ``json_schema_transformer``
        (:class:`AnthropicJsonSchemaTransformer`) is applied to strip Anthropic-
        rejected keywords and canonicalize key order for the 24h schema cache.

        When ``schema`` is ``None``, Anthropic does not support a ``json_object``-
        style free-form JSON mode, so we route through the prompted fallback —
        caller should supply a schema for deterministic output.

        Mutex: Anthropic returns HTTP 400 when ``output_config.format`` is
        combined with document-block citations. This method does not inspect
        messages directly; :meth:`_check_citation_mutex` is invoked from
        :meth:`_build_request` before wire-encoding.
        """
        kwargs = dict(kwargs)

        if schema is None:
            # No schema — Anthropic has no equivalent of OpenAI's json_object
            # mode; fall back to prompted guidance. Caller sees the same
            # behavior whether this method was invoked or not.
            return kwargs

        transformed_schema = schema
        if self.profile.json_schema_transformer is not None:
            transformer = self.profile.json_schema_transformer(schema, strict=True)
            transformed_schema = transformer.transform()

        kwargs["output_config"] = {
            "format": {
                "type": "json_schema",
                "schema": transformed_schema,
            }
        }
        return kwargs

    def _check_citation_mutex(self, body: dict[str, Any]) -> None:
        """Raise if ``output_config.format`` + document-block citations coexist.

        Anthropic's structured-outputs doc:
        "Incompatible with: Citations (returns 400 error), message prefilling."

        We enforce client-side so the caller sees a diagnostic error with a
        fix, not an opaque 400 from the provider.
        """
        if "output_config" not in body:
            return

        messages = body.get("messages", [])
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                citations = block.get("citations")
                if isinstance(citations, dict) and citations.get("enabled"):
                    raise KaosLLMValidationError(
                        "Anthropic rejects `output_config.format` combined with "
                        "document-block `citations: {enabled: true}` (HTTP 400). "
                        "Fix: pick one — use structured outputs for typed JSON, "
                        "or use citations for evidence-grounded free text. "
                        "Alternative: make two passes (free-text + citations, "
                        "then structured-outputs on the extracted text).",
                    )

    # --- Response parsing ---

    def _parse_response(self, raw: dict[str, Any], request: ProviderRequest) -> ProviderResponse:
        """Parse an Anthropic Messages API response into a ProviderResponse.

        Response format::

            {
                "id": "msg_...",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "..."},
                    {"type": "thinking", "thinking": "..."},
                    {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 50, ...}
            }
        """
        parts: list[ContentPart] = []

        for block in raw.get("content", []):
            block_type = block.get("type")

            if block_type == "text":
                parts.append(ContentPart(type="text", text=block.get("text", ""), raw=block))

            elif block_type == "thinking":
                parts.append(
                    ContentPart(type="thinking", thinking=block.get("thinking", ""), raw=block)
                )

            elif block_type == "tool_use":
                tool_call = ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=block.get("input", {}),
                    raw=block,
                )
                parts.append(ContentPart(type="tool_use", tool_call=tool_call, raw=block))

        # Parse usage
        raw_usage = raw.get("usage", {})
        usage = UsageInfo(
            input_tokens=raw_usage.get("input_tokens", 0),
            output_tokens=raw_usage.get("output_tokens", 0),
            total_tokens=(raw_usage.get("input_tokens", 0) + raw_usage.get("output_tokens", 0)),
            cache_read_tokens=raw_usage.get("cache_read_input_tokens"),
            cache_creation_tokens=raw_usage.get("cache_creation_input_tokens"),
        )

        return ProviderResponse(
            provider=self._provider_name,
            model=self.model,
            raw=raw,
            parts=parts,
            usage=usage,
            stop_reason=raw.get("stop_reason"),
            response_id=raw.get("id"),
            # Plan §Issue 3 — capture the served snapshot. Anthropic's
            # Messages API returns ``model`` as the resolved versioned
            # snapshot (e.g. requesting ``claude-sonnet-4-6`` may return
            # ``claude-sonnet-4-6-20260415``).
            model_snapshot=raw.get("model"),
            request_id=request.request_id,
        )

    # --- Stream parsing ---

    def _parse_stream_chunk(self, data: dict[str, Any]) -> StreamChunk:
        """Parse an Anthropic SSE event dict into a StreamChunk.

        Anthropic SSE event types:
        - ``message_start`` — contains response id and initial usage
        - ``content_block_start`` — new content block (text, thinking, tool_use)
        - ``content_block_delta`` — incremental content update
        - ``content_block_stop`` — block finished
        - ``message_delta`` — stop_reason and final usage
        - ``message_stop`` — message complete
        - ``ping`` — keep-alive
        """
        event_type = data.get("type", "")

        if event_type == "content_block_delta":
            return self._parse_content_block_delta(data)

        if event_type == "content_block_start":
            return self._parse_content_block_start(data)

        if event_type == "message_delta":
            return self._parse_message_delta(data)

        if event_type == "message_start":
            return self._parse_message_start(data)

        # ping, content_block_stop, message_stop — no actionable content
        return StreamChunk(type="text_delta", text="", raw=data)

    def _parse_content_block_delta(self, data: dict[str, Any]) -> StreamChunk:
        """Parse a content_block_delta event.

        Delta types:
        - ``text_delta`` — incremental text: ``{"type": "text_delta", "text": "..."}``
        - ``thinking_delta`` — incremental thinking:
          ``{"type": "thinking_delta", "thinking": "..."}``
        - ``input_json_delta`` — tool input: ``{"type": "input_json_delta", "partial_json": "..."}``
        """
        delta = data.get("delta", {})
        delta_type = delta.get("type", "")

        if delta_type == "text_delta":
            return StreamChunk(
                type="text_delta",
                text=delta.get("text", ""),
                raw=data,
            )

        if delta_type == "thinking_delta":
            return StreamChunk(
                type="thinking_delta",
                thinking=delta.get("thinking", ""),
                raw=data,
            )

        if delta_type == "input_json_delta":
            return StreamChunk(
                type="tool_call_delta",
                tool_call_delta={"arguments": delta.get("partial_json", "")},
                raw=data,
            )

        return StreamChunk(type="text_delta", text="", raw=data)

    def _parse_content_block_start(self, data: dict[str, Any]) -> StreamChunk:
        """Parse a content_block_start event.

        When a tool_use block starts, emit a tool_call_delta with the tool id and name
        so the StreamAccumulator can begin tracking a new tool call.
        """
        content_block = data.get("content_block", {})
        block_type = content_block.get("type", "")

        if block_type == "tool_use":
            return StreamChunk(
                type="tool_call_delta",
                tool_call_delta={
                    "id": content_block.get("id", ""),
                    "name": content_block.get("name", ""),
                    "arguments": "",
                },
                raw=data,
            )

        # text or thinking block start — no content yet
        return StreamChunk(type="text_delta", text="", raw=data)

    def _parse_message_delta(self, data: dict[str, Any]) -> StreamChunk:
        """Parse a message_delta event — carries stop_reason and final usage."""
        raw_usage = data.get("usage", {})

        usage = None
        if raw_usage:
            usage = UsageInfo(
                output_tokens=raw_usage.get("output_tokens", 0),
            )

        chunk = StreamChunk(
            type="usage",
            usage=usage,
            raw=data,
        )
        # Store stop_reason in raw for accumulator to pick up
        # The accumulator reads stop_reason from raw chunks
        return chunk

    def _parse_message_start(self, data: dict[str, Any]) -> StreamChunk:
        """Parse a message_start event — carries response id and initial usage."""
        message = data.get("message", {})
        raw_usage = message.get("usage", {})

        usage = None
        if raw_usage:
            usage = UsageInfo(
                input_tokens=raw_usage.get("input_tokens", 0),
            )

        return StreamChunk(
            type="usage",
            usage=usage,
            raw=data,
        )
