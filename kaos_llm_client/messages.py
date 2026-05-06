"""Typed message models for kaos-llm-client.

Optional typed input models that coexist with raw dicts. Callers can use
either — raw dicts pass through as before, typed models serialize to the
same OpenAI-format dicts the providers already accept.

Usage::

    from kaos_llm_client.messages import UserMessage, SystemMessage

    # Typed (catches errors at construction)
    messages = [
        SystemMessage("Be concise."),
        UserMessage("What is 2+2?"),
        UserMessage(["Describe this image:", image_from_path("photo.jpg")]),
    ]
    response = client.chat(messages)

    # Raw dicts (still works, zero conversion)
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    response = client.chat(messages)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Content types for multimodal messages
# ---------------------------------------------------------------------------

# A single content part: plain text, or a dict content part (image_url, input_audio, etc.)
ContentItem = str | dict[str, Any]

# Message content: a plain string, or a list of content parts (for multimodal)
MessageContent = str | Sequence[ContentItem]


# ---------------------------------------------------------------------------
# Message classes
# ---------------------------------------------------------------------------


class Message(dict[str, Any]):
    """Base class for typed messages.

    Subclasses ``dict`` so it can be passed directly to providers without
    conversion. The typed constructor validates fields; the result is a
    plain dict that providers already understand.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)


class SystemMessage(Message):
    """System prompt message.

    Args:
        content: The system prompt text.
        name: Optional sender name.
    """

    def __init__(self, content: str, *, name: str | None = None) -> None:
        data: dict[str, Any] = {"role": "system", "content": content}
        if name is not None:
            data["name"] = name
        super().__init__(data)


class UserMessage(Message):
    """User message, optionally with multimodal content.

    Args:
        content: Plain text string, or a list of content parts for multimodal
            messages (text dicts, image_url dicts, input_audio dicts, etc.).
        name: Optional sender name.

    Examples::

        # Simple text
        UserMessage("Hello!")

        # Multimodal with image
        UserMessage([
            {"type": "text", "text": "What is this?"},
            image_from_path("photo.jpg"),
        ])
    """

    def __init__(self, content: MessageContent, *, name: str | None = None) -> None:
        # Normalize: if content is a sequence, convert to list of dicts
        if isinstance(content, str):
            normalized: str | list[dict[str, Any]] = content
        else:
            parts: list[dict[str, Any]] = []
            for item in content:
                if isinstance(item, str):
                    parts.append({"type": "text", "text": item})
                elif isinstance(item, dict):
                    parts.append(item)
                else:
                    raise TypeError(f"Content item must be str or dict, got {type(item).__name__}")
            normalized = parts

        data: dict[str, Any] = {"role": "user", "content": normalized}
        if name is not None:
            data["name"] = name
        super().__init__(data)


class AssistantMessage(Message):
    """Assistant message (for multi-turn conversations).

    Args:
        content: The assistant's text response, or None for tool-call-only messages.
        tool_calls: Tool calls from the assistant (OpenAI format).
        name: Optional sender name.
    """

    def __init__(
        self,
        content: str | list[dict[str, Any]] | None = None,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
        name: str | None = None,
    ) -> None:
        data: dict[str, Any] = {"role": "assistant"}
        if content is not None:
            data["content"] = content
        if tool_calls is not None:
            data["tool_calls"] = tool_calls
        if name is not None:
            data["name"] = name
        super().__init__(data)

    @classmethod
    def from_response(cls, response: Any) -> AssistantMessage:
        """Create from a ProviderResponse for multi-turn conversations.

        Preserves the full response structure including thinking blocks and
        tool calls in a provider-appropriate format. The provider field on
        the response determines the serialization:

        - **OpenAI**: ``tool_calls`` array in OpenAI function-calling format
        - **Anthropic**: ``content`` as a list of content blocks (text,
          thinking, tool_use) — required for thinking replay in tool-use
          continuations per Anthropic docs
        - **Google/other**: ``content`` blocks with tool calls embedded

        Usage::

            response = client.chat(messages)
            messages.append(AssistantMessage.from_response(response))
            messages.append(ToolResultMessage(...))
            response2 = client.chat(messages)
        """
        import json as _json

        provider = getattr(response, "provider", "openai")

        if provider == "anthropic":
            # Anthropic needs content blocks: thinking + text + tool_use
            # Thinking blocks MUST be replayed during tool-use continuations
            content_blocks: list[dict[str, Any]] = []
            for part in response.parts:
                if part.type == "thinking" and part.thinking:
                    block: dict[str, Any] = {
                        "type": "thinking",
                        "thinking": part.thinking,
                    }
                    # Preserve signature if present in raw (required for replay)
                    if part.raw and "signature" in part.raw:
                        block["signature"] = part.raw["signature"]
                    content_blocks.append(block)
                elif part.type == "text" and part.text:
                    content_blocks.append({"type": "text", "text": part.text})
                elif part.type == "tool_use" and part.tool_call:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": part.tool_call.id,
                            "name": part.tool_call.name,
                            "input": part.tool_call.arguments,
                        }
                    )
            return cls(content=content_blocks)  # type: ignore[arg-type]

        if provider == "openai-responses":
            # Responses API: assistant message carries content blocks,
            # tool calls are separate output items referenced by call_id.
            # The _messages_to_input_items converter handles the wire format.
            content_parts: list[dict[str, Any]] = []
            if response.text:
                content_parts.append({"type": "output_text", "text": response.text})
            for tc in response.tool_calls:
                content_parts.append(
                    {
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": _json.dumps(tc.arguments),
                    }
                )
            return cls(content=content_parts)  # type: ignore[arg-type]

        if provider == "google":
            # Google needs content as parts with functionCall + thoughtSignature
            # preserved. Per Google SDK: Part.thought_signature must be echoed.
            google_parts: list[dict[str, Any]] = []
            for part in response.parts:
                if part.type == "thinking" and part.thinking:
                    p: dict[str, Any] = {"text": part.thinking, "thought": True}
                    if part.raw and "thoughtSignature" in part.raw:
                        p["thoughtSignature"] = part.raw["thoughtSignature"]
                    google_parts.append(p)
                elif part.type == "text" and part.text:
                    p = {"text": part.text}
                    if part.raw and "thoughtSignature" in part.raw:
                        p["thoughtSignature"] = part.raw["thoughtSignature"]
                    google_parts.append(p)
                elif part.type == "tool_use" and part.tool_call:
                    fc: dict[str, Any] = {
                        "name": part.tool_call.name,
                        "args": part.tool_call.arguments,
                    }
                    if part.tool_call.id:
                        fc["id"] = part.tool_call.id
                    p: dict[str, Any] = {"functionCall": fc}
                    # Preserve thoughtSignature — it's on the raw Part dict,
                    # stored in ToolCall.raw (not ContentPart.raw)
                    tc_raw = part.tool_call.raw if part.tool_call.raw else {}
                    if "thoughtSignature" in tc_raw:
                        p["thoughtSignature"] = tc_raw["thoughtSignature"]
                    google_parts.append(p)
            return cls(content=google_parts)  # type: ignore[arg-type]

        # OpenAI Chat Completions / others: standard format
        content = response.text or None
        tool_calls = None
        if response.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": _json.dumps(tc.arguments),
                    },
                }
                for tc in response.tool_calls
            ]
        return cls(content=content, tool_calls=tool_calls)


class ToolResultMessage(Message):
    """Tool result message (for tool-calling workflows).

    Args:
        tool_call_id: The ID of the tool call this result responds to.
        content: The tool result content (string or JSON-serializable).
        name: Tool name. **Required for Google Gemini** (used as the
            ``functionResponse`` name). Optional for OpenAI/Anthropic but
            recommended for all providers.

    The provider layer handles format translation:
    - OpenAI: ``role: "tool"`` with ``tool_call_id``
    - Anthropic: rewrites to ``role: "user"`` with ``tool_result`` content block
    - Google: ``role: "function"`` with ``functionResponse`` (uses ``name``)

    Convenience: use ``ToolCall`` from the response to fill both id and name::

        for tc in response.tool_calls:
            result = execute_tool(tc.name, tc.arguments)
            messages.append(ToolResultMessage(tc.id, result, name=tc.name))
    """

    def __init__(
        self,
        tool_call_id: str,
        content: str | dict[str, Any] | list[Any],
        *,
        name: str | None = None,
    ) -> None:
        if not isinstance(content, str):
            content = __import__("json").dumps(content)

        data: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
        if name is not None:
            data["name"] = name
        super().__init__(data)


class CachePoint(Message):
    """Prompt cache breakpoint marker for Anthropic.

    Insert between messages to mark a cache boundary. Anthropic caches
    all content before this point, reducing cost on subsequent requests
    with the same prefix.

    Not supported by all providers — providers that don't support it
    silently ignore it.

    Usage::

        messages = [
            SystemMessage("Long system prompt..."),
            CachePoint(),  # Cache the system prompt
            UserMessage("New question each time"),
        ]
    """

    def __init__(self) -> None:
        super().__init__({"role": "cache_point"})


# ---------------------------------------------------------------------------
# Type aliases for provider method signatures
# ---------------------------------------------------------------------------

# What chat() accepts: raw dicts or typed messages, mixed freely
ChatMessages = Sequence[dict[str, Any] | Message]
