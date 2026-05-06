"""OpenAI-compatible provider client.

Shared base for all APIs that follow the OpenAI chat completions format:
OpenAI, xAI/Grok, vLLM, OpenRouter, Ollama, and other compatible endpoints.
"""

from __future__ import annotations

import json
import re
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_client.errors import KaosLLMAuthError
from kaos_llm_client.providers.base import BaseProviderClient
from kaos_llm_client.types import (
    BinaryData,
    ContentPart,
    EmbeddingResponse,
    ProviderRequest,
    ProviderResponse,
    RequestOptions,
    StreamChunk,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    UsageInfo,
)

logger = get_logger("kaos_llm_client.providers.openai_compat")

# Regex to extract <think>...</think> blocks (DeepSeek-R1, Qwen3, etc.)
_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _split_thinking_from_text(
    text: str,
) -> tuple[str | None, str]:
    """Extract ``<think>...</think>`` blocks from text content.

    Models like DeepSeek-R1 and Qwen3 embed thinking in text rather than
    sending it as a separate content block type. This function splits the
    thinking out so it can be represented as a proper ``ContentPart``.

    Returns:
        (thinking_text, remaining_text) — thinking is None if no tags found.
    """
    matches = _THINK_TAG_RE.findall(text)
    if not matches:
        return None, text
    thinking = "\n".join(m.strip() for m in matches if m.strip())
    remaining = _THINK_TAG_RE.sub("", text).strip()
    return (thinking or None), remaining


# ---------------------------------------------------------------------------
# Tool format helpers
# ---------------------------------------------------------------------------


def _tool_def_to_openai(tool: ToolDefinition) -> dict[str, Any]:
    """Convert a ``ToolDefinition`` to OpenAI function-calling format."""
    func: dict[str, Any] = {
        "name": tool.name,
        "parameters": tool.parameters,
    }
    if tool.description is not None:
        func["description"] = tool.description
    if tool.strict is not None:
        func["strict"] = tool.strict
    return {"type": "function", "function": func}


def _tool_choice_to_openai(choice: ToolChoice) -> str | dict[str, Any]:
    """Convert a ``ToolChoice`` to the OpenAI ``tool_choice`` format.

    Returns:
        - ``"auto"`` / ``"none"`` / ``"required"`` for those modes
        - ``{"type": "function", "function": {"name": "..."}}`` for specific
    """
    if choice.type == "specific" and choice.name:
        return {"type": "function", "function": {"name": choice.name}}
    # "auto", "none", "required" pass through as strings
    return choice.type


# ---------------------------------------------------------------------------
# OpenAI-compatible client
# ---------------------------------------------------------------------------


class OpenAICompatibleClient(BaseProviderClient):
    """Client for any OpenAI-compatible chat completions API.

    Works with OpenAI, xAI/Grok, vLLM, OpenRouter, Ollama, and other
    endpoints that implement the ``/v1/chat/completions`` contract.
    """

    _provider_name: str = "openai-compatible"

    # --- Abstract method implementations ---

    def _get_default_base_url(self) -> str:
        return self._settings.openai_base_url

    def _get_api_key_from_settings(self) -> str:
        key = self._settings.openai_api_key
        if key is None:
            raise KaosLLMAuthError(
                "OpenAI-compatible API key is not configured.",
                provider=self._provider_name,
                fix="Set KAOS_LLM_OPENAI_API_KEY environment variable or pass api_key= "
                "to the client constructor.",
            )
        secret = key.get_secret_value()
        if not secret:
            raise KaosLLMAuthError(
                "OpenAI-compatible API key is empty.",
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
        return "/v1/chat/completions"

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
        """Build an OpenAI-compatible chat completions request body.

        System prompts stay in messages as ``role: "system"``.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        # Streaming
        if stream:
            body["stream"] = True
            # Request usage in streamed responses (OpenAI extension, widely supported)
            body["stream_options"] = {"include_usage": True}

        # Max tokens — use profile-configured field name
        max_tokens_field = self.profile.max_tokens_field
        if max_tokens_field in kwargs:
            body[max_tokens_field] = kwargs.pop(max_tokens_field)
        elif "max_tokens" in kwargs:
            # Accept generic max_tokens kwarg, map to profile field name
            body[max_tokens_field] = kwargs.pop("max_tokens")
        elif self.profile.requires_max_tokens:
            # Anthropic requires max_tokens; OpenAI/Google do not
            body[max_tokens_field] = self.profile.default_max_tokens

        # Tool definitions
        if tools:
            body["tools"] = [_tool_def_to_openai(t) for t in tools]

        # Tool choice
        if tool_choice is not None:
            body["tool_choice"] = _tool_choice_to_openai(tool_choice)

        # Default service_tier from settings (flex saves ~50%).
        # Reasoning models (o3-mini, o3, o4-mini) reject service_tier with 400
        # — gated by profile.supports_service_tier=False. All other flex failures
        # (500 on mini/nano, 500 on structured output) are handled by the
        # graceful fallback in transport.execute_with_retry: on a 500 with
        # service_tier in the body, it retries once without service_tier.
        supports_service_tier = getattr(self.profile, "supports_service_tier", True)
        if (
            supports_service_tier
            and "service_tier" not in kwargs
            and self._settings.default_service_tier
        ):
            body["service_tier"] = self._settings.default_service_tier

        # Merge remaining kwargs into body (temperature, top_p, etc.)
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
        """Parse an OpenAI-compatible chat completions response."""
        parts: list[ContentPart] = []

        # Extract from choices[0].message
        choices = raw.get("choices", [])
        message: dict[str, Any] = {}
        stop_reason: str | None = None

        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            stop_reason = choice.get("finish_reason")

        # Text content — extract <think> tags from DeepSeek/Qwen models
        content = message.get("content")
        if content:
            thinking, text_content = _split_thinking_from_text(content)
            if thinking:
                parts.append(ContentPart(type="thinking", thinking=thinking))
            if text_content:
                parts.append(ContentPart(type="text", text=text_content))

        # Tool calls
        tool_calls_raw = message.get("tool_calls", [])
        if tool_calls_raw:
            for tc in tool_calls_raw:
                func = tc.get("function", {})
                # Parse arguments from JSON string
                args_str = func.get("arguments", "{}")
                try:
                    arguments = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, ValueError):
                    arguments = {}

                parts.append(
                    ContentPart(
                        type="tool_use",
                        tool_call=ToolCall(
                            id=tc.get("id", ""),
                            name=func.get("name", ""),
                            arguments=arguments,
                            raw=tc,
                        ),
                    )
                )

        # Audio output (OpenAI audio models)
        audio_output = message.get("audio")
        if audio_output:
            audio_data = audio_output.get("data")
            audio_transcript = audio_output.get("transcript")
            if audio_data:
                parts.append(
                    ContentPart(
                        type="audio",
                        binary=BinaryData(data=audio_data, media_type="audio/wav"),
                        transcript=audio_transcript,
                        raw=audio_output,
                    )
                )

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
        """Parse OpenAI-format usage into normalized ``UsageInfo``."""
        if not usage_raw:
            return UsageInfo()

        input_tokens = usage_raw.get("prompt_tokens", 0)
        output_tokens = usage_raw.get("completion_tokens", 0)
        total_tokens = usage_raw.get("total_tokens", input_tokens + output_tokens)

        # Reasoning tokens (o-series models)
        reasoning_tokens: int | None = None
        completion_details = usage_raw.get("completion_tokens_details")
        if isinstance(completion_details, dict):
            reasoning_tokens = completion_details.get("reasoning_tokens")

        # Cache tokens (OpenAI prompt caching)
        cache_read_tokens: int | None = None
        prompt_details = usage_raw.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            cache_read_tokens = prompt_details.get("cached_tokens")

        return UsageInfo(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
            cache_read_tokens=cache_read_tokens,
        )

    # --- Stream parsing ---

    def _parse_stream_chunk(self, data: dict[str, Any]) -> StreamChunk | list[StreamChunk]:
        """Parse one OpenAI SSE chunk into one or more ``StreamChunk`` instances.

        OpenAI streaming format::

            {"choices": [{"delta": {"content": "...", "tool_calls": [...]}, "finish_reason": ...}],
             "usage": {...}}  # only in final chunk when stream_options.include_usage is set
        """
        choices = data.get("choices", [])

        # Usage-only chunk (final chunk with stream_options.include_usage)
        if not choices and "usage" in data:
            return StreamChunk(
                type="usage",
                usage=self._parse_usage(data.get("usage")),
                raw=data,
            )

        if not choices:
            # Empty chunk — treat as no-op text delta
            return StreamChunk(type="text_delta", text="", raw=data)

        choice = choices[0]
        delta = choice.get("delta", {})

        # Tool call deltas — OpenAI may send multiple tool_calls entries
        # with different indices in a single chunk (parallel tool calls).
        # Return a list so the streaming loop emits one chunk per delta.
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            chunks: list[StreamChunk] = []
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_delta: dict[str, Any] = {}
                idx = tc.get("index")
                if idx is not None:
                    tool_delta["index"] = idx
                if "id" in tc:
                    tool_delta["id"] = tc["id"]
                if "name" in func:
                    tool_delta["name"] = func["name"]
                if "arguments" in func:
                    tool_delta["arguments"] = func["arguments"]
                chunks.append(
                    StreamChunk(type="tool_call_delta", tool_call_delta=tool_delta, raw=data)
                )
            return chunks if len(chunks) > 1 else chunks[0]

        # Text delta
        content = delta.get("content")
        if content is not None:
            return StreamChunk(type="text_delta", text=content, raw=data)

        # Finish reason without content (e.g., stop)
        if choice.get("finish_reason") is not None:
            # Check for usage in the same chunk
            usage_raw = data.get("usage")
            if usage_raw:
                return StreamChunk(
                    type="usage",
                    usage=self._parse_usage(usage_raw),
                    raw=data,
                )
            return StreamChunk(type="text_delta", text="", raw=data)

        return StreamChunk(type="text_delta", text="", raw=data)

    # --- Embeddings ---

    async def embed_async(
        self,
        input: str | list[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        options: RequestOptions | None = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Create embeddings using the ``/v1/embeddings`` endpoint.

        Args:
            input: A single string or list of strings to embed.
            model: Optional model override (defaults to ``self.model``).
            dimensions: Optional output dimensionality (provider-dependent).
            options: Transport-level options (timeout, retries).
            **kwargs: Additional provider-specific parameters.

        Returns:
            An ``EmbeddingResponse`` with the embedding vectors.
        """
        from kaos_llm_client.transport import execute_with_retry

        # Normalize input to list
        if isinstance(input, str):
            input = [input]

        body: dict[str, Any] = {
            "model": model or self.model,
            "input": input,
        }
        if dimensions is not None:
            body["dimensions"] = dimensions
        body.update(kwargs)

        request = ProviderRequest(
            provider=self._provider_name,
            model=model or self.model,
            endpoint="/v1/embeddings",
            body=body,
        )
        request.headers.update(self._build_headers())

        timeout = options.timeout if options and options.timeout else None
        client = self._get_async_client()
        response = await execute_with_retry(
            client,
            request,
            retry_policy=self._retry_policy,
            provider=self._provider_name,
            timeout=timeout,
        )

        raw = response.json()

        # Parse embeddings from OpenAI format
        embeddings: list[list[float]] = []
        for item in raw.get("data", []):
            embeddings.append(item.get("embedding", []))

        # Parse usage
        usage_raw = raw.get("usage", {})
        usage = UsageInfo(
            input_tokens=usage_raw.get("prompt_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        )

        return EmbeddingResponse(
            provider=self._provider_name,
            model=raw.get("model", model or self.model),
            embeddings=embeddings,
            usage=usage,
            raw=raw,
            request_id=request.request_id,
        )

    # --- Structured output ---

    def _apply_native_json_mode(
        self, kwargs: dict[str, Any], schema: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Apply ``response_format`` for native JSON mode.

        Sets ``{"type": "json_object"}`` to request JSON output. The
        OpenAI-specific subclass overrides this to add ``json_schema`` support.
        """
        kwargs = dict(kwargs)
        kwargs["response_format"] = {"type": "json_object"}
        return kwargs
