"""Google Gemini API provider client.

Targets Google AI Studio (generativelanguage.googleapis.com) using API key
authentication. Translates between the kaos-llm-client message format and
Google's ``generateContent`` wire protocol.
"""

from __future__ import annotations

import json
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_client.errors import KaosLLMAuthError
from kaos_llm_client.providers.base import BaseProviderClient
from kaos_llm_client.types import (
    BinaryData,
    ContentPart,
    ProviderRequest,
    ProviderResponse,
    StreamChunk,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    UsageInfo,
)

logger = get_logger("kaos_llm_client.providers.google")

# Mapping from OpenAI audio format strings to MIME types
_AUDIO_FORMAT_MIME: dict[str, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
}


# ---------------------------------------------------------------------------
# Message format helpers
# ---------------------------------------------------------------------------


def _role_to_google(role: str) -> str:
    """Map standard roles to Google's role names.

    Google uses ``"model"`` where other providers use ``"assistant"``.
    System messages are handled separately via ``systemInstruction``.
    """
    if role == "assistant":
        return "model"
    if role == "tool":
        return "function"
    return role


def _convert_content_to_parts(content: Any) -> list[dict[str, Any]]:
    """Convert message content (string or list-of-parts) to Google ``parts``.

    Handles:
    - Plain text strings -> ``[{"text": "..."}]``
    - Lists of content parts with text, image_url, or inline_data
    """
    if isinstance(content, str):
        return [{"text": content}]

    if not isinstance(content, list):
        return [{"text": str(content)}]

    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            parts.append({"text": part})
        elif isinstance(part, dict):
            if part.get("type") == "text":
                parts.append({"text": part.get("text", "")})
            elif part.get("type") == "image_url":
                # Convert OpenAI-style image_url to Google inline_data
                url_info = part.get("image_url", {})
                url = url_info.get("url", "") if isinstance(url_info, dict) else str(url_info)
                if url.startswith("data:"):
                    # data:image/png;base64,... -> inline_data
                    mime_end = url.index(";")
                    mime_type = url[5:mime_end]
                    data = url.split(",", 1)[1] if "," in url else ""
                    parts.append({"inline_data": {"mime_type": mime_type, "data": data}})
                else:
                    # URL reference -- Google requires inline_data, not URLs
                    # Pass as text fallback
                    parts.append({"text": f"[Image: {url}]"})
            elif part.get("type") == "document":
                # Document part (PDF, etc.)
                source = part.get("source", {})
                source_type = source.get("type", "")
                if source_type == "base64":
                    parts.append(
                        {
                            "inline_data": {
                                "mime_type": source.get("media_type", "application/pdf"),
                                "data": source.get("data", ""),
                            }
                        }
                    )
                elif source_type == "url":
                    url = source.get("url", "")
                    if url.startswith("gs://"):
                        parts.append(
                            {
                                "file_data": {
                                    "file_uri": url,
                                    "mime_type": source.get("media_type", "application/pdf"),
                                }
                            }
                        )
                    else:
                        parts.append({"text": f"[Document: {url}]"})
                else:
                    parts.append({"text": str(part)})
            elif part.get("type") == "input_audio":
                # OpenAI-style audio input -> Google inline_data
                audio_info = part.get("input_audio", {})
                audio_data = audio_info.get("data", "")
                audio_format = audio_info.get("format", "wav")
                mime_type = _AUDIO_FORMAT_MIME.get(audio_format, f"audio/{audio_format}")
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": audio_data,
                        }
                    }
                )
            elif "file_data" in part:
                # Pre-formatted Google file_data — pass through as-is
                parts.append({"file_data": part["file_data"]})
            elif "inline_data" in part:
                # Already in Google format
                parts.append({"inline_data": part["inline_data"]})
            elif "functionCall" in part:
                # Google-native functionCall from AssistantMessage.from_response()
                # Pass through as-is (preserves id + thoughtSignature)
                parts.append(part)
            elif part.get("thought") is True:
                # Google-native thought part — pass through
                parts.append(part)
            else:
                parts.append({"text": str(part)})
        else:
            parts.append({"text": str(part)})
    return parts


def _convert_messages(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Split messages into system instruction and Google-format contents.

    Returns:
        Tuple of (system_instruction_or_None, google_contents_list).
    """
    system_instruction: dict[str, Any] | None = None
    contents: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")

        if role == "system":
            # Accumulate system messages into systemInstruction
            text = msg.get("content", "")
            if isinstance(text, list):
                text = " ".join(
                    p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in text
                )
            if system_instruction is None:
                system_instruction = {"parts": [{"text": text}]}
            else:
                system_instruction["parts"].append({"text": text})
            continue

        if role == "tool":
            # Tool result messages -> functionResponse parts
            # Per Google SDK: FunctionResponse has id, name, and response fields
            func_resp: dict[str, Any] = {
                "name": msg.get("name", msg.get("tool_call_id", "")),
                "response": _parse_tool_response_content(msg.get("content", "")),
            }
            # Echo the function call id if available (required for Gemini 3+)
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id:
                func_resp["id"] = tool_call_id
            parts = [{"functionResponse": func_resp}]
            contents.append({"role": "function", "parts": parts})
            continue

        google_role = _role_to_google(role)
        parts = _convert_content_to_parts(msg.get("content", ""))

        # Append tool_calls as functionCall parts (for assistant messages)
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            func = tc.get("function", tc)
            name = func.get("name", "")
            args_raw = func.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except (json.JSONDecodeError, ValueError):
                args = {}
            parts.append({"functionCall": {"name": name, "args": args}})

        contents.append({"role": google_role, "parts": parts})

    return system_instruction, contents


def _parse_tool_response_content(content: Any) -> dict[str, Any]:
    """Parse tool response content into a dict for functionResponse."""
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return {"result": content}
    return {"result": str(content)}


# ---------------------------------------------------------------------------
# Tool format helpers
# ---------------------------------------------------------------------------


def _tool_def_to_google(
    tool: ToolDefinition,
    *,
    schema_transformer: type | None = None,
) -> dict[str, Any]:
    """Convert a ``ToolDefinition`` to Google ``functionDeclarations`` format.

    0.1.1 (R0.3 — reliability roadmap #560): Google Gemini's
    ``generateContent`` rejects raw JSONSchema parameter blocks
    that contain ``$ref``, ``$defs``, ``title``, ``const``,
    ``default``, or several other keywords (see
    :class:`GoogleJsonSchemaTransformer` for the full list of
    quirks). Pre-fix, ``_tool_def_to_google`` passed ``tool.parameters``
    verbatim — every tool turn returned HTTP 400 from Google and
    both Gemini Pro and Flash were unusable for tool-using legal
    research in the SPA. ``GoogleJsonSchemaTransformer`` already
    existed but was only applied to the structured-output
    ``responseSchema`` path (see line ~640 of this module). Now we
    apply the same transformer to tool parameter blocks so Gemini
    can actually dispatch.

    Args:
        tool: The :class:`~kaos_llm_client.tools.ToolDefinition` to
            translate.
        schema_transformer: Optional transformer class (typically
            ``profile.json_schema_transformer``) applied to
            ``tool.parameters`` before forwarding to Google.
            Defaults to None (legacy behavior) so callers that build
            tool definitions through profile-aware paths can pass
            the transformer explicitly. The caller
            :func:`_build_request` passes the profile's transformer
            when one is configured (which it is for every
            Gemini-family model in :data:`PROFILES`).
    """
    parameters = tool.parameters
    if schema_transformer is not None and parameters:
        try:
            transformer = schema_transformer(parameters)
            parameters = transformer.transform()
        except Exception:  # never let a transform error
            # take down a real turn; fall back to the raw JSONSchema
            # and let Google's error surface naturally. Transform
            # failures here are rare (typically malformed schemas)
            # and the failure shape on the wire is the same either
            # way: HTTP 400 with a parameter validation message.
            parameters = tool.parameters
    decl: dict[str, Any] = {
        "name": tool.name,
        "parameters": parameters,
    }
    if tool.description is not None:
        decl["description"] = tool.description
    return decl


def _tool_choice_to_google(choice: ToolChoice) -> dict[str, Any]:
    """Convert a ``ToolChoice`` to Google ``toolConfig`` format."""
    if choice.type == "none":
        return {"functionCallingConfig": {"mode": "NONE"}}
    if choice.type == "required":
        return {"functionCallingConfig": {"mode": "ANY"}}
    if choice.type == "specific" and choice.name:
        return {
            "functionCallingConfig": {
                "mode": "ANY",
                "allowedFunctionNames": [choice.name],
            }
        }
    # "auto" is the default
    return {"functionCallingConfig": {"mode": "AUTO"}}


# ---------------------------------------------------------------------------
# Google Gemini client
# ---------------------------------------------------------------------------


class GoogleClient(BaseProviderClient):
    """Google Gemini API client supporting both AI Studio and Vertex AI.

    **AI Studio** (default): Uses ``x-goog-api-key`` header authentication.
    Endpoint: ``/v1beta/models/{model}:generateContent``

    **Vertex AI**: Activated when ``base_url`` contains ``aiplatform.googleapis.com``.
    Uses ``Authorization: Bearer {token}`` header authentication.
    Endpoint:
    ``/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent``

    Vertex mode requires ``google_project`` in settings (or ``KAOS_LLM_GOOGLE_PROJECT``
    env var). ``google_location`` defaults to ``us-central1``.
    """

    _provider_name: str = "google"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tool_call_counter = 0

    # --- Vertex AI detection ---

    @property
    def _is_vertex(self) -> bool:
        """True when the base URL targets Vertex AI (aiplatform.googleapis.com)."""
        return "aiplatform.googleapis.com" in self._base_url

    @property
    def _vertex_project(self) -> str:
        """Return the Google Cloud project ID for Vertex AI requests."""
        project = self._settings.google_project
        if not project:
            raise KaosLLMAuthError(
                "Vertex AI requires a Google Cloud project ID.",
                provider=self._provider_name,
                fix="Set KAOS_LLM_GOOGLE_PROJECT environment variable or google_project "
                "in KaosLLMSettings.",
            )
        return project

    @property
    def _vertex_location(self) -> str:
        """Return the Google Cloud location for Vertex AI requests."""
        return self._settings.google_location

    # --- Abstract method implementations ---

    def _get_default_base_url(self) -> str:
        return self._settings.google_base_url

    def _get_api_key_from_settings(self) -> str:
        key = self._settings.google_api_key
        if key is None:
            if self._is_vertex:
                raise KaosLLMAuthError(
                    "Vertex AI access token is not configured.",
                    provider=self._provider_name,
                    fix="Pass api_key= with a valid access token (from gcloud auth "
                    "print-access-token), or set KAOS_LLM_GOOGLE_API_KEY.",
                )
            raise KaosLLMAuthError(
                "Google API key is not configured.",
                provider=self._provider_name,
                fix="Set KAOS_LLM_GOOGLE_API_KEY environment variable or pass api_key= "
                "to the client constructor.",
            )
        secret = key.get_secret_value()
        if not secret:
            raise KaosLLMAuthError(
                "Google API key is empty.",
                provider=self._provider_name,
                fix="Set KAOS_LLM_GOOGLE_API_KEY to a valid API key.",
            )
        return secret

    def _build_headers(self) -> dict[str, str]:
        api_key = self._resolve_api_key()
        if self._is_vertex:
            return {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        return {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }

    def _default_endpoint(self) -> str:
        if self._is_vertex:
            project = self._vertex_project
            location = self._vertex_location
            return (
                f"/v1/projects/{project}/locations/{location}"
                f"/publishers/google/models/{self.model}:generateContent"
            )
        return f"/v1beta/models/{self.model}:generateContent"

    def _stream_endpoint(self) -> str:
        """Streaming uses a different endpoint suffix with SSE alt parameter."""
        if self._is_vertex:
            project = self._vertex_project
            location = self._vertex_location
            return (
                f"/v1/projects/{project}/locations/{location}"
                f"/publishers/google/models/{self.model}:streamGenerateContent?alt=sse"
            )
        return f"/v1beta/models/{self.model}:streamGenerateContent?alt=sse"

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
        """Build a Google Gemini ``generateContent`` request body."""
        system_instruction, contents = _convert_messages(messages)

        body: dict[str, Any] = {
            "contents": contents,
        }

        # System instruction at top level
        if system_instruction is not None:
            body["systemInstruction"] = system_instruction

        # Generation config
        generation_config: dict[str, Any] = {}

        # Max tokens -> maxOutputTokens
        max_tokens_field = self.profile.max_tokens_field  # "maxOutputTokens"
        if max_tokens_field in kwargs:
            generation_config["maxOutputTokens"] = kwargs.pop(max_tokens_field)
        elif "max_tokens" in kwargs:
            generation_config["maxOutputTokens"] = kwargs.pop("max_tokens")
        else:
            generation_config["maxOutputTokens"] = self.profile.default_max_tokens

        # Common generation params
        for param in ("temperature", "topP", "topK", "top_p", "top_k"):
            if param in kwargs:
                # Normalize to Google's camelCase
                google_key = param.replace("_", "")
                google_key = {"topp": "topP", "topk": "topK"}.get(google_key, google_key)
                generation_config[google_key] = kwargs.pop(param)

        # Stop sequences
        if "stop" in kwargs:
            generation_config["stopSequences"] = kwargs.pop("stop")
        if "stop_sequences" in kwargs:
            generation_config["stopSequences"] = kwargs.pop("stop_sequences")

        if generation_config:
            body["generationConfig"] = generation_config

        # Tool definitions
        # 0.1.1 (R0.3): pass the profile's JSON schema transformer so
        # tool parameter blocks get sanitized for Gemini's stricter
        # JSONSchema subset (no $ref/$defs, no const, no title,
        # no default). Without this, every tool turn returns HTTP 400
        # and both Gemini Pro and Flash are unusable for tool-using
        # work in the SPA — confirmed by the worker-honesty audit
        # 2026-05-21 (see ``kaos-modules/docs/audits/2026-05-21-worker-honesty.md``).
        if tools:
            transformer = self.profile.json_schema_transformer
            body["tools"] = [
                {
                    "functionDeclarations": [
                        _tool_def_to_google(t, schema_transformer=transformer) for t in tools
                    ]
                }
            ]

        # Tool choice
        if tool_choice is not None:
            body["toolConfig"] = _tool_choice_to_google(tool_choice)

        # Merge Google JSON mode config INTO generationConfig (not overwrite)
        google_json_config = kwargs.pop("_google_json_config", None)
        if google_json_config:
            if "generationConfig" not in body:
                body["generationConfig"] = {}
            body["generationConfig"].update(google_json_config)

        # Merge remaining kwargs
        body.update(kwargs)

        # Choose endpoint based on streaming
        endpoint = self._stream_endpoint() if stream else self._default_endpoint()

        return ProviderRequest(
            provider=self._provider_name,
            model=self.model,
            endpoint=endpoint,
            body=body,
            stream=stream,
        )

    # --- Response parsing ---

    def _parse_response(self, raw: dict[str, Any], request: ProviderRequest) -> ProviderResponse:
        """Parse a Google Gemini ``generateContent`` response."""
        parts: list[ContentPart] = []
        stop_reason: str | None = None

        candidates = raw.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            stop_reason = candidate.get("finishReason")

            content = candidate.get("content", {})
            for part in content.get("parts", []):
                # Thinking part (Gemini 2.5+/3.x with thought=True)
                if part.get("thought") is True and "text" in part:
                    parts.append(ContentPart(type="thinking", thinking=part["text"], raw=part))
                    continue

                # Text part (preserve thoughtSignature in raw for replay)
                if "text" in part:
                    parts.append(ContentPart(type="text", text=part["text"], raw=part))

                # Function call part — use provider's id if present (Gemini 3+),
                # fall back to synthetic counter for older models
                if "functionCall" in part:
                    fc = part["functionCall"]
                    call_id = fc.get("id")
                    if not call_id:
                        self._tool_call_counter += 1
                        call_id = f"google_tc_{self._tool_call_counter}"
                    parts.append(
                        ContentPart(
                            type="tool_use",
                            tool_call=ToolCall(
                                id=call_id,
                                name=fc.get("name", ""),
                                arguments=fc.get("args", {}),
                                raw=part,
                            ),
                        )
                    )

                # Inline binary data (image/audio output from Gemini)
                if "inline_data" in part:
                    inline = part["inline_data"]
                    mime = inline.get("mime_type", "")
                    if mime.startswith("image/"):
                        parts.append(
                            ContentPart(
                                type="image",
                                binary=BinaryData(data=inline["data"], media_type=mime),
                                raw=part,
                            )
                        )
                    elif mime.startswith("audio/"):
                        parts.append(
                            ContentPart(
                                type="audio",
                                binary=BinaryData(data=inline["data"], media_type=mime),
                                raw=part,
                            )
                        )

        # Usage metadata
        usage = self._parse_usage(raw.get("usageMetadata"))

        return ProviderResponse(
            provider=self._provider_name,
            model=raw.get("modelVersion", self.model),
            raw=raw,
            parts=parts,
            usage=usage,
            stop_reason=stop_reason,
            request_id=request.request_id,
        )

    def _parse_usage(self, usage_raw: dict[str, Any] | None) -> UsageInfo:
        """Parse Google ``usageMetadata`` into normalized ``UsageInfo``."""
        if not usage_raw:
            return UsageInfo()

        input_tokens = usage_raw.get("promptTokenCount", 0)
        output_tokens = usage_raw.get("candidatesTokenCount", 0)
        total_tokens = usage_raw.get("totalTokenCount", input_tokens + output_tokens)

        return UsageInfo(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    # --- Stream parsing ---

    def _parse_stream_chunk(self, data: dict[str, Any]) -> StreamChunk:
        """Parse one Google SSE chunk into a ``StreamChunk``.

        Google streaming format mirrors the non-streaming response structure:
        each SSE event contains ``candidates[0].content.parts`` and optionally
        ``usageMetadata``.
        """
        candidates = data.get("candidates", [])

        # Usage metadata (may appear in any chunk, typically the last)
        usage_raw = data.get("usageMetadata")

        if not candidates:
            if usage_raw:
                return StreamChunk(
                    type="usage",
                    usage=self._parse_usage(usage_raw),
                    raw=data,
                )
            return StreamChunk(type="text_delta", text="", raw=data)

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        # Check for function call parts
        for part in parts:
            if "functionCall" in part:
                fc = part["functionCall"]
                call_id = fc.get("id")
                if not call_id:
                    self._tool_call_counter += 1
                    call_id = f"google_tc_{self._tool_call_counter}"
                return StreamChunk(
                    type="tool_call_delta",
                    tool_call_delta={
                        "id": call_id,
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {})),
                    },
                    raw=data,
                )

        # Check for inline binary data (image/audio output)
        for part in parts:
            if "inline_data" in part:
                inline = part["inline_data"]
                mime = inline.get("mime_type", "")
                if mime.startswith("image/"):
                    return StreamChunk(type="text_delta", text="", raw=data)
                if mime.startswith("audio/"):
                    return StreamChunk(type="text_delta", text="", raw=data)

        # Text parts
        text_parts = [p.get("text", "") for p in parts if "text" in p]
        text = "".join(text_parts)

        # If this is the final chunk with usage, emit usage
        if usage_raw and not text:
            return StreamChunk(
                type="usage",
                usage=self._parse_usage(usage_raw),
                raw=data,
            )

        return StreamChunk(type="text_delta", text=text, raw=data)

    # --- Structured output ---

    def _apply_native_json_mode(
        self, kwargs: dict[str, Any], schema: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Apply Google's native structured output via ``generationConfig``.

        Sets ``responseMimeType: "application/json"`` and optionally
        ``responseSchema`` if a schema is provided.

        Uses ``_google_json_config`` to store the JSON mode settings, which
        ``_build_request()`` merges INTO (not overwrites) the generationConfig.
        This avoids clobbering maxOutputTokens and other generation params.
        """
        kwargs = dict(kwargs)

        json_config: dict[str, Any] = {"responseMimeType": "application/json"}

        if schema is not None:
            # Apply Google schema transformer if the profile defines one
            transformed = schema
            if self.profile.json_schema_transformer is not None:
                transformer = self.profile.json_schema_transformer(schema)
                transformed = transformer.transform()
            json_config["responseSchema"] = transformed

        # Store separately — _build_request merges this into generationConfig
        kwargs["_google_json_config"] = json_config
        return kwargs
