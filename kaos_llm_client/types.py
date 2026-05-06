"""Core type system for kaos-llm-client.

All types inherit from ``KaosModel`` (Pydantic BaseModel with extra="forbid").
"""

from __future__ import annotations

import base64
import json
import mimetypes
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from kaos_core.types.content import KaosModel
from pydantic import Field

# ---------------------------------------------------------------------------
# Request types
# ---------------------------------------------------------------------------


class CachePolicy(StrEnum):
    """Cache behavior for a single request."""

    DEFAULT = "default"  # use client-level setting
    SKIP = "skip"  # bypass cache for this request
    FORCE = "force"  # always cache, even if client default is off


class RequestOptions(KaosModel):
    """Transport-level options, separate from provider body."""

    timeout: float | None = None
    max_retries: int | None = None
    retry_backoff_base: float | None = None
    cache_policy: CachePolicy = CachePolicy.DEFAULT
    extra_headers: dict[str, str] | None = None
    max_response_bytes: int | None = None
    """Per-request override for ``KaosLLMSettings.max_response_bytes``. ``None``
    falls back to the settings value."""
    stream_max_duration: float | None = None
    """Per-request override for ``KaosLLMSettings.stream_max_duration``. ``None``
    falls back to the settings value."""


class ProviderRequest(KaosModel):
    """Internal typed request representation."""

    provider: str
    model: str
    endpoint: str  # e.g., "/v1/chat/completions", "/v1/messages"
    body: dict[str, Any]  # provider-native request body
    headers: dict[str, str] = Field(default_factory=dict)
    stream: bool = False
    request_id: str = Field(default_factory=lambda: str(uuid4()))


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------


class RequestHooks:
    """Optional lifecycle callbacks for request/response observation.

    Attach to a client to observe requests, responses, errors, and retries
    without modifying behavior. Foundation for instrumentation and cost tracking.

    Usage::

        def log_usage(request, response):
            print(f"{response.usage.total_tokens} tokens")

        hooks = RequestHooks(on_response=log_usage)
        client = create_client("openai:gpt-5", hooks=hooks)

    Header redaction
    ----------------

    By default, ``on_request`` / ``on_response`` / ``on_error`` receive a
    copy of the request whose ``Authorization`` and ``api-key`` headers
    are replaced with ``<redacted>``. This prevents accidental key
    leakage when hooks log the request object (which is a common pattern
    for instrumentation). Set ``include_auth_headers=True`` to opt out
    if you need raw headers for a specific use case (e.g. transport
    debugging) — but never log the resulting object verbatim.
    """

    __slots__ = (
        "include_auth_headers",
        "on_error",
        "on_request",
        "on_response",
        "on_retry",
    )

    def __init__(
        self,
        *,
        on_request: Any = None,
        on_response: Any = None,
        on_error: Any = None,
        on_retry: Any = None,
        include_auth_headers: bool = False,
    ) -> None:
        self.on_request = on_request
        self.on_response = on_response
        self.on_error = on_error
        self.on_retry = on_retry
        self.include_auth_headers = include_auth_headers


# ---------------------------------------------------------------------------
# Tool types
# ---------------------------------------------------------------------------


class ToolDefinition(KaosModel):
    """A tool/function definition to send to the model."""

    name: str
    description: str | None = None
    parameters: dict[str, Any]  # JSON Schema
    strict: bool | None = None  # OpenAI strict mode


class ToolChoice(KaosModel):
    """Control which tool the model should call."""

    type: Literal["auto", "none", "required", "specific"] = "auto"
    name: str | None = None  # for type="specific"


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------


class UsageInfo(KaosModel):
    """Token usage from any provider, normalized."""

    model_config = KaosModel.model_config.copy()
    model_config["extra"] = "allow"  # type: ignore[assignment]

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None


class ToolCall(KaosModel):
    """A tool/function call from the model."""

    id: str
    name: str
    arguments: dict[str, Any]
    raw: dict[str, Any] | None = None


class BinaryData(KaosModel):
    """Binary content with media type metadata.

    Represents base64-encoded binary data (images, audio, documents)
    in both input helpers and output response parts.

    Follows pydantic-ai's ``BinaryContent`` pattern but stays lighter —
    data is always base64, not raw bytes.
    """

    data: str  # base64-encoded
    media_type: str  # e.g. "image/png", "audio/wav", "application/pdf"

    @property
    def data_uri(self) -> str:
        """Return ``data:{media_type};base64,{data}``."""
        return f"data:{self.media_type};base64,{self.data}"

    @property
    def is_image(self) -> bool:
        return self.media_type.startswith("image/")

    @property
    def is_audio(self) -> bool:
        return self.media_type.startswith("audio/")

    @property
    def is_document(self) -> bool:
        return self.media_type in (
            "application/pdf",
            "text/plain",
            "text/csv",
            "text/html",
            "text/markdown",
        )

    @classmethod
    def from_data_uri(cls, uri: str) -> BinaryData:
        """Parse a ``data:`` URI into BinaryData."""
        if not uri.startswith("data:"):
            raise ValueError("Data URI must start with 'data:'")
        header, _, data = uri.partition(",")
        media_type = (
            header.split(":")[1].split(";")[0] if ":" in header else "application/octet-stream"
        )
        return cls(data=data, media_type=media_type)

    @classmethod
    def from_bytes(cls, raw: bytes, media_type: str) -> BinaryData:
        """Create from raw bytes."""
        return cls(data=base64.b64encode(raw).decode("ascii"), media_type=media_type)

    @classmethod
    def from_path(cls, path: str | Path) -> BinaryData:
        """Read a file and infer media type from extension."""
        p = Path(path)
        media_type, _ = mimetypes.guess_type(str(p))
        if media_type is None:
            media_type = "application/octet-stream"
        data = base64.b64encode(p.read_bytes()).decode("ascii")
        return cls(data=data, media_type=media_type)

    def to_bytes(self) -> bytes:
        """Decode base64 data to raw bytes."""
        return base64.b64decode(self.data)


class ContentPart(KaosModel):
    """One part of the model's response content."""

    type: Literal["text", "thinking", "tool_use", "image", "audio", "document"]
    text: str | None = None
    thinking: str | None = None
    tool_call: ToolCall | None = None
    binary: BinaryData | None = None  # image/audio/document data
    transcript: str | None = None  # audio transcript (OpenAI)
    raw: dict[str, Any] | None = None


class EmbeddingResponse(KaosModel):
    """Response from an embedding request."""

    model_config = KaosModel.model_config.copy()
    model_config["extra"] = "allow"  # type: ignore[assignment]

    provider: str
    model: str
    embeddings: list[list[float]]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    raw: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None

    @property
    def embedding(self) -> list[float]:
        """Convenience accessor for the first embedding vector."""
        return self.embeddings[0] if self.embeddings else []


class ProviderResponse(KaosModel):
    """Complete response from any provider.

    ``.raw`` always contains the unmodified provider payload.
    Structured fields are convenience accessors parsed from raw.
    """

    model_config = KaosModel.model_config.copy()
    model_config["extra"] = "allow"  # type: ignore[assignment]

    provider: str
    model: str
    raw: dict[str, Any]

    # Structured content
    parts: list[ContentPart] = Field(default_factory=list)
    usage: UsageInfo = Field(default_factory=UsageInfo)
    stop_reason: str | None = None
    response_id: str | None = None

    # Transport metadata
    status_code: int = 200
    response_headers: dict[str, str] = Field(default_factory=dict)
    request_id: str | None = None
    latency_ms: float | None = None

    # --- Convenience properties ---

    @property
    def text(self) -> str:
        """Concatenated text from all text parts."""
        return "".join(p.text for p in self.parts if p.type == "text" and p.text)

    @property
    def thinking(self) -> str | None:
        """Concatenated thinking content, or None."""
        parts = [p.thinking for p in self.parts if p.type == "thinking" and p.thinking]
        return "".join(parts) if parts else None

    @property
    def tool_calls(self) -> list[ToolCall]:
        """All tool calls in response order."""
        return [p.tool_call for p in self.parts if p.type == "tool_use" and p.tool_call]

    @property
    def images(self) -> list[BinaryData]:
        """All image outputs from the response."""
        return [p.binary for p in self.parts if p.type == "image" and p.binary]

    @property
    def audio(self) -> BinaryData | None:
        """First audio output, or None."""
        for p in self.parts:
            if p.type == "audio" and p.binary:
                return p.binary
        return None

    @property
    def audio_transcript(self) -> str | None:
        """Audio transcript from OpenAI audio models, or None."""
        for p in self.parts:
            if p.type == "audio" and p.transcript:
                return p.transcript
        return None

    @property
    def output_json(self) -> dict[str, Any] | list[Any] | None:
        """Parsed JSON from text content, or None if not valid JSON."""
        text = self.text
        if not text:
            return None
        try:
            result = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            # Try stripping markdown code fences
            from kaos_llm_client.json_utils import extract_json

            result = extract_json(text)
        if isinstance(result, dict | list):
            return result
        return None


# ---------------------------------------------------------------------------
# Streaming types
# ---------------------------------------------------------------------------


class StreamChunk(KaosModel):
    """One chunk from an SSE stream."""

    type: Literal["text_delta", "thinking_delta", "tool_call_delta", "usage", "done", "error"]
    text: str | None = None
    thinking: str | None = None
    tool_call_delta: dict[str, Any] | None = None
    usage: UsageInfo | None = None
    raw: dict[str, Any] | None = None


class StreamAccumulator:
    """Accumulates ``StreamChunk`` instances into a final ``ProviderResponse``.

    Usage::

        accumulator = StreamAccumulator(provider="openai", model="gpt-5", request_id="...")
        async for chunk in stream:
            accumulator.feed(chunk)
            if chunk.text:
                print(chunk.text, end="", flush=True)
        response = accumulator.accumulated
    """

    def __init__(
        self,
        provider: str,
        model: str,
        request_id: str,
        *,
        strip_leading_whitespace: bool = False,
    ) -> None:
        self.provider = provider
        self.model = model
        self.request_id = request_id
        self._strip_leading_ws = strip_leading_whitespace
        self._text_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._tool_calls: list[dict[str, Any]] = []
        self._tool_calls_by_index: dict[int, dict[str, Any]] = {}
        self._current_tool_call: dict[str, Any] | None = None
        self._usage: UsageInfo | None = None
        self._stop_reason: str | None = None
        self._raw_chunks: list[dict[str, Any]] = []
        self._response_id: str | None = None

    def feed(self, chunk: StreamChunk) -> None:
        """Feed a stream chunk into the accumulator."""
        if chunk.raw:
            self._raw_chunks.append(chunk.raw)
            # Extract metadata from raw chunk data
            self._extract_metadata(chunk.raw)

        if chunk.type == "text_delta" and chunk.text:
            self._text_parts.append(chunk.text)
        elif chunk.type == "thinking_delta" and chunk.thinking:
            self._thinking_parts.append(chunk.thinking)
        elif chunk.type == "tool_call_delta" and chunk.tool_call_delta:
            self._accumulate_tool_call(chunk.tool_call_delta)
        elif chunk.type == "usage" and chunk.usage:
            self._merge_usage(chunk.usage)
        elif chunk.type == "done":
            self._finalize_current_tool_call()

    def _extract_metadata(self, raw: dict[str, Any]) -> None:
        """Extract stop_reason and response_id from raw SSE data.

        Different providers embed this in different places:
        - OpenAI: choices[0].finish_reason, id
        - Anthropic: type=="message_delta" → delta.stop_reason; type=="message_start" → message.id
        - Google: candidates[0].finishReason
        """
        # OpenAI: response id at top level
        if "id" in raw and self._response_id is None:
            self._response_id = raw["id"]

        # OpenAI: finish_reason in choices
        choices = raw.get("choices", [])
        if choices:
            fr = choices[0].get("finish_reason")
            if fr:
                self._stop_reason = fr

        # Anthropic: message_start carries id
        if raw.get("type") == "message_start":
            msg = raw.get("message", {})
            if "id" in msg:
                self._response_id = msg["id"]

        # Anthropic: message_delta carries stop_reason
        if raw.get("type") == "message_delta":
            delta = raw.get("delta", {})
            sr = delta.get("stop_reason")
            if sr:
                self._stop_reason = sr

        # Google: finishReason in candidates
        candidates = raw.get("candidates", [])
        if candidates:
            fr = candidates[0].get("finishReason")
            if fr:
                self._stop_reason = fr

    def _accumulate_tool_call(self, delta: dict[str, Any]) -> None:
        """Accumulate incremental tool call deltas.

        Handles both index-based (OpenAI parallel tool calls) and
        id-based (Anthropic, Google) accumulation. When an ``index``
        field is present, each index slot is tracked independently
        so interleaved deltas for different tool calls don't clobber
        each other.
        """
        idx = delta.get("index")
        if idx is not None:
            # Index-based: OpenAI parallel tool calls
            if idx not in self._tool_calls_by_index:
                self._tool_calls_by_index[idx] = {
                    "id": delta.get("id", ""),
                    "name": delta.get("name", ""),
                    "arguments_json": delta.get("arguments", ""),
                }
            else:
                tc = self._tool_calls_by_index[idx]
                if delta.get("id"):
                    tc["id"] = delta["id"]
                if delta.get("name"):
                    tc["name"] += delta["name"]
                if delta.get("arguments"):
                    tc["arguments_json"] += delta["arguments"]
        elif delta.get("id"):
            # Id-based: new tool call starting (Anthropic, Google)
            self._finalize_current_tool_call()
            self._current_tool_call = {
                "id": delta.get("id", ""),
                "name": delta.get("name", ""),
                "arguments_json": delta.get("arguments", ""),
            }
        elif self._current_tool_call is not None:
            # Continue accumulating current call
            if delta.get("name"):
                self._current_tool_call["name"] += delta["name"]
            if delta.get("arguments"):
                self._current_tool_call["arguments_json"] += delta["arguments"]

    def _finalize_current_tool_call(self) -> None:
        """Finalize the current in-progress tool call."""
        if self._current_tool_call is not None:
            self._tool_calls.append(self._current_tool_call)
            self._current_tool_call = None

    def _merge_usage(self, new_usage: UsageInfo) -> None:
        """Merge usage info instead of overwriting.

        Anthropic sends input_tokens in ``message_start`` and output_tokens
        in ``message_delta``. We must accumulate both, not lose the first.
        """
        if self._usage is None:
            self._usage = new_usage
            return
        # Take the max of each field — Anthropic sends partial updates
        input_t = max(self._usage.input_tokens, new_usage.input_tokens)
        output_t = max(self._usage.output_tokens, new_usage.output_tokens)
        total_t = max(self._usage.total_tokens, new_usage.total_tokens)
        # If total is still 0 but we have input+output, compute it
        if total_t == 0 and (input_t > 0 or output_t > 0):
            total_t = input_t + output_t
        self._usage = UsageInfo(
            input_tokens=input_t,
            output_tokens=output_t,
            total_tokens=total_t,
            reasoning_tokens=new_usage.reasoning_tokens or self._usage.reasoning_tokens,
            cache_read_tokens=new_usage.cache_read_tokens or self._usage.cache_read_tokens,
            cache_creation_tokens=new_usage.cache_creation_tokens
            or self._usage.cache_creation_tokens,
        )

    @property
    def text_so_far(self) -> str:
        """Text accumulated so far."""
        return "".join(self._text_parts)

    @property
    def accumulated(self) -> ProviderResponse:
        """Build the final ``ProviderResponse`` from accumulated chunks."""
        self._finalize_current_tool_call()

        parts: list[ContentPart] = []

        # Thinking comes first — skip empty thinking
        thinking_text = "".join(self._thinking_parts).strip()
        if thinking_text:
            parts.append(ContentPart(type="thinking", thinking=thinking_text))

        # Text — strip leading whitespace noise from models that emit
        # empty <think></think> blocks before real content
        text = "".join(self._text_parts)
        if self._strip_leading_ws:
            text = text.lstrip()
        if text:
            parts.append(ContentPart(type="text", text=text))

        # Tool calls — merge index-based (OpenAI parallel) and id-based
        all_tool_calls = list(self._tool_calls)
        for _idx in sorted(self._tool_calls_by_index):
            all_tool_calls.append(self._tool_calls_by_index[_idx])

        for tc in all_tool_calls:
            try:
                args = json.loads(tc.get("arguments_json", "{}"))
            except (json.JSONDecodeError, ValueError):
                args = {}
            parts.append(
                ContentPart(
                    type="tool_use",
                    tool_call=ToolCall(
                        id=tc.get("id", ""),
                        name=tc.get("name", ""),
                        arguments=args,
                    ),
                )
            )

        # Use last raw chunk as the representative raw payload (typically
        # contains usage/finish_reason). Falls back to summary if empty.
        raw: dict[str, Any] = self._raw_chunks[-1] if self._raw_chunks else {"streamed_chunks": 0}

        return ProviderResponse(
            provider=self.provider,
            model=self.model,
            raw=raw,
            parts=parts,
            usage=self._usage or UsageInfo(),
            stop_reason=self._stop_reason,
            response_id=self._response_id,
            request_id=self.request_id,
        )
