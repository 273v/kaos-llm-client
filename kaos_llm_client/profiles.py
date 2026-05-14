"""Model profiles encode provider and model behavioral differences.

Profiles drive structured output strategy, parameter mapping, and schema
transformations. They are small, static, and cheap to maintain.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class StructuredOutputMode(StrEnum):
    """Strategy for getting structured (JSON/Pydantic) output from models."""

    TOOL = "tool"  # return structured data via tool call
    NATIVE = "native"  # provider-native JSON schema (OpenAI response_format)
    PROMPTED = "prompted"  # instruction-based: "return JSON matching this schema"


# ---------------------------------------------------------------------------
# JSON Schema Transformers
# ---------------------------------------------------------------------------


class JsonSchemaTransformer:
    """Base class for provider-specific JSON schema transformations."""

    def __init__(self, schema: dict[str, Any], *, strict: bool = False) -> None:
        self.schema = schema
        self.strict = strict

    def transform(self) -> dict[str, Any]:
        """Return the transformed schema. Base impl returns as-is."""
        return self.schema


class OpenAIJsonSchemaTransformer(JsonSchemaTransformer):
    """OpenAI strict mode: additionalProperties=False, all properties required,
    remove unsupported keys, canonicalize for cache stability.

    Stripped keywords match OpenAI's Structured Outputs "unsupported schemas"
    doc (https://platform.openai.com/docs/guides/structured-outputs) plus the
    keys Pydantic emits for date/time/list fields (``format``, ``default``,
    ``minItems``, ``maxItems``) which OpenAI also rejects.

    The transformed schema is canonicalized via :func:`schema_cache.canonicalize`
    so identical schemas with different input key ordering produce identical
    bytes — letting OpenAI's global schema cache hit across callers.
    """

    # Keywords OpenAI rejects in strict mode. Kept as a class attribute so
    # tests can assert the exact set and callers can extend if a future
    # release allows some of them.
    _STRIPPED_KEYS: frozenset[str] = frozenset(
        {
            # String constraints
            "minLength",
            "maxLength",
            "pattern",
            "patternProperties",
            # Numeric constraints
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            # Array constraints — OpenAI strict rejects these but Pydantic
            # emits them for ``list`` fields with length bounds.
            "minItems",
            "maxItems",
            # Field-level format hints — OpenAI strict does not enforce and
            # rejects the keyword. Pydantic emits ``format: "date"`` /
            # ``"date-time"`` for date-typed fields.
            "format",
            # Default values — OpenAI strict requires every property in
            # ``required``, so defaults are meaningless and rejected.
            "default",
        }
    )

    def transform(self) -> dict[str, Any]:
        if not self.strict:
            return self.schema
        import copy

        from kaos_llm_client.schema_cache import canonicalize

        strict = self._make_strict(copy.deepcopy(self.schema))
        # Canonicalize for provider-side cache stability.
        return canonicalize(strict)

    def _make_strict(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Recursively apply OpenAI strict mode constraints."""
        schema_type = schema.get("type")

        if schema_type == "object":
            schema["additionalProperties"] = False
            # Make all properties required
            if "properties" in schema:
                schema["required"] = list(schema["properties"].keys())
                for prop_schema in schema["properties"].values():
                    if isinstance(prop_schema, dict):
                        self._make_strict(prop_schema)

        elif schema_type == "array" and "items" in schema:
            items = schema["items"]
            if isinstance(items, dict):
                self._make_strict(items)

        # Recurse into composition keywords (anyOf / allOf / oneOf).
        for key in ("anyOf", "allOf", "oneOf"):
            for sub in schema.get(key, ()):
                if isinstance(sub, dict):
                    self._make_strict(sub)

        # Recurse into $defs so ref'd schemas also get strict-ified.
        for sub in schema.get("$defs", {}).values():
            if isinstance(sub, dict):
                self._make_strict(sub)

        # Remove keys unsupported by OpenAI strict mode.
        for unsupported in self._STRIPPED_KEYS:
            schema.pop(unsupported, None)

        return schema


class AnthropicJsonSchemaTransformer(JsonSchemaTransformer):
    """Anthropic structured outputs (``output_config.format``) schema transformer.

    Anthropic's native structured-outputs GA (April 2026) accepts JSON Schema
    on ``output_config.format.schema`` but rejects a subset of keywords
    (https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs):

    - **No recursive schemas** — caller must ensure their schema is acyclic.
    - **No numerical / string constraints** — ``minimum`` / ``maximum`` /
      ``minLength`` / ``maxLength`` / ``pattern`` are rejected.
    - **No ``additionalProperties: true``** on objects — must be ``false`` or
      absent (we set it to ``false`` in strict mode for parity with OpenAI).

    The strip list matches OpenAI's since the overlap is near-total, and we
    canonicalize for cache stability (Anthropic keeps a 24h schema cache).
    """

    # Same set as OpenAI — the two providers reject the same keyword surface.
    _STRIPPED_KEYS: frozenset[str] = OpenAIJsonSchemaTransformer._STRIPPED_KEYS

    def transform(self) -> dict[str, Any]:
        if not self.strict:
            return self.schema
        import copy

        from kaos_llm_client.schema_cache import canonicalize

        strict = self._make_strict(copy.deepcopy(self.schema))
        return canonicalize(strict)

    def _make_strict(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Recursively apply Anthropic strict-mode constraints."""
        schema_type = schema.get("type")

        if schema_type == "object":
            # Anthropic requires additionalProperties to be explicitly false.
            schema["additionalProperties"] = False
            if "properties" in schema:
                # Every property required for deterministic parsing.
                schema["required"] = list(schema["properties"].keys())
                for prop_schema in schema["properties"].values():
                    if isinstance(prop_schema, dict):
                        self._make_strict(prop_schema)

        elif schema_type == "array" and "items" in schema:
            items = schema["items"]
            if isinstance(items, dict):
                self._make_strict(items)

        for key in ("anyOf", "allOf", "oneOf"):
            for sub in schema.get(key, ()):
                if isinstance(sub, dict):
                    self._make_strict(sub)

        for sub in schema.get("$defs", {}).values():
            if isinstance(sub, dict):
                self._make_strict(sub)

        for unsupported in self._STRIPPED_KEYS:
            schema.pop(unsupported, None)

        return schema


class GoogleJsonSchemaTransformer(JsonSchemaTransformer):
    """Google Gemini schema transformer.

    Google's ``generateContent`` JSON schema support has several quirks:
    - ``const`` is not supported -- convert to ``enum: [value]``
    - ``title`` is not supported -- strip it
    - ``format`` is not natively enforced -- convert to a description annotation
    - ``$defs`` / ``$ref`` are not supported -- inline all references
    - ``default`` is not supported -- strip it
    """

    def transform(self) -> dict[str, Any]:
        import copy

        from kaos_llm_client.schema_cache import canonicalize

        schema = copy.deepcopy(self.schema)
        # Resolve all $ref/$defs first, then clean the result
        defs = schema.pop("$defs", None) or schema.pop("definitions", None) or {}
        if defs:
            schema = self._resolve_refs(schema, defs)
        cleaned = self._clean(schema)
        # Canonicalize for provider-side cache stability (Google caches
        # per-request but canonical input still yields stable trace hashes).
        return canonicalize(cleaned)

    def _resolve_refs(self, node: Any, defs: dict[str, Any]) -> Any:
        """Recursively inline ``$ref`` references using the ``$defs`` map."""
        if isinstance(node, dict):
            if "$ref" in node:
                ref_path = node["$ref"]  # e.g. "#/$defs/Address"
                ref_name = ref_path.rsplit("/", 1)[-1]
                if ref_name in defs:
                    import copy

                    resolved = copy.deepcopy(defs[ref_name])
                    # Merge any sibling keys (e.g. description override) onto the resolved def
                    for key, value in node.items():
                        if key != "$ref":
                            resolved[key] = value
                    return self._resolve_refs(resolved, defs)
                # Unknown ref -- drop it, return remaining keys
                result = {k: v for k, v in node.items() if k != "$ref"}
                return {k: self._resolve_refs(v, defs) for k, v in result.items()}
            return {k: self._resolve_refs(v, defs) for k, v in node.items()}
        if isinstance(node, list):
            return [self._resolve_refs(item, defs) for item in node]
        return node

    def _clean(self, node: Any) -> Any:
        """Recursively apply Google-specific schema transformations."""
        if isinstance(node, dict):
            # const -> enum: [value]
            if "const" in node:
                node["enum"] = [node.pop("const")]

            # title -> remove
            node.pop("title", None)

            # default -> remove (Google does not support defaults)
            node.pop("default", None)

            # format -> fold into description
            fmt = node.pop("format", None)
            if fmt:
                existing = node.get("description", "")
                suffix = f"Format: {fmt}"
                node["description"] = f"{existing}  ({suffix})" if existing else suffix

            # Remove any leftover $defs/$ref that might survive
            node.pop("$defs", None)
            node.pop("definitions", None)

            # Recurse into property maps (dict of name -> sub-schema)
            if "properties" in node:
                node["properties"] = {k: self._clean(v) for k, v in node["properties"].items()}

            # Recurse into sub-schemas that are single schema nodes or lists
            for key in (
                "items",
                "additionalProperties",
                "allOf",
                "anyOf",
                "oneOf",
                "not",
                "if",
                "then",
                "else",
                "prefixItems",
            ):
                if key in node:
                    node[key] = self._clean(node[key])

            return node

        if isinstance(node, list):
            return [self._clean(item) for item in node]

        return node


# ---------------------------------------------------------------------------
# ModelProfile
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class ModelProfile:
    """Behavioral profile for a provider/model combination."""

    # Feature support
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_thinking: bool = False
    supports_native_structured_output: bool = False

    # Structured output
    default_structured_output_mode: StructuredOutputMode = StructuredOutputMode.TOOL
    json_schema_transformer: type[JsonSchemaTransformer] | None = None

    # Parameter mapping
    max_tokens_field: str = "max_tokens"
    system_prompt_location: str = "messages"  # "messages", "top_level", "system_instruction"
    requires_max_tokens: bool = False

    # Thinking/reasoning
    thinking_parameter: str | None = None  # "thinking" (Anthropic) or "reasoning" (OpenAI)

    # Streaming
    stream_format: str = "openai_sse"  # "openai_sse", "anthropic_sse", "google_sse"
    strip_leading_whitespace: bool = False  # Qwen3/DeepSeek emit empty think blocks

    # Provider metadata
    provider_name: str = ""
    # Output token budget. Frontier models in 2026 support 64K-200K+ output;
    # 100K is the right default — it lets the model finish a long deliverable
    # and matches what every major provider supports natively (OpenAI gpt-5.x:
    # 128K, OpenAI reasoning o3/o4: 200K, Google gemini-2.5/3: 64K-128K,
    # xAI grok-4: 128K). Anthropic Claude 4.x caps at 64K *without* beta
    # headers — the per-model resolver clamps Anthropic profiles down to
    # 64K, but everywhere else this is the floor we operate from.
    # Historical 4096 default dated to Claude 2 / GPT-3.5 era and silently
    # truncated multi-page deliverables — see the Harvey CoC bench
    # 2026-05-06 truncation incident: `docs/benchmarks/harvey-coc-2026-05-06.json`
    # deliverable cuts off mid-sentence at exactly the 4096-token boundary.
    default_max_tokens: int = 100_000

    def update(self, **kwargs: Any) -> ModelProfile:
        """Return a new profile with the given fields replaced."""
        import dataclasses

        return dataclasses.replace(self, **kwargs)


# ---------------------------------------------------------------------------
# Provider-specific profile subclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class OpenAIModelProfile(ModelProfile):
    """Profile for OpenAI models with OpenAI-specific capabilities."""

    supports_reasoning_effort: bool = False
    supports_strict_mode: bool = True
    supports_response_format: bool = True
    supports_service_tier: bool = True
    supports_audio_output: bool = False
    # Reasoning models (o3, o4, gpt-5.5, gpt-5-thinking) reject the
    # ``temperature`` and ``top_p`` request parameters outright with
    # "temperature is not supported for this model". Standard chat
    # models accept them. The OpenAI-compatible client strips these
    # kwargs from the request body when this flag is ``False``.
    supports_temperature: bool = True


@dataclass(frozen=True, kw_only=True)
class AnthropicModelProfile(ModelProfile):
    """Profile for Anthropic models with Anthropic-specific capabilities."""

    supports_prompt_caching: bool = True
    supports_extended_thinking: bool = False
    max_thinking_budget: int = 32768
    supports_pdf_input: bool = True
    anthropic_version: str = "2023-06-01"


@dataclass(frozen=True, kw_only=True)
class GoogleModelProfile(ModelProfile):
    """Profile for Google models with Google-specific capabilities."""

    supports_grounding: bool = False
    supports_code_execution: bool = False
    google_api_version: str = "v1beta"


# ---------------------------------------------------------------------------
# Provider Profile Constants
# ---------------------------------------------------------------------------


OPENAI_DEFAULT = OpenAIModelProfile(
    supports_vision=True,
    supports_native_structured_output=True,
    default_structured_output_mode=StructuredOutputMode.NATIVE,
    max_tokens_field="max_tokens",
    json_schema_transformer=OpenAIJsonSchemaTransformer,
    provider_name="openai",
    # gpt-5.x supports 128K output; gpt-4o/4.1 support 16K-32K. Default
    # high; specific gpt-4 variants can pin lower per call if needed.
    default_max_tokens=128_000,
)

OPENAI_REASONING = OpenAIModelProfile(
    supports_vision=True,
    supports_thinking=True,
    supports_native_structured_output=True,
    default_structured_output_mode=StructuredOutputMode.NATIVE,
    max_tokens_field="max_completion_tokens",
    thinking_parameter="reasoning",
    json_schema_transformer=OpenAIJsonSchemaTransformer,
    provider_name="openai",
    # Reasoning models (o-series, gpt-5.5) burn output tokens on hidden
    # chain-of-thought before the visible answer. 200K matches the
    # published o3/o4 reasoning ceiling and gives the visible answer
    # plenty of room after thinking.
    default_max_tokens=200_000,
    supports_reasoning_effort=True,
    supports_service_tier=False,
    supports_temperature=False,
)

# Azure-hosted OpenAI chat models reject ``max_tokens`` (the legacy
# chat-completions field) and require ``max_completion_tokens`` even for
# non-reasoning deployments. Azure also does not honour OpenAI's
# ``service_tier`` (flex pricing). Mirrors OPENAI_DEFAULT otherwise.
AZURE_OPENAI_DEFAULT = OpenAIModelProfile(
    supports_vision=True,
    supports_native_structured_output=True,
    default_structured_output_mode=StructuredOutputMode.NATIVE,
    max_tokens_field="max_completion_tokens",
    json_schema_transformer=OpenAIJsonSchemaTransformer,
    provider_name="azure-openai",
    # Azure tracks OpenAI's ceilings for the underlying model.
    default_max_tokens=128_000,
    supports_service_tier=False,
)

ANTHROPIC_DEFAULT = AnthropicModelProfile(
    supports_vision=True,
    supports_thinking=True,
    # Native structured outputs GA April 2026 on Claude 4.x / Haiku 4.5 — use
    # ``output_config.format = {type: "json_schema", schema: ...}``. See
    # ``AnthropicClient._apply_native_json_mode`` for wire details. Falls back
    # to tool-based output for older models via the profile override table.
    supports_native_structured_output=True,
    default_structured_output_mode=StructuredOutputMode.NATIVE,
    json_schema_transformer=AnthropicJsonSchemaTransformer,
    max_tokens_field="max_tokens",
    system_prompt_location="top_level",
    requires_max_tokens=True,
    thinking_parameter="thinking",
    stream_format="anthropic_sse",
    provider_name="anthropic",
    # Anthropic raised the header-free ceiling on Sonnet 4.5/4.6 and
    # Opus 4.7 to 100K output (verified live 2026-05-06). Haiku 4.5
    # still hard-caps at 64K — the per-model resolver pins Haiku down.
    # The 1M-token output beta (`anthropic-beta: max-tokens-1m-*`) is
    # available for deployments that need to go higher, but 100K is
    # the right floor for everyday work — long deliverables fit
    # comfortably in this budget.
    default_max_tokens=100_000,
    supports_extended_thinking=True,
)

# Legacy Anthropic profile for pre-4.x models that do not support
# ``output_config.format``. Kept exported for tests + caller pinning.
# Claude 3.5/3.7 supported 8K output; Claude 3 supported 4K. We pick
# 8K as the safe value that works across all 3.x models — callers
# pinning to 3.x can override per-call.
ANTHROPIC_TOOL_FALLBACK = AnthropicModelProfile(
    supports_vision=True,
    supports_thinking=True,
    supports_native_structured_output=False,
    default_structured_output_mode=StructuredOutputMode.TOOL,
    max_tokens_field="max_tokens",
    system_prompt_location="top_level",
    requires_max_tokens=True,
    thinking_parameter="thinking",
    stream_format="anthropic_sse",
    provider_name="anthropic",
    default_max_tokens=8_192,
    supports_extended_thinking=True,
)

GOOGLE_DEFAULT = GoogleModelProfile(
    supports_vision=True,
    supports_native_structured_output=True,
    default_structured_output_mode=StructuredOutputMode.NATIVE,
    max_tokens_field="maxOutputTokens",
    system_prompt_location="system_instruction",
    json_schema_transformer=GoogleJsonSchemaTransformer,
    stream_format="google_sse",
    provider_name="google",
    # Gemini 2.0 / 1.5 supported 8K-32K output; this default also covers
    # generic Gemini callers. 2.5+ models pick up GOOGLE_THINKING (below).
    default_max_tokens=100_000,
)

OPENAI_COMPATIBLE_DEFAULT = ModelProfile(
    supports_vision=False,
    supports_native_structured_output=False,
    default_structured_output_mode=StructuredOutputMode.PROMPTED,
    max_tokens_field="max_tokens",
    provider_name="openai-compatible",
    # Generic OpenAI-compatible (vLLM, Ollama, LiteLLM, etc.). Many
    # serve frontier models too — default high.
    default_max_tokens=100_000,
)

XAI_DEFAULT = ModelProfile(
    supports_vision=True,
    supports_native_structured_output=False,
    default_structured_output_mode=StructuredOutputMode.TOOL,
    max_tokens_field="max_tokens",
    provider_name="xai",
    # grok-2/3 caps. grok-4 picks up XAI_GROK4 below.
    default_max_tokens=32_768,
)


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------

# Gemini 2.5+ supports thinking
GOOGLE_THINKING = GoogleModelProfile(
    supports_vision=True,
    supports_thinking=True,
    supports_native_structured_output=True,
    default_structured_output_mode=StructuredOutputMode.NATIVE,
    max_tokens_field="maxOutputTokens",
    system_prompt_location="system_instruction",
    json_schema_transformer=GoogleJsonSchemaTransformer,
    stream_format="google_sse",
    thinking_parameter="thinkingConfig",
    provider_name="google",
    # Gemini 2.5/3 thinking models support 64K-1M output. Default high.
    default_max_tokens=100_000,
)

# Grok 4 supports builtin tools
XAI_GROK4 = ModelProfile(
    supports_vision=True,
    supports_tools=True,
    supports_native_structured_output=True,
    default_structured_output_mode=StructuredOutputMode.TOOL,
    max_tokens_field="max_tokens",
    provider_name="xai",
    # Grok 4 supports 128K output.
    default_max_tokens=131_072,
)


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------

# Provider defaults
_PROVIDER_PROFILES: dict[str, ModelProfile] = {
    "openai": OPENAI_DEFAULT,
    "openai-responses": OPENAI_DEFAULT,
    "anthropic": ANTHROPIC_DEFAULT,
    "google": GOOGLE_DEFAULT,
    "xai": XAI_DEFAULT,
    "groq": OPENAI_DEFAULT,
    "mistral": OPENAI_DEFAULT,
    "openrouter": OPENAI_COMPATIBLE_DEFAULT,
    "openai-compatible": OPENAI_COMPATIBLE_DEFAULT,
}


# ---------------------------------------------------------------------------
# Per-provider resolver functions
# ---------------------------------------------------------------------------


def _resolve_openai_profile(model: str) -> ModelProfile:
    """Resolve profile for OpenAI models.

    Output-token ceilings as of 2026:
    - o1 / o3 / o4 reasoning: 200K (most allotted to hidden CoT)
    - gpt-5.5 (reasoning): 200K
    - gpt-5.x (non-reasoning): 128K
    - gpt-4.1: 32K
    - gpt-4o: 16K
    - gpt-4 / gpt-3.5: 4K-8K (legacy)
    """
    if model.startswith(("o1", "o3", "o4", "gpt-5.5")):
        return OPENAI_REASONING  # 200K
    if model.startswith("gpt-5"):
        return OPENAI_DEFAULT  # 128K
    if model.startswith("gpt-4.1"):
        return OPENAI_DEFAULT.update(default_max_tokens=32_768)
    if model.startswith("gpt-4o"):
        return OPENAI_DEFAULT.update(default_max_tokens=16_384)
    if model.startswith("gpt-4"):
        return OPENAI_DEFAULT.update(default_max_tokens=8_192)
    if model.startswith("gpt-3.5"):
        return OPENAI_DEFAULT.update(default_max_tokens=4_096)
    return OPENAI_DEFAULT


def _resolve_azure_openai_profile(model: str) -> ModelProfile:
    """Resolve profile for Azure-hosted OpenAI deployments.

    Azure rejects ``max_tokens`` for current chat-completions models, so
    every deployment uses ``max_completion_tokens``. Reasoning models still
    pick up the reasoning-specific profile.
    """
    if model.startswith(("o1", "o3", "o4", "gpt-5.5")):
        return OPENAI_REASONING
    return AZURE_OPENAI_DEFAULT


def _resolve_anthropic_profile(model: str) -> ModelProfile:
    """Resolve profile for Anthropic models.

    Output-token ceilings as of 2026-05 (verified live, no beta headers):
    - Claude Sonnet 4.5+ / Opus 4.5+: 100K output (header-free)
    - Claude Sonnet 4 (May 2025 dated): 64K output (legacy)
    - Claude Haiku 4.5: 64K output (hard cap below Sonnet/Opus)
    - Claude 3.x family: 8K output (Sonnet 3.5/3.7) or 4K (3.0)
    - Claude 2.x: 4K output (legacy)

    The 1M-token output beta header (`anthropic-beta:
    max-tokens-1m-*`) is available for deployments that need to go
    higher than 100K — flip it on at call time.
    """
    # Sonnet 4.5+ and Opus 4.5+ — 100K output (verified live).
    # Strict prefix match: only models we've verified accept 100K
    # without a beta header. Older dated 4.x models (e.g.
    # ``claude-sonnet-4-20250514``) still cap at 64K and 400 if we
    # send 100K, so they fall through to the conservative default.
    if model.startswith(
        (
            "claude-sonnet-4-5",
            "claude-sonnet-4-6",
            "claude-sonnet-4-7",
            "claude-sonnet-4-8",
            "claude-sonnet-4-9",
            "sonnet-4-5",
            "sonnet-4-6",
            "sonnet-4-7",
            "sonnet-4-8",
            "sonnet-4-9",
            "claude-opus-4-5",
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-4-9",
            "opus-4-5",
            "opus-4-6",
            "opus-4-7",
            "opus-4-8",
            "opus-4-9",
        )
    ):
        return ANTHROPIC_DEFAULT  # 100K

    # Other Claude 4.x (Haiku 4.x, dated Sonnet/Opus 4.0) — 64K cap.
    if model.startswith(
        (
            "claude-haiku-4",
            "claude-sonnet-4",
            "claude-opus-4",
            "haiku-4",
            "sonnet-4",
            "opus-4",
        )
    ):
        return ANTHROPIC_DEFAULT.update(default_max_tokens=64_000)

    # Claude 3.5 / 3.7 — 8K output, no native structured output
    if model.startswith(
        (
            "claude-3-5",
            "claude-3.5",
            "claude-3-7",
            "claude-3.7",
        )
    ):
        return ANTHROPIC_TOOL_FALLBACK  # 8K, tool-based JSON

    # Claude 3.0 / 2.x — 4K output legacy
    if model.startswith(("claude-3", "claude-2")):
        return ANTHROPIC_TOOL_FALLBACK.update(default_max_tokens=4_096)

    # Unknown claude-* model — default to current-gen defaults
    return ANTHROPIC_DEFAULT


def _resolve_google_profile(model: str) -> ModelProfile:
    """Resolve profile for Google models.

    Output-token ceilings as of 2026:
    - Gemini 3.x: 64K-128K output
    - Gemini 2.5 Pro: 64K output (with thinking)
    - Gemini 2.0: 8K output
    - Gemini 1.5: 8K output
    """
    if model.startswith("gemini-3"):
        return GOOGLE_THINKING  # 64K
    if model.startswith("gemini-2.5"):
        return GOOGLE_THINKING  # 64K
    if model.startswith("gemini-2.0"):
        return GOOGLE_DEFAULT.update(default_max_tokens=8_192)
    if model.startswith("gemini-1"):
        return GOOGLE_DEFAULT.update(default_max_tokens=8_192)
    return GOOGLE_DEFAULT


def _resolve_xai_profile(model: str) -> ModelProfile:
    """Resolve profile for xAI models.

    Output-token ceilings as of 2026:
    - grok-4 / grok-4-1 / grok-4.20: 128K output
    - grok-3: 16K output
    - grok-2: 8K output
    """
    if model.startswith("grok-4"):
        return XAI_GROK4  # 128K
    if model.startswith("grok-3"):
        return XAI_DEFAULT.update(default_max_tokens=16_384)
    if model.startswith("grok-2"):
        return XAI_DEFAULT.update(default_max_tokens=8_192)
    return XAI_DEFAULT


_PROVIDER_RESOLVERS: dict[str, Callable[[str], ModelProfile]] = {
    "openai": _resolve_openai_profile,
    "openai-responses": _resolve_openai_profile,
    # ``azure:``/``azure-openai:`` (chat completions) need ``max_completion_tokens``
    # (Azure rejects ``max_tokens`` on current chat models). The dedicated
    # AZURE_OPENAI_DEFAULT profile encodes that + ``service_tier=False``.
    "azure": _resolve_azure_openai_profile,
    "azure-openai": _resolve_azure_openai_profile,
    # ``azure-responses:``/``azure-foundry:`` (Responses API) use the standard
    # OpenAI profile resolution: parameter is ``max_output_tokens`` regardless,
    # so the chat-completions ``max_tokens_field`` rename does not apply.
    "azure-responses": _resolve_openai_profile,
    "azure-foundry": _resolve_openai_profile,
    # AWS Bedrock — OpenAI-compatible Responses API. Same Responses-API wire
    # shape as direct OpenAI, so the standard resolver applies. Bedrock
    # model ids (e.g. ``openai.gpt-oss-120b``) don't match the openai prefix
    # rules in ``infer_provider`` — the explicit ``bedrock:`` prefix is
    # required.
    "bedrock": _resolve_openai_profile,
    "anthropic": _resolve_anthropic_profile,
    "google": _resolve_google_profile,
    "xai": _resolve_xai_profile,
    "groq": lambda model: OPENAI_DEFAULT,
    "mistral": lambda model: OPENAI_DEFAULT,
}

# Model name → provider inference patterns.
# Covers the full model landscape as of May 2026:
# - OpenAI: gpt-5.5 (reasoning), gpt-5.4/5.3/5.2/5/4.1/4o/4/3.5, o1/o3/o4,
#            chatgpt-*, computer-use-*, gpt-image-*, gpt-realtime-*, davinci/babbage
# - Anthropic: claude-sonnet/opus/haiku 3/3.5/3.7/4/4.1/4.5/4.6, claude-opus-4-7
# - Google: gemini-3.1/3/2.5/2.0/1.5/1.0, imagen-*, veo-*
# - xAI: grok-4.20/4-1/4/3/2, grok-beta, grok-imagine-*
_MODEL_PREFIXES: dict[str, str] = {
    "gpt-": "openai",
    "chatgpt-": "openai",
    "computer-use-": "openai",
    "codex-": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "davinci": "openai",
    "babbage": "openai",
    "claude-": "anthropic",
    "gemini-": "google",
    "imagen": "google",
    "veo-": "google",
    "grok-": "xai",
    "llama-": "groq",
    "mixtral-": "mistral",
    "mistral-": "mistral",
}


def resolve_profile(provider: str, model: str) -> ModelProfile:
    """Resolve the best profile for a provider/model combination.

    Uses per-provider resolver functions for model-specific overrides,
    then falls back to provider defaults. Unknown providers get
    ``OPENAI_COMPATIBLE_DEFAULT``.
    """
    resolver = _PROVIDER_RESOLVERS.get(provider)
    if resolver is not None:
        return resolver(model)

    return _PROVIDER_PROFILES.get(provider, OPENAI_COMPATIBLE_DEFAULT)


def infer_provider(model: str) -> str | None:
    """Infer provider from model name patterns.

    Recognizes all major model families:
    - OpenAI: gpt-5.5/5.4/5/4.1/4o/4, o1/o3/o4, chatgpt-*
    - Anthropic: claude-sonnet/opus/haiku 3/3.5/3.7/4/4.5/4.6, claude-opus-4-7
    - Google: gemini-1/1.5/2/2.5/3
    - xAI: grok-2/3/4

    Returns the provider string or None if unrecognized.
    """
    for prefix, provider in _MODEL_PREFIXES.items():
        if model.startswith(prefix):
            return provider
    return None
