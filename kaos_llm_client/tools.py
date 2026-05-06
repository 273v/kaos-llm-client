"""MCP tool definitions for LLM client operations.

KaosTool implementations registered with KaosRuntime and exposed via kaos-mcp.
Each tool provides LLM inference capabilities (chat, structured output, embeddings)
across multiple providers.
"""

from __future__ import annotations

import json
from typing import Any

from kaos_core import KaosContext, KaosRuntime, KaosTool, ToolMetadata, ToolResult
from kaos_core.logging import get_logger
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

logger = get_logger("kaos_llm_client.tools")

_MODULE = "kaos-llm"
_VERSION = "0.1.0"


# Canonical structured-log keys used across kaos-llm-client. Mirroring
# the set documented on ``BaseProviderClient._log_extra``: any new log
# call site SHOULD pull from this set so a single grep finds them
# everywhere (Splunk/Datadog/OTel exporters can index them without
# parsing the message string).
#
#   provider, model, request_id, response_id, session_id, trace_id,
#   tool_name, attempt, latency_ms, cache_hit, error, retry_after_s,
#   input_tokens, output_tokens, total_tokens, estimated_usd
def _tool_log_extra(
    context: KaosContext | None,
    *,
    tool_name: str,
    **extra: Any,
) -> dict[str, Any]:
    """Build the ``extra=`` payload for a tool-layer structured log record.

    Pulls ``session_id`` / ``trace_id`` from the supplied
    :class:`KaosContext` so kaos-core's ``ContextFilter`` can attach them
    to the emitted log record. Tool-layer logs do not have a
    ``ProviderRequest`` in scope (the request lives one layer below in
    the provider client), so the only fallback is the context's own
    ``trace_id``.
    """
    session_id: str | None = None
    trace_id: str | None = None
    if context is not None:
        session_id = getattr(context, "session_id", None)
        trace_id = getattr(context, "trace_id", None)

    payload: dict[str, Any] = {
        "session_id": session_id,
        "trace_id": trace_id,
        "tool_name": tool_name,
    }
    payload.update(extra)
    return payload


# LLM tools call external APIs (openWorld) but don't modify anything (readOnly).
_LLM_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
)

# Known providers for error messages.
_KNOWN_PROVIDERS = "openai, anthropic, google, xai, groq, mistral, openrouter, openai-compatible"


async def _store_artifact(
    context: Any,
    data: dict[str, Any],
    *,
    tool_name: str,
    model: str,
) -> dict[str, str] | None:
    """Store a tool result as a session artifact if runtime is available.

    Returns artifact info dict, or None if storage is not available.
    """
    try:
        runtime = context.runtime
        if runtime is None or not hasattr(runtime, "artifacts"):
            return None

        from kaos_core.types.enums import ArtifactRole

        name = f"llm-{tool_name.split('-')[-1]}-{model.replace(':', '-')}"
        vfs_path = f"llm-responses/{name}.json"

        body = json.dumps(data, indent=2).encode()
        vfs = getattr(context, "_vfs", None) or getattr(runtime, "vfs", None)
        if vfs is None:
            return None

        await vfs.write(vfs_path, body, context_id=context.session_id)

        manifest = await runtime.artifacts.create_from_path(
            vfs_path,
            context_id=context.session_id,
            session_id=context.session_id,
            name=name,
            description=f"LLM response from {model} via {tool_name}",
            mime_type="application/json",
            role=ArtifactRole.BODY,
            provenance={"tool": tool_name, "model": model},
        )
        return {
            "artifact_id": str(manifest.artifact_id),
            "uri": str(manifest.uri),
            "name": name,
        }
    except Exception:
        logger.debug(
            "Failed to store artifact",
            exc_info=True,
            extra=_tool_log_extra(
                context,
                tool_name=tool_name,
                model=model,
            ),
        )
        return None


class KaosLLMChatTool(KaosTool):
    """Send a chat message to an LLM provider and get a response."""

    def __init__(self, *, default_model: str | None = None) -> None:
        super().__init__()
        self._default_model = default_model

    @property
    def metadata(self) -> ToolMetadata:
        model_desc = "Model string in 'provider:model' format (e.g., 'openai:gpt-5')."
        if self._default_model:
            model_desc += f" Defaults to '{self._default_model}' if omitted."

        return ToolMetadata(
            name="kaos-llm-chat",
            display_name="LLM Chat",
            description=(
                "Send a chat message to an LLM provider and get a response. "
                "Supports all major providers (OpenAI, Anthropic, Google, xAI, Groq, "
                "Mistral, OpenRouter). Use 'provider:model' format for the model string. "
                "For structured JSON output, use 'kaos-llm-json' instead. "
                "For embeddings, use 'kaos-llm-embed'."
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
                    description="The user message to send to the model.",
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
                    description=(
                        "Sampling temperature (0.0-2.0). Lower = more deterministic, "
                        "higher = more creative. Default varies by provider."
                    ),
                    required=False,
                    constraints={"minimum": 0.0, "maximum": 2.0},
                ),
                ParameterSchema(
                    name="top_p",
                    type="number",
                    description=(
                        "Nucleus sampling: only consider tokens with cumulative "
                        "probability >= top_p (0.0-1.0). Alternative to temperature."
                    ),
                    required=False,
                    constraints={"minimum": 0.0, "maximum": 1.0},
                ),
                ParameterSchema(
                    name="reasoning_effort",
                    type="string",
                    description=(
                        "Reasoning effort for o-series models (o3, o4-mini). "
                        "Values: 'low', 'medium', 'high'. Ignored by non-reasoning models."
                    ),
                    required=False,
                    constraints={"enum": ["low", "medium", "high"]},
                ),
                ParameterSchema(
                    name="service_tier",
                    type="string",
                    description=(
                        "Processing tier for cost/latency tradeoff. "
                        "'flex' = ~50%% cost savings with variable latency (OpenAI, Google). "
                        "'priority' = premium pricing with lowest latency. "
                        "'default' = standard processing. "
                        "Supported by OpenAI (GPT-5, o3, o4-mini) and Google Gemini."
                    ),
                    required=False,
                    constraints={"enum": ["auto", "default", "flex", "priority"]},
                ),
                ParameterSchema(
                    name="provider",
                    type="string",
                    description=(
                        "Override the provider (e.g., 'openai', 'anthropic'). "
                        "Normally inferred from the model string prefix."
                    ),
                    required=False,
                    constraints={
                        "enum": [
                            "openai",
                            "anthropic",
                            "google",
                            "xai",
                            "groq",
                            "mistral",
                            "openrouter",
                            "openai-compatible",
                        ]
                    },
                ),
                ParameterSchema(
                    name="store_artifact",
                    type="boolean",
                    description=(
                        "Store the response as a session artifact. "
                        "When true, the full response is written to VFS and "
                        "discoverable via kaos://session/{session_id}/artifacts. "
                        "Useful for large responses or when you need to reference "
                        "the result later."
                    ),
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
                "Provide the text message to send to the model."
            )

        # Build model string with optional provider override
        provider = inputs.get("provider")
        if provider and ":" not in model:
            model = f"{provider}:{model}"

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

        try:
            from kaos_llm_client.providers import create_client
            from kaos_llm_client.settings import KaosLLMSettings

            settings = KaosLLMSettings()
            client = create_client(model, settings=settings)
            response = await client.chat_async(messages=messages, **kwargs)
        except ImportError as exc:
            return ToolResult.create_error(
                f"Missing provider dependency: {exc}. "
                "Install the required provider package (e.g., pip install openai)."
            )
        except Exception as exc:
            return _format_llm_error(exc, model)

        # Build structured result
        usage_dict = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        if response.usage.reasoning_tokens:
            usage_dict["reasoning_tokens"] = response.usage.reasoning_tokens
        if response.usage.cache_read_tokens:
            usage_dict["cache_read_tokens"] = response.usage.cache_read_tokens

        result_data: dict[str, Any] = {
            "model": response.model,
            "provider": response.provider,
            "text": response.text,
            "usage": usage_dict,
            "stop_reason": response.stop_reason,
            "latency_ms": response.latency_ms,
        }

        logger.info(
            "LLM chat completed: model=%s, tokens=%d",
            response.model,
            response.usage.total_tokens,
            extra=_tool_log_extra(
                context,
                tool_name="kaos-llm-chat",
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

        # Optionally store as artifact
        if inputs.get("store_artifact") and context is not None:
            artifact_info = await _store_artifact(
                context, result_data, tool_name="kaos-llm-chat", model=model
            )
            if artifact_info:
                result_data["artifact"] = artifact_info

        return ToolResult.create_success(
            output=result_data,
            summary=response.text,
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
            from kaos_llm_client.settings import KaosLLMSettings

            settings = KaosLLMSettings()
            client = create_client(model, settings=settings)
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

        logger.info(
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
_LOCAL_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)

# Pricing table now lives in ``kaos_llm_client.cost`` so that
# ``BaseProviderClient`` can emit a per-call USD-cost log without pulling
# in the MCP tool layer. ``_MODEL_PRICING`` is preserved as a local
# alias for back-compat with downstream tests / scripts that imported it
# from this module.
from kaos_llm_client.cost import MODEL_PRICING as _MODEL_PRICING  # noqa: E402
from kaos_llm_client.cost import lookup_pricing as _lookup_pricing  # noqa: E402

# Average characters per token for English text. Most tokenizers (BPE/SentencePiece)
# produce ~3.5-4.5 chars/token for English prose; 4 is a safe middle estimate.
_CHARS_PER_TOKEN_ESTIMATE = 4


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate based on _CHARS_PER_TOKEN_ESTIMATE."""
    return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)


class KaosLLMEmbedTool(KaosTool):
    """Generate text embeddings using an LLM provider."""

    def __init__(self, *, default_model: str | None = None) -> None:
        super().__init__()
        self._default_model = default_model

    @property
    def metadata(self) -> ToolMetadata:
        model_desc = (
            "Embedding model string (e.g., 'openai:text-embedding-3-small'). "
            "Must be an embedding-capable model."
        )
        if self._default_model:
            model_desc += f" Defaults to '{self._default_model}' if omitted."

        return ToolMetadata(
            name="kaos-llm-embed",
            display_name="LLM Embeddings",
            description=(
                "Generate text embeddings from an LLM provider. "
                "Accepts a single string or a list of strings. "
                "Returns embedding vectors suitable for semantic search, "
                "clustering, or similarity comparison. "
                "Requires an embedding-capable model (e.g., OpenAI text-embedding-3-small, "
                "Mistral mistral-embed). Not all providers support embeddings."
            ),
            category=ToolCategory.INTEGRATION,
            capability=ToolCapability.ANALYZE,
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
                    name="input",
                    type="string",
                    description=(
                        "Text to embed. Provide a single string for one embedding, "
                        "or a JSON array of strings for batch embedding."
                    ),
                ),
                ParameterSchema(
                    name="dimensions",
                    type="integer",
                    description=(
                        "Optional output dimensionality. "
                        "Only supported by some models (e.g., OpenAI text-embedding-3-*)."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="store_artifact",
                    type="boolean",
                    description="Store the embeddings as a session artifact.",
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
                "Provide an embedding model string like 'openai:text-embedding-3-small'. "
                "Not all models support embeddings; use an embedding-specific model."
            )

        raw_input = inputs.get("input")
        if not raw_input:
            return ToolResult.create_error(
                "Missing required parameter 'input'. "
                "Provide the text to embed as a string or JSON array of strings."
            )

        # Parse input: accept a string or a JSON array of strings
        embed_input: str | list[str]
        if isinstance(raw_input, list):
            embed_input = raw_input
        elif isinstance(raw_input, str):
            # Try parsing as JSON array
            try:
                parsed = json.loads(raw_input)
                if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
                    embed_input = parsed
                else:
                    embed_input = raw_input
            except (json.JSONDecodeError, ValueError):
                embed_input = raw_input
        else:
            embed_input = str(raw_input)

        kwargs: dict[str, Any] = {}
        dimensions = inputs.get("dimensions")
        if dimensions is not None:
            kwargs["dimensions"] = dimensions

        try:
            from kaos_llm_client.providers import create_client
            from kaos_llm_client.settings import KaosLLMSettings

            settings = KaosLLMSettings()
            client = create_client(model, settings=settings)
            response = await client.embed_async(embed_input, **kwargs)
        except NotImplementedError:
            return ToolResult.create_error(
                f"Model '{model}' does not support embeddings. "
                "Use an embedding-capable model such as 'openai:text-embedding-3-small' "
                "or 'mistral:mistral-embed'."
            )
        except ImportError as exc:
            return ToolResult.create_error(
                f"Missing provider dependency: {exc}. "
                "Install the required provider package (e.g., pip install openai)."
            )
        except Exception as exc:
            return _format_llm_error(exc, model)

        n_embeddings = len(response.embeddings)
        dimensions_actual = len(response.embeddings[0]) if response.embeddings else 0

        usage_dict = {
            "input_tokens": response.usage.input_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        result_data: dict[str, Any] = {
            "model": response.model,
            "provider": response.provider,
            "embeddings": response.embeddings,
            "count": n_embeddings,
            "dimensions": dimensions_actual,
            "usage": usage_dict,
        }

        logger.info(
            "LLM embed completed: model=%s, count=%d, dims=%d",
            response.model,
            n_embeddings,
            dimensions_actual,
            extra=_tool_log_extra(
                context,
                tool_name="kaos-llm-embed",
                provider=response.provider,
                model=response.model,
                request_id=(response.request_id or "")[:16] or None,
                input_tokens=response.usage.input_tokens,
                total_tokens=response.usage.total_tokens,
                embedding_count=n_embeddings,
                embedding_dimensions=dimensions_actual,
            ),
        )

        if inputs.get("store_artifact") and context is not None:
            artifact_info = await _store_artifact(
                context, result_data, tool_name="kaos-llm-embed", model=model
            )
            if artifact_info:
                result_data["artifact"] = artifact_info

        return ToolResult.create_success(
            output=result_data,
            summary=(
                f"Generated {n_embeddings} embedding(s) with {dimensions_actual} dimensions "
                f"using {response.model} ({response.usage.total_tokens} tokens)"
            ),
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
            from kaos_llm_client.settings import KaosLLMSettings

            settings = KaosLLMSettings()
            client = create_client(model, settings=settings)
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
            from kaos_llm_client.settings import KaosLLMSettings

            settings = KaosLLMSettings()
            client = create_client(model, settings=settings)

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

        logger.info(
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


class KaosLLMProviderCheckTool(KaosTool):
    """Check which LLM providers have API keys configured."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-llm-provider-check",
            display_name="LLM Provider Check",
            description=(
                "Check which LLM providers have valid API keys configured. "
                "Does NOT make API calls -- only checks if keys are set in the environment. "
                "Use this to discover available providers before making LLM calls. "
                "For credential verification via actual API call, use 'kaos-llm-chat' "
                "with a simple test message."
            ),
            category=ToolCategory.INTEGRATION,
            capability=ToolCapability.VALIDATE,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_LOCAL_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="providers",
                    type="array",
                    description=(
                        "Optional list of provider names to check "
                        "(e.g., ['openai', 'anthropic']). "
                        "If empty or omitted, checks all known providers."
                    ),
                    required=False,
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        from kaos_llm_client.settings import KaosLLMSettings

        settings = KaosLLMSettings()

        # Map provider name -> (settings field name, base URL field name)
        provider_map: dict[str, tuple[str, str]] = {
            "openai": ("openai_api_key", "openai_base_url"),
            "anthropic": ("anthropic_api_key", "anthropic_base_url"),
            "google": ("google_api_key", "google_base_url"),
            "xai": ("xai_api_key", "xai_base_url"),
            "groq": ("groq_api_key", "groq_base_url"),
            "mistral": ("mistral_api_key", "mistral_base_url"),
            "openrouter": ("openrouter_api_key", "openrouter_base_url"),
        }

        requested = inputs.get("providers")
        if requested and isinstance(requested, list):
            # Validate requested providers
            unknown = [p for p in requested if p not in provider_map]
            if unknown:
                return ToolResult.create_error(
                    f"Unknown provider(s): {', '.join(unknown)}. "
                    f"Known providers: {', '.join(sorted(provider_map))}."
                )
            check_providers = {k: v for k, v in provider_map.items() if k in requested}
        else:
            check_providers = provider_map

        results: list[dict[str, Any]] = []
        configured_count = 0

        for name, (key_field, url_field) in sorted(check_providers.items()):
            key_value = getattr(settings, key_field, None)
            base_url = getattr(settings, url_field, None)
            is_configured = key_value is not None and key_value.get_secret_value() != ""
            if is_configured:
                configured_count += 1
            results.append(
                {
                    "name": name,
                    "configured": is_configured,
                    "base_url": base_url,
                }
            )

        result_data = {
            "providers": results,
            "configured_count": configured_count,
            "total_checked": len(results),
        }

        configured_names = [r["name"] for r in results if r["configured"]]
        if configured_names:
            summary = f"{configured_count} provider(s) configured: {', '.join(configured_names)}"
        else:
            summary = "No providers configured. Set API keys in environment variables."

        logger.info(
            "Provider check: %d/%d configured",
            configured_count,
            len(results),
            extra=_tool_log_extra(
                context,
                tool_name="kaos-llm-provider-check",
                configured_count=configured_count,
                total_checked=len(results),
            ),
        )

        return ToolResult.create_success(output=result_data, summary=summary)


class KaosLLMCostEstimateTool(KaosTool):
    """Estimate token count and cost for an LLM request."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-llm-cost-estimate",
            display_name="LLM Cost Estimate",
            description=(
                "Estimate the token count and cost for an LLM request. "
                "Uses approximate tokenization (~4 chars/token) and known model pricing. "
                "Does NOT make API calls. Useful for cost planning before making requests. "
                "Pricing data covers major models from OpenAI, Anthropic, Google, and xAI. "
                "For actual usage tracking, check the 'usage' field in tool responses."
            ),
            category=ToolCategory.INTEGRATION,
            capability=ToolCapability.ANALYZE,
            module_name=_MODULE,
            version=_VERSION,
            annotations=_LOCAL_ANNOTATIONS,
            input_schema=[
                ParameterSchema(
                    name="model",
                    type="string",
                    description=(
                        "Model name to estimate cost for. "
                        "Use bare model name (e.g., 'gpt-5') or provider:model format "
                        "(e.g., 'openai:gpt-5'). Provider prefix is stripped for pricing lookup."
                    ),
                ),
                ParameterSchema(
                    name="input_text",
                    type="string",
                    description="The input text to estimate token count for.",
                ),
                ParameterSchema(
                    name="max_output_tokens",
                    type="integer",
                    description=(
                        "Expected maximum output tokens (default: 1000). "
                        "Used to estimate output cost."
                    ),
                    required=False,
                    constraints={"minimum": 1},
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        model = inputs.get("model")
        if not model:
            return ToolResult.create_error(
                "Missing required parameter 'model'. "
                "Provide a model name like 'gpt-5' or 'openai:gpt-5'."
            )

        input_text = inputs.get("input_text")
        if not input_text:
            return ToolResult.create_error(
                "Missing required parameter 'input_text'. "
                "Provide the text you plan to send to the model."
            )

        max_output_tokens = inputs.get("max_output_tokens", 1000)

        # Strip provider prefix for pricing lookup
        model_name = model.split(":", 1)[1] if ":" in model else model

        estimated_input_tokens = _estimate_tokens(input_text)
        pricing = _lookup_pricing(model_name)

        if pricing is None:
            known_models = ", ".join(sorted(_MODEL_PRICING.keys()))
            return ToolResult.create_error(
                f"Unknown model '{model_name}' -- no pricing data available. "
                f"Known models: {known_models}. "
                "For models not in the pricing table, check the provider's pricing page."
            )

        input_cost = (estimated_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (max_output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        result_data = {
            "model": model_name,
            "estimated_input_tokens": estimated_input_tokens,
            "max_output_tokens": max_output_tokens,
            "estimated_cost_usd": round(total_cost, 6),
            "pricing_per_1m": {
                "input": pricing["input"],
                "output": pricing["output"],
            },
        }

        summary = (
            f"Estimated cost for {model_name}: ~{estimated_input_tokens} input tokens + "
            f"{max_output_tokens} max output tokens = ${total_cost:.6f} USD"
        )

        logger.info(
            "Cost estimate: model=%s, input_tokens=%d, cost=$%.6f",
            model_name,
            estimated_input_tokens,
            total_cost,
            extra=_tool_log_extra(
                context,
                tool_name="kaos-llm-cost-estimate",
                model=model_name,
                input_tokens=estimated_input_tokens,
                output_tokens=max_output_tokens,
                estimated_usd=round(total_cost, 6),
            ),
        )

        return ToolResult.create_success(output=result_data, summary=summary)


def _format_llm_error(exc: Exception, model: str) -> ToolResult:
    """Format an LLM error into an agent-friendly ToolResult.

    Follows the three-part rule: what went wrong, how to fix it, alternatives.
    """
    from kaos_llm_client.errors import (
        KaosLLMAuthError,
        KaosLLMError,
        KaosLLMProviderError,
        KaosLLMRetryExhaustedError,
    )

    if isinstance(exc, KaosLLMAuthError):
        return ToolResult.create_error(
            f"Authentication failed for model '{model}': {exc}. "
            "Verify that the correct API key is set in environment variables "
            "(e.g., KAOS_LLM_OPENAI_API_KEY or OPENAI_API_KEY). "
            "Run 'kaos-llm-client check' to verify credentials."
        )

    if isinstance(exc, KaosLLMProviderError):
        msg = f"Provider error for model '{model}' (HTTP {exc.status_code}): {exc}. "
        if exc.status_code == 429:
            msg += "Rate limited. Wait a moment and retry, or use a different model."
        elif exc.status_code >= 500:
            msg += "Provider server error. Retry after a moment, or try a different provider."
        else:
            msg += "Check the model name and request parameters."
        if exc.fix:
            msg += f" {exc.fix}"
        return ToolResult.create_error(msg)

    if isinstance(exc, KaosLLMRetryExhaustedError):
        return ToolResult.create_error(
            f"All retry attempts exhausted for model '{model}': {exc}. "
            "The provider may be experiencing issues. "
            "Try again later or use a different provider/model."
        )

    if isinstance(exc, KaosLLMError):
        return ToolResult.create_error(
            f"LLM client error for model '{model}': {exc}. Supported providers: {_KNOWN_PROVIDERS}."
        )

    # Unexpected error
    return ToolResult.create_error(
        f"Unexpected error calling model '{model}': {type(exc).__name__}: {exc}. "
        "Verify the model string format ('provider:model') and that the provider "
        "package is installed."
    )


def register_llm_tools(
    runtime: KaosRuntime,
    *,
    default_model: str | None = None,
) -> int:
    """Register all LLM tools with the runtime. Returns count.

    Args:
        runtime: The KaosRuntime to register tools with.
        default_model: Optional default model string for tools.
            When set, tools can be called without specifying a model.
    """
    from kaos_llm_client.settings import KaosLLMSettings

    runtime.module_settings["llm"] = KaosLLMSettings()

    tools: list[KaosTool] = [
        KaosLLMChatTool(default_model=default_model),
        KaosLLMStructuredOutputTool(default_model=default_model),
        KaosLLMEmbedTool(default_model=default_model),
        KaosLLMToolCallTool(default_model=default_model),
        KaosLLMPydanticTool(default_model=default_model),
        KaosLLMProviderCheckTool(),
        KaosLLMCostEstimateTool(),
    ]
    for tool in tools:
        runtime.tools.register_tool(tool)
    return len(tools)
