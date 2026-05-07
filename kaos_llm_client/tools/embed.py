"""KaosLLMEmbedTool — extracted from tools.py per audit-01 KLC-03."""

from __future__ import annotations

import json
from typing import Any

from kaos_core import KaosContext, KaosTool, ToolMetadata, ToolResult
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.parameters import ParameterSchema

from kaos_llm_client.tools._common import (
    _LLM_ANNOTATIONS,
    _MODULE,
    _VERSION,
    _format_llm_error,
    _store_artifact,
    _tool_log_extra,
    logger,
)

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

            # Pass `context=` (not `settings=`) so KaosLLMSettings.from_context
            # is invoked inside BaseProviderClient and any KaosContext._config
            # overrides win over env vars (KLC-01 fix). See providers/base.py.
            client = create_client(model, context=context)
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
