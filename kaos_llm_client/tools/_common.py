"""Shared helpers + module-level constants for the kaos-llm-client tool layer.

Extracted from the historical monolithic ``kaos_llm_client/tools.py`` as
part of audit-01 KLC-03. Each tool class lives in its own sibling module
under ``kaos_llm_client.tools.*`` and imports the helpers it needs from
this private module.

Public names from the historical ``tools.py`` (the seven tool classes,
``register_llm_tools``, ``_estimate_tokens``, ``_lookup_pricing``) remain
reachable via :mod:`kaos_llm_client.tools` re-exports — see
``tools/__init__.py``.
"""

from __future__ import annotations

import json
from typing import Any

from kaos_core import KaosContext, ToolResult
from kaos_core.logging import get_logger
from kaos_core.types.annotations import ToolAnnotations

# Pricing table lives in ``kaos_llm_client.cost`` so that
# ``BaseProviderClient`` can emit a per-call USD-cost log without pulling
# in the MCP tool layer. ``_MODEL_PRICING`` is preserved as a local
# alias for back-compat with downstream tests / scripts that imported it
# from this module.
from kaos_llm_client.cost import MODEL_PRICING as _MODEL_PRICING  # noqa: F401
from kaos_llm_client.cost import lookup_pricing as _lookup_pricing  # noqa: F401

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

# Local / read-only tools (provider-check, cost-estimate) — do not call out.
_LOCAL_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
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
