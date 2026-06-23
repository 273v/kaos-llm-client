"""Per-call cost estimation for LLM provider responses.

This module owns the model-pricing table and the small helpers that turn
a :class:`UsageInfo` into a USD estimate. It exists separately from
``tools.py`` so that ``BaseProviderClient`` can emit a cost-tagged
"LLM call complete" structured log on every successful request without
pulling in the full MCP tool layer (which depends on kaos-core's
``KaosTool`` / ``ToolMetadata`` machinery).

The pricing values come from each provider's public pricing page as of
:data:`PRICING_LAST_UPDATED`. Update the table when adding new models or
after a provider price change. The schema is deliberately minimal —
``input`` / ``output`` USD-per-million-tokens — so the lookup path stays
short on the hot path of every chat completion.

OpenTelemetry alignment
-----------------------

The structured-log keys emitted alongside the cost estimate are loosely
aligned with the OpenTelemetry ``gen_ai.*`` semantic conventions:

- ``gen_ai.usage.input_tokens`` → ``input_tokens``
- ``gen_ai.usage.output_tokens`` → ``output_tokens``
- ``gen_ai.request.model`` → ``model``
- ``gen_ai.system`` → ``provider``

We DO NOT prefix our keys with ``gen_ai.`` because the rest of kaos-core
log records use flat field names; an OTel exporter sitting downstream
can map either way. The mapping is documented here so future tooling
doesn't have to guess.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaos_llm_client.types import UsageInfo

logger = logging.getLogger("kaos.llm_client.cost")


# STALENESS WARNING: Update this table when adding new models or after provider
# pricing changes. Check provider pricing pages for current rates.
PRICING_LAST_UPDATED = "2026-05"

# Approximate pricing per 1M tokens (USD) as of :data:`PRICING_LAST_UPDATED`.
# Used by both ``BaseProviderClient`` (for the per-call cost log) and the
# ``kaos-llm-cost-estimate`` MCP tool.
#
# Schema (per model):
#   input            base per-token rate for fresh (non-cached) input
#   output           base per-token rate for generated output
#   cache_read       (optional) discounted rate for prompt-cache hits
#   cache_creation   (optional) premium rate for prompt-cache writes
#                     (Anthropic 5-min ephemeral; GPT-5.5 1-hour; etc.)
#
# When a per-model entry omits ``cache_read`` / ``cache_creation`` the
# ``input`` rate is used as a conservative upper bound — see
# :func:`estimate_call_cost`. Anthropic Claude documents read at 0.1x and
# 5-minute writes at 1.25x of the base input rate; GPT-5.5 documents
# cache reads at $0.50 / MTok with no separate write rate (writes free).
# Sources captured 2026-05 — refresh ``PRICING_LAST_UPDATED`` when these
# rows are touched.
MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI — gpt-5.5 explicitly publishes a cached-input rate; treat
    # cache writes as free (no documented premium tier).
    "gpt-5.5": {
        "input": 5.00,
        "output": 30.00,
        "cache_read": 0.50,
        "cache_creation": 5.00,
    },
    "gpt-5.4": {"input": 2.50, "output": 10.00},
    "gpt-5.4-mini": {"input": 0.40, "output": 1.60},
    "gpt-5.4-nano": {"input": 0.10, "output": 0.40},
    "gpt-5": {"input": 2.00, "output": 8.00},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "o3": {"input": 2.00, "output": 8.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    # Anthropic — claude-opus-4-7 publishes 5m / 1h cache write rates +
    # cache-hit rate. v1 collapses both write tiers into a single
    # ``cache_creation`` entry using the 5m rate (the more common case).
    "claude-opus-4-7": {
        "input": 5.00,
        "output": 25.00,
        "cache_read": 0.50,
        "cache_creation": 6.25,  # 5m write rate; 1h is $10
    },
    "claude-sonnet-4-6": {
        "input": 3.00,
        "output": 15.00,
        "cache_read": 0.30,
        "cache_creation": 3.75,
    },
    "claude-opus-4-6": {
        "input": 15.00,
        "output": 75.00,
        "cache_read": 1.50,
        "cache_creation": 18.75,
    },
    "claude-haiku-4-5": {
        "input": 0.80,
        "output": 4.00,
        "cache_read": 0.08,
        "cache_creation": 1.00,
    },
    "claude-sonnet-4-5": {
        "input": 3.00,
        "output": 15.00,
        "cache_read": 0.30,
        "cache_creation": 3.75,
    },
    # Google
    "gemini-3.1-pro-preview": {"input": 1.25, "output": 10.00},
    "gemini-3-flash-preview": {"input": 0.15, "output": 0.60},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    # xAI
    "grok-3": {"input": 3.00, "output": 15.00},
    "grok-3-mini": {"input": 0.30, "output": 0.50},
}


def _strip_provider_prefix(model: str) -> str:
    """Return the bare model name with any ``provider:`` prefix removed.

    ``openai:gpt-5.4-mini`` → ``gpt-5.4-mini``. Idempotent.

    Plan §8 (Pricing registry) of
    ``kaos-modules/docs/plans/2026-05-19-lateral-redesign-capability-layer.md``:
    previously callers were "responsible" for stripping the prefix,
    which produced the documented #466 defect where
    ``lookup_pricing('openai:gpt-5.4-mini')`` returned ``None`` even
    though ``MODEL_PRICING`` had ``gpt-5.4-mini``. The lookup now
    strips the prefix internally so callers cannot silently miss.
    """
    if ":" in model:
        return model.split(":", 1)[1]
    return model


def lookup_pricing(
    model: str,
    *,
    pricing_table: dict[str, dict[str, float]] | None = None,
) -> dict[str, float] | None:
    """Look up pricing for a model, trying exact match then prefix match.

    The prefix fallback handles versioned model identifiers — e.g.,
    ``gpt-5-0125`` is priced as ``gpt-5``. Longest-prefix wins so
    ``gpt-4.1-mini`` is never confused with ``gpt-4.1``.

    Args:
        model: Either the bare model name (e.g., ``gpt-5``) or a
            ``provider:model`` form (e.g., ``openai:gpt-5.4-mini``).
            The function strips the provider prefix internally before
            matching, so callers can pass either form safely.
        pricing_table: Optional override of the default
            :data:`MODEL_PRICING` table. Lets tests pin pricing without
            mutating the module-level dict.

    Returns:
        A ``{"input": float, "output": float}`` mapping, or ``None`` when
        the model is not in the table.

    Resolution order:
        1. Exact match on ``model`` as provided
        2. Exact match on ``model`` with any ``provider:`` prefix
           stripped
        3. Longest-prefix match against the stripped name
        4. ``None``
    """
    table = pricing_table if pricing_table is not None else MODEL_PRICING
    if model in table:
        return table[model]
    stripped = _strip_provider_prefix(model)
    if stripped != model and stripped in table:
        return table[stripped]
    # Prefix match — sort by descending length so longer prefixes win.
    # Match against the stripped name only: ``openai:gpt-5.4-mini``
    # should prefix-match ``gpt-5.4-mini``, not the ``openai:`` portion.
    table_keys: list[str] = list(table.keys())
    table_keys.sort(key=lambda k: -len(k))
    for key in table_keys:
        if stripped.startswith(key):
            return table[key]
    return None


def estimate_call_cost(
    usage: UsageInfo | None,
    model: str,
    *,
    pricing_table: dict[str, dict[str, float]] | None = None,
) -> float | None:
    """Estimate USD cost for a single LLM call.

    Args:
        usage: Parsed token usage. ``None`` returns ``0.0`` — the caller
            had no usage to bill against (likely a cached or no-op
            response). We deliberately do NOT return ``None`` for missing
            usage because that is indistinguishable from "unknown
            pricing" downstream.
        model: The bare model name. Provider prefixes (``openai:``,
            ``anthropic:``) must be stripped by the caller.
        pricing_table: Optional pricing-table override; see
            :func:`lookup_pricing`.

    Returns:
        Estimated USD cost as a ``float``, or ``None`` when the model is
        not in the pricing table. Callers should treat ``None`` as
        "unknown" and emit ``estimated_usd=null`` in their structured
        log so dashboards can spot pricing-table gaps.

    Notes:
        When the model entry includes ``cache_read`` / ``cache_creation``
        rates, ``UsageInfo.cache_read_tokens`` and
        ``UsageInfo.cache_creation_tokens`` are billed at those rates and
        SUBTRACTED from ``input_tokens`` so cache reads / writes aren't
        double-counted. When a model has no published cache rates, the
        cache token columns are billed at the base ``input`` rate (the
        existing upper-bound behaviour) — the result still beats
        ignoring cache tokens entirely.
    """
    if usage is None:
        return 0.0

    # Strip provider prefix defensively — many call sites pass
    # ``self.model`` which can include a colon for synthetic models.
    bare_model = model.split(":", 1)[1] if ":" in model else model
    pricing = lookup_pricing(bare_model, pricing_table=pricing_table)
    if pricing is None:
        return None

    input_tokens = max(0, getattr(usage, "input_tokens", 0) or 0)
    output_tokens = max(0, getattr(usage, "output_tokens", 0) or 0)
    cache_read = max(0, getattr(usage, "cache_read_tokens", 0) or 0)
    cache_creation = max(0, getattr(usage, "cache_creation_tokens", 0) or 0)

    # Cache-token columns are reported by providers as a SUBSET of the
    # ``input_tokens`` count (the usage record carries both, and input
    # already covers cache reads). Subtract them out so we don't bill
    # the same token twice.
    fresh_input = max(0, input_tokens - cache_read - cache_creation)

    input_rate = pricing["input"]
    output_rate = pricing["output"]
    cache_read_rate = pricing.get("cache_read", input_rate)
    cache_creation_rate = pricing.get("cache_creation", input_rate)

    fresh_cost = (fresh_input / 1_000_000.0) * input_rate
    output_cost = (output_tokens / 1_000_000.0) * output_rate
    cache_read_cost = (cache_read / 1_000_000.0) * cache_read_rate
    cache_creation_cost = (cache_creation / 1_000_000.0) * cache_creation_rate
    total = fresh_cost + output_cost + cache_read_cost + cache_creation_cost
    return round(total, 8)


__all__ = [
    "MODEL_PRICING",
    "PRICING_LAST_UPDATED",
    "apply_pricing_overlay",
    "estimate_call_cost",
    "load_pricing_overlay",
    "lookup_pricing",
]


# Plan §8 — pricing as a hot-reloadable registry.
#
# In addition to the baked-in MODEL_PRICING dict, callers can ship an
# overlay JSON file that adds or overrides entries without a release.
# This lets the table catch up with model launches between releases.
#
# Overlay file format (mirrors MODEL_PRICING shape):
#
#   {
#       "gpt-5.4-nano": {"input": 0.10, "output": 0.40},
#       "claude-haiku-5": {"input": 0.50, "output": 2.50}
#   }
#
# Resolution: when ``KAOS_LLM_PRICING_OVERLAY_PATH`` is set in the
# environment, :func:`apply_pricing_overlay` is invoked at module import
# time to merge the overlay into MODEL_PRICING. Existing keys are
# replaced; new keys are added. Unknown / malformed file is a WARN log,
# not a fatal error — production should never crash on a bad overlay.
_OVERLAY_ENV_VAR = "KAOS_LLM_PRICING_OVERLAY_PATH"


def load_pricing_overlay(path: str) -> dict[str, dict[str, float]]:
    """Parse a pricing overlay JSON file.

    Args:
        path: Filesystem path to a JSON file mapping model name to a
            ``{"input": float, "output": float, ...}`` dict.

    Returns:
        The parsed overlay dict. Empty when the file is missing,
        unreadable, or malformed.
    """
    try:
        with Path(path).open(encoding="utf-8") as f:
            raw = json.load(f)
    except (FileNotFoundError, PermissionError) as exc:
        logger.warning(
            "Pricing overlay path %r is not readable (%s); skipping overlay",
            path,
            exc,
        )
        return {}
    except json.JSONDecodeError as exc:
        logger.warning(
            "Pricing overlay at %r is not valid JSON (%s); skipping overlay",
            path,
            exc,
        )
        return {}
    if not isinstance(raw, dict):
        logger.warning(
            "Pricing overlay at %r is not a JSON object; got %s",
            path,
            type(raw).__name__,
        )
        return {}
    # Validate each entry is a dict with at least input/output floats.
    validated: dict[str, dict[str, float]] = {}
    for model, rates in raw.items():
        if not isinstance(model, str) or not isinstance(rates, dict):
            continue
        try:
            cleaned = {k: float(v) for k, v in rates.items() if isinstance(v, int | float)}
        except (TypeError, ValueError):
            continue
        if "input" not in cleaned or "output" not in cleaned:
            continue
        validated[model] = cleaned
    return validated


def apply_pricing_overlay(
    overlay: dict[str, dict[str, float]] | None = None,
    *,
    target: dict[str, dict[str, float]] | None = None,
) -> int:
    """Merge an overlay into a target pricing dict.

    Args:
        overlay: Mapping of model name to rate dict. When ``None``,
            loads from the path in ``KAOS_LLM_PRICING_OVERLAY_PATH`` if
            set, else no-op.
        target: Destination dict. Defaults to :data:`MODEL_PRICING`
            (mutates module state).

    Returns:
        Count of entries merged (added or replaced).
    """
    if overlay is None:
        path = os.environ.get(_OVERLAY_ENV_VAR)
        if not path:
            return 0
        overlay = load_pricing_overlay(path)
    if not overlay:
        return 0
    if target is None:
        target = MODEL_PRICING
    target.update(overlay)
    logger.debug(
        "Pricing overlay merged: %d models (added/replaced)",
        len(overlay),
    )
    return len(overlay)


# Auto-apply on import when the env var is set. Safe to be no-op when
# the env var is absent — and safe to call multiple times because dict
# update is idempotent on identical content.
apply_pricing_overlay()
