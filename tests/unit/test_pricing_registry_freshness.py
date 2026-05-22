"""Pricing registry freshness gate (plan §Issue 9).

Plan §Issue 9 acceptance row:

    Pricing currency — every shipped model — pricing entry present
    in ``kaos_llm_client/pricing.py``; no ``cost_usd=0`` for known
    models.

The historical incident (#466) was: ``MODEL_PRICING`` shipped without
the ``gpt-5.4-mini`` row after the SPA default model rotated to it,
so every turn through the SPA reported ``cost_usd=0`` even though the
provider was charging real money. The cost-cap budget gate then
permitted unlimited spend because "current cost is $0".

These tests are a CI gate that catches that regression class:

1. The current-generation model list (the SPA default + the 4 models
   advertised on the live integration tier) MUST have a pricing
   entry. Adding a new default model without updating MODEL_PRICING
   fails the build.
2. Each entry must carry strictly-positive ``input`` and ``output``
   rates — zero is the lie that hid #466.
3. The provider-prefix lookup (``openai:gpt-5.4-mini`` resolves to
   the bare ``gpt-5.4-mini`` row) must work, defending against #466
   where an early version of ``lookup_pricing`` required the caller
   to strip the prefix and silently returned ``None`` if they didn't.
4. ``estimate_call_cost`` on a non-zero token count for any known
   model must return strictly-positive USD — the integration test of
   the pricing gate.
"""

from __future__ import annotations

import pytest

from kaos_llm_client.cost import (
    MODEL_PRICING,
    PRICING_LAST_UPDATED,
    estimate_call_cost,
    lookup_pricing,
)
from kaos_llm_client.types import UsageInfo

# The minimum set of models that MUST have pricing entries today.
# Pinned against the kaos-modules CLAUDE.md "Always use the latest
# model families" guidance + the test_live.py integration matrix
# (OpenAI, Anthropic, Google × 1 cheap each, plus the SPA default).
REQUIRED_MODELS: tuple[tuple[str, str], ...] = (
    # (provider_prefixed_id, bare_model_name)
    ("openai:gpt-5.4-mini", "gpt-5.4-mini"),  # SPA default — #466 root cause
    ("openai:gpt-5.4-nano", "gpt-5.4-nano"),  # cheapest current-gen
    ("openai:gpt-5.4", "gpt-5.4"),
    ("openai:o3", "o3"),
    ("openai:o4-mini", "o4-mini"),
    ("anthropic:claude-haiku-4-5", "claude-haiku-4-5"),  # integration default
    ("anthropic:claude-sonnet-4-6", "claude-sonnet-4-6"),
    ("anthropic:claude-opus-4-7", "claude-opus-4-7"),
    ("google:gemini-2.5-flash", "gemini-2.5-flash"),  # integration default
    ("google:gemini-2.5-pro", "gemini-2.5-pro"),
)


@pytest.mark.unit
def test_pricing_last_updated_is_recent_string() -> None:
    """A reminder string lives in the module so contributors who
    touch ``MODEL_PRICING`` remember to refresh it. We don't gate
    on an exact date (that would flake), but we DO require a
    non-empty YYYY-MM string so the comment block in cost.py stays
    truthful."""
    assert isinstance(PRICING_LAST_UPDATED, str)
    assert len(PRICING_LAST_UPDATED) >= len("2026-05"), (
        f"PRICING_LAST_UPDATED looks malformed: {PRICING_LAST_UPDATED!r}"
    )
    assert PRICING_LAST_UPDATED.startswith("20"), (
        f"PRICING_LAST_UPDATED should start with year: {PRICING_LAST_UPDATED!r}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("_prefixed, bare", REQUIRED_MODELS)
def test_every_required_model_has_pricing_entry(_prefixed: str, bare: str) -> None:
    """Each model in the required set has a row in MODEL_PRICING."""
    assert bare in MODEL_PRICING, (
        f"Model {bare!r} is in the REQUIRED_MODELS set but has no "
        f"entry in MODEL_PRICING. This is the #466 defect class: "
        f"every cost calculation for this model will silently report "
        f"$0, defeating the cost-cap gate. Add the row to "
        f"kaos_llm_client/cost.py and refresh PRICING_LAST_UPDATED."
    )


@pytest.mark.unit
@pytest.mark.parametrize("_prefixed, bare", REQUIRED_MODELS)
def test_required_model_rates_are_positive(_prefixed: str, bare: str) -> None:
    """Both ``input`` and ``output`` rates strictly > 0 — zero is
    the lie that hid #466."""
    entry = MODEL_PRICING[bare]
    assert "input" in entry, f"{bare}: missing 'input' rate"
    assert "output" in entry, f"{bare}: missing 'output' rate"
    assert entry["input"] > 0.0, (
        f"{bare}: 'input' rate is {entry['input']} — must be > 0; "
        f"a $0 input rate masks #466-class regressions"
    )
    assert entry["output"] > 0.0, f"{bare}: 'output' rate is {entry['output']} — must be > 0"


@pytest.mark.unit
@pytest.mark.parametrize("prefixed, _bare", REQUIRED_MODELS)
def test_lookup_pricing_strips_provider_prefix(prefixed: str, _bare: str) -> None:
    """``lookup_pricing('openai:gpt-5.4-mini')`` MUST resolve the
    bare ``gpt-5.4-mini`` row — the historical #466 defect was the
    lookup returning None on the provider-prefixed form, with the
    caller falling back silently to $0 cost.

    The lookup_pricing function should handle the provider prefix
    internally so callers cannot miss."""
    pricing = lookup_pricing(prefixed)
    assert pricing is not None, (
        f"lookup_pricing({prefixed!r}) returned None — this is the "
        f"#466 regression class. Provider-prefixed model ids MUST "
        f"resolve to their bare-name pricing row."
    )
    assert "input" in pricing and "output" in pricing


@pytest.mark.unit
@pytest.mark.parametrize("prefixed, _bare", REQUIRED_MODELS)
def test_estimate_call_cost_strictly_positive_on_real_tokens(prefixed: str, _bare: str) -> None:
    """End-to-end pricing integration: ``estimate_call_cost`` on
    1k input + 1k output tokens for every required model returns
    strictly-positive USD. Zero-cost regression catcher across the
    full lookup + math path, not just the table row.

    Why both this AND the table-row tests: the table can be right
    and the lookup wrong (the #466 shape); the lookup can be right
    and the math wrong (cost rounding to $0 on tiny rates). This
    test catches both."""
    usage = UsageInfo(input_tokens=1_000, output_tokens=1_000, total_tokens=2_000)
    cost = estimate_call_cost(usage, model=prefixed)
    assert cost is not None and cost > 0.0, (
        f"estimate_call_cost({prefixed!r}, 1k+1k tokens) returned "
        f"{cost} — must be > 0. Zero-cost lies defeat the per-tool "
        f"and per-loop cost gates. See #466."
    )


@pytest.mark.unit
def test_spa_default_model_pricing_is_present() -> None:
    """Defense-in-depth: the SPA's documented default model is
    ``openai:gpt-5.4-mini`` per the 2026-05-19 rotation. If the SPA
    default rotates, this test must be updated AT THE SAME TIME as
    MODEL_PRICING — otherwise we silently re-introduce #466.

    A failing assertion here means either (a) MODEL_PRICING dropped
    the row, or (b) the SPA default rotated without updating the
    pricing registry. Both are P0 cost-control failures."""
    pricing = lookup_pricing("openai:gpt-5.4-mini")
    assert pricing is not None
    assert pricing["input"] > 0.0
    assert pricing["output"] > 0.0
    # And the actual SPA-spend math returns dollars, not zeroes.
    usage = UsageInfo(input_tokens=10_000, output_tokens=2_000, total_tokens=12_000)
    spend = estimate_call_cost(usage, model="openai:gpt-5.4-mini")
    assert spend is not None and spend > 0.0
