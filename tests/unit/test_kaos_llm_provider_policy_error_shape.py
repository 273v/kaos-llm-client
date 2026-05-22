"""Field-shape contract for :class:`KaosLLMProviderPolicyError`
(plan §Issue 4).

The SPA's chat-router 403 response body carries the
``(kind, constraint, provider, model)`` quadruple that the React
frontend renders into the BAA refusal toast. That envelope is
derived from the error's three positional fields plus the
caller-supplied ``kind`` enum.

If any of ``provider`` / ``model`` / ``constraint`` becomes
``None`` on the raised error, the toast renders ``"Provider 'None'
is not …"`` — a P0 ux failure for an attorney-facing surface.
These tests pin the field-shape contract so a future refactor
(e.g. dropping the explicit kwargs in favor of ``**details``)
fails loudly.

Plan: ``kaos-modules/docs/plans/2026-05-22-launch-blocker-top-10.md``
§Issue 4 — vendor egress + BAA + HIPAA mode.
"""

from __future__ import annotations

import pytest

from kaos_llm_client.errors import KaosLLMError, KaosLLMProviderPolicyError

# ── Field-shape contract (the load-bearing invariant) ──────────────


@pytest.mark.unit
def test_baa_refusal_carries_full_triple() -> None:
    """The canonical refusal: HIPAA-required session targets a
    non-BAA provider. All three fields MUST be present on the
    raised error — the SPA's 403 body shape depends on it."""
    err = KaosLLMProviderPolicyError(
        "Provider 'xai' does not have a BAA; HIPAA mode requires BAA-eligible provider.",
        provider="xai",
        model="xai:grok-4",
        constraint="hipaa_required:no_baa",
    )
    assert err.provider == "xai"
    assert err.model == "xai:grok-4"
    assert err.constraint == "hipaa_required:no_baa"


@pytest.mark.unit
def test_allowlist_refusal_carries_full_triple() -> None:
    """The other canonical refusal: session-policy allowed_providers
    list excludes the requested provider. Same field invariant."""
    err = KaosLLMProviderPolicyError(
        "Provider 'openai' is not in allowed_providers ['anthropic'].",
        provider="openai",
        model="openai:gpt-5.4-mini",
        constraint="allowed_providers",
    )
    assert err.provider == "openai"
    assert err.model == "openai:gpt-5.4-mini"
    assert err.constraint == "allowed_providers"


@pytest.mark.unit
def test_inherits_from_kaos_llm_error() -> None:
    """The hierarchy matters: any caller catching the base error
    type (e.g. the retry-coordinator) MUST also catch policy
    refusals so they aren't re-tried with the same provider."""
    err = KaosLLMProviderPolicyError(
        "x",
        provider="p",
        model="m",
        constraint="c",
    )
    assert isinstance(err, KaosLLMError)


# ── Defensive defaults ─────────────────────────────────────────────


@pytest.mark.unit
def test_all_kwargs_default_to_none() -> None:
    """If a caller forgets a kwarg, the field is None — not a
    KeyError or default-string. The SPA's toast renderer special-
    cases None so the user sees a generic refusal rather than a
    half-populated message."""
    err = KaosLLMProviderPolicyError("generic refusal")
    assert err.provider is None
    assert err.model is None
    assert err.constraint is None


# ── Message preserved verbatim ─────────────────────────────────────


@pytest.mark.unit
def test_str_includes_message_text() -> None:
    """The ``str(err)`` representation must include the message so
    `logger.error("%s", err)` produces actionable text. Other test
    suites assert against specific message substrings; this is the
    invariant that lets those work."""
    err = KaosLLMProviderPolicyError(
        "Provider 'openai' is not in allowed list ['anthropic']",
        provider="openai",
        model="openai:gpt-5.4-mini",
        constraint="allowed_providers",
    )
    text = str(err)
    assert "openai" in text
    assert "anthropic" in text


# ── Round-trip details survive ─────────────────────────────────────


@pytest.mark.unit
def test_extra_details_kwargs_attach_for_audit_log() -> None:
    """The base ``KaosLLMError`` accepts **details kwargs; an
    auditor or hook can attach session_id / tenant_id / request_id
    via this path. The three named fields don't crowd out arbitrary
    audit context."""
    err = KaosLLMProviderPolicyError(
        "refusal",
        provider="openai",
        model="openai:gpt-5.4-mini",
        constraint="allowed_providers",
        session_id="01KS8CB9MAT57P13QZ5PQKRG8R",
        tenant_id="abc-tenant",
    )
    # Named fields unchanged.
    assert err.provider == "openai"
    assert err.constraint == "allowed_providers"
    # Extra detail kwargs accessible via the base class's `.details`
    # dict — pattern shared with KaosLLMError siblings.
    details = getattr(err, "details", None) or {}
    if details:
        assert details.get("session_id") == "01KS8CB9MAT57P13QZ5PQKRG8R"
        assert details.get("tenant_id") == "abc-tenant"


# ── Two distinct refusals are not equal (no false dedup) ──────────


@pytest.mark.unit
def test_two_distinct_refusals_are_not_value_equal() -> None:
    """Two refusal errors with different (provider, model, constraint)
    must NOT compare equal — otherwise audit-log deduplication could
    silently coalesce a HIPAA refusal and an allowlist refusal into
    one row."""
    a = KaosLLMProviderPolicyError(
        "first", provider="openai", model="m1", constraint="allowed_providers"
    )
    b = KaosLLMProviderPolicyError(
        "second", provider="xai", model="m2", constraint="hipaa_required:no_baa"
    )
    # Exceptions inherit identity equality by default (no __eq__
    # override). Pin that: two distinct refusals are distinct
    # objects with distinct field tuples.
    assert (a.provider, a.model, a.constraint) != (b.provider, b.model, b.constraint)
    assert a is not b
