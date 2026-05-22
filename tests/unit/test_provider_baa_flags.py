"""Unit tests for the per-provider BAA flag (plan §Issue 4).

The ``ModelProfile.baa_available`` flag plus ``assert_baa_compliance``
helper let tenant policy enforce that PHI never exits the boundary
through a provider that doesn't carry a Business Associate Agreement
(HIPAA §164.314(a)(1)).
"""

from __future__ import annotations

import pytest

from kaos_llm_client.errors import KaosLLMProviderPolicyError
from kaos_llm_client.profiles import (
    ANTHROPIC_DEFAULT,
    GOOGLE_DEFAULT,
    OPENAI_DEFAULT,
    XAI_DEFAULT,
    ModelProfile,
    assert_baa_compliance,
)


@pytest.mark.unit
def test_baa_available_default_is_false_for_every_provider_default() -> None:
    """Every shipping provider profile MUST default to ``baa_available=False``.
    Operators must explicitly opt a profile in based on a real BAA
    contract — defaulting to True would silently allow PHI through
    providers that may not actually be BAA-eligible on a given tenant
    contract.
    """
    for profile in (
        OPENAI_DEFAULT,
        ANTHROPIC_DEFAULT,
        GOOGLE_DEFAULT,
        XAI_DEFAULT,
    ):
        assert profile.baa_available is False, (
            f"{profile.provider_name} default profile must NOT claim baa_available; "
            "operators opt in per signed contract."
        )


@pytest.mark.unit
def test_assert_baa_compliance_passes_when_hipaa_not_required() -> None:
    """Non-PHI workloads bypass the check — no constraint, no raise."""
    profile = OPENAI_DEFAULT  # baa_available=False
    assert_baa_compliance(profile, hipaa_required=False, model="gpt-5.4-mini")


@pytest.mark.unit
def test_assert_baa_compliance_passes_when_provider_is_baa_eligible() -> None:
    """A profile that an operator has marked baa_available=True passes
    even with hipaa_required."""
    profile = OPENAI_DEFAULT.update(baa_available=True)
    assert_baa_compliance(profile, hipaa_required=True, model="gpt-5.4-mini")


@pytest.mark.unit
def test_assert_baa_compliance_raises_on_unmarked_provider() -> None:
    """A profile with the conservative default fails closed — the
    error carries provider, model, and constraint for actionable
    remediation by the SPA backend."""
    profile = XAI_DEFAULT  # baa_available=False
    with pytest.raises(KaosLLMProviderPolicyError) as exc_info:
        assert_baa_compliance(profile, hipaa_required=True, model="grok-4")
    err = exc_info.value
    assert err.provider == "xai"
    assert err.model == "grok-4"
    assert err.constraint == "hipaa_required"
    assert "BAA" in str(err)
    assert "Azure" in str(err) or "Bedrock" in str(err)  # remediation hint


@pytest.mark.unit
def test_baa_flag_round_trips_through_update() -> None:
    """``ModelProfile.update`` preserves the flag — important because
    the per-model resolvers chain ``update`` calls when narrowing
    max_tokens / supports_temperature."""
    base = OPENAI_DEFAULT.update(baa_available=True)
    narrowed = base.update(default_max_tokens=8192)
    assert narrowed.baa_available is True


@pytest.mark.unit
def test_assert_baa_compliance_uses_unknown_when_provider_name_missing() -> None:
    """A bare ModelProfile() with no provider_name still produces a
    legible error message — operators see ``'unknown'`` instead of
    an empty string."""
    profile = ModelProfile()  # provider_name="" by default
    with pytest.raises(KaosLLMProviderPolicyError) as exc_info:
        assert_baa_compliance(profile, hipaa_required=True)
    assert "'unknown'" in str(exc_info.value)
