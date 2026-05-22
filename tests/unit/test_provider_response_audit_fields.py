"""ProviderResponse audit-trail field-shape contract (plan §Issue 3 +
§Issue 6).

Plan §Issue 3 acceptance row: every shipped model populates
``model_snapshot`` on ``ProviderResponse``. Plan §Issue 6 calls out
the provider's ``request_id`` capture as the operator's
"reproduce yesterday's turn" anchor — auditors index by it to
correlate vendor billing receipts against in-process audit logs.

This file pins the three audit-trail fields that span Issues 3 and
6 — ``model_snapshot`` (Issue 3), ``request_id`` (Issue 6), and
``response_id`` (also Issue 6, the vendor-assigned correlation id
that appears in the provider dashboard).

Existing tests in ``test_provider_model_snapshot.py`` exercise each
provider parser; this file pins the field-level contract on the
``ProviderResponse`` Pydantic model itself so a future refactor
that drops one of the fields fails loudly.
"""

from __future__ import annotations

import pytest

from kaos_llm_client.types import ProviderResponse

# ── Field presence and default-None contract ───────────────────────


@pytest.mark.unit
def test_response_id_field_defaults_to_none() -> None:
    """``response_id`` is the vendor-assigned correlation id (e.g.
    Anthropic ``msg_01ABC...`` or OpenAI ``chatcmpl-abc...``). A
    parser that hasn't been updated for a new provider must still
    construct ProviderResponse without explicitly setting it."""
    r = ProviderResponse(provider="t", model="m", raw={})
    assert r.response_id is None


@pytest.mark.unit
def test_request_id_field_defaults_to_none() -> None:
    """``request_id`` is the per-call audit anchor the transport
    layer attaches when it dispatches. Default None — populated
    by the transport, not the parser."""
    r = ProviderResponse(provider="t", model="m", raw={})
    assert r.request_id is None


@pytest.mark.unit
def test_model_snapshot_field_defaults_to_none() -> None:
    """Defended by the existing model_snapshot tests too; pin here
    so this file is self-contained and a future grep for the
    field name finds both files."""
    r = ProviderResponse(provider="t", model="m", raw={})
    assert r.model_snapshot is None


# ── All three audit fields populate independently ─────────────────


@pytest.mark.unit
def test_all_three_audit_fields_populate_round_trip() -> None:
    """All three fields land on the response and round-trip through
    model_dump for persistence to the audit JSONL."""
    r = ProviderResponse(
        provider="anthropic",
        model="claude-haiku-4-5",
        raw={},
        response_id="msg_01ABC123XYZ",
        request_id="req_internal_42",
        model_snapshot="claude-haiku-4-5-20260415",
    )
    assert r.response_id == "msg_01ABC123XYZ"
    assert r.request_id == "req_internal_42"
    assert r.model_snapshot == "claude-haiku-4-5-20260415"

    dumped = r.model_dump(mode="json")
    assert dumped["response_id"] == "msg_01ABC123XYZ"
    assert dumped["request_id"] == "req_internal_42"
    assert dumped["model_snapshot"] == "claude-haiku-4-5-20260415"


@pytest.mark.unit
def test_response_id_independent_of_request_id() -> None:
    """``response_id`` is vendor-assigned; ``request_id`` is
    internal. They MUST NOT silently alias. A regression that
    collapsed both into a single field would break the
    "diff billed-vs-emitted" audit step."""
    r = ProviderResponse(
        provider="openai",
        model="gpt-5.4-nano",
        raw={},
        response_id="chatcmpl-abc",
        request_id="kaos-internal-xyz",
    )
    assert r.response_id != r.request_id


# ── Audit-trail completeness invariant ─────────────────────────────


@pytest.mark.unit
def test_audit_complete_when_all_three_populated() -> None:
    """An auditor 18 months later must be able to re-bind a
    response to the exact provider call. The three fields together
    are necessary AND sufficient:
    - ``request_id`` says "which of our outbound calls"
    - ``response_id`` says "which line on the provider's billing"
    - ``model_snapshot`` says "exactly which versioned model"
    """
    r = ProviderResponse(
        provider="google",
        model="gemini-2.5-flash",
        raw={},
        response_id="gen-cand-99",
        request_id="kaos-req-1",
        model_snapshot="gemini-2.5-flash-001",
    )
    # All three present → audit-complete.
    audit_complete = (
        r.response_id is not None and r.request_id is not None and r.model_snapshot is not None
    )
    assert audit_complete is True


@pytest.mark.unit
def test_audit_partial_flagged_when_any_field_missing() -> None:
    """A response missing any of the three audit anchors is
    audit-partial — the operator alert sink should flag it. Pin
    that the field-level invariant (all three present) is easy to
    derive."""
    r_missing_response = ProviderResponse(
        provider="t",
        model="m",
        raw={},
        request_id="r",
        model_snapshot="m-001",
    )
    audit_complete = (
        r_missing_response.response_id is not None
        and r_missing_response.request_id is not None
        and r_missing_response.model_snapshot is not None
    )
    assert audit_complete is False


# ── Serialization stability ───────────────────────────────────────


@pytest.mark.unit
def test_model_dump_emits_field_keys_even_when_none() -> None:
    """The audit JSONL writer dumps the full schema for forward
    compatibility — even None-valued fields should serialize
    (the operator's reader can distinguish "no value recorded"
    from "key missing")."""
    r = ProviderResponse(provider="t", model="m", raw={})
    dumped = r.model_dump(mode="json")
    assert "response_id" in dumped
    assert "request_id" in dumped
    assert "model_snapshot" in dumped


@pytest.mark.unit
def test_distinct_responses_have_distinct_audit_anchors() -> None:
    """Two real provider calls produce distinct ``response_id`` AND
    distinct ``request_id`` values. A regression that produced
    duplicate ids would silently collapse audit rows."""
    r1 = ProviderResponse(
        provider="anthropic",
        model="claude-haiku-4-5",
        raw={},
        response_id="msg_001",
        request_id="req_001",
    )
    r2 = ProviderResponse(
        provider="anthropic",
        model="claude-haiku-4-5",
        raw={},
        response_id="msg_002",
        request_id="req_002",
    )
    assert r1.response_id != r2.response_id
    assert r1.request_id != r2.request_id
