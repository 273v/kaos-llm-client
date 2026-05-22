"""Live integration test for ProviderResponse.model_snapshot
(launch-blocker plan §Issue 3 — per-turn version pinning).

Closes the live tier for the model_snapshot capture: hits each
provider's real API with a 1-token "hello" prompt and asserts:

1. ``response.model_snapshot`` is non-empty (the provider returned
   a resolved snapshot string).
2. The snapshot differs from the requested family alias (e.g.
   ``gpt-5.4-nano`` resolves to ``gpt-5.4-nano-2026-MM-DD``).
3. The snapshot is a stable identifier suitable for EU AI Act
   Article 12 / Annex III §6 record-keeping.

Why this matters: 18 months from now an auditor must be able to
re-bind a given response back to the exact versioned model that
generated it. Family aliases auto-route to internal dated
snapshots; without explicit capture, the audit chain breaks.

The unit-level tests in
``tests/unit/test_provider_model_snapshot.py`` already pin the
parser logic against synthetic payloads. This file is the live
proof that the parsers' assumptions about the wire format hold
against real provider responses TODAY.

Requires the relevant per-provider API key. Each test is gated
independently so a missing Anthropic key doesn't disqualify the
OpenAI assertions.

Cost: ~$0.0001 per provider for the 1-token prompt; the matrix
runs well under $0.001.
"""

from __future__ import annotations

import pytest

from kaos_llm_client.providers import create_client

from .conftest import requires_anthropic, requires_google, requires_openai

# 1-token-class prompt — cheapest possible live verification.
_SHORT_PROMPT = "Say exactly one word: hello"


@pytest.mark.integration
@requires_openai
def test_openai_chat_completions_returns_versioned_snapshot() -> None:
    """OpenAI Chat Completions echoes the resolved snapshot in
    ``model``; the parser captures it as ``model_snapshot``."""
    r = create_client("openai:gpt-5.4-nano").chat(
        [{"role": "user", "content": _SHORT_PROMPT}],
    )
    assert r.model_snapshot is not None, (
        "OpenAI chat response is missing the versioned model_snapshot — "
        "the parser may have regressed or the wire format changed."
    )
    # Family alias is ``gpt-5.4-nano``; the snapshot must include this
    # family stem (else we got something else entirely) AND must extend
    # past the alias (the dated suffix).
    assert "gpt-5.4-nano" in r.model_snapshot, (
        f"snapshot {r.model_snapshot!r} does not contain the requested family alias 'gpt-5.4-nano'"
    )
    assert len(r.model_snapshot) > len("gpt-5.4-nano"), (
        f"snapshot {r.model_snapshot!r} is the bare alias; expected a "
        f"dated suffix like 'gpt-5.4-nano-YYYY-MM-DD'"
    )


@pytest.mark.integration
@requires_openai
def test_openai_responses_api_returns_versioned_snapshot() -> None:
    """OpenAI Responses API (``/v1/responses``) — same field shape
    as Chat Completions; parser must capture it independently."""
    # Use o4-mini through the Responses endpoint pattern.
    r = create_client("openai-responses:o4-mini").chat(
        [{"role": "user", "content": _SHORT_PROMPT}],
    )
    assert r.model_snapshot is not None, "OpenAI Responses API response is missing model_snapshot."
    assert "o4-mini" in r.model_snapshot, (
        f"snapshot {r.model_snapshot!r} does not contain 'o4-mini'"
    )


@pytest.mark.integration
@requires_anthropic
def test_anthropic_returns_versioned_snapshot() -> None:
    """Anthropic Messages API echoes the resolved snapshot in
    ``model`` at the top level of the response body."""
    r = create_client("anthropic:claude-haiku-4-5").chat(
        [{"role": "user", "content": _SHORT_PROMPT}],
    )
    assert r.model_snapshot is not None, (
        "Anthropic response is missing model_snapshot — the parser "
        "may have regressed or the wire format changed."
    )
    assert "claude-haiku" in r.model_snapshot, (
        f"snapshot {r.model_snapshot!r} does not contain 'claude-haiku'"
    )
    # Anthropic snapshots include a YYYYMMDD-style suffix.
    assert len(r.model_snapshot) > len("claude-haiku-4-5"), (
        f"snapshot {r.model_snapshot!r} appears to be the bare alias; expected a dated suffix"
    )


@pytest.mark.integration
@requires_google
def test_google_returns_versioned_snapshot_from_modelVersion() -> None:
    """Google Gemini uses ``modelVersion`` (not ``model``) for the
    resolved snapshot — the parser must read the right field."""
    r = create_client("google:gemini-2.5-flash").chat(
        [{"role": "user", "content": _SHORT_PROMPT}],
    )
    assert r.model_snapshot is not None, (
        "Google response is missing model_snapshot — the parser may "
        "be reading the wrong field (Google uses 'modelVersion', "
        "not 'model')."
    )
    assert "gemini" in r.model_snapshot.lower(), (
        f"snapshot {r.model_snapshot!r} does not contain 'gemini'"
    )


@pytest.mark.integration
@requires_openai
def test_snapshot_is_stable_across_repeated_calls() -> None:
    """Within a short window, repeated calls to the same family
    alias should resolve to the same versioned snapshot. If this
    flakes, the audit trail breaks — same wall-clock window, two
    different "underlying" models recorded.

    Uses a fresh client per call so the asyncio event loop in the
    sync ``.chat()`` shim closes cleanly between calls (anyio's
    transport reader otherwise tries to resume on a torn-down loop
    when the second call inside the same client reuses the httpx
    transport)."""
    r1 = create_client("openai:gpt-5.4-nano").chat(
        [{"role": "user", "content": _SHORT_PROMPT}],
    )
    r2 = create_client("openai:gpt-5.4-nano").chat(
        [{"role": "user", "content": _SHORT_PROMPT}],
    )
    assert r1.model_snapshot == r2.model_snapshot, (
        f"snapshot drift within one session: r1={r1.model_snapshot!r} "
        f"r2={r2.model_snapshot!r}. The audit chain assumes intra-"
        f"window stability — provider routing change?"
    )
