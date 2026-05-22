"""Unit tests for ProviderResponse.model_snapshot (plan §Issue 3).

The ``model`` field on ``ProviderResponse`` carries what we ASKED for.
The new ``model_snapshot`` field carries what the provider actually
served. Auditors 18 months later must be able to identify the exact
versioned model snapshot used to generate a given response —
required for EU AI Act Article 12 / Annex III §6.

These tests directly exercise each provider's ``_parse_response``
against synthetic raw payloads that mirror the real wire formats.
Live integration tests against actual provider APIs live in
``tests/integration/test_provider_model_snapshot.py`` (separate
file, separate tier).
"""

from __future__ import annotations

from typing import Any

import pytest

from kaos_llm_client.types import ProviderRequest


def _make_request(model: str = "test-model") -> ProviderRequest:
    """Minimal ProviderRequest stub for parser tests."""
    return ProviderRequest(
        provider="test",
        model=model,
        endpoint="/v1/messages",
        body={"model": model, "messages": []},
        headers={"Content-Type": "application/json"},
    )


# ── Field presence on the typed model ───────────────────────────────


@pytest.mark.unit
def test_provider_response_model_snapshot_field_defaults_to_none() -> None:
    """Field is optional — a provider parser that hasn't been
    updated yet must continue to construct ProviderResponse without
    explicitly passing model_snapshot."""
    from kaos_llm_client.types import ProviderResponse

    resp = ProviderResponse(provider="test", model="m", raw={})
    assert resp.model_snapshot is None


@pytest.mark.unit
def test_provider_response_model_snapshot_round_trips_through_model_dump() -> None:
    """The field must serialise to JSON for the audit trail."""
    from kaos_llm_client.types import ProviderResponse

    resp = ProviderResponse(
        provider="anthropic",
        model="claude-sonnet-4-6",
        raw={},
        model_snapshot="claude-sonnet-4-6-20260415",
    )
    dumped = resp.model_dump(mode="json")
    assert dumped["model_snapshot"] == "claude-sonnet-4-6-20260415"


# ── Anthropic ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_anthropic_captures_response_model_field() -> None:
    """Anthropic Messages API returns the resolved versioned snapshot
    in the top-level ``model`` field on the response body."""
    from kaos_llm_client.providers.anthropic import AnthropicClient

    client = AnthropicClient(
        api_key="sk-ant-test",
        model="claude-sonnet-4-6",
        base_url="https://api.anthropic.com",
    )
    raw: dict[str, Any] = {
        "id": "msg_01ABC123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6-20260415",  # versioned snapshot
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 1},
    }
    resp = client._parse_response(raw, _make_request("claude-sonnet-4-6"))
    assert resp.model_snapshot == "claude-sonnet-4-6-20260415"
    assert resp.model == "claude-sonnet-4-6"  # what we asked for, unchanged


@pytest.mark.unit
def test_anthropic_model_snapshot_is_none_when_field_missing() -> None:
    """Defensive: if Anthropic ever omits the field, we get None
    rather than KeyError."""
    from kaos_llm_client.providers.anthropic import AnthropicClient

    client = AnthropicClient(
        api_key="sk-ant-test",
        model="claude-sonnet-4-6",
        base_url="https://api.anthropic.com",
    )
    raw: dict[str, Any] = {
        "id": "msg_01ABC123",
        "type": "message",
        "role": "assistant",
        # NO model field — simulate the missing-field edge case
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 1},
    }
    resp = client._parse_response(raw, _make_request("claude-sonnet-4-6"))
    assert resp.model_snapshot is None


# ── OpenAI (chat completions, openai_compat) ─────────────────────────


@pytest.mark.unit
def test_openai_compat_captures_response_model_field() -> None:
    """OpenAI Chat Completions API returns the resolved dated rev in
    the top-level ``model`` field."""
    from kaos_llm_client.providers.openai_compat import OpenAICompatibleClient

    client = OpenAICompatibleClient(
        api_key="sk-test",
        model="gpt-5.4-mini",
        base_url="https://api.openai.com/v1",
        provider_name="openai",
    )
    raw: dict[str, Any] = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1747000000,
        "model": "gpt-5.4-mini-2026-04-30",  # versioned snapshot
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }
    resp = client._parse_response(raw, _make_request("gpt-5.4-mini"))
    assert resp.model_snapshot == "gpt-5.4-mini-2026-04-30"


# ── OpenAI Responses API ────────────────────────────────────────────


@pytest.mark.unit
def test_openai_responses_captures_response_model_field() -> None:
    """OpenAI Responses API also echoes the resolved model in ``model``."""
    from kaos_llm_client.providers.openai_responses import OpenAIResponsesClient

    client = OpenAIResponsesClient(
        api_key="sk-test",
        model="o3",
        base_url="https://api.openai.com/v1",
    )
    raw: dict[str, Any] = {
        "id": "resp_abc123",
        "object": "response",
        "model": "o3-2026-03-15",  # versioned snapshot
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            },
        ],
        "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
    }
    resp = client._parse_response(raw, _make_request("o3"))
    assert resp.model_snapshot == "o3-2026-03-15"


# ── Google Gemini ───────────────────────────────────────────────────


@pytest.mark.unit
def test_google_captures_modelVersion_field() -> None:
    """Google Gemini API returns the resolved versioned snapshot as
    ``modelVersion`` (NOT ``model`` — different from OpenAI/Anthropic)."""
    from kaos_llm_client.providers.google import GoogleClient

    client = GoogleClient(
        api_key="test-api-key",
        model="gemini-2.5-flash",
    )
    raw: dict[str, Any] = {
        "modelVersion": "gemini-2.5-flash-001",
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "hello"}],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            },
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 1,
            "totalTokenCount": 6,
        },
    }
    resp = client._parse_response(raw, _make_request("gemini-2.5-flash"))
    assert resp.model_snapshot == "gemini-2.5-flash-001"
    # ``model`` already used modelVersion as the canonical id — pin
    # this behaviour to avoid divergence with the new field.
    assert resp.model == "gemini-2.5-flash-001"
