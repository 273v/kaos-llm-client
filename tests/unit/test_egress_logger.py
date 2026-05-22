"""Unit tests for the ``vendor_egress`` audit log (plan §Issue 4).

Every outbound LLM provider call must emit exactly one structured
``vendor_egress`` log line carrying:

  - provider
  - model
  - bytes (serialised body length)
  - request_hash (sha256 of canonical-serialised body)
  - scrub_patterns (list of pattern names, may be empty)
  - scrubbed_chars (int, may be 0)
  - timestamp (ISO-8601 UTC)

The hash + bytes pair lets an auditor diff what was billed vs what
left the process without storing prompt text itself.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from kaos_llm_client.transport import (
    _request_body_digest,
    emit_vendor_egress_log,
)


@pytest.fixture
def egress_handler() -> Iterator[list[logging.LogRecord]]:
    """Attach a buffering handler to the kaos egress logger so tests
    can inspect the structured ``extra=`` fields directly.

    ``caplog`` alone is unreliable here: ``kaos_core.logging.get_logger``
    installs its own handler chain and may set ``propagate=False``, so
    records never reach the pytest root handler. Attaching a direct
    handler to the logger name we know is used gives us a stable
    assertion surface that survives kaos-core logging refactors.
    """
    name = "kaos.llm_client.transport.egress"
    logger_ = logging.getLogger(name)
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture(level=logging.DEBUG)
    logger_.addHandler(handler)
    prev_level = logger_.level
    logger_.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        logger_.removeHandler(handler)
        logger_.setLevel(prev_level)


@pytest.mark.unit
def test_request_body_digest_is_stable_across_dict_order() -> None:
    """sha256 of the serialised body must NOT depend on dict iteration
    order. An auditor re-hashing a captured cassette gets the same
    digest as production regardless of CPython hash randomisation."""
    body_a = {"model": "claude-sonnet-4-6", "max_tokens": 100, "messages": []}
    body_b = {"messages": [], "max_tokens": 100, "model": "claude-sonnet-4-6"}
    assert _request_body_digest(body_a) == _request_body_digest(body_b)


@pytest.mark.unit
def test_request_body_digest_changes_on_content_change() -> None:
    """A single-byte change to the body MUST change the hash —
    otherwise the audit trail is forgeable."""
    body_a = {"prompt": "tell me about Mata v. Avianca"}
    body_b = {"prompt": "tell me about Mata v. Avianca."}  # period
    _, hash_a = _request_body_digest(body_a)
    _, hash_b = _request_body_digest(body_b)
    assert hash_a != hash_b


@pytest.mark.unit
def test_request_body_digest_byte_count_is_utf8_length() -> None:
    """Non-ASCII content must contribute its UTF-8 byte length, not
    character length. A 1-character emoji is 4 bytes — this is the
    metric the wire actually carries."""
    body = {"prompt": "🦀"}  # U+1F980 — 4 UTF-8 bytes
    size, _ = _request_body_digest(body)
    # JSON envelope = {"prompt":"🦀"} = 14 ASCII chars + 4 emoji bytes = 17 bytes
    # (canonical-serialised form uses ensure_ascii=False).
    assert size == 17


@pytest.mark.unit
def test_request_body_digest_handles_non_serialisable_values() -> None:
    """``default=str`` fallback means a body containing bytes/datetime
    still produces SOME digest — we never silently lose audit trail."""

    class _NotSerialisable:
        def __repr__(self) -> str:
            return "not-json-serialisable"

    body = {"messages": [{"role": "user", "content": _NotSerialisable()}]}
    size, hash_ = _request_body_digest(body)
    assert size > 0
    assert len(hash_) == 64  # sha256 hex


@pytest.mark.unit
def test_emit_vendor_egress_log_writes_structured_extras(
    egress_handler: list[logging.LogRecord],
) -> None:
    """The egress logger must emit on the ``kaos.llm_client.transport.egress``
    channel at INFO level, with every required structured field set."""
    emit_vendor_egress_log(
        provider="anthropic",
        model="claude-sonnet-4-6",
        body={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]},
        attempt=0,
    )

    assert len(egress_handler) == 1
    fields = vars(egress_handler[0])
    assert fields["event"] == "vendor_egress"
    assert fields["provider"] == "anthropic"
    assert fields["model"] == "claude-sonnet-4-6"
    assert isinstance(fields["bytes"], int) and fields["bytes"] > 0
    assert fields["request_hash"].startswith("sha256:")
    assert len(fields["request_hash"]) == 7 + 64  # "sha256:" + 64 hex chars
    assert fields["scrub_patterns"] == []
    assert fields["scrubbed_chars"] == 0
    assert fields["attempt"] == 0
    assert "T" in fields["timestamp"]  # ISO-8601


@pytest.mark.unit
def test_emit_vendor_egress_log_carries_scrub_metadata(
    egress_handler: list[logging.LogRecord],
) -> None:
    """When upstream callers ran a PII scrubber over the body, they
    can pass the pattern list + char count for the audit record."""
    emit_vendor_egress_log(
        provider="openai",
        model="gpt-5.4-mini",
        body={"prompt": "redacted body"},
        scrub_patterns=("ssn", "ein", "credit_card"),
        scrubbed_chars=42,
    )

    assert len(egress_handler) == 1
    fields = vars(egress_handler[-1])
    assert fields["scrub_patterns"] == ["ssn", "ein", "credit_card"]
    assert fields["scrubbed_chars"] == 42


@pytest.mark.unit
def test_emit_vendor_egress_log_swallows_internal_errors(
    egress_handler: list[logging.LogRecord],
) -> None:
    """A logging-side failure (e.g. a body containing self-referential
    cycles after fallback) MUST NOT propagate — audit logging is a
    best-effort sidecar to the real LLM call, not a gate."""
    # Construct a self-referential dict that even the default=str
    # fallback can choke on if both branches fail.
    cycle: dict = {"a": 1}
    cycle["self"] = cycle

    # Should not raise — defensive ``except Exception`` catches anything.
    emit_vendor_egress_log(provider="x", model="y", body=cycle)
