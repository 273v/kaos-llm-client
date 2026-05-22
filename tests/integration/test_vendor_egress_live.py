"""Live integration test for vendor_egress audit logging
(launch-blocker plan §Issue 4 — per-vendor PII egress log).

Closes the live tier for the egress logger: hits a real LLM
provider and asserts the ``kaos.llm_client.transport.egress``
logger emits exactly one ``vendor_egress`` INFO record per
outbound HTTP POST with the full audit-trail payload.

Schema we pin (each per-call record must carry):

- ``event`` == ``"vendor_egress"``
- ``provider`` (string, e.g. ``"openai"``)
- ``model`` (string, the family alias we requested)
- ``bytes`` (int, > 0 — body byte count)
- ``request_hash`` (string, ``sha256:<hex>`` 64-char digest)
- ``scrub_patterns`` (list, may be empty)
- ``scrubbed_chars`` (int, ≥ 0)
- ``attempt`` (int, retry counter — 0 on first try)
- ``timestamp`` (string, ISO-8601 UTC)

Unit tests in ``tests/unit/test_vendor_egress.py`` already pin
the helper logic against synthetic inputs. This file is the
**live** proof that the helper fires on the real
``execute_with_retry`` path against a real provider call,
end-to-end through the retry/transport stack.

Why this matters: the audit row is GDPR Art. 30 + HIPAA §164.312
evidence that every outbound prompt was recorded with a stable
hash that lets an auditor diff what was billed vs what left the
process — without storing the prompt text itself.

Cost: ~$0.0001 (one 1-token-class call per test).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

import pytest

from kaos_llm_client.providers import create_client

from .conftest import requires_openai

_SHORT_PROMPT = "Say exactly one word: hello"

# Logger name pinned by transport.py:33.
_EGRESS_LOGGER_NAME = "kaos.llm_client.transport.egress"


class _ListHandler(logging.Handler):
    """Capture all records emitted on the egress logger.

    We use a custom handler attached directly to the egress logger
    instead of ``caplog`` because the kaos-core ``get_logger()``
    wrapper sets ``propagate=False`` on the wrapped Python logger,
    so caplog's root-handler never sees the records.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def egress_records() -> Iterator[list[logging.LogRecord]]:
    """Attach a list-handler to the egress logger for the test body
    and detach it afterward."""
    logger = logging.getLogger(_EGRESS_LOGGER_NAME)
    prior_level = logger.level
    logger.setLevel(logging.DEBUG)
    handler = _ListHandler()
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)


def _find_egress_records(
    records: list[logging.LogRecord], event: str = "vendor_egress"
) -> list[logging.LogRecord]:
    """Filter records to those carrying the ``event=<event>`` attr."""
    return [r for r in records if getattr(r, "event", None) == event]


@pytest.mark.integration
@requires_openai
def test_openai_live_call_emits_one_vendor_egress_record(
    egress_records: list[logging.LogRecord],
) -> None:
    """A successful first-try OpenAI call must produce exactly one
    ``vendor_egress`` log record with the full audit-trail payload."""
    r = create_client("openai:gpt-5.4-nano").chat(
        [{"role": "user", "content": _SHORT_PROMPT}],
    )
    assert "hello" in r.text.lower()

    matching = _find_egress_records(egress_records)
    assert len(matching) == 1, (
        f"Expected exactly 1 vendor_egress record on a first-try success, "
        f"got {len(matching)}. records={egress_records!r}"
    )

    rec = matching[0]
    # Pin every field the auditor will rely on.
    assert getattr(rec, "provider", None) == "openai"
    assert getattr(rec, "model", None) == "gpt-5.4-nano"
    body_bytes = getattr(rec, "bytes", None)
    assert isinstance(body_bytes, int)
    assert body_bytes > 0, "request body byte count must be > 0"

    request_hash = getattr(rec, "request_hash", "")
    assert isinstance(request_hash, str)
    assert request_hash.startswith("sha256:"), (
        f"request_hash must be sha256-prefixed, got {request_hash!r}"
    )
    # sha256 hex digest = 64 chars after the prefix.
    assert len(request_hash) == len("sha256:") + 64, (
        f"request_hash hex length wrong: {request_hash!r}"
    )

    # scrub_patterns is empty by default (no scrubber installed in this
    # test); scrubbed_chars is 0.
    assert getattr(rec, "scrub_patterns", None) == []
    assert getattr(rec, "scrubbed_chars", -1) == 0

    # attempt=0 on first try, ISO timestamp.
    assert getattr(rec, "attempt", None) == 0
    ts = getattr(rec, "timestamp", "")
    # Match e.g. 2026-05-22T13:32:00.123456+00:00 — anchor the date prefix
    # rather than the whole tail so a +HH:MM or Z suffix doesn't fail.
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", ts), f"timestamp not ISO-8601: {ts!r}"


@pytest.mark.integration
@requires_openai
def test_egress_records_unique_request_hash_per_call(
    egress_records: list[logging.LogRecord],
) -> None:
    """Two distinct prompts must produce two distinct request_hash
    values — the hash is the auditor's only handle on which prompt
    text was sent in each call."""
    client_a = create_client("openai:gpt-5.4-nano")
    client_a.chat([{"role": "user", "content": "Say A"}])
    client_b = create_client("openai:gpt-5.4-nano")
    client_b.chat([{"role": "user", "content": "Say B"}])

    matching = _find_egress_records(egress_records)
    assert len(matching) == 2, (
        f"Expected 2 vendor_egress records (one per call), got {len(matching)}"
    )

    hashes = [getattr(r, "request_hash", "") for r in matching]
    assert hashes[0] != hashes[1], (
        f"Distinct prompts produced identical hashes: {hashes}. The "
        f"audit chain depends on hash distinctness per outbound prompt."
    )
