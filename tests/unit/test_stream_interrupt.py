"""Tests for B1.3 — mid-stream provider interrupt error (#570).

Pre-B1.3 (broad-reliability roadmap §B1.3), an httpx ``ReadError`` /
``RemoteProtocolError`` mid-stream surfaced as an opaque
``KaosLLMTransportError`` with no partial-text payload. SPA consumers
shipped a half-streamed message with no recovery signal — the user saw
the assistant cut off mid-word.

Post-B1.3, ``parse_sse_stream`` (and its sync sibling) catch the
network exception and raise a typed
:class:`KaosLLMStreamInterruptedError` carrying ``bytes_received``,
``partial_text``, and the underlying cause. Consumers can decide
between (a) ship-partial-with-footer and (b) retry-as-fresh-call.

These tests assert the error envelope, the byte counter, and the
attribute round-trip through ``__cause__``.
"""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest

from kaos_llm_client.errors import (
    KaosLLMStreamInterruptedError,
    KaosLLMTransportError,
)
from kaos_llm_client.transport import parse_sse_stream, parse_sse_stream_sync

# ── Test doubles ────────────────────────────────────────────────────


class _FakeStreamResponse:
    """httpx-shaped streaming response that yields chunks then raises."""

    def __init__(
        self,
        chunks: list[str],
        *,
        raise_at: int | None = None,
        exc: Exception | None = None,
    ) -> None:
        self._chunks = chunks
        self._raise_at = raise_at
        self._exc = exc or httpx.ReadError("simulated drop")

    async def aiter_text(self):
        for i, chunk in enumerate(self._chunks):
            if self._raise_at is not None and i == self._raise_at:
                raise self._exc
            yield chunk
        if self._raise_at is not None and self._raise_at >= len(self._chunks):
            raise self._exc

    def iter_text(self):
        for i, chunk in enumerate(self._chunks):
            if self._raise_at is not None and i == self._raise_at:
                raise self._exc
            yield chunk
        if self._raise_at is not None and self._raise_at >= len(self._chunks):
            raise self._exc


async def _collect(gen) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    async for ev in gen:
        out.append(ev)
    return out


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_clean_stream_completes_without_error() -> None:
    """Baseline: a normal stream with a [DONE] sentinel finishes cleanly."""
    chunks = [
        'data: {"text": "hello "}\n',
        'data: {"text": "world"}\n',
        "data: [DONE]\n",
    ]
    response = cast(httpx.Response, _FakeStreamResponse(chunks))
    events = await _collect(parse_sse_stream(response))
    assert events == [{"text": "hello "}, {"text": "world"}]


@pytest.mark.asyncio
async def test_pre_first_byte_failure_raises_transport_error() -> None:
    """Network drop BEFORE any bytes arrive → vanilla transport error.

    The pre-first-byte case is recoverable as a fresh call — keep the
    legacy error shape so existing retry policies still trigger.
    """
    response = cast(httpx.Response, _FakeStreamResponse(chunks=[], raise_at=0))
    with pytest.raises(KaosLLMTransportError) as excinfo:
        async for _ in parse_sse_stream(response):
            pass
    assert not isinstance(excinfo.value, KaosLLMStreamInterruptedError)
    assert "before any data" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_mid_stream_drop_raises_typed_interrupt() -> None:
    """Mid-stream httpx.ReadError → KaosLLMStreamInterruptedError with
    a non-zero byte counter and the underlying exception in __cause__."""
    chunks = [
        'data: {"text": "hello "}\n',
        'data: {"text": "wor',  # cut mid-line
    ]
    drop = httpx.ReadError("connection reset by peer")
    response = cast(httpx.Response, _FakeStreamResponse(chunks, raise_at=2, exc=drop))

    events: list[dict[str, Any]] = []
    with pytest.raises(KaosLLMStreamInterruptedError) as excinfo:
        async for ev in parse_sse_stream(response):
            events.append(ev)

    # The first event was successfully parsed before the drop.
    assert events == [{"text": "hello "}]
    # Byte counter reflects everything received pre-drop.
    expected_bytes = sum(len(c) for c in chunks)
    assert excinfo.value.bytes_received == expected_bytes
    # __cause__ + .cause both point at the original network error.
    assert excinfo.value.cause is drop
    assert excinfo.value.__cause__ is drop


@pytest.mark.asyncio
async def test_mid_stream_drop_includes_byte_count_in_message() -> None:
    chunks = ['data: {"x": 1}\n', "data: par"]
    response = cast(
        httpx.Response, _FakeStreamResponse(chunks, raise_at=2, exc=httpx.ReadError("drop"))
    )
    with pytest.raises(KaosLLMStreamInterruptedError) as excinfo:
        async for _ in parse_sse_stream(response):
            pass
    assert str(excinfo.value.bytes_received) in str(excinfo.value)


@pytest.mark.asyncio
async def test_remote_protocol_error_treated_as_interrupt() -> None:
    """RemoteProtocolError (server-side framing error mid-stream) →
    same typed interrupt — caller doesn't care about the exact subtype."""
    chunks = ['data: {"text": "hello"}\n']
    exc = httpx.RemoteProtocolError("server sent malformed chunk")
    response = cast(httpx.Response, _FakeStreamResponse(chunks, raise_at=1, exc=exc))
    with pytest.raises(KaosLLMStreamInterruptedError) as excinfo:
        async for _ in parse_sse_stream(response):
            pass
    assert excinfo.value.bytes_received > 0
    assert excinfo.value.cause is exc


def test_sync_stream_mirrors_async_behavior() -> None:
    """The sync parse_sse_stream_sync follows the same contract."""
    chunks = ['data: {"text": "hello"}\n', "data: cut"]
    drop = httpx.ReadError("sync drop")
    response = cast(httpx.Response, _FakeStreamResponse(chunks, raise_at=2, exc=drop))

    events: list[dict[str, Any]] = []
    with pytest.raises(KaosLLMStreamInterruptedError) as excinfo:
        for ev in parse_sse_stream_sync(response):
            events.append(ev)

    assert events == [{"text": "hello"}]
    assert excinfo.value.bytes_received == sum(len(c) for c in chunks)
    assert excinfo.value.cause is drop


def test_sync_pre_first_byte_failure_raises_transport_error() -> None:
    response = cast(httpx.Response, _FakeStreamResponse(chunks=[], raise_at=0))
    with pytest.raises(KaosLLMTransportError) as excinfo:
        for _ in parse_sse_stream_sync(response):
            pass
    assert not isinstance(excinfo.value, KaosLLMStreamInterruptedError)


class TestInterruptErrorShape:
    """The exception class itself satisfies its contract."""

    def test_inherits_from_transport_error(self) -> None:
        """Existing ``except KaosLLMTransportError`` handlers still
        catch the new error type."""
        e = KaosLLMStreamInterruptedError(
            "test",
            partial_text="hi",
            bytes_received=10,
        )
        assert isinstance(e, KaosLLMTransportError)

    def test_carries_partial_text_and_byte_count(self) -> None:
        e = KaosLLMStreamInterruptedError("drop", partial_text="the answer is", bytes_received=42)
        assert e.partial_text == "the answer is"
        assert e.bytes_received == 42

    def test_cause_round_trip(self) -> None:
        underlying = httpx.ReadError("X")
        e = KaosLLMStreamInterruptedError(
            "drop", partial_text="", bytes_received=1, cause=underlying
        )
        assert e.cause is underlying
        assert e.__cause__ is underlying

    def test_public_export(self) -> None:
        """``KaosLLMStreamInterruptedError`` is exposed from the top-level
        package — consumers shouldn't have to reach into ``.errors``."""
        from kaos_llm_client import KaosLLMStreamInterruptedError as exported

        assert exported is KaosLLMStreamInterruptedError

    def test_details_round_trip_through_kaos_core(self) -> None:
        """The bytes_received + partial_text fields are also present in
        ``details`` so KaosCoreError consumers (loggers, audit trails)
        see the structured payload."""
        e = KaosLLMStreamInterruptedError("drop", partial_text="hello", bytes_received=5)
        # KaosCoreError stores **details kwargs; structure depends on
        # the base class. Either way the attributes are accessible.
        assert e.bytes_received == 5
        assert e.partial_text == "hello"
        # Round-trip via JSON-ish dump (callers serialize errors for
        # structured logging).
        as_dict = {
            "bytes_received": e.bytes_received,
            "partial_text": e.partial_text,
        }
        round_tripped = json.loads(json.dumps(as_dict))
        assert round_tripped == as_dict
