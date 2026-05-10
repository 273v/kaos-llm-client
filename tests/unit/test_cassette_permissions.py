"""Tests for owner-only file permissions on cassette writes.

A cassette is a near-verbatim transcript of every prompt sent to a
model and every response received. Even with sensitive auth headers
stripped (see ``_sanitize_request``), the prompt body is preserved
in full, which means cassettes contain whatever the user typed —
PII, source code, internal docs, etc.

To avoid leaking that to other local users on shared hosts, the
cassette write path mirrors ``FileCache.put`` in ``cache.py``: parent
directory ``0o700``, file ``0o600``.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from kaos_llm_client.cassette import (
    Cassette,
    CassetteEntry,
    CassetteRecorder,
    use_cassette,
)
from kaos_llm_client.types import (
    ContentPart,
    ProviderRequest,
    ProviderResponse,
    UsageInfo,
)

# POSIX permission bits (0o600 / 0o700) don't translate to Windows
# ACLs — ``Path.chmod`` is essentially a no-op there, and freshly
# created files report 0o666 / directories 0o777. The cassette
# permission contract is documented as POSIX-only (shared/multi-tenant
# Unix hosts); the Windows path relies on NTFS / per-user profile
# isolation instead. Skip the assertions on Windows so the test
# matrix's Windows leg doesn't false-fail; the POSIX behavior is
# still covered on Linux + macOS.
_posix_only = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX permission bits don't apply on Windows (NTFS ACLs)",
)


def _make_request(**overrides: Any) -> ProviderRequest:
    defaults: dict[str, Any] = {
        "provider": "openai",
        "model": "gpt-5",
        "endpoint": "/v1/chat/completions",
        "body": {"messages": [{"role": "user", "content": "hello"}]},
    }
    defaults.update(overrides)
    return ProviderRequest(**defaults)


def _make_response(**overrides: Any) -> ProviderResponse:
    defaults: dict[str, Any] = {
        "provider": "openai",
        "model": "gpt-5",
        "raw": {"choices": [{"message": {"content": "hi"}}]},
        "parts": [ContentPart(type="text", text="hi")],
        "usage": UsageInfo(input_tokens=5, output_tokens=1, total_tokens=6),
    }
    defaults.update(overrides)
    return ProviderResponse(**defaults)


def _file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# ---------------------------------------------------------------------------
# Make the umask deterministic for these tests. Without this, an
# unusually-permissive umask (e.g. ``0o000``) would let the file be
# created world-writable BEFORE our explicit chmod fires, and on some
# filesystems ``os.open(..., 0o600)`` does not actually clear the world
# bits. The test asserts the *final* mode after our chmod, so we set a
# normal umask here for repeatability.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _normal_umask() -> Any:
    old = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(old)


@_posix_only
class TestCassetteSavePermissions:
    def test_file_mode_is_0o600(self, tmp_path: Path) -> None:
        cassette = Cassette()
        cassette.add(
            CassetteEntry(
                key="abc123",
                sequence=0,
                request={"provider": "openai", "model": "gpt-5"},
                response={"provider": "openai", "model": "gpt-5"},
                metadata={},
            )
        )
        out = tmp_path / "session.jsonl"
        cassette.save(out)

        assert out.exists()
        assert _file_mode(out) == 0o600

    def test_directory_mode_is_0o700(self, tmp_path: Path) -> None:
        cassette = Cassette()
        cassette.add(
            CassetteEntry(
                key="abc",
                sequence=0,
                request={},
                response={},
                metadata={},
            )
        )
        out = tmp_path / "nested" / "session.jsonl"
        cassette.save(out)

        assert out.parent.exists()
        assert _file_mode(out.parent) == 0o700

    def test_file_overwrite_keeps_0o600(self, tmp_path: Path) -> None:
        """Overwriting an existing cassette must not relax permissions."""
        cassette = Cassette()
        cassette.add(CassetteEntry(key="abc", sequence=0, request={}, response={}, metadata={}))
        out = tmp_path / "session.jsonl"
        cassette.save(out)
        # Pre-corrupt the mode to simulate a stale wide-open file.
        out.chmod(0o644)
        cassette.save(out)
        assert _file_mode(out) == 0o600

    def test_recorder_writes_with_0o600(self, tmp_path: Path) -> None:
        out = tmp_path / "rec.jsonl"
        rec = CassetteRecorder(out)
        req = _make_request()
        rec._on_request(req)
        rec._on_response(req, _make_response())
        rec.save()

        assert out.exists()
        assert _file_mode(out) == 0o600

    def test_use_cassette_writes_with_0o600(self, tmp_path: Path) -> None:
        out = tmp_path / "ctx.jsonl"
        with use_cassette(out) as ctx:
            req = _make_request()
            assert ctx.hooks.on_request is not None
            assert ctx.hooks.on_response is not None
            ctx.hooks.on_request(req)
            ctx.hooks.on_response(req, _make_response())

        assert out.exists()
        assert _file_mode(out) == 0o600

    def test_content_round_trips_through_secure_write(self, tmp_path: Path) -> None:
        """The hardened write must not corrupt the JSONL payload."""
        out = tmp_path / "rt.jsonl"
        cassette = Cassette()
        cassette.add(
            CassetteEntry(
                key="key-1",
                sequence=0,
                request={"provider": "openai", "model": "gpt-5"},
                response={"provider": "openai", "model": "gpt-5", "ok": True},
                metadata={"recorded_at": 0.0},
            )
        )
        cassette.save(out)

        loaded = Cassette.load(out)
        assert loaded.size == 1
        assert loaded.entries[0].key == "key-1"
        assert loaded.entries[0].response["ok"] is True

    def test_empty_cassette_still_creates_secure_file(self, tmp_path: Path) -> None:
        """Saving an empty cassette truncates / creates the file with 0o600."""
        out = tmp_path / "empty.jsonl"
        Cassette().save(out)
        assert out.exists()
        assert _file_mode(out) == 0o600
        assert out.read_text() == ""
