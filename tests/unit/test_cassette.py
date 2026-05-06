"""Tests for kaos_llm_client.cassette — record/replay cassette system."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kaos_llm_client.cassette import (
    Cassette,
    CassetteEntry,
    CassetteMissError,
    CassetteMode,
    CassetteRecorder,
    CassetteReplayClient,
    _sanitize_request,
    _sanitize_response,
    cassette_key,
    use_cassette,
)
from kaos_llm_client.types import ContentPart, ProviderRequest, ProviderResponse, UsageInfo


def _make_request(**overrides: Any) -> ProviderRequest:  # type: ignore[no-any-explicit]
    defaults: dict[str, Any] = {
        "provider": "openai",
        "model": "gpt-5",
        "endpoint": "/v1/chat/completions",
        "body": {"messages": [{"role": "user", "content": "hello"}]},
    }
    defaults.update(overrides)
    return ProviderRequest(**defaults)


def _make_response(**overrides: Any) -> ProviderResponse:  # type: ignore[no-any-explicit]
    defaults: dict[str, Any] = {
        "provider": "openai",
        "model": "gpt-5",
        "raw": {"choices": [{"message": {"content": "hi"}}]},
        "parts": [ContentPart(type="text", text="hi")],
        "usage": UsageInfo(input_tokens=5, output_tokens=1, total_tokens=6),
    }
    defaults.update(overrides)
    return ProviderResponse(**defaults)


class TestCassetteKey:
    def test_deterministic(self) -> None:
        req = _make_request()
        assert cassette_key(req) == cassette_key(req)

    def test_different_bodies(self) -> None:
        req1 = _make_request(body={"messages": [{"role": "user", "content": "hello"}]})
        req2 = _make_request(body={"messages": [{"role": "user", "content": "world"}]})
        assert cassette_key(req1) != cassette_key(req2)

    def test_different_models(self) -> None:
        req1 = _make_request(model="gpt-5")
        req2 = _make_request(model="gpt-4.1")
        assert cassette_key(req1) != cassette_key(req2)

    def test_ignores_headers(self) -> None:
        req1 = _make_request()
        req2 = _make_request(headers={"Authorization": "Bearer secret"})
        assert cassette_key(req1) == cassette_key(req2)

    def test_ignores_request_id(self) -> None:
        req1 = _make_request(request_id="id-1")
        req2 = _make_request(request_id="id-2")
        assert cassette_key(req1) == cassette_key(req2)

    def test_ignores_stream_flag(self) -> None:
        req1 = _make_request(stream=False)
        req2 = _make_request(stream=True)
        assert cassette_key(req1) == cassette_key(req2)

    def test_hex_string(self) -> None:
        key = cassette_key(_make_request())
        assert isinstance(key, str)
        assert len(key) == 32
        int(key, 16)


class TestSanitization:
    def test_request_strips_auth_headers(self) -> None:
        req = _make_request(
            headers={
                "Authorization": "Bearer sk-secret",
                "X-Api-Key": "secret",
                "Content-Type": "application/json",
            }
        )
        sanitized = _sanitize_request(req)
        assert "Authorization" not in sanitized["headers"]
        assert "X-Api-Key" not in sanitized["headers"]
        assert sanitized["headers"]["Content-Type"] == "application/json"

    def test_response_strips_transport_metadata(self) -> None:
        resp = _make_response(
            status_code=200,
            response_headers={"x-request-id": "abc"},
            latency_ms=150.5,
        )
        sanitized = _sanitize_response(resp)
        assert "status_code" not in sanitized
        assert "response_headers" not in sanitized
        assert "latency_ms" not in sanitized
        assert sanitized["provider"] == "openai"

    def test_response_preserves_content(self) -> None:
        resp = _make_response()
        sanitized = _sanitize_response(resp)
        assert sanitized["parts"][0]["text"] == "hi"
        assert sanitized["usage"]["input_tokens"] == 5


class TestCassetteEntry:
    def test_roundtrip_json(self) -> None:
        entry = CassetteEntry(
            key="abc123",
            sequence=0,
            request={"provider": "openai", "model": "gpt-5"},
            response={"parts": [{"type": "text", "text": "hi"}]},
            metadata={"recorded_at": 1234567890.0},
        )
        line = entry.to_json_line()
        restored = CassetteEntry.from_json_line(line)
        assert restored.key == entry.key
        assert restored.sequence == entry.sequence
        assert restored.request == entry.request
        assert restored.response == entry.response
        assert restored.metadata == entry.metadata

    def test_json_is_single_line(self) -> None:
        entry = CassetteEntry(
            key="abc123",
            sequence=0,
            request={"body": {"messages": [{"role": "user", "content": "multi\nline"}]}},
            response={},
            metadata={},
        )
        line = entry.to_json_line()
        assert "\n" not in line


class TestCassette:
    def test_load_save_roundtrip(self, tmp_path: Path) -> None:
        cassette = Cassette()
        for i in range(3):
            cassette.add(
                CassetteEntry(
                    key=f"key-{i}",
                    sequence=i,
                    request={"i": i},
                    response={"r": i},
                    metadata={},
                )
            )
        path = tmp_path / "test.jsonl"
        cassette.save(path)

        loaded = Cassette.load(path)
        assert loaded.size == 3
        assert loaded.entries[0].key == "key-0"
        assert loaded.entries[2].sequence == 2

    def test_load_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        loaded = Cassette.load(tmp_path / "nonexistent.jsonl")
        assert loaded.size == 0

    def test_hash_lookup(self) -> None:
        cassette = Cassette()
        cassette.add(
            CassetteEntry(
                key="abc",
                sequence=0,
                request={},
                response={"text": "first"},
                metadata={},
            )
        )
        cassette.add(
            CassetteEntry(
                key="def",
                sequence=1,
                request={},
                response={"text": "second"},
                metadata={},
            )
        )

        result = cassette.lookup("abc")
        assert result is not None
        assert result.response["text"] == "first"

        assert cassette.lookup("nonexistent") is None

    def test_hash_lookup_same_key_returns_sequential(self) -> None:
        cassette = Cassette()
        for i in range(3):
            cassette.add(
                CassetteEntry(
                    key="same",
                    sequence=i,
                    request={},
                    response={"i": i},
                    metadata={},
                )
            )

        first = cassette.lookup("same")
        second = cassette.lookup("same")
        third = cassette.lookup("same")
        assert first is not None
        assert second is not None
        assert third is not None
        assert first.response["i"] == 0
        assert second.response["i"] == 1
        assert third.response["i"] == 2
        assert cassette.lookup("same") is None

    def test_sequential_mode(self) -> None:
        cassette = Cassette()
        for i in range(3):
            cassette.add(
                CassetteEntry(
                    key=f"key-{i}",
                    sequence=i,
                    request={},
                    response={"i": i},
                    metadata={},
                )
            )

        first = cassette.next_sequential()
        second = cassette.next_sequential()
        third = cassette.next_sequential()
        assert first is not None
        assert second is not None
        assert third is not None
        assert first.response["i"] == 0
        assert second.response["i"] == 1
        assert third.response["i"] == 2
        assert cassette.next_sequential() is None

    def test_reset_cursors(self) -> None:
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
        cassette.lookup("abc")
        assert cassette.lookup("abc") is None

        cassette.reset_cursors()
        assert cassette.lookup("abc") is not None

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        cassette = Cassette()
        cassette.add(
            CassetteEntry(
                key="k",
                sequence=0,
                request={},
                response={},
                metadata={},
            )
        )
        deep_path = tmp_path / "a" / "b" / "c" / "test.jsonl"
        cassette.save(deep_path)
        assert deep_path.exists()


class TestCassetteRecorder:
    def test_records_request_response_pair(self, tmp_path: Path) -> None:
        recorder = CassetteRecorder(tmp_path / "test.jsonl")
        req = _make_request()
        resp = _make_response()

        recorder._on_request(req)
        recorder._on_response(req, resp)

        assert recorder.entry_count == 1
        entry = recorder.cassette.entries[0]
        assert entry.key == cassette_key(req)
        assert entry.sequence == 0
        assert entry.metadata["provider"] == "openai"
        assert entry.metadata["model"] == "gpt-5"

    def test_records_multiple_pairs(self, tmp_path: Path) -> None:
        recorder = CassetteRecorder(tmp_path / "test.jsonl")

        for i in range(3):
            req = _make_request(body={"messages": [{"role": "user", "content": f"msg-{i}"}]})
            resp = _make_response()
            recorder._on_request(req)
            recorder._on_response(req, resp)

        assert recorder.entry_count == 3
        for i, entry in enumerate(recorder.cassette.entries):
            assert entry.sequence == i

    def test_save_writes_to_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        recorder = CassetteRecorder(path)

        req = _make_request()
        resp = _make_response()
        recorder._on_request(req)
        recorder._on_response(req, resp)
        recorder.save()

        assert path.exists()
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["key"] == cassette_key(req)

    def test_close_flushes(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        recorder = CassetteRecorder(path)

        req = _make_request()
        resp = _make_response()
        recorder._on_request(req)
        recorder._on_response(req, resp)
        recorder.close()

        assert path.exists()

    def test_hooks_attribute(self, tmp_path: Path) -> None:
        recorder = CassetteRecorder(tmp_path / "test.jsonl")
        assert recorder.hooks.on_request is not None
        assert recorder.hooks.on_response is not None

    def test_sanitizes_auth_headers(self, tmp_path: Path) -> None:
        recorder = CassetteRecorder(tmp_path / "test.jsonl")
        req = _make_request(headers={"Authorization": "Bearer sk-secret"})
        resp = _make_response()
        recorder._on_request(req)
        recorder._on_response(req, resp)

        entry = recorder.cassette.entries[0]
        assert "Authorization" not in entry.request.get("headers", {})


class TestCassetteReplayClient:
    def test_hash_lookup_hit(self) -> None:
        req = _make_request()
        key = cassette_key(req)
        cassette = Cassette()
        cassette.add(
            CassetteEntry(
                key=key,
                sequence=0,
                request=_sanitize_request(req),
                response=_sanitize_response(_make_response()),
                metadata={},
            )
        )

        replay = CassetteReplayClient(cassette)
        result = replay.lookup(req)
        assert result is not None
        assert result.text == "hi"

    def test_hash_lookup_miss(self) -> None:
        cassette = Cassette()
        replay = CassetteReplayClient(cassette)
        result = replay.lookup(_make_request())
        assert result is None

    def test_sequential_mode(self) -> None:
        cassette = Cassette()
        for i in range(2):
            cassette.add(
                CassetteEntry(
                    key=f"key-{i}",
                    sequence=i,
                    request={},
                    response=_sanitize_response(
                        _make_response(parts=[ContentPart(type="text", text=f"resp-{i}")])
                    ),
                    metadata={},
                )
            )

        replay = CassetteReplayClient(cassette, sequential=True)
        r0 = replay.lookup(_make_request())
        assert r0 is not None
        assert r0.text == "resp-0"

        r1 = replay.lookup(_make_request(body={"different": True}))
        assert r1 is not None
        assert r1.text == "resp-1"


class TestUseCassette:
    def test_record_mode_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        with use_cassette(path, mode=CassetteMode.RECORD) as ctx:
            assert ctx.mode == CassetteMode.RECORD
            req = _make_request()
            resp = _make_response()
            recorder = ctx._recorder
            assert recorder is not None
            recorder._on_request(req)
            recorder._on_response(req, resp)

        assert path.exists()
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_replay_mode_reads_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        req = _make_request()
        key = cassette_key(req)

        # Pre-write a cassette
        cassette = Cassette()
        cassette.add(
            CassetteEntry(
                key=key,
                sequence=0,
                request=_sanitize_request(req),
                response=_sanitize_response(_make_response()),
                metadata={},
            )
        )
        cassette.save(path)

        with use_cassette(path, mode=CassetteMode.REPLAY) as ctx:
            result = ctx.lookup(req)
            assert result is not None
            assert result.text == "hi"

    def test_replay_mode_miss_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        path.write_text("")

        with use_cassette(path, mode=CassetteMode.REPLAY) as ctx:
            result = ctx.lookup(_make_request())
            assert result is None

    def test_auto_mode_saves_new_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"

        with use_cassette(path, mode=CassetteMode.AUTO) as ctx:
            req = _make_request()
            resp = _make_response()
            recorder = ctx._recorder
            assert recorder is not None
            recorder._on_request(req)
            recorder._on_response(req, resp)

        loaded = Cassette.load(path)
        assert loaded.size == 1

    def test_auto_mode_replays_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        req = _make_request()
        key = cassette_key(req)

        cassette = Cassette()
        cassette.add(
            CassetteEntry(
                key=key,
                sequence=0,
                request=_sanitize_request(req),
                response=_sanitize_response(_make_response()),
                metadata={},
            )
        )
        cassette.save(path)

        with use_cassette(path, mode=CassetteMode.AUTO) as ctx:
            result = ctx.lookup(req)
            assert result is not None
            assert result.text == "hi"

    def test_entry_count(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        with use_cassette(path, mode=CassetteMode.RECORD) as ctx:
            assert ctx.entry_count == 0
            req = _make_request()
            resp = _make_response()
            recorder = ctx._recorder
            assert recorder is not None
            recorder._on_request(req)
            recorder._on_response(req, resp)
            assert ctx.entry_count == 1


class TestCassetteMissError:
    def test_error_message(self) -> None:
        err = CassetteMissError("abc123", provider="openai", model="gpt-5")
        assert "abc123" in str(err)
        assert "openai" in str(err)
        assert "RECORD" in str(err)
        assert err.key == "abc123"
        assert err.provider == "openai"
        assert err.model == "gpt-5"
