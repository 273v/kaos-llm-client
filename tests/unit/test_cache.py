"""Tests for kaos_llm_client.cache — cache key generation and file cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import kaos_llm_client.cache as cache_module
from kaos_llm_client.cache import FileCache, NullCache, cache_key
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


class TestCacheKey:
    def test_deterministic(self):
        req = _make_request()
        key1 = cache_key(req)
        key2 = cache_key(req)
        assert key1 == key2

    def test_different_bodies_different_keys(self):
        req1 = _make_request(body={"messages": [{"role": "user", "content": "hello"}]})
        req2 = _make_request(body={"messages": [{"role": "user", "content": "world"}]})
        assert cache_key(req1) != cache_key(req2)

    def test_different_models_different_keys(self):
        req1 = _make_request(model="gpt-5")
        req2 = _make_request(model="gpt-4.1")
        assert cache_key(req1) != cache_key(req2)

    def test_different_providers_different_keys(self):
        req1 = _make_request(provider="openai")
        req2 = _make_request(provider="anthropic")
        assert cache_key(req1) != cache_key(req2)

    def test_key_is_hex_string(self):
        key = cache_key(_make_request())
        assert isinstance(key, str)
        assert len(key) == 32  # 16 bytes hex = 32 chars
        int(key, 16)  # should not raise

    def test_ignores_headers(self):
        req1 = _make_request()
        req2 = _make_request(headers={"Authorization": "Bearer secret"})
        assert cache_key(req1) == cache_key(req2)

    def test_ignores_request_id(self):
        req1 = _make_request(request_id="id1")
        req2 = _make_request(request_id="id2")
        assert cache_key(req1) == cache_key(req2)


class TestNullCache:
    def test_get_returns_none(self):
        cache = NullCache()
        assert cache.get("any_key") is None

    def test_put_does_nothing(self):
        cache = NullCache()
        cache.put("key", _make_response())  # should not raise

    def test_clear_does_nothing(self):
        cache = NullCache()
        cache.clear()  # should not raise


class TestFileCache:
    def test_roundtrip(self, tmp_path: Path):
        cache = FileCache(tmp_path / "cache")
        resp = _make_response()
        cache.put("testkey123456789a", resp)
        result = cache.get("testkey123456789a")
        assert result is not None
        assert result.text == "hi"
        assert result.provider == "openai"
        assert result.usage.input_tokens == 5

    def test_miss(self, tmp_path: Path):
        cache = FileCache(tmp_path / "cache")
        assert cache.get("nonexistent1234567") is None

    def test_clear(self, tmp_path: Path):
        cache = FileCache(tmp_path / "cache")
        cache.put("testkey123456789a", _make_response())
        cache.clear()
        assert cache.get("testkey123456789a") is None

    def test_clear_rejects_sibling_with_matching_prefix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        allowed_root = tmp_path / "allowed"
        sibling = tmp_path / "allowed_evil"
        monkeypatch.setattr(cache_module, "_CLEAR_ALLOWLIST_ROOTS", (allowed_root,))

        cache = FileCache(sibling)
        cache.put("testkey123456789a", _make_response())

        with pytest.raises(RuntimeError):
            cache.clear()

        assert sibling.exists()

    def test_directory_structure(self, tmp_path: Path):
        cache = FileCache(tmp_path / "cache")
        cache.put("abcdef1234567890ab", _make_response())
        # Should be stored at ab/abcdef1234567890ab.json.gz
        expected = tmp_path / "cache" / "ab" / "abcdef1234567890ab.json.gz"
        assert expected.exists()

    def test_gzip_compressed(self, tmp_path: Path):
        import gzip

        cache = FileCache(tmp_path / "cache")
        cache.put("abcdef1234567890ab", _make_response())
        file_path = tmp_path / "cache" / "ab" / "abcdef1234567890ab.json.gz"
        # Verify it's valid gzip
        data = gzip.decompress(file_path.read_bytes())
        parsed = json.loads(data)
        assert parsed["provider"] == "openai"

    def test_corrupt_entry_returns_none(self, tmp_path: Path):
        cache = FileCache(tmp_path / "cache")
        file_path = tmp_path / "cache" / "ab" / "abcdef1234567890ab.json.gz"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"not-a-gzip-payload")

        assert cache.get("abcdef1234567890ab") is None
