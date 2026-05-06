"""Verify ``BaseProviderClient`` propagates KaosContext IDs into log records.

kaos-core's ``ContextFilter`` reads ``record.session_id`` and
``record.trace_id`` from each ``LogRecord``. Before this test, kaos-llm-
client never set them, so every log line rendered ``[session=- trace=-]``
and the structured-logging contract documented in CLAUDE.md was silently
broken.

These tests assert that:

- ``BaseProviderClient._log_extra(...)`` returns ``session_id`` /
  ``trace_id`` populated from ``self._context``.
- The cache-hit log line emitted by ``request_async`` carries those
  attributes on the captured ``LogRecord``.
"""

from __future__ import annotations

import logging
import logging.handlers
from dataclasses import dataclass, field
from typing import Any

from kaos_llm_client.cache import CacheBackend
from kaos_llm_client.providers import base as base_module
from kaos_llm_client.providers.openai import OpenAIClient
from kaos_llm_client.settings import KaosLLMSettings
from kaos_llm_client.types import (
    ContentPart,
    ProviderResponse,
    UsageInfo,
)


@dataclass
class _FakeContext:
    """Minimal stand-in for ``kaos_core.context.KaosContext``."""

    session_id: str | None = None
    trace_id: str | None = None
    _config: dict[str, Any] = field(default_factory=dict)


class _MemoryCache(CacheBackend):
    """In-memory cache so the cache-hit branch fires without disk I/O."""

    def __init__(self) -> None:
        self._store: dict[str, ProviderResponse] = {}

    def get(self, key: str) -> ProviderResponse | None:
        return self._store.get(key)

    def put(self, key: str, response: ProviderResponse) -> None:
        self._store[key] = response

    def clear(self) -> None:
        self._store.clear()


def _canned_response() -> ProviderResponse:
    return ProviderResponse(
        provider="openai",
        model="gpt-5",
        raw={"id": "cached"},
        parts=[ContentPart(type="text", text="cached hello")],
        usage=UsageInfo(input_tokens=1, output_tokens=1, total_tokens=2),
    )


class TestLogExtraHelper:
    """Direct unit coverage on ``_log_extra``."""

    def test_log_extra_pulls_ids_from_context(self) -> None:
        ctx = _FakeContext(session_id="s-test", trace_id="t-test")
        client = OpenAIClient(model="gpt-5", api_key="test-key", context=ctx)

        extra = client._log_extra(provider="openai")

        assert extra["session_id"] == "s-test"
        assert extra["trace_id"] == "t-test"
        assert extra["provider"] == "openai"

    def test_log_extra_no_context_yields_dash_placeholder(self) -> None:
        """Missing session/trace IDs render as ``"-"`` (matching kaos-core's
        StructuredFormatter convention) rather than ``None``, so the
        formatted output reads ``[session=- trace=-]`` instead of
        ``[session=None trace=None]``.
        """
        client = OpenAIClient(model="gpt-5", api_key="test-key")

        extra = client._log_extra()

        assert extra["session_id"] == "-"
        assert extra["trace_id"] == "-"

    def test_log_extra_falls_back_to_request_id_for_trace(self) -> None:
        """When context has no trace_id, use the first 16 chars of request_id."""
        ctx = _FakeContext(session_id="s-test", trace_id=None)
        client = OpenAIClient(model="gpt-5", api_key="test-key", context=ctx)

        from kaos_llm_client.types import ProviderRequest

        req = ProviderRequest(
            provider="openai",
            model="gpt-5",
            endpoint="/v1/chat/completions",
            body={},
            request_id="abcdef0123456789FFFFFFFFFFFFFF",
        )
        extra = client._log_extra(request=req)

        assert extra["session_id"] == "s-test"
        assert extra["trace_id"] == "abcdef0123456789"


class TestCacheHitLogContext:
    """Cache-hit log line carries session_id / trace_id."""

    def test_cache_hit_log_carries_session_and_trace_ids(self) -> None:
        ctx = _FakeContext(session_id="s-test", trace_id="t-test")
        cache = _MemoryCache()

        # cache_enabled=True so request_async actually consults the cache
        # backend. Without this, _resolve_cache_policy short-circuits to
        # SKIP and the cache-hit branch never fires.
        settings = KaosLLMSettings(cache_enabled=True)
        client = OpenAIClient(
            model="gpt-5",
            api_key="test-key",
            context=ctx,
            cache=cache,
            settings=settings,
        )

        # Pre-seed the cache so request_async takes the cache-hit branch
        # before any HTTP transport is touched. We compute the cache key
        # exactly the way request_async does.
        from kaos_llm_client.cache import cache_key

        request = client._build_request(
            [{"role": "user", "content": "Hi"}],
            stream=False,
        )
        request.headers.update(client._build_headers())
        key = cache_key(
            request,
            base_url=client._base_url,
            auth_scope=client._cache_auth_scope(),
        )
        cache.put(key, _canned_response())

        # Capture log records emitted by the providers.base logger. The
        # kaos-core hierarchy uses propagate=False, so we attach a
        # MemoryHandler to the module logger directly (same trick as
        # test_transport_coverage.py:test_retry_log_includes_error_message).
        memory_handler = logging.handlers.MemoryHandler(capacity=100)
        memory_handler.setLevel(logging.DEBUG)
        # The handler must run the module's filters first so the
        # ContextFilter has a chance to populate record.session_id /
        # trace_id from the kaos-core formatter; but our _log_extra path
        # already sets them via ``extra=``, so just attaching is enough.
        base_module.logger.addHandler(memory_handler)
        previous_level = base_module.logger.level
        base_module.logger.setLevel(logging.DEBUG)
        try:
            response = client.chat([{"role": "user", "content": "Hi"}])
        finally:
            base_module.logger.removeHandler(memory_handler)
            base_module.logger.setLevel(previous_level)

        assert response.text == "cached hello"

        cache_hit_records = [r for r in memory_handler.buffer if "Cache hit" in r.getMessage()]
        assert cache_hit_records, (
            "expected a 'Cache hit' log record; "
            f"saw: {[r.getMessage() for r in memory_handler.buffer]}"
        )

        record = cache_hit_records[0]
        assert getattr(record, "session_id", None) == "s-test"
        assert getattr(record, "trace_id", None) == "t-test"
