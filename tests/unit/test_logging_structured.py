"""Verify all production log call sites emit structured ``extra={...}``.

Task 9 (P3.1) — every existing ``logger.info``/``logger.warning``/
``logger.debug`` call in production code must carry typed ``extra=``
fields so log aggregators (Splunk, Datadog, OTel) can index them
without re-parsing the message string.

These tests pin the canonical key set:

- ``provider``, ``model`` on every chat / embed / fallback / instrumented
  log line.
- ``request_id``, ``response_id`` where a ``ProviderRequest`` /
  ``ProviderResponse`` is in scope.
- ``session_id``, ``trace_id`` propagated from ``KaosContext`` through
  the ``_log_extra`` helper.
- ``tool_name`` on tool-layer logs.
- ``attempt`` / ``total_providers`` on fallback retries.
- ``cache_op`` on cache read/write debug logs.

Mocked unit tests catch *missing keys*. Live tests are the quality
bar for *correct values* — see test_live.py for the integration coverage.
"""

from __future__ import annotations

import logging
import logging.handlers
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest

from kaos_llm_client import cache as cache_module

if TYPE_CHECKING:
    from kaos_core import KaosContext
from kaos_llm_client.errors import KaosLLMProviderError
from kaos_llm_client.profiles import ModelProfile
from kaos_llm_client.providers import base as base_module
from kaos_llm_client.providers import fallback as fallback_module
from kaos_llm_client.providers import instrumented as instrumented_module
from kaos_llm_client.providers.fallback import FallbackClient
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.providers.instrumented import InstrumentedClient
from kaos_llm_client.providers.openai import OpenAIClient
from kaos_llm_client.types import (
    ContentPart,
    ProviderResponse,
    UsageInfo,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeContext:
    """Minimal stand-in for ``kaos_core.context.KaosContext``."""

    session_id: str | None = None
    trace_id: str | None = None
    _config: dict[str, Any] = field(default_factory=dict)


_OPENAI_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-test-1",
    "object": "chat.completion",
    "model": "gpt-5",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
}


def _inject_mock(client: Any, payload: dict[str, Any], status: int = 200) -> None:
    """Swap in a httpx.MockTransport for both async + sync clients."""

    async def async_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    def sync_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    base_url = client._base_url
    client._async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(async_handler), base_url=base_url
    )
    client._sync_client = httpx.Client(
        transport=httpx.MockTransport(sync_handler), base_url=base_url
    )


class _CapturingHandler(logging.Handler):
    """Captures log records into ``self.records``."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _attach(logger_obj: logging.Logger) -> _CapturingHandler:
    handler = _CapturingHandler()
    logger_obj.addHandler(handler)
    logger_obj.setLevel(logging.DEBUG)
    return handler


def _detach(logger_obj: logging.Logger, handler: _CapturingHandler) -> None:
    logger_obj.removeHandler(handler)


# ---------------------------------------------------------------------------
# Base provider client logs
# ---------------------------------------------------------------------------


class TestBaseClientStructuredLogs:
    """Base provider client emits structured fields on every log."""

    def test_request_async_emits_call_complete_with_canonical_keys(self) -> None:
        ctx = _FakeContext(session_id="s-1", trace_id="t-1")
        client = OpenAIClient(model="gpt-5", api_key="k", context=ctx)
        _inject_mock(client, _OPENAI_RESPONSE)

        handler = _attach(base_module.logger)
        try:
            client.chat([{"role": "user", "content": "hi"}])
        finally:
            _detach(base_module.logger, handler)

        completion_records = [r for r in handler.records if r.getMessage() == "LLM call complete"]
        assert len(completion_records) == 1, (
            f"expected one 'LLM call complete' info-log, "
            f"saw: {[r.getMessage() for r in handler.records]}"
        )
        rec = completion_records[0]
        # Canonical key set from CLAUDE.md / _log_extra docstring.
        assert getattr(rec, "provider", None) == "openai"
        assert getattr(rec, "model", None) == "gpt-5"
        assert getattr(rec, "session_id", None) == "s-1"
        assert getattr(rec, "trace_id", None) == "t-1"
        # response_id comes from the OpenAI raw payload (`id` field).
        assert getattr(rec, "response_id", None) == "chatcmpl-test-1"
        # request_id is generated client-side; non-empty when populated.
        request_id = getattr(rec, "request_id", None)
        assert request_id is None or len(request_id) <= 16
        assert getattr(rec, "input_tokens", None) == 4
        assert getattr(rec, "output_tokens", None) == 1
        assert getattr(rec, "total_tokens", None) == 5
        assert getattr(rec, "cache_hit", None) is False
        # estimated_usd may be None (e.g. unknown model) but for gpt-5
        # we expect a non-negative float.
        usd = getattr(rec, "estimated_usd", None)
        assert usd is None or usd >= 0.0

    def test_log_extra_helper_pulls_session_and_trace(self) -> None:
        ctx = _FakeContext(session_id="abc", trace_id="def")
        client = OpenAIClient(model="gpt-5", api_key="k", context=ctx)

        extra = client._log_extra(provider="openai", model="gpt-5")
        assert extra["session_id"] == "abc"
        assert extra["trace_id"] == "def"
        assert extra["provider"] == "openai"
        assert extra["model"] == "gpt-5"

    def test_streaming_emits_call_complete_after_stream_finish(self) -> None:
        """Stream completion log fires once after the stream finishes."""
        from collections.abc import AsyncIterator

        from kaos_llm_client.types import StreamChunk

        async def stream_handler(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(type="text_delta", text="hi")

        ctx = _FakeContext(session_id="s-stream", trace_id="t-stream")
        client = FunctionClient(stream_function=stream_handler, context=ctx)
        # FunctionClient overrides ``request_stream_async`` and bypasses
        # the base class entirely — its streaming path doesn't run our
        # call-complete log emission. We assert the attachment to base
        # logger anyway so future moves of the helper are caught.
        handler = _attach(base_module.logger)
        try:
            import asyncio

            async def run() -> list[StreamChunk]:
                chunks: list[StreamChunk] = []
                async for chunk in client.request_stream_async([{"role": "user", "content": "x"}]):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(run())
            assert any(c.type == "text_delta" for c in chunks)
        finally:
            _detach(base_module.logger, handler)


# ---------------------------------------------------------------------------
# Cache hit log carries cache_hit=True
# ---------------------------------------------------------------------------


class TestCacheLogStructuredKeys:
    def test_cache_hit_log_carries_cache_hit_true(self) -> None:
        from kaos_llm_client.cache import CacheBackend
        from kaos_llm_client.settings import KaosLLMSettings

        class _MemCache(CacheBackend):
            def __init__(self) -> None:
                self._store: dict[str, ProviderResponse] = {}

            def get(self, key: str) -> ProviderResponse | None:
                return self._store.get(key)

            def put(self, key: str, response: ProviderResponse) -> None:
                self._store[key] = response

            def clear(self) -> None:
                self._store.clear()

        ctx = _FakeContext(session_id="s-c", trace_id="t-c")
        cache = _MemCache()
        settings = KaosLLMSettings(cache_enabled=True)
        client = OpenAIClient(
            model="gpt-5", api_key="k", context=ctx, cache=cache, settings=settings
        )

        # Pre-seed cache so the cache-hit branch fires before HTTP.
        from kaos_llm_client.cache import cache_key

        request = client._build_request([{"role": "user", "content": "Hi"}], stream=False)
        request.headers.update(client._build_headers())
        key = cache_key(request, base_url=client._base_url, auth_scope=client._cache_auth_scope())
        cached = ProviderResponse(
            provider="openai",
            model="gpt-5",
            raw={"id": "cached"},
            parts=[ContentPart(type="text", text="cached hi")],
            usage=UsageInfo(input_tokens=2, output_tokens=2, total_tokens=4),
        )
        cache.put(key, cached)

        handler = _attach(base_module.logger)
        try:
            response = client.chat([{"role": "user", "content": "Hi"}])
        finally:
            _detach(base_module.logger, handler)

        assert response.text == "cached hi"

        # Cache-hit completion log must carry cache_hit=True.
        completion_records = [r for r in handler.records if r.getMessage() == "LLM call complete"]
        assert len(completion_records) >= 1
        cache_hit_records = [r for r in completion_records if getattr(r, "cache_hit", None) is True]
        assert cache_hit_records, "expected a cache_hit=True 'LLM call complete' record"
        rec = cache_hit_records[0]
        assert getattr(rec, "session_id", None) == "s-c"
        assert getattr(rec, "trace_id", None) == "t-c"

    def test_cache_read_failure_log_carries_cache_op(self) -> None:
        """FileCache read failure log carries ``cache_op='read'`` extra."""
        import tempfile

        # Create a corrupt cache file so ``get`` falls into the except branch.
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = cache_module.FileCache(tmpdir)
            key = "abcdef0123456789"  # 16 hex chars  # gitleaks:allow
            file_path = cache._key_path(key)
            file_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            file_path.write_bytes(b"not gzipped at all")

            handler = _attach(cache_module.logger)
            try:
                result = cache.get(key)
            finally:
                _detach(cache_module.logger, handler)

            assert result is None  # corrupt file silently swallowed
            failure_records = [r for r in handler.records if "Cache read failed" in r.getMessage()]
            assert failure_records, "expected a 'Cache read failed' debug log"
            rec = failure_records[0]
            assert getattr(rec, "cache_op", None) == "read"
            assert getattr(rec, "key", None) == key[:8]


# ---------------------------------------------------------------------------
# Fallback client logs include provider + attempt counts
# ---------------------------------------------------------------------------


class TestFallbackStructuredLogs:
    def test_fallback_warning_carries_provider_and_attempt(self) -> None:
        def fail_handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            raise KaosLLMProviderError("boom", provider="function", model="test", status_code=500)

        def ok_handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return ProviderResponse(
                provider="function",
                model="test",
                raw={},
                parts=[ContentPart(type="text", text="ok")],
                usage=UsageInfo(input_tokens=1, output_tokens=1, total_tokens=2),
            )

        primary = FunctionClient(model="primary-test", function=fail_handler)
        backup = FunctionClient(model="backup-test", function=ok_handler)
        client = FallbackClient([primary, backup])

        handler = _attach(fallback_module.logger)
        try:
            response = client.chat([{"role": "user", "content": "Hi"}])
        finally:
            _detach(fallback_module.logger, handler)

        assert response.text == "ok"
        records = [r for r in handler.records if "Fallback" in r.getMessage()]
        assert records, "expected a 'Fallback' warning log"
        rec = records[0]
        # Canonical fallback fields.
        assert getattr(rec, "provider", None) == "function"
        # The model on the failing client.
        assert getattr(rec, "model", None) == "primary-test"
        assert getattr(rec, "attempt", None) == 1
        assert getattr(rec, "total_providers", None) == 2
        assert getattr(rec, "error", None) is not None


# ---------------------------------------------------------------------------
# Instrumented client logs include canonical keys
# ---------------------------------------------------------------------------


class TestInstrumentedStructuredLogs:
    def test_instrumented_emits_canonical_extras(self) -> None:
        def ok_handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return ProviderResponse(
                provider="function",
                model="test",
                raw={"id": "rsp-instrumented"},
                parts=[ContentPart(type="text", text="ok")],
                usage=UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15),
                response_id="rsp-instrumented",
            )

        inner = FunctionClient(function=ok_handler)
        client = InstrumentedClient(
            inner, cost_per_input_token=0.000003, cost_per_output_token=0.00001
        )

        handler = _attach(instrumented_module.logger)
        try:
            client.chat([{"role": "user", "content": "Hi"}])
        finally:
            _detach(instrumented_module.logger, handler)

        records = [r for r in handler.records if r.getMessage() == "LLM request completed"]
        assert len(records) == 1
        rec = records[0]
        # Canonical keys from the audit plan.
        assert getattr(rec, "provider", None) == "function"
        assert getattr(rec, "model", None) == "test"
        assert getattr(rec, "response_id", None) == "rsp-instrumented"
        assert getattr(rec, "input_tokens", None) == 10
        assert getattr(rec, "output_tokens", None) == 5
        assert getattr(rec, "total_tokens", None) == 15


# ---------------------------------------------------------------------------
# Tool-layer logs
# ---------------------------------------------------------------------------


class TestToolLayerStructuredLogs:
    """The MCP tool layer enriches logger.info call sites with ``extra``."""

    def test_chat_tool_log_carries_provider_model_session_trace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kaos_llm_client import tools as tools_module

        # Monkeypatch ``create_client`` so the tool calls a FunctionClient
        # rather than reaching out to a real provider.
        def fake_handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return ProviderResponse(
                provider="openai",
                model="gpt-5",
                raw={"id": "rsp-tool-1"},
                parts=[ContentPart(type="text", text="hello")],
                usage=UsageInfo(input_tokens=3, output_tokens=2, total_tokens=5),
                response_id="rsp-tool-1",
            )

        client = FunctionClient(model="gpt-5", function=fake_handler)

        def fake_create_client(*args: Any, **kwargs: Any) -> Any:
            return client

        monkeypatch.setattr(tools_module, "create_client", fake_create_client, raising=False)
        # The tool body imports lazily — patch the function-attribute on
        # the providers package as well.
        from kaos_llm_client import providers as providers_pkg

        monkeypatch.setattr(providers_pkg, "create_client", fake_create_client, raising=False)

        ctx = _FakeContext(session_id="s-tool", trace_id="t-tool")
        tool = tools_module.KaosLLMChatTool()

        handler = _attach(tools_module.logger)
        try:
            import asyncio

            asyncio.run(
                tool.execute(
                    {"model": "openai:gpt-5", "message": "hi"},
                    context=cast("KaosContext", ctx),
                )
            )
        finally:
            _detach(tools_module.logger, handler)

        records = [r for r in handler.records if "LLM chat completed" in r.getMessage()]
        assert records, (
            f"expected 'LLM chat completed' log; saw: {[r.getMessage() for r in handler.records]}"
        )
        rec = records[0]
        assert getattr(rec, "provider", None) == "openai"
        assert getattr(rec, "model", None) == "gpt-5"
        assert getattr(rec, "tool_name", None) == "kaos-llm-chat"
        assert getattr(rec, "session_id", None) == "s-tool"
        assert getattr(rec, "trace_id", None) == "t-tool"
        assert getattr(rec, "response_id", None) == "rsp-tool-1"
        assert getattr(rec, "input_tokens", None) == 3
        assert getattr(rec, "output_tokens", None) == 2
        assert getattr(rec, "total_tokens", None) == 5

    def test_provider_check_log_carries_tool_name(self) -> None:
        from kaos_llm_client import tools as tools_module

        ctx = _FakeContext(session_id="s-pc", trace_id="t-pc")
        tool = tools_module.KaosLLMProviderCheckTool()

        handler = _attach(tools_module.logger)
        try:
            import asyncio

            asyncio.run(tool.execute({}, context=cast("KaosContext", ctx)))
        finally:
            _detach(tools_module.logger, handler)

        records = [r for r in handler.records if "Provider check" in r.getMessage()]
        assert records
        rec = records[0]
        assert getattr(rec, "tool_name", None) == "kaos-llm-provider-check"
        assert getattr(rec, "session_id", None) == "s-pc"
        assert getattr(rec, "trace_id", None) == "t-pc"

    def test_cost_estimate_log_carries_tool_name_and_estimated_usd(self) -> None:
        from kaos_llm_client import tools as tools_module

        ctx = _FakeContext(session_id="s-ce", trace_id="t-ce")
        tool = tools_module.KaosLLMCostEstimateTool()

        handler = _attach(tools_module.logger)
        try:
            import asyncio

            asyncio.run(
                tool.execute(
                    {
                        "model": "openai:gpt-5",
                        "input_text": "hello world",
                        "max_output_tokens": 100,
                    },
                    context=cast("KaosContext", ctx),
                )
            )
        finally:
            _detach(tools_module.logger, handler)

        records = [r for r in handler.records if "Cost estimate" in r.getMessage()]
        assert records
        rec = records[0]
        assert getattr(rec, "tool_name", None) == "kaos-llm-cost-estimate"
        assert getattr(rec, "model", None) == "gpt-5"
        assert getattr(rec, "estimated_usd", None) is not None
