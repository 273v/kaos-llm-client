"""Cost-estimation + per-call observability tests.

Task 10 (P3.2) — every successful provider call must emit one
``LLM call complete`` info-log carrying token counts and an estimated
USD cost. This is the foundation the live-tier $50/run cost ceiling
documented in ``docs/oss/40-ci-cd/live-tier.yml.md`` builds on.

Coverage:

- ``estimate_call_cost()`` and ``lookup_pricing()`` happy / sad paths.
- One ``LLM call complete`` info-log per successful chat call (mock
  transport, no real provider hit).
- Cache hits emit a separate completion log with ``cache_hit=True``;
  ``client.metrics()`` increments accordingly.
- Streaming chats emit one completion log after the stream finishes.
- Unknown models surface ``estimated_usd=None`` in the log.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from kaos_llm_client.cache import CacheBackend, cache_key
from kaos_llm_client.cost import (
    MODEL_PRICING,
    estimate_call_cost,
    lookup_pricing,
)
from kaos_llm_client.profiles import ModelProfile
from kaos_llm_client.providers import base as base_module
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.providers.openai import OpenAIClient
from kaos_llm_client.settings import KaosLLMSettings
from kaos_llm_client.types import (
    ContentPart,
    ProviderResponse,
    StreamChunk,
    UsageInfo,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_OPENAI_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-cost-1",
    "object": "chat.completion",
    "model": "gpt-5",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    # 1M / 1M tokens — easy multiplication to verify cost rounding.
    "usage": {
        "prompt_tokens": 1_000_000,
        "completion_tokens": 1_000_000,
        "total_tokens": 2_000_000,
    },
}


def _inject_mock(client: Any, payload: dict[str, Any], status: int = 200) -> None:
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
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _MemCache(CacheBackend):
    def __init__(self) -> None:
        self._store: dict[str, ProviderResponse] = {}

    def get(self, key: str) -> ProviderResponse | None:
        return self._store.get(key)

    def put(self, key: str, response: ProviderResponse) -> None:
        self._store[key] = response

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# estimate_call_cost / lookup_pricing
# ---------------------------------------------------------------------------


class TestEstimateCallCost:
    def test_known_model_full_million_tokens_each_way(self) -> None:
        """1M input + 1M output on a known model returns expected USD sum."""
        usage = UsageInfo(input_tokens=1_000_000, output_tokens=1_000_000, total_tokens=2_000_000)
        cost = estimate_call_cost(usage, "gpt-4.1-mini")
        # gpt-4.1-mini: $0.40 input + $1.60 output per 1M = $2.00 total.
        assert cost is not None
        assert pytest.approx(cost, abs=1e-6) == 2.00

    def test_none_usage_returns_zero(self) -> None:
        """Cost helper returns 0.0 (NOT None) when usage is missing."""
        assert estimate_call_cost(None, "gpt-5") == 0.0

    def test_unknown_model_returns_none(self) -> None:
        """Unknown models surface as ``None`` so dashboards can spot gaps."""
        usage = UsageInfo(input_tokens=100, output_tokens=50)
        assert estimate_call_cost(usage, "totally-made-up-2099") is None

    def test_provider_prefix_stripped(self) -> None:
        """``openai:gpt-5`` looks up the same as ``gpt-5``."""
        usage = UsageInfo(input_tokens=1_000_000, output_tokens=0, total_tokens=1_000_000)
        cost_with_prefix = estimate_call_cost(usage, "openai:gpt-5")
        cost_without_prefix = estimate_call_cost(usage, "gpt-5")
        assert cost_with_prefix == cost_without_prefix
        assert cost_with_prefix is not None and cost_with_prefix > 0

    def test_versioned_model_falls_back_to_prefix(self) -> None:
        """``gpt-5-0125`` should be priced as ``gpt-5``."""
        usage = UsageInfo(input_tokens=1_000_000, output_tokens=0)
        priced = estimate_call_cost(usage, "gpt-5-0125")
        plain = estimate_call_cost(usage, "gpt-5")
        assert priced == plain

    def test_pricing_table_override(self) -> None:
        """Caller-supplied pricing tables override the module default."""
        custom = {"my-model": {"input": 100.0, "output": 200.0}}
        usage = UsageInfo(input_tokens=1_000_000, output_tokens=0)
        cost = estimate_call_cost(usage, "my-model", pricing_table=custom)
        assert cost == pytest.approx(100.0)

    def test_lookup_pricing_longest_prefix_wins(self) -> None:
        """``gpt-4.1-mini`` should not be confused with ``gpt-4.1``."""
        # Both keys are in the default table.
        assert lookup_pricing("gpt-4.1-mini") is MODEL_PRICING["gpt-4.1-mini"]
        assert lookup_pricing("gpt-4.1") is MODEL_PRICING["gpt-4.1"]

    # --- Cache-token pricing (added in v1.1) -------------------------

    def test_cache_read_billed_at_discounted_rate_when_published(self) -> None:
        """Anthropic claude-opus-4-7 publishes a cache-read rate of $0.50/MTok.
        With 1M cache_read tokens (counted within input_tokens) and 0 output,
        the cost should equal exactly the cache-read rate, not the input rate.
        """
        usage = UsageInfo(
            input_tokens=1_000_000,
            output_tokens=0,
            total_tokens=1_000_000,
            cache_read_tokens=1_000_000,
        )
        cost = estimate_call_cost(usage, "claude-opus-4-7")
        # Cache-read rate is 0.50 (vs base input 5.00) → exactly $0.50.
        assert cost == pytest.approx(0.50, abs=1e-6)

    def test_cache_creation_billed_at_premium_rate_when_published(self) -> None:
        """claude-opus-4-7 publishes a 5m cache-write rate of $6.25/MTok.
        With 1M cache_creation tokens (subset of input_tokens) and 0 output,
        the cost should equal that premium rate.
        """
        usage = UsageInfo(
            input_tokens=1_000_000,
            output_tokens=0,
            total_tokens=1_000_000,
            cache_creation_tokens=1_000_000,
        )
        cost = estimate_call_cost(usage, "claude-opus-4-7")
        assert cost == pytest.approx(6.25, abs=1e-6)

    def test_fresh_input_subtracts_cache_columns(self) -> None:
        """Mixed call: 1M input total, 600k cache reads + 100k cache writes
        + 300k fresh. Cost should be the per-rate sum, not 1M x base.
        """
        usage = UsageInfo(
            input_tokens=1_000_000,
            output_tokens=0,
            cache_read_tokens=600_000,
            cache_creation_tokens=100_000,
        )
        cost = estimate_call_cost(usage, "claude-opus-4-7")
        # 300k x 5.00 + 600k x 0.50 + 100k x 6.25 (all per million)
        # = 1.5 + 0.3 + 0.625 = 2.425
        assert cost == pytest.approx(2.425, abs=1e-6)

    def test_no_published_cache_rate_falls_back_to_input(self) -> None:
        """gpt-4.1-mini has no cache_read / cache_creation column. Cache
        tokens should be billed at the base input rate (upper bound).
        """
        usage = UsageInfo(
            input_tokens=1_000_000,
            output_tokens=0,
            cache_read_tokens=400_000,
            cache_creation_tokens=200_000,
        )
        cost = estimate_call_cost(usage, "gpt-4.1-mini")
        # All 1M tokens billed at $0.40/MTok — same as if cache columns
        # were ignored (this is the documented upper-bound fallback).
        assert cost == pytest.approx(0.40, abs=1e-6)

    def test_negative_cache_tokens_clamped(self) -> None:
        """Defensive: a malformed usage record with negative cache tokens
        should not yield a negative cost.
        """
        usage = UsageInfo(
            input_tokens=1_000_000,
            output_tokens=0,
            cache_read_tokens=-50,
            cache_creation_tokens=-100,
        )
        cost = estimate_call_cost(usage, "claude-opus-4-7")
        assert cost is not None
        assert cost > 0  # all 1M billed at the base input rate
        assert cost == pytest.approx(5.00, abs=1e-6)

    def test_pricing_table_explicit_cache_rates(self) -> None:
        """Override-supplied table can carry cache rates too."""
        custom = {
            "test-cache-model": {
                "input": 10.00,
                "output": 30.00,
                "cache_read": 1.00,
                "cache_creation": 12.00,
            }
        }
        usage = UsageInfo(
            input_tokens=2_000_000,
            output_tokens=1_000_000,
            cache_read_tokens=1_000_000,
            cache_creation_tokens=500_000,
        )
        cost = estimate_call_cost(usage, "test-cache-model", pricing_table=custom)
        # fresh = 500k x 10 = 5.0; output = 1M x 30 = 30.0
        # cache_read = 1M x 1 = 1.0; cache_create = 500k x 12 = 6.0
        # total = 42.0
        assert cost == pytest.approx(42.0, abs=1e-6)


# ---------------------------------------------------------------------------
# request_async emits exactly one LLM call complete per chat call
# ---------------------------------------------------------------------------


class TestRequestAsyncCallCompleteLog:
    def test_one_call_complete_per_chat_call_with_cost(self) -> None:
        client = OpenAIClient(model="gpt-5", api_key="k")
        _inject_mock(client, _OPENAI_RESPONSE)

        handler = _CapturingHandler()
        base_module.logger.addHandler(handler)
        base_module.logger.setLevel(logging.DEBUG)
        try:
            client.chat([{"role": "user", "content": "hi"}])
        finally:
            base_module.logger.removeHandler(handler)

        completion_records = [r for r in handler.records if r.getMessage() == "LLM call complete"]
        assert len(completion_records) == 1
        rec = completion_records[0]
        assert getattr(rec, "cache_hit", None) is False
        assert getattr(rec, "input_tokens", None) == 1_000_000
        assert getattr(rec, "output_tokens", None) == 1_000_000
        assert getattr(rec, "total_tokens", None) == 2_000_000
        # gpt-5 default pricing: $2 input + $8 output per 1M = $10 total.
        usd = getattr(rec, "estimated_usd", None)
        assert usd is not None
        assert pytest.approx(usd, abs=1e-6) == 10.00

    def test_call_complete_log_does_not_break_on_unknown_model(self) -> None:
        """Unknown model -> ``estimated_usd=None``, log still emits."""
        client = OpenAIClient(model="totally-made-up-9999", api_key="k")
        _inject_mock(client, _OPENAI_RESPONSE)

        handler = _CapturingHandler()
        base_module.logger.addHandler(handler)
        base_module.logger.setLevel(logging.DEBUG)
        try:
            client.chat([{"role": "user", "content": "hi"}])
        finally:
            base_module.logger.removeHandler(handler)

        completion_records = [r for r in handler.records if r.getMessage() == "LLM call complete"]
        assert len(completion_records) == 1
        assert getattr(completion_records[0], "estimated_usd", None) is None


# ---------------------------------------------------------------------------
# Cache hit emits cache_hit=True + bumps metrics counters
# ---------------------------------------------------------------------------


class TestCacheHitMetrics:
    def test_cache_hit_log_and_counter(self) -> None:
        cache = _MemCache()
        settings = KaosLLMSettings(cache_enabled=True)
        client = OpenAIClient(model="gpt-5", api_key="k", cache=cache, settings=settings)
        _inject_mock(client, _OPENAI_RESPONSE)

        # First call → cache miss; populate the cache.
        client.chat([{"role": "user", "content": "Hi"}])
        assert client.metrics()["cache_misses"] == 1
        assert client.metrics()["cache_hits"] == 0

        # Second call with the same body → cache hit; emits cache_hit=True log.
        handler = _CapturingHandler()
        base_module.logger.addHandler(handler)
        base_module.logger.setLevel(logging.DEBUG)
        try:
            client.chat([{"role": "user", "content": "Hi"}])
        finally:
            base_module.logger.removeHandler(handler)

        assert client.metrics()["cache_hits"] == 1
        # Hit-rate should be 0.5 after one miss + one hit.
        assert client.metrics()["cache_hit_rate"] == pytest.approx(0.5)

        completion_records = [r for r in handler.records if r.getMessage() == "LLM call complete"]
        cache_hit_records = [r for r in completion_records if getattr(r, "cache_hit", None) is True]
        assert cache_hit_records, "expected at least one cache_hit=True completion record"

    def test_metrics_zero_state(self) -> None:
        client = OpenAIClient(model="gpt-5", api_key="k")
        m = client.metrics()
        assert m == {"cache_hits": 0, "cache_misses": 0, "cache_hit_rate": 0.0}

    def test_metrics_pre_seeded_cache_records_hit(self) -> None:
        """A pre-seeded cache emits cache_hit=True on the first call."""
        cache = _MemCache()
        settings = KaosLLMSettings(cache_enabled=True)
        client = OpenAIClient(model="gpt-5", api_key="k", cache=cache, settings=settings)
        request = client._build_request([{"role": "user", "content": "Hi"}], stream=False)
        request.headers.update(client._build_headers())
        key = cache_key(request, base_url=client._base_url, auth_scope=client._cache_auth_scope())
        cached = ProviderResponse(
            provider="openai",
            model="gpt-5",
            raw={"id": "cached"},
            parts=[ContentPart(type="text", text="cached")],
            usage=UsageInfo(input_tokens=2, output_tokens=2, total_tokens=4),
            response_id="cached",
        )
        cache.put(key, cached)

        handler = _CapturingHandler()
        base_module.logger.addHandler(handler)
        base_module.logger.setLevel(logging.DEBUG)
        try:
            response = client.chat([{"role": "user", "content": "Hi"}])
        finally:
            base_module.logger.removeHandler(handler)

        assert response.text == "cached"
        assert client.metrics()["cache_hits"] == 1
        assert client.metrics()["cache_misses"] == 0


# ---------------------------------------------------------------------------
# Streaming emits one completion log after the stream finishes
# ---------------------------------------------------------------------------


class TestStreamingCallComplete:
    def test_streaming_emits_one_completion_log(self) -> None:
        async def stream_handler(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(type="text_delta", text="hi ")
            yield StreamChunk(type="text_delta", text="there")
            yield StreamChunk(
                type="usage", usage=UsageInfo(input_tokens=2, output_tokens=2, total_tokens=4)
            )

        # FunctionClient overrides ``request_stream_async`` and bypasses
        # the base streaming path. To test BaseProviderClient's streaming
        # call-complete emission, we use OpenAIClient with a mocked SSE
        # transport that returns a few delta chunks.
        client = OpenAIClient(model="gpt-5", api_key="k")

        # Build an SSE response body the OpenAI parser will accept.
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"hi "},"index":0}]}',
            'data: {"choices":[{"delta":{"content":"there"},"index":0}]}',
            (
                'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}'
            ),
            "data: [DONE]",
        ]
        sse_body = "\n\n".join(sse_lines).encode() + b"\n\n"

        async def stream_transport(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_body,
                headers={"content-type": "text/event-stream"},
            )

        client._async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(stream_transport), base_url=client._base_url
        )

        handler = _CapturingHandler()
        base_module.logger.addHandler(handler)
        base_module.logger.setLevel(logging.DEBUG)
        try:
            import asyncio

            async def run() -> list[StreamChunk]:
                chunks: list[StreamChunk] = []
                async for chunk in client.request_stream_async([{"role": "user", "content": "hi"}]):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(run())
            assert any(c.type == "text_delta" for c in chunks)
        finally:
            base_module.logger.removeHandler(handler)

        completion_records = [r for r in handler.records if r.getMessage() == "LLM call complete"]
        assert len(completion_records) == 1
        rec = completion_records[0]
        assert getattr(rec, "cache_hit", None) is False
        # Token counts should reflect the final usage chunk.
        assert getattr(rec, "input_tokens", None) == 2
        assert getattr(rec, "output_tokens", None) == 2

    def test_function_client_streaming_path_does_not_crash(self) -> None:
        """FunctionClient bypasses base streaming — sanity check it still works."""

        async def stream_handler(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(type="text_delta", text="x")

        client = FunctionClient(stream_function=stream_handler)
        import asyncio

        async def run() -> list[StreamChunk]:
            chunks: list[StreamChunk] = []
            async for chunk in client.request_stream_async([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        assert any(c.type == "text_delta" for c in chunks)
