"""Tests for ConcurrencyLimitedClient — semaphore limiting."""

from __future__ import annotations

import asyncio
from typing import Any

from kaos_llm_client.profiles import ModelProfile
from kaos_llm_client.providers.concurrency import ConcurrencyLimitedClient
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart, ProviderResponse, UsageInfo


def _make_response(text: str = "ok") -> ProviderResponse:
    return ProviderResponse(
        provider="function",
        model="test",
        raw={},
        parts=[ContentPart(type="text", text=text)],
        usage=UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15),
    )


class TestConcurrencyLimitedClient:
    def test_basic_request(self) -> None:
        """A basic request passes through the semaphore."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response("concurrent ok")

        inner = FunctionClient(function=handler)
        client = ConcurrencyLimitedClient(inner, limit=5)
        response = client.chat([{"role": "user", "content": "Hi"}])
        assert response.text == "concurrent ok"

    async def test_limits_concurrency(self) -> None:
        """Launch limit+1 concurrent tasks, verify at most limit run simultaneously."""
        limit = 2
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def slow_handler(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> ProviderResponse:
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
            # Simulate work
            await asyncio.sleep(0.05)
            async with lock:
                current_concurrent -= 1
            return _make_response("done")

        inner = FunctionClient(function=slow_handler)
        client = ConcurrencyLimitedClient(inner, limit=limit)

        # Launch limit + 1 tasks concurrently
        tasks = [
            asyncio.create_task(client.request_async([{"role": "user", "content": f"req {i}"}]))
            for i in range(limit + 1)
        ]

        results = await asyncio.gather(*tasks)
        assert all(r.text == "done" for r in results)
        assert max_concurrent <= limit
