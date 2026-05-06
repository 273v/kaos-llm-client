"""Concurrency-limited wrapper that caps parallel async requests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from kaos_llm_client.providers.base import BaseProviderClient
from kaos_llm_client.providers.wrapper import WrapperClient
from kaos_llm_client.transport import run_sync
from kaos_llm_client.types import (
    ProviderResponse,
    RequestOptions,
    StreamChunk,
    ToolChoice,
    ToolDefinition,
)


class ConcurrencyLimitedClient(WrapperClient):
    """Wraps a client with asyncio.Semaphore to limit concurrent requests."""

    def __init__(self, wrapped: BaseProviderClient, *, limit: int = 10) -> None:
        super().__init__(wrapped)
        self._semaphore = asyncio.Semaphore(limit)
        self._limit = limit

    async def request_async(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: RequestOptions | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        async with self._semaphore:
            return await self._wrapped.request_async(
                messages, tools=tools, tool_choice=tool_choice, options=options, **kwargs
            )

    async def request_stream_async(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: RequestOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        async with self._semaphore:
            async for chunk in self._wrapped.request_stream_async(
                messages, tools=tools, tool_choice=tool_choice, options=options, **kwargs
            ):
                yield chunk

    def request(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: RequestOptions | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        return run_sync(
            self.request_async(
                messages, tools=tools, tool_choice=tool_choice, options=options, **kwargs
            )
        )
