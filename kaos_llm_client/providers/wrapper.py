"""Base wrapper client that delegates all behavior to a wrapped client.

Enables composition patterns (concurrency limiting, caching layers,
fallback chains) without requiring subclasses to re-implement every
abstract method.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from kaos_llm_client.cache import NullCache
from kaos_llm_client.providers.base import BaseProviderClient
from kaos_llm_client.types import (
    ProviderRequest,
    ProviderResponse,
    RequestOptions,
    StreamChunk,
    ToolChoice,
    ToolDefinition,
)


class WrapperClient(BaseProviderClient):
    """Base class that wraps another client, delegating all behavior."""

    def __init__(self, wrapped: BaseProviderClient, **kwargs: Any) -> None:
        self._wrapped = wrapped
        super().__init__(
            model=wrapped.model,
            settings=wrapped._settings,
            profile=wrapped.profile,
            cache=NullCache(),
            base_url=wrapped._base_url,
            api_key=wrapped._resolve_api_key(),
            timeout=wrapped._timeout,
            hooks=wrapped._hooks,
            **kwargs,
        )

    @property
    def _provider_name(self) -> str:  # type: ignore[override]
        return self._wrapped._provider_name

    # --- Delegate request methods ---

    async def request_async(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: RequestOptions | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
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
        return self._wrapped.request(
            messages, tools=tools, tool_choice=tool_choice, options=options, **kwargs
        )

    # --- Delegate ABC methods ---

    def _build_request(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ProviderRequest:
        return self._wrapped._build_request(
            messages, tools=tools, tool_choice=tool_choice, stream=stream, **kwargs
        )

    def _parse_response(self, raw: dict[str, Any], request: ProviderRequest) -> ProviderResponse:
        return self._wrapped._parse_response(raw, request)

    def _parse_stream_chunk(self, data: dict[str, Any]) -> StreamChunk | list[StreamChunk]:
        return self._wrapped._parse_stream_chunk(data)

    def _build_headers(self) -> dict[str, str]:
        return self._wrapped._build_headers()

    def _default_endpoint(self) -> str:
        return self._wrapped._default_endpoint()

    def _get_default_base_url(self) -> str:
        return self._wrapped._get_default_base_url()

    def _get_api_key_from_settings(self) -> str:
        return self._wrapped._get_api_key_from_settings()

    # --- Delegate embeddings ---

    async def embed_async(self, input: Any, **kwargs: Any) -> Any:
        return await self._wrapped.embed_async(input, **kwargs)

    def embed(self, input: Any, **kwargs: Any) -> Any:
        return self._wrapped.embed(input, **kwargs)

    # --- Delegate lifecycle ---

    def close(self) -> None:
        self._wrapped.close()

    async def aclose(self) -> None:
        await self._wrapped.aclose()
