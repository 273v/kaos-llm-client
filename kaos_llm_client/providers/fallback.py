"""Fallback client that tries providers in order.

When the primary client fails with a retryable error, subsequent clients
are tried until one succeeds or all have been exhausted.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_client.cache import NullCache
from kaos_llm_client.errors import (
    KaosLLMProviderError,
    KaosLLMRetryExhaustedError,
    KaosLLMTransportError,
)
from kaos_llm_client.providers.base import BaseProviderClient
from kaos_llm_client.transport import run_sync
from kaos_llm_client.types import (
    ProviderRequest,
    ProviderResponse,
    RequestOptions,
    StreamChunk,
    ToolChoice,
    ToolDefinition,
)

logger = get_logger("kaos_llm_client.providers.fallback")

_DEFAULT_FALLBACK_ON: tuple[type[BaseException], ...] = (
    KaosLLMProviderError,
    KaosLLMTransportError,
    KaosLLMRetryExhaustedError,
)


class FallbackClient(BaseProviderClient):
    """Tries clients in order, falling back on configurable exceptions."""

    _provider_name: str = "fallback"

    def __init__(
        self,
        clients: list[BaseProviderClient],
        *,
        fallback_on: tuple[type[BaseException], ...] = _DEFAULT_FALLBACK_ON,
        **kwargs: Any,
    ) -> None:
        if not clients:
            raise ValueError("FallbackClient requires at least one client")
        self._clients = clients
        self._fallback_on = fallback_on
        primary = clients[0]
        super().__init__(
            model=primary.model,
            settings=primary._settings,
            profile=primary.profile,
            cache=NullCache(),
            base_url=primary._base_url,
            api_key=primary._resolve_api_key(),
            timeout=primary._timeout,
            hooks=primary._hooks,
            **kwargs,
        )

    # --- Request methods with fallback ---

    async def request_async(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: RequestOptions | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        last_error: BaseException = RuntimeError("unreachable")
        for i, client in enumerate(self._clients):
            try:
                return await client.request_async(
                    messages, tools=tools, tool_choice=tool_choice, options=options, **kwargs
                )
            except BaseException as exc:
                if not isinstance(exc, self._fallback_on):
                    raise
                last_error = exc
                logger.warning(
                    "Fallback: client %d (%s) failed, trying next",
                    i,
                    client._provider_name,
                    extra=self._log_extra(
                        provider=client._provider_name,
                        model=client.model,
                        attempt=i + 1,
                        total_providers=len(self._clients),
                        error=str(exc),
                    ),
                )
        raise last_error

    async def request_stream_async(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: RequestOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        last_error: BaseException = RuntimeError("unreachable")
        for i, client in enumerate(self._clients):
            yielded = False
            try:
                async for chunk in client.request_stream_async(
                    messages, tools=tools, tool_choice=tool_choice, options=options, **kwargs
                ):
                    yielded = True
                    yield chunk
                return  # stream completed successfully
            except BaseException as exc:
                # Once we've yielded chunks, we cannot transparently switch
                # providers — the caller already has partial data. Propagate.
                if yielded:
                    raise
                if not isinstance(exc, self._fallback_on):
                    raise
                last_error = exc
                logger.warning(
                    "Fallback stream: client %d (%s) failed, trying next",
                    i,
                    client._provider_name,
                    extra=self._log_extra(
                        provider=client._provider_name,
                        model=client.model,
                        attempt=i + 1,
                        total_providers=len(self._clients),
                        error=str(exc),
                    ),
                )
        raise last_error

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

    # --- ABC methods delegate to primary client ---

    def _build_request(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ProviderRequest:
        return self._clients[0]._build_request(
            messages, tools=tools, tool_choice=tool_choice, stream=stream, **kwargs
        )

    def _parse_response(self, raw: dict[str, Any], request: ProviderRequest) -> ProviderResponse:
        return self._clients[0]._parse_response(raw, request)

    def _parse_stream_chunk(self, data: dict[str, Any]) -> StreamChunk | list[StreamChunk]:
        return self._clients[0]._parse_stream_chunk(data)

    def _build_headers(self) -> dict[str, str]:
        return self._clients[0]._build_headers()

    def _default_endpoint(self) -> str:
        return self._clients[0]._default_endpoint()

    def _get_default_base_url(self) -> str:
        return self._clients[0]._get_default_base_url()

    def _get_api_key_from_settings(self) -> str:
        return self._clients[0]._get_api_key_from_settings()

    # --- Delegate embeddings to primary ---

    async def embed_async(self, input: Any, **kwargs: Any) -> Any:
        return await self._clients[0].embed_async(input, **kwargs)

    def embed(self, input: Any, **kwargs: Any) -> Any:
        return self._clients[0].embed(input, **kwargs)

    # --- Lifecycle ---

    def close(self) -> None:
        for client in self._clients:
            client.close()

    async def aclose(self) -> None:
        for client in self._clients:
            await client.aclose()
