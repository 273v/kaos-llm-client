"""Function-based test double for deterministic unit testing.

Inspired by pydantic-ai's FunctionModel. Executes Python callables instead
of making HTTP requests, enabling deterministic tests without mocks or
network access.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable
from typing import Any, Protocol, cast, runtime_checkable

from kaos_core.logging import get_logger

from kaos_llm_client.profiles import ModelProfile, resolve_profile
from kaos_llm_client.providers.base import BaseProviderClient
from kaos_llm_client.settings import KaosLLMSettings
from kaos_llm_client.types import (
    ProviderRequest,
    ProviderResponse,
    StreamChunk,
    ToolChoice,
    ToolDefinition,
)

logger = get_logger("kaos_llm_client.providers.function")


# ---------------------------------------------------------------------------
# Callable protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class FunctionDef(Protocol):
    """Sync or async callable that produces a complete response."""

    def __call__(
        self, messages: list[dict[str, Any]], profile: ModelProfile
    ) -> ProviderResponse | Awaitable[ProviderResponse]: ...


@runtime_checkable
class StreamFunctionDef(Protocol):
    """Async callable that yields stream chunks."""

    def __call__(
        self, messages: list[dict[str, Any]], profile: ModelProfile
    ) -> AsyncIterator[StreamChunk]: ...


# ---------------------------------------------------------------------------
# FunctionClient
# ---------------------------------------------------------------------------


class FunctionClient(BaseProviderClient):
    """Test double that executes Python functions instead of HTTP requests.

    Usage::

        def my_handler(messages, profile):
            return ProviderResponse(
                provider="function", model="test", raw={},
                parts=[ContentPart(type="text", text="Hello!")],
            )

        client = FunctionClient("test-model", function=my_handler)
        response = client.chat([{"role": "user", "content": "Hi"}])
        assert response.text == "Hello!"
        assert len(client.call_history) == 1
    """

    _provider_name: str = "function"

    def __init__(
        self,
        model: str = "function-test",
        *,
        function: FunctionDef | None = None,
        stream_function: StreamFunctionDef | None = None,
        settings: KaosLLMSettings | None = None,
        profile: ModelProfile | None = None,
        **kwargs: Any,
    ) -> None:
        # FunctionClient doesn't need real settings, provide defaults
        if settings is None:
            settings = KaosLLMSettings()

        super().__init__(
            model=model,
            settings=settings,
            profile=profile or resolve_profile("function", model),
            # Prevent base from trying to resolve a real API key
            api_key="function-test-key",
            base_url="function://test",
            **kwargs,
        )

        self._function = function
        self._stream_function = stream_function

        # Call history for test assertions: list of (messages, kwargs) tuples
        self.call_history: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    # --- Abstract method implementations (no-ops for function client) ---

    def _get_default_base_url(self) -> str:
        return "function://test"

    def _get_api_key_from_settings(self) -> str:
        return "function-test-key"

    def _build_headers(self) -> dict[str, str]:
        return {}

    def _default_endpoint(self) -> str:
        return "function://test"

    def _build_request(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ProviderRequest:
        """Build a minimal request — just stores the messages and kwargs for history."""
        body: dict[str, Any] = {
            "messages": messages,
        }
        if tools is not None:
            body["tools"] = [t.model_dump() for t in tools]
        if tool_choice is not None:
            body["tool_choice"] = tool_choice.model_dump()
        body.update(kwargs)

        return ProviderRequest(
            provider=self._provider_name,
            model=self.model,
            endpoint="function://test",
            body=body,
            stream=stream,
        )

    def _parse_response(self, raw: dict[str, Any], request: ProviderRequest) -> ProviderResponse:
        """Not used — request_async is overridden to bypass HTTP."""
        raise NotImplementedError(
            "FunctionClient bypasses HTTP; _parse_response should not be called."
        )

    def _parse_stream_chunk(self, data: dict[str, Any]) -> StreamChunk:
        """Not used — request_stream_async is overridden to bypass HTTP."""
        raise NotImplementedError(
            "FunctionClient bypasses HTTP; _parse_stream_chunk should not be called."
        )

    # --- Override request methods to call functions directly ---

    async def request_async(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: Any = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Call the function directly instead of making an HTTP request."""
        if self._function is None:
            raise NotImplementedError(
                "FunctionClient requires a `function` callable. Pass function= to the constructor."
            )

        # Record call for test assertions
        self.call_history.append((messages, kwargs))

        # Call the function — handle both sync and async callables
        result = self._function(messages, self.profile)

        if inspect.isawaitable(result):
            response = cast(ProviderResponse, await result)
        else:
            response = cast(ProviderResponse, result)

        return response

    async def request_stream_async(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Call the stream function directly instead of making an HTTP request."""
        if self._stream_function is None:
            raise NotImplementedError(
                "FunctionClient requires a `stream_function` callable for streaming. "
                "Pass stream_function= to the constructor."
            )

        # Record call for test assertions
        self.call_history.append((messages, kwargs))

        async for chunk in self._stream_function(messages, self.profile):
            yield chunk

        # Yield final done chunk
        yield StreamChunk(type="done")
