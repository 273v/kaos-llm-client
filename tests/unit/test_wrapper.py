"""Tests for WrapperClient — delegation and lifecycle."""

from __future__ import annotations

from typing import Any

from kaos_llm_client.profiles import ModelProfile
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.providers.wrapper import WrapperClient
from kaos_llm_client.types import ContentPart, ProviderResponse, UsageInfo


def _make_response(text: str = "wrapped") -> ProviderResponse:
    return ProviderResponse(
        provider="function",
        model="test",
        raw={},
        parts=[ContentPart(type="text", text=text)],
        usage=UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def _handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
    return _make_response("hello from wrapper")


class _TrackingClient(FunctionClient):
    """FunctionClient subclass that tracks close/aclose calls."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.close_called = False
        self.aclose_called = False

    def close(self) -> None:
        self.close_called = True
        super().close()

    async def aclose(self) -> None:
        self.aclose_called = True
        await super().aclose()


class TestWrapperClient:
    def test_delegates_chat(self) -> None:
        """WrapperClient(FunctionClient) delegates chat() to wrapped client."""
        inner = FunctionClient(function=_handler)
        wrapper = WrapperClient(inner)
        response = wrapper.chat([{"role": "user", "content": "Hi"}])
        assert response.text == "hello from wrapper"
        assert len(inner.call_history) == 1

    async def test_delegates_chat_async(self) -> None:
        """WrapperClient delegates chat_async() to wrapped client."""
        inner = FunctionClient(function=_handler)
        wrapper = WrapperClient(inner)
        response = await wrapper.chat_async([{"role": "user", "content": "Hi"}])
        assert response.text == "hello from wrapper"
        assert len(inner.call_history) == 1

    def test_model_proxied(self) -> None:
        """wrapper.model matches wrapped client's model."""
        inner = FunctionClient(model="my-model", function=_handler)
        wrapper = WrapperClient(inner)
        assert wrapper.model == "my-model"

    def test_provider_name_proxied(self) -> None:
        """wrapper._provider_name matches wrapped client's _provider_name."""
        inner = FunctionClient(function=_handler)
        wrapper = WrapperClient(inner)
        assert wrapper._provider_name == "function"

    def test_close_delegates(self) -> None:
        """close() propagates to the wrapped client."""
        inner = _TrackingClient(function=_handler)
        wrapper = WrapperClient(inner)
        wrapper.close()
        assert inner.close_called

    async def test_aclose_delegates(self) -> None:
        """aclose() propagates to the wrapped client."""
        inner = _TrackingClient(function=_handler)
        wrapper = WrapperClient(inner)
        await wrapper.aclose()
        assert inner.aclose_called
