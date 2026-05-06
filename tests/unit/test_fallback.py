"""Tests for FallbackClient — fallback chains and error propagation."""

from __future__ import annotations

from typing import Any

import pytest

from kaos_llm_client.errors import (
    KaosLLMAuthError,
    KaosLLMProviderError,
    KaosLLMTransportError,
)
from kaos_llm_client.profiles import ModelProfile
from kaos_llm_client.providers.fallback import FallbackClient
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import (
    ContentPart,
    ProviderResponse,
    StreamChunk,
    UsageInfo,
)


def _make_response(text: str = "ok") -> ProviderResponse:
    return ProviderResponse(
        provider="function",
        model="test",
        raw={},
        parts=[ContentPart(type="text", text=text)],
        usage=UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15),
    )


class TestFallbackClient:
    def test_primary_succeeds(self) -> None:
        """First client works, result returned directly."""

        def ok_handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response("primary ok")

        primary = FunctionClient(function=ok_handler)
        fallback = FunctionClient(function=ok_handler)
        client = FallbackClient([primary, fallback])
        response = client.chat([{"role": "user", "content": "Hi"}])
        assert response.text == "primary ok"
        assert len(primary.call_history) == 1
        assert len(fallback.call_history) == 0

    def test_fallback_on_provider_error(self) -> None:
        """First fails with KaosLLMProviderError, second succeeds."""

        def fail_handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            raise KaosLLMProviderError(
                "Server error", provider="function", model="test", status_code=500
            )

        def ok_handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response("backup ok")

        primary = FunctionClient(function=fail_handler)
        backup = FunctionClient(function=ok_handler)
        client = FallbackClient([primary, backup])
        response = client.chat([{"role": "user", "content": "Hi"}])
        assert response.text == "backup ok"
        assert len(primary.call_history) == 1
        assert len(backup.call_history) == 1

    def test_all_fail_raises_last(self) -> None:
        """Both clients fail, last error is raised."""

        def fail_1(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            raise KaosLLMProviderError(
                "Error from client 1", provider="function", model="test", status_code=500
            )

        def fail_2(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            raise KaosLLMProviderError(
                "Error from client 2", provider="function", model="test", status_code=503
            )

        c1 = FunctionClient(function=fail_1)
        c2 = FunctionClient(function=fail_2)
        client = FallbackClient([c1, c2])
        with pytest.raises(KaosLLMProviderError, match="Error from client 2"):
            client.chat([{"role": "user", "content": "Hi"}])

    def test_auth_error_no_fallback(self) -> None:
        """KaosLLMAuthError is NOT in default fallback_on; propagates immediately."""

        def auth_fail(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            raise KaosLLMAuthError("Bad API key", provider="function")

        def ok_handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response("should not reach")

        primary = FunctionClient(function=auth_fail)
        backup = FunctionClient(function=ok_handler)
        client = FallbackClient([primary, backup])
        with pytest.raises(KaosLLMAuthError, match="Bad API key"):
            client.chat([{"role": "user", "content": "Hi"}])
        # backup was never called
        assert len(backup.call_history) == 0

    def test_custom_fallback_on(self) -> None:
        """Custom fallback_on tuple enables fallback for specific exception types."""

        def transport_fail(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> ProviderResponse:
            raise KaosLLMTransportError("Connection refused", provider="function")

        def ok_handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response("custom fallback ok")

        primary = FunctionClient(function=transport_fail)
        backup = FunctionClient(function=ok_handler)
        # Only fall back on transport errors
        client = FallbackClient([primary, backup], fallback_on=(KaosLLMTransportError,))
        response = client.chat([{"role": "user", "content": "Hi"}])
        assert response.text == "custom fallback ok"

    async def test_streaming_fallback(self) -> None:
        """First client's stream setup fails, second works."""
        from collections.abc import AsyncIterator

        async def fail_stream(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> AsyncIterator[StreamChunk]:
            raise KaosLLMProviderError(
                "Stream setup failed", provider="function", model="test", status_code=500
            )
            # Make this a generator so the type signature is correct
            yield StreamChunk(type="done")  # pragma: no cover

        async def ok_stream(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(type="text_delta", text="streamed ok")

        # The fail stream function is used as stream_function, but FunctionClient
        # will call it via request_stream_async. We need the function= for non-streaming,
        # and stream_function= for streaming.
        def dummy_handler(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> ProviderResponse:
            return _make_response("not used")

        primary = FunctionClient(stream_function=fail_stream, function=dummy_handler)
        backup = FunctionClient(stream_function=ok_stream, function=dummy_handler)
        client = FallbackClient([primary, backup])

        chunks: list[StreamChunk] = []
        async for chunk in client.request_stream_async([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)

        text_chunks = [c for c in chunks if c.type == "text_delta"]
        assert len(text_chunks) == 1
        assert text_chunks[0].text == "streamed ok"

    def test_empty_clients_raises(self) -> None:
        """FallbackClient with empty list raises ValueError."""
        with pytest.raises(ValueError, match="at least one client"):
            FallbackClient([])
