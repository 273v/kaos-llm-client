"""Coverage-focused tests for WrapperClient (wrapper.py) and FallbackClient (fallback.py).

Targets uncovered lines in wrapper delegation (ABC methods, embed, close/aclose,
properties, request_stream_async) and fallback streaming edge cases (yield-then-error,
setup-failure, embed delegation, and ABC delegates).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from kaos_llm_client.errors import (
    KaosLLMProviderError,
    KaosLLMTransportError,
)
from kaos_llm_client.profiles import ModelProfile
from kaos_llm_client.providers.concurrency import ConcurrencyLimitedClient
from kaos_llm_client.providers.fallback import FallbackClient
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.providers.instrumented import InstrumentedClient
from kaos_llm_client.providers.wrapper import WrapperClient
from kaos_llm_client.types import (
    ContentPart,
    ProviderResponse,
    StreamChunk,
    UsageInfo,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(text: str = "ok") -> ProviderResponse:
    return ProviderResponse(
        provider="function",
        model="test",
        raw={},
        parts=[ContentPart(type="text", text=text)],
        usage=UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def _handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
    return _make_response("hello")


class _TrackingClient(FunctionClient):
    """FunctionClient that tracks close/aclose calls."""

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


# ===========================================================================
# WrapperClient comprehensive delegation
# ===========================================================================


class TestWrapperDelegatesAllABCMethods:
    """Cover wrapper delegation of _build_request, _parse_response, _build_headers, etc."""

    def test_build_request_delegates(self) -> None:
        """_build_request delegates to wrapped client."""
        inner = FunctionClient(function=_handler)
        wrapper = WrapperClient(inner)
        req = wrapper._build_request([{"role": "user", "content": "Hi"}])
        assert req.provider == "function"
        assert req.endpoint == "function://test"

    def test_parse_stream_chunk_delegates(self) -> None:
        """_parse_stream_chunk delegates to wrapped client."""
        inner = FunctionClient(function=_handler)
        wrapper = WrapperClient(inner)
        with pytest.raises(NotImplementedError):
            wrapper._parse_stream_chunk({"data": "test"})

    def test_build_headers_delegates(self) -> None:
        """_build_headers delegates to wrapped client."""
        inner = FunctionClient(function=_handler)
        wrapper = WrapperClient(inner)
        headers = wrapper._build_headers()
        assert isinstance(headers, dict)

    def test_default_endpoint_delegates(self) -> None:
        """_default_endpoint delegates to wrapped client."""
        inner = FunctionClient(function=_handler)
        wrapper = WrapperClient(inner)
        assert wrapper._default_endpoint() == "function://test"

    def test_get_default_base_url_delegates(self) -> None:
        """_get_default_base_url delegates to wrapped client."""
        inner = FunctionClient(function=_handler)
        wrapper = WrapperClient(inner)
        assert wrapper._get_default_base_url() == "function://test"

    def test_get_api_key_from_settings_delegates(self) -> None:
        """_get_api_key_from_settings delegates to wrapped client."""
        inner = FunctionClient(function=_handler)
        wrapper = WrapperClient(inner)
        assert wrapper._get_api_key_from_settings() == "function-test-key"

    def test_parse_response_delegates(self) -> None:
        """_parse_response delegates to wrapped client."""
        inner = FunctionClient(function=_handler)
        wrapper = WrapperClient(inner)
        # FunctionClient._parse_response raises NotImplementedError
        with pytest.raises(NotImplementedError):
            wrapper._parse_response({}, wrapper._build_request([]))

    def test_request_sync_delegates(self) -> None:
        """Sync request() delegates to wrapped client."""
        inner = FunctionClient(function=_handler)
        wrapper = WrapperClient(inner)
        result = wrapper.request([{"role": "user", "content": "Hi"}])
        assert result.text == "hello"
        assert len(inner.call_history) == 1


class TestWrapperEmbedDelegates:
    """Verify embed_async and embed delegate to the wrapped client."""

    async def test_embed_async_delegates(self) -> None:
        """embed_async() delegates to wrapped client (raises NotImplementedError)."""
        inner = FunctionClient(function=_handler)
        wrapper = WrapperClient(inner)
        with pytest.raises(NotImplementedError, match="does not support embeddings"):
            await wrapper.embed_async("test text")

    def test_embed_sync_delegates(self) -> None:
        """embed() delegates to wrapped client (raises NotImplementedError)."""
        inner = FunctionClient(function=_handler)
        wrapper = WrapperClient(inner)
        with pytest.raises(NotImplementedError, match="does not support embeddings"):
            wrapper.embed("test text")


class TestWrapperCloseDelegates:
    """Verify close and aclose propagate to the wrapped client."""

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


class TestWrapperPropertiesProxy:
    """Cover model, profile, _provider_name from wrapped."""

    def test_model_from_wrapped(self) -> None:
        """wrapper.model matches wrapped client model."""
        inner = FunctionClient(model="my-model", function=_handler)
        wrapper = WrapperClient(inner)
        assert wrapper.model == "my-model"

    def test_profile_from_wrapped(self) -> None:
        """wrapper.profile matches wrapped client profile."""
        inner = FunctionClient(function=_handler)
        wrapper = WrapperClient(inner)
        assert wrapper.profile is not None
        assert wrapper.profile == inner.profile

    def test_provider_name_from_wrapped(self) -> None:
        """wrapper._provider_name matches wrapped client."""
        inner = FunctionClient(function=_handler)
        wrapper = WrapperClient(inner)
        assert wrapper._provider_name == "function"


class TestWrapperStreamingDelegates:
    """Cover request_stream_async delegation in WrapperClient."""

    async def test_stream_delegates_to_wrapped(self) -> None:
        """request_stream_async delegates to wrapped client."""

        async def stream_fn(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(type="text_delta", text="streamed")

        inner = FunctionClient(function=_handler, stream_function=stream_fn)
        wrapper = WrapperClient(inner)

        chunks: list[StreamChunk] = []
        async for chunk in wrapper.request_stream_async([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)

        text_chunks = [c for c in chunks if c.type == "text_delta" and c.text]
        assert len(text_chunks) >= 1
        assert text_chunks[0].text == "streamed"


# ===========================================================================
# FallbackClient comprehensive coverage
# ===========================================================================


class TestFallbackStreamNoSpliceAfterYield:
    """When the first client yields chunks then errors, error propagates (no fallback)."""

    async def test_stream_no_splice_after_yield(self) -> None:
        """First client yields a chunk then raises; error propagates to caller."""

        async def fail_after_yield(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(type="text_delta", text="partial")
            raise KaosLLMProviderError(
                "Mid-stream failure",
                provider="function",
                model="test",
                status_code=500,
            )

        async def ok_stream(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(type="text_delta", text="backup")

        primary = FunctionClient(stream_function=fail_after_yield, function=_handler)
        backup = FunctionClient(stream_function=ok_stream, function=_handler)
        client = FallbackClient([primary, backup])

        chunks: list[StreamChunk] = []
        with pytest.raises(KaosLLMProviderError, match="Mid-stream failure"):
            async for chunk in client.request_stream_async([{"role": "user", "content": "Hi"}]):
                chunks.append(chunk)

        # The partial chunk was yielded before the error
        text_chunks = [c for c in chunks if c.type == "text_delta"]
        assert len(text_chunks) >= 1
        assert text_chunks[0].text == "partial"


class TestFallbackStreamSetupFailure:
    """First client fails before yielding; second client succeeds."""

    async def test_stream_setup_failure_falls_back(self) -> None:
        """Stream setup error triggers fallback to the second client."""

        async def fail_stream(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> AsyncIterator[StreamChunk]:
            raise KaosLLMTransportError("Connection refused", provider="function")
            yield StreamChunk(type="done")  # pragma: no cover — make it a generator

        async def ok_stream(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(type="text_delta", text="fallback ok")

        primary = FunctionClient(stream_function=fail_stream, function=_handler)
        backup = FunctionClient(stream_function=ok_stream, function=_handler)
        client = FallbackClient([primary, backup])

        chunks: list[StreamChunk] = []
        async for chunk in client.request_stream_async([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)

        text_chunks = [c for c in chunks if c.type == "text_delta" and c.text]
        assert len(text_chunks) >= 1
        assert text_chunks[0].text == "fallback ok"


class TestFallbackEmbedDelegates:
    """embed_async goes to the first (primary) client."""

    async def test_embed_delegates_to_primary(self) -> None:
        """embed_async() delegates to clients[0]."""
        primary = FunctionClient(function=_handler)
        backup = FunctionClient(function=_handler)
        client = FallbackClient([primary, backup])

        with pytest.raises(NotImplementedError, match="does not support embeddings"):
            await client.embed_async("test text")

    def test_embed_sync_delegates_to_primary(self) -> None:
        """embed() delegates to clients[0]."""
        primary = FunctionClient(function=_handler)
        backup = FunctionClient(function=_handler)
        client = FallbackClient([primary, backup])

        with pytest.raises(NotImplementedError, match="does not support embeddings"):
            client.embed("test text")


class TestFallbackABCDelegates:
    """Cover all ABC method delegates in FallbackClient."""

    def test_build_request_delegates(self) -> None:
        """_build_request delegates to clients[0]."""
        primary = FunctionClient(function=_handler)
        client = FallbackClient([primary])
        req = client._build_request([{"role": "user", "content": "Hi"}])
        assert req.provider == "function"

    def test_parse_response_delegates(self) -> None:
        """_parse_response delegates to clients[0]."""
        primary = FunctionClient(function=_handler)
        client = FallbackClient([primary])
        with pytest.raises(NotImplementedError):
            client._parse_response({}, client._build_request([]))

    def test_parse_stream_chunk_delegates(self) -> None:
        """_parse_stream_chunk delegates to clients[0]."""
        primary = FunctionClient(function=_handler)
        client = FallbackClient([primary])
        with pytest.raises(NotImplementedError):
            client._parse_stream_chunk({})

    def test_build_headers_delegates(self) -> None:
        """_build_headers delegates to clients[0]."""
        primary = FunctionClient(function=_handler)
        client = FallbackClient([primary])
        assert isinstance(client._build_headers(), dict)

    def test_default_endpoint_delegates(self) -> None:
        """_default_endpoint delegates to clients[0]."""
        primary = FunctionClient(function=_handler)
        client = FallbackClient([primary])
        assert client._default_endpoint() == "function://test"

    def test_get_default_base_url_delegates(self) -> None:
        """_get_default_base_url delegates to clients[0]."""
        primary = FunctionClient(function=_handler)
        client = FallbackClient([primary])
        assert client._get_default_base_url() == "function://test"

    def test_get_api_key_from_settings_delegates(self) -> None:
        """_get_api_key_from_settings delegates to clients[0]."""
        primary = FunctionClient(function=_handler)
        client = FallbackClient([primary])
        assert client._get_api_key_from_settings() == "function-test-key"


class TestFallbackLifecycle:
    """Cover close/aclose on FallbackClient (both clients get closed)."""

    def test_close_all_clients(self) -> None:
        """close() closes all clients in the chain."""
        c1 = _TrackingClient(function=_handler)
        c2 = _TrackingClient(function=_handler)
        client = FallbackClient([c1, c2])
        client.close()
        assert c1.close_called
        assert c2.close_called

    async def test_aclose_all_clients(self) -> None:
        """aclose() closes all clients in the chain."""
        c1 = _TrackingClient(function=_handler)
        c2 = _TrackingClient(function=_handler)
        client = FallbackClient([c1, c2])
        await client.aclose()
        assert c1.aclose_called
        assert c2.aclose_called


class TestFallbackSyncRequest:
    """Cover sync request() path in FallbackClient."""

    def test_request_sync(self) -> None:
        """Sync request() uses run_sync to call request_async."""
        primary = FunctionClient(function=_handler)
        client = FallbackClient([primary])
        result = client.request([{"role": "user", "content": "Hi"}])
        assert result.text == "hello"


class TestFallbackNonMatchingExceptionPropagates:
    """When the exception is not in fallback_on, it propagates immediately."""

    async def test_non_matching_exception_no_fallback(self) -> None:
        """RuntimeError is not in default fallback_on; propagates immediately."""

        def fail(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            raise RuntimeError("unexpected")

        primary = FunctionClient(function=fail)
        backup = FunctionClient(function=_handler)
        client = FallbackClient([primary, backup])

        with pytest.raises(RuntimeError, match="unexpected"):
            await client.request_async([{"role": "user", "content": "Hi"}])

        assert len(backup.call_history) == 0

    async def test_non_matching_exception_stream_no_fallback(self) -> None:
        """Non-matching exception in stream propagates without fallback."""

        async def fail_stream(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> AsyncIterator[StreamChunk]:
            raise RuntimeError("stream boom")
            yield StreamChunk(type="done")  # pragma: no cover

        async def ok_stream(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(type="text_delta", text="backup")

        primary = FunctionClient(stream_function=fail_stream, function=_handler)
        backup = FunctionClient(stream_function=ok_stream, function=_handler)
        client = FallbackClient([primary, backup])

        with pytest.raises(RuntimeError, match="stream boom"):
            async for _ in client.request_stream_async([{"role": "user", "content": "Hi"}]):
                pass  # pragma: no cover


# ===========================================================================
# ConcurrencyLimitedClient streaming
# ===========================================================================


class TestConcurrencyStreamingSemaphore:
    """Verify the semaphore is held for the duration of a stream."""

    async def test_semaphore_held_for_stream_duration(self) -> None:
        """Verify that during streaming, the semaphore slot is occupied."""

        async def slow_stream(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(type="text_delta", text="a")
            await asyncio.sleep(0.05)
            yield StreamChunk(type="text_delta", text="b")

        inner = FunctionClient(function=_handler, stream_function=slow_stream)
        client = ConcurrencyLimitedClient(inner, limit=1)

        # With limit=1, a concurrent stream should be blocked
        results: list[list[str]] = [[], []]
        order: list[int] = []

        async def consume_stream(idx: int) -> None:
            async for chunk in client.request_stream_async(
                [{"role": "user", "content": f"req {idx}"}]
            ):
                if chunk.type == "text_delta" and chunk.text:
                    results[idx].append(chunk.text)
                    order.append(idx)

        t1 = asyncio.create_task(consume_stream(0))
        t2 = asyncio.create_task(consume_stream(1))
        await asyncio.gather(t1, t2)

        # Both streams completed
        assert results[0] == ["a", "b"]
        assert results[1] == ["a", "b"]

        # With limit=1, they must run sequentially: all of one before the other
        # Check that the first 2 entries in `order` share the same idx
        assert order[0] == order[1], (
            f"Expected sequential execution with limit=1, got interleaved: {order}"
        )


# ===========================================================================
# InstrumentedClient edge cases
# ===========================================================================


class TestInstrumentedEmbedDelegates:
    """InstrumentedClient.embed() passes through to wrapped (raises)."""

    def test_embed_passthrough(self) -> None:
        """embed() delegates to wrapped FunctionClient (raises NotImplementedError)."""
        inner = FunctionClient(function=_handler)
        client = InstrumentedClient(inner)
        with pytest.raises(NotImplementedError, match="does not support embeddings"):
            client.embed("test")


class TestInstrumentedRecordsOnError:
    """Error during request_async does not crash InstrumentedClient internals."""

    async def test_error_propagates_cleanly(self) -> None:
        """Error from wrapped client propagates; counters remain consistent."""

        def fail(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            raise KaosLLMProviderError("boom", provider="function", model="test", status_code=500)

        inner = FunctionClient(function=fail)
        client = InstrumentedClient(inner)

        with pytest.raises(KaosLLMProviderError, match="boom"):
            await client.request_async([{"role": "user", "content": "Hi"}])

        # No successful request was recorded
        assert client.total_requests == 0
        assert client.total_input_tokens == 0

    def test_sync_error_propagates_cleanly(self) -> None:
        """Sync error from wrapped client propagates; counters remain consistent."""

        def fail(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            raise KaosLLMProviderError(
                "sync boom", provider="function", model="test", status_code=500
            )

        inner = FunctionClient(function=fail)
        client = InstrumentedClient(inner)

        with pytest.raises(KaosLLMProviderError, match="sync boom"):
            client.request([{"role": "user", "content": "Hi"}])

        assert client.total_requests == 0
