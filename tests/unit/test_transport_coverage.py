"""Extended coverage tests for kaos_llm_client.transport.

Targets uncovered lines from transport.py to raise coverage from ~63% to 90%+.
Tests exercise execute_with_retry, parse_sse_stream, parse_sse_stream_sync,
run_sync, raise_for_status, and httpx client factories.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import httpx
import pytest

from kaos_llm_client.errors import (
    KaosLLMAuthError,
    KaosLLMProviderError,
    KaosLLMTransportError,
)
from kaos_llm_client.transport import (
    RetryPolicy,
    create_async_http_client,
    create_http_client,
    execute_with_retry,
    parse_sse_stream,
    parse_sse_stream_sync,
    raise_for_status,
    run_sync,
)
from kaos_llm_client.types import ProviderRequest


def _make_request(endpoint: str = "/v1/test", body: dict | None = None) -> ProviderRequest:
    return ProviderRequest(
        provider="test",
        model="test-model",
        endpoint=endpoint,
        body=body or {"prompt": "hi"},
    )


# ---------------------------------------------------------------------------
# execute_with_retry tests
# ---------------------------------------------------------------------------


_MOCK_BASE = "https://api.test.local"


class TestExecuteWithRetry:
    """Tests for execute_with_retry using httpx.MockTransport."""

    async def test_success_first_try(self):
        """200 response on first attempt returns response directly."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, json={"result": "ok"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url=_MOCK_BASE) as client:
            resp = await execute_with_retry(
                client,
                _make_request(),
                retry_policy=RetryPolicy(max_retries=3),
                provider="test",
            )

        assert resp.status_code == 200
        assert resp.json() == {"result": "ok"}
        assert call_count == 1
        assert "latency_ms" in resp.extensions

    async def test_429_retries_then_succeeds(self):
        """429 on first attempt, 200 on second -> retries once."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(429, json={"error": {"message": "rate limited"}})
            return httpx.Response(200, json={"result": "ok"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url=_MOCK_BASE) as client:
            resp = await execute_with_retry(
                client,
                _make_request(),
                retry_policy=RetryPolicy(max_retries=3, backoff_base=0.01),
                provider="test",
            )

        assert resp.status_code == 200
        assert call_count == 2

    async def test_auth_error_no_retry(self):
        """401 raises immediately, never retried."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(401, json={"error": {"message": "Invalid API key"}})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url=_MOCK_BASE) as client:
            with pytest.raises(KaosLLMAuthError):
                await execute_with_retry(
                    client,
                    _make_request(),
                    retry_policy=RetryPolicy(max_retries=3),
                    provider="test",
                )

        assert call_count == 1

    async def test_all_retries_exhausted(self):
        """All 500s exhausts retries -- raises KaosLLMRetryExhaustedError.

        WS-TR.PR-6f post-mortem fix: previously this re-raised the raw
        KaosLLMProviderError on the final attempt, which made it
        impossible to distinguish "exhausted retries" from "first-try
        failure". Now we always raise KaosLLMRetryExhaustedError after a
        retry loop completes without success — the underlying provider
        error is on .last_error / __cause__.
        """
        from kaos_llm_client.errors import KaosLLMRetryExhaustedError

        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(500, json={"error": {"message": "server error"}})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url=_MOCK_BASE) as client:
            with pytest.raises(KaosLLMRetryExhaustedError) as exc_info:
                await execute_with_retry(
                    client,
                    _make_request(),
                    retry_policy=RetryPolicy(max_retries=2, backoff_base=0.01),
                    provider="test",
                )

        # Underlying provider error is exposed as .last_error
        assert isinstance(exc_info.value.last_error, KaosLLMProviderError)
        assert exc_info.value.last_error.status_code == 500
        # max_retries=2 means 3 total attempts (0, 1, 2)
        assert call_count == 3

    async def test_on_retry_callback_fires(self):
        """on_retry callback is called on each retry."""
        call_count = 0
        retry_calls: list[tuple] = []

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return httpx.Response(500, json={"error": {"message": "error"}})
            return httpx.Response(200, json={"result": "ok"})

        def on_retry(req, attempt, exc):
            retry_calls.append((req.endpoint, attempt, type(exc).__name__))

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url=_MOCK_BASE) as client:
            await execute_with_retry(
                client,
                _make_request(endpoint="/v1/chat"),
                retry_policy=RetryPolicy(max_retries=3, backoff_base=0.01),
                provider="test",
                on_retry=on_retry,
            )

        assert len(retry_calls) == 2
        assert retry_calls[0][0] == "/v1/chat"
        assert retry_calls[0][1] == 0  # first retry is attempt 0
        assert retry_calls[1][1] == 1

    async def test_transport_error_retried(self):
        """httpx.ConnectError is retried and eventually succeeds."""
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("Connection refused")
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url=_MOCK_BASE) as client:
            resp = await execute_with_retry(
                client,
                _make_request(),
                retry_policy=RetryPolicy(max_retries=3, backoff_base=0.01),
                provider="test",
            )

        assert resp.status_code == 200
        assert call_count == 2

    async def test_transport_error_exhausted(self):
        """ConnectError exhausts all retries -- raises KaosLLMRetryExhaustedError
        (a KaosLLMTransportError subclass) carrying the underlying
        connect error as .last_error.
        """
        from kaos_llm_client.errors import KaosLLMRetryExhaustedError

        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url=_MOCK_BASE) as client:
            with pytest.raises(KaosLLMRetryExhaustedError) as exc_info:
                await execute_with_retry(
                    client,
                    _make_request(),
                    retry_policy=RetryPolicy(max_retries=1, backoff_base=0.01),
                    provider="test",
                )

        # The wrapped underlying error is on .last_error
        assert isinstance(exc_info.value.last_error, KaosLLMTransportError)
        assert "Connection error" in str(exc_info.value.last_error)

    async def test_transport_error_on_retry_callback(self):
        """on_retry callback fires for transport errors too."""
        call_count = 0
        retry_calls: list = []

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("refused")
            return httpx.Response(200, json={"ok": True})

        def on_retry(req, attempt, exc):
            retry_calls.append(type(exc).__name__)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url=_MOCK_BASE) as client:
            await execute_with_retry(
                client,
                _make_request(),
                retry_policy=RetryPolicy(max_retries=3, backoff_base=0.01),
                provider="test",
                on_retry=on_retry,
            )

        assert len(retry_calls) == 1
        assert retry_calls[0] == "KaosLLMTransportError"

    async def test_non_retryable_provider_error_raises_immediately(self):
        """400 error is not retryable and raises immediately."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(400, json={"error": {"message": "bad request"}})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url=_MOCK_BASE) as client:
            with pytest.raises(KaosLLMProviderError) as exc_info:
                await execute_with_retry(
                    client,
                    _make_request(),
                    retry_policy=RetryPolicy(max_retries=3),
                    provider="test",
                )

        assert exc_info.value.status_code == 400
        assert call_count == 1


class TestServiceTierFallback:
    """Tests for the graceful flex-tier fallback. The bug exposed by
    WS-TR.PR-6f.6 was that the fallback only fired on 500, not on the
    full set of retryable status codes — so persistent 503/504/429s
    on flex tier slipped through and the request failed entirely."""

    @pytest.mark.parametrize("status", [500, 502, 503, 504, 429])
    async def test_fallback_fires_on_any_retryable_status(self, status: int):
        """Flex tier returning ANY retryable 5xx/429 should trigger the
        fallback to default tier on the FINAL retry."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            body = json.loads(request.content)
            # The fallback retry omits service_tier — return success then.
            if "service_tier" not in body:
                return httpx.Response(200, json={"result": "fallback ok"})
            return httpx.Response(status, json={"error": {"message": f"err {status}"}})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url=_MOCK_BASE) as client:
            resp = await execute_with_retry(
                client,
                _make_request(body={"prompt": "hi", "service_tier": "flex"}),
                retry_policy=RetryPolicy(max_retries=2, backoff_base=0.001),
                provider="test",
            )

        assert resp.status_code == 200
        assert resp.json() == {"result": "fallback ok"}
        # 3 attempts (max_retries=2 → 0,1,2) on flex + 1 fallback = 4.
        assert call_count == 4

    async def test_fallback_carries_failure_reason_when_both_fail(self):
        """When the fallback also fails, the raised
        KaosLLMRetryExhaustedError MUST surface the fallback error, not
        the pre-fallback one. Otherwise operators see the wrong cause
        and chase the wrong bug (this is what cost two PR-6f.6 reruns)."""
        from kaos_llm_client.errors import KaosLLMRetryExhaustedError

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if "service_tier" not in body:
                # Fallback hit — return 502 with a distinctive message.
                return httpx.Response(502, json={"error": {"message": "fallback also broken"}})
            return httpx.Response(503, json={"error": {"message": "flex broken"}})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url=_MOCK_BASE) as client:
            with pytest.raises(KaosLLMRetryExhaustedError) as exc_info:
                await execute_with_retry(
                    client,
                    _make_request(body={"prompt": "hi", "service_tier": "flex"}),
                    retry_policy=RetryPolicy(max_retries=1, backoff_base=0.001),
                    provider="test",
                )

        # The exhaustion message tells the caller fallback was tried.
        assert "service_tier fallback" in str(exc_info.value)
        # The underlying error reflects the FALLBACK failure (502), not
        # the original (503) — operators need the actual final state.
        last = exc_info.value.last_error
        assert last is not None
        assert "fallback also broken" in str(last)

    async def test_no_fallback_when_no_service_tier(self):
        """If the request body never had service_tier, the fallback
        path is skipped entirely and we go straight to exhaustion."""
        from kaos_llm_client.errors import KaosLLMRetryExhaustedError

        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(503, json={"error": {"message": "down"}})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url=_MOCK_BASE) as client:
            with pytest.raises((KaosLLMProviderError, KaosLLMRetryExhaustedError)):
                await execute_with_retry(
                    client,
                    _make_request(body={"prompt": "hi"}),  # no service_tier
                    retry_policy=RetryPolicy(max_retries=1, backoff_base=0.001),
                    provider="test",
                )

        # max_retries=1 → 2 standard attempts + 0 fallback = 2.
        assert call_count == 2

    async def test_retry_log_includes_error_message(self):
        """The 'LLM retry' log line must include the actual error
        message — opaque 'LLM retry' lines were what hid the flex bug
        across two PR-6f.6 benchmark reruns. Verified by attaching a
        MemoryHandler directly to the kaos-llm-client transport logger
        (the kaos-core logger hierarchy uses propagate=False so caplog
        doesn't see records via the root logger)."""
        import logging
        import logging.handlers as _lh

        from kaos_llm_client import transport as _t

        memory_handler = _lh.MemoryHandler(capacity=100)
        memory_handler.setLevel(logging.WARNING)
        _t.logger.addHandler(memory_handler)
        try:

            def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(503, json={"error": {"message": "DISTINCTIVE_ERR"}})

            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport, base_url=_MOCK_BASE) as client:
                from kaos_llm_client.errors import KaosLLMRetryExhaustedError

                with pytest.raises((KaosLLMProviderError, KaosLLMRetryExhaustedError)):
                    await execute_with_retry(
                        client,
                        _make_request(body={"prompt": "hi", "service_tier": "flex"}),
                        retry_policy=RetryPolicy(max_retries=1, backoff_base=0.001),
                        provider="test",
                    )

            messages = [r.getMessage() for r in memory_handler.buffer]
        finally:
            _t.logger.removeHandler(memory_handler)

        # The WARNING-level log message must carry both the upstream
        # error string AND the service_tier in the FORMATTED message
        # (not buried in extras) so plain `grep` finds them.
        retry_messages = [m for m in messages if "retry" in m.lower()]
        assert any("DISTINCTIVE_ERR" in m for m in retry_messages), (
            f"retry log lines did not include error message; saw: {retry_messages}"
        )
        assert any("flex" in m.lower() for m in retry_messages), (
            f"retry log lines did not include service tier; saw: {retry_messages}"
        )


# ---------------------------------------------------------------------------
# SSE stream parsing tests
# ---------------------------------------------------------------------------


class TestParseSSEStream:
    """Tests for parse_sse_stream (async)."""

    async def test_basic_sse(self):
        """Parses data: lines into JSON dicts."""
        sse_content = 'data: {"text": "hello"}\n\ndata: {"text": "world"}\n\ndata: [DONE]\n\n'

        async def mock_aiter_text():
            yield sse_content

        response = MagicMock(spec=httpx.Response)
        response.aiter_text = mock_aiter_text

        chunks = []
        async for chunk in parse_sse_stream(response):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0] == {"text": "hello"}
        assert chunks[1] == {"text": "world"}

    async def test_done_stops_iteration(self):
        """[DONE] sentinel stops iteration."""
        sse_content = 'data: {"a": 1}\ndata: [DONE]\ndata: {"b": 2}\n'

        async def mock_aiter_text():
            yield sse_content

        response = MagicMock(spec=httpx.Response)
        response.aiter_text = mock_aiter_text

        chunks = []
        async for chunk in parse_sse_stream(response):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0] == {"a": 1}

    async def test_skip_empty_lines(self):
        """Empty and whitespace-only lines are skipped."""
        sse_content = '\n\n   \ndata: {"val": 42}\n\n'

        async def mock_aiter_text():
            yield sse_content

        response = MagicMock(spec=httpx.Response)
        response.aiter_text = mock_aiter_text

        chunks = []
        async for chunk in parse_sse_stream(response):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0] == {"val": 42}

    async def test_skip_event_and_id_lines(self):
        """event: and id: lines are silently ignored."""
        sse_content = 'event: message\nid: 1\ndata: {"ok": true}\n\n'

        async def mock_aiter_text():
            yield sse_content

        response = MagicMock(spec=httpx.Response)
        response.aiter_text = mock_aiter_text

        chunks = []
        async for chunk in parse_sse_stream(response):
            chunks.append(chunk)

        assert len(chunks) == 1

    async def test_skip_unparseable_data(self):
        """Invalid JSON data lines are silently skipped."""
        sse_content = 'data: not-json\ndata: {"valid": true}\n\n'

        async def mock_aiter_text():
            yield sse_content

        response = MagicMock(spec=httpx.Response)
        response.aiter_text = mock_aiter_text

        chunks = []
        async for chunk in parse_sse_stream(response):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0] == {"valid": True}

    async def test_chunked_delivery(self):
        """Data arriving in multiple chunks is reassembled correctly."""

        async def mock_aiter_text():
            yield 'data: {"k"'
            yield ': "v"}\n\n'

        response = MagicMock(spec=httpx.Response)
        response.aiter_text = mock_aiter_text

        chunks = []
        async for chunk in parse_sse_stream(response):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0] == {"k": "v"}


class TestParseSSEStreamSync:
    """Tests for parse_sse_stream_sync (sync version)."""

    def test_basic_sync(self):
        """Parses data: lines synchronously."""
        sse_content = 'data: {"text": "sync"}\ndata: [DONE]\n'

        def mock_iter_text():
            yield sse_content

        response = MagicMock(spec=httpx.Response)
        response.iter_text = mock_iter_text

        chunks = list(parse_sse_stream_sync(response))

        assert len(chunks) == 1
        assert chunks[0] == {"text": "sync"}

    def test_done_stops_sync(self):
        """[DONE] stops sync iteration."""
        sse_content = 'data: {"a": 1}\ndata: [DONE]\ndata: {"b": 2}\n'

        def mock_iter_text():
            yield sse_content

        response = MagicMock(spec=httpx.Response)
        response.iter_text = mock_iter_text

        chunks = list(parse_sse_stream_sync(response))
        assert len(chunks) == 1

    def test_skip_empty_sync(self):
        """Empty lines are skipped in sync parsing."""
        sse_content = '\n\ndata: {"x": 1}\n\n'

        def mock_iter_text():
            yield sse_content

        response = MagicMock(spec=httpx.Response)
        response.iter_text = mock_iter_text

        chunks = list(parse_sse_stream_sync(response))
        assert len(chunks) == 1

    def test_skip_unparseable_sync(self):
        """Invalid JSON is skipped in sync parsing."""
        sse_content = 'data: broken!\ndata: {"ok": true}\n\n'

        def mock_iter_text():
            yield sse_content

        response = MagicMock(spec=httpx.Response)
        response.iter_text = mock_iter_text

        chunks = list(parse_sse_stream_sync(response))
        assert len(chunks) == 1
        assert chunks[0] == {"ok": True}


# ---------------------------------------------------------------------------
# run_sync tests
# ---------------------------------------------------------------------------


class TestRunSync:
    """Tests for run_sync helper."""

    def test_run_sync_no_loop(self):
        """run_sync works when no event loop is running."""

        async def coro():
            return 42

        result = run_sync(coro())
        assert result == 42

    def test_run_sync_returns_value(self):
        """run_sync propagates the return value of the coroutine."""

        async def coro():
            return {"key": "value"}

        result = run_sync(coro())
        assert result == {"key": "value"}


# ---------------------------------------------------------------------------
# raise_for_status tests
# ---------------------------------------------------------------------------


class TestRaiseForStatusExtended:
    """Extended tests for raise_for_status."""

    def _mock_response(self, status_code: int, json_body: dict | None = None) -> httpx.Response:
        resp = httpx.Response(
            status_code=status_code,
            request=httpx.Request("POST", "https://api.test.com/v1/test"),
        )
        if json_body is not None:
            resp._content = json.dumps(json_body).encode()
        return resp

    def test_success_no_op(self):
        """200 response does not raise."""
        resp = self._mock_response(200)
        raise_for_status(resp, provider="test")

    def test_openai_error_format(self):
        """Parses OpenAI error.message format."""
        resp = self._mock_response(
            400,
            {"error": {"message": "Invalid model: gpt-99", "type": "invalid_request"}},
        )
        with pytest.raises(KaosLLMProviderError) as exc_info:
            raise_for_status(resp, provider="openai")

        assert "Invalid model: gpt-99" in str(exc_info.value)
        assert exc_info.value.status_code == 400

    def test_anthropic_error_format(self):
        """Parses Anthropic type=error format."""
        resp = self._mock_response(
            400,
            {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "max_tokens required"},
            },
        )
        with pytest.raises(KaosLLMProviderError) as exc_info:
            raise_for_status(resp, provider="anthropic")

        assert "max_tokens required" in str(exc_info.value)

    def test_error_string_body(self):
        """Non-JSON error body handled gracefully."""
        resp = self._mock_response(500)
        resp._content = b"Internal Server Error"
        with pytest.raises(KaosLLMProviderError) as exc_info:
            raise_for_status(resp, provider="test")

        assert exc_info.value.status_code == 500

    def test_auth_error_401(self):
        """401 -> KaosLLMAuthError."""
        resp = self._mock_response(401, {"error": {"message": "Invalid key"}})
        with pytest.raises(KaosLLMAuthError):
            raise_for_status(resp, provider="test")

    def test_auth_error_403(self):
        """403 -> KaosLLMAuthError."""
        resp = self._mock_response(403)
        with pytest.raises(KaosLLMAuthError):
            raise_for_status(resp, provider="test")

    def test_error_with_model(self):
        """Model name is passed through to error."""
        resp = self._mock_response(404, {"error": {"message": "not found"}})
        with pytest.raises(KaosLLMProviderError) as exc_info:
            raise_for_status(resp, provider="openai", model="gpt-5")

        assert exc_info.value.model == "gpt-5"

    def test_error_includes_fix_suggestion(self):
        """Error includes fix suggestion based on status code."""
        resp = self._mock_response(429, {"error": {"message": "rate limited"}})
        with pytest.raises(KaosLLMProviderError) as exc_info:
            raise_for_status(resp, provider="test")

        assert exc_info.value.fix is not None
        assert "rate" in exc_info.value.fix.lower() or "retry" in exc_info.value.fix.lower()


# ---------------------------------------------------------------------------
# httpx client creation tests
# ---------------------------------------------------------------------------


class TestCreateHttpClient:
    """Tests for create_http_client and create_async_http_client."""

    def test_create_http_client(self):
        """Returns a configured httpx.Client."""
        client = create_http_client(
            base_url="https://api.example.com",
            timeout=30.0,
            headers={"X-Custom": "val"},
        )
        try:
            assert isinstance(client, httpx.Client)
        finally:
            client.close()

    def test_create_http_client_defaults(self):
        """Works with default arguments."""
        client = create_http_client(base_url="https://api.example.com")
        try:
            assert isinstance(client, httpx.Client)
        finally:
            client.close()

    async def test_create_async_http_client(self):
        """Returns a configured httpx.AsyncClient."""
        client = create_async_http_client(
            base_url="https://api.example.com",
            timeout=60.0,
            headers={"Authorization": "Bearer tok"},
        )
        try:
            assert isinstance(client, httpx.AsyncClient)
        finally:
            await client.aclose()

    async def test_create_async_http_client_defaults(self):
        """Works with default arguments."""
        client = create_async_http_client(base_url="https://api.example.com")
        try:
            assert isinstance(client, httpx.AsyncClient)
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# Additional coverage: _extract_error_message edge cases
# ---------------------------------------------------------------------------


class TestExtractErrorMessageEdgeCases:
    """Tests for _extract_error_message via raise_for_status edge cases."""

    def _mock_response(self, status_code: int, json_body: dict | None = None) -> httpx.Response:
        resp = httpx.Response(
            status_code=status_code,
            request=httpx.Request("POST", "https://api.test.com/v1/test"),
        )
        if json_body is not None:
            resp._content = json.dumps(json_body).encode()
        return resp

    def test_error_field_as_string(self):
        """error field as plain string (not dict) is returned directly."""
        resp = self._mock_response(500, {"error": "Something went wrong"})
        with pytest.raises(KaosLLMProviderError) as exc_info:
            raise_for_status(resp, provider="test")

        assert "Something went wrong" in str(exc_info.value)

    def test_anthropic_error_as_string(self):
        """Anthropic format with error as string falls to generic handler."""
        resp = self._mock_response(
            400,
            {"type": "error", "error": "plain text error"},
        )
        with pytest.raises(KaosLLMProviderError) as exc_info:
            raise_for_status(resp, provider="test")

        # "error" key matches OpenAI handler first, returns str(error)
        assert "plain text error" in str(exc_info.value)

    def test_unknown_error_format(self):
        """Unknown error format falls through to str(raw_error)."""
        resp = self._mock_response(502, {"status": "bad", "details": "unknown"})
        with pytest.raises(KaosLLMProviderError) as exc_info:
            raise_for_status(resp, provider="test")

        # Falls through all specific handlers, returns str(raw_error)
        assert "bad" in str(exc_info.value) or "unknown" in str(exc_info.value)

    def test_no_json_body(self):
        """Non-JSON response body defaults to HTTP {status_code}."""
        resp = self._mock_response(503)
        resp._content = b"Service Unavailable"
        with pytest.raises(KaosLLMProviderError) as exc_info:
            raise_for_status(resp, provider="test")

        assert "503" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Additional coverage: run_sync in running loop
# ---------------------------------------------------------------------------


class TestRunSyncInLoop:
    """Test run_sync when called from within a running event loop."""

    async def test_run_sync_from_async_context(self):
        """run_sync works when called from within an async context (running loop).

        This exercises the ThreadPoolExecutor branch (lines 87-91).
        """

        async def inner_coro():
            return "from_thread"

        # We're already in an async context, so run_sync will detect the running loop
        # and use the thread pool executor path
        result = run_sync(inner_coro())
        assert result == "from_thread"


# ---------------------------------------------------------------------------
# Additional coverage: _get_or_create_event_loop
# ---------------------------------------------------------------------------


class TestGetOrCreateEventLoop:
    """Test _get_or_create_event_loop helper."""

    def test_creates_loop_when_none_running(self):
        """Creates a new event loop when none is running."""
        from kaos_llm_client.transport import _get_or_create_event_loop

        loop = _get_or_create_event_loop()
        assert loop is not None
        assert isinstance(loop, asyncio.AbstractEventLoop)

    async def test_returns_running_loop(self):
        """Returns the running loop when one exists."""
        from kaos_llm_client.transport import _get_or_create_event_loop

        running_loop = asyncio.get_running_loop()
        result = _get_or_create_event_loop()
        assert result is running_loop


# ---------------------------------------------------------------------------
# Additional coverage: RetryPolicy edge cases
# ---------------------------------------------------------------------------


class TestRetryPolicyEdgeCases:
    """Additional RetryPolicy edge cases for coverage."""

    def test_should_retry_read_timeout(self):
        """ReadTimeout is retryable."""
        policy = RetryPolicy()
        err = httpx.ReadTimeout("read timed out")
        assert policy.should_retry(err, 0) is True

    def test_should_retry_connect_timeout(self):
        """ConnectTimeout is retryable."""
        policy = RetryPolicy()
        err = httpx.ConnectTimeout("connect timed out")
        assert policy.should_retry(err, 0) is True

    def test_should_not_retry_generic_exception(self):
        """Generic exceptions are not retried."""
        policy = RetryPolicy()
        err = ValueError("something")
        assert policy.should_retry(err, 0) is False
