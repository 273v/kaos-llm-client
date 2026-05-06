"""Tests for kaos_llm_client.transport — retry, SSE parsing."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
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
    execute_with_retry,
    parse_retry_after,
    parse_sse_stream,
    raise_for_status,
)
from kaos_llm_client.types import ProviderRequest

_MOCK_BASE = "https://api.test.local"


def _make_request(endpoint: str = "/v1/test", body: dict | None = None) -> ProviderRequest:
    return ProviderRequest(
        provider="test",
        model="test-model",
        endpoint=endpoint,
        body=body or {"prompt": "hi"},
    )


class TestRetryPolicy:
    def test_defaults(self):
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.backoff_base == 1.0
        assert 429 in policy.retryable_status_codes
        assert 500 in policy.retryable_status_codes

    def test_should_retry_auth_error(self):
        policy = RetryPolicy()
        err = KaosLLMAuthError("bad key")
        assert policy.should_retry(err, 0) is False

    def test_should_retry_retryable_status(self):
        policy = RetryPolicy()
        err = KaosLLMProviderError("rate limited", provider="test", status_code=429)
        assert policy.should_retry(err, 0) is True

    def test_should_retry_non_retryable_status(self):
        policy = RetryPolicy()
        err = KaosLLMProviderError("bad request", provider="test", status_code=400)
        assert policy.should_retry(err, 0) is False

    def test_should_retry_transport_error(self):
        policy = RetryPolicy()
        err = KaosLLMTransportError("connection refused")
        assert policy.should_retry(err, 0) is True

    def test_should_not_retry_past_max(self):
        policy = RetryPolicy(max_retries=2)
        err = KaosLLMProviderError("error", provider="test", status_code=500)
        assert policy.should_retry(err, 2) is False

    def test_should_retry_httpx_connect_error(self):
        policy = RetryPolicy()
        err = httpx.ConnectError("failed")
        assert policy.should_retry(err, 0) is True

    def test_backoff_full_jitter_in_range(self):
        """Full-jitter backoff returns a uniform value in [0, base * 2**attempt]."""
        policy = RetryPolicy(backoff_base=1.0, max_backoff=60.0)
        for attempt, expo in [(0, 1.0), (1, 2.0), (2, 4.0), (3, 8.0)]:
            for _ in range(50):
                v = policy.backoff_seconds(attempt)
                assert 0.0 <= v <= expo, f"attempt={attempt} got {v}, expo={expo}"

    def test_backoff_full_jitter_custom_base(self):
        """Custom base scales the jitter window."""
        policy = RetryPolicy(backoff_base=0.5, max_backoff=60.0)
        for attempt, expo in [(0, 0.5), (1, 1.0), (2, 2.0)]:
            for _ in range(50):
                v = policy.backoff_seconds(attempt)
                assert 0.0 <= v <= expo, f"attempt={attempt} got {v}, expo={expo}"


class TestRaiseForStatus:
    def _mock_response(self, status_code: int, json_body: dict | None = None) -> httpx.Response:
        """Create a mock httpx.Response."""
        response = httpx.Response(
            status_code=status_code,
            request=httpx.Request("POST", "https://api.test.com/v1/test"),
        )
        if json_body is not None:
            response._content = __import__("json").dumps(json_body).encode()
        return response

    def test_success_no_raise(self):
        resp = self._mock_response(200)
        raise_for_status(resp, provider="test")  # should not raise

    def test_auth_error_401(self):
        resp = self._mock_response(401, {"error": {"message": "Invalid API key"}})
        with pytest.raises(KaosLLMAuthError) as exc_info:
            raise_for_status(resp, provider="openai")
        assert "authentication failed" in str(exc_info.value).lower()

    def test_auth_error_403(self):
        resp = self._mock_response(403)
        with pytest.raises(KaosLLMAuthError):
            raise_for_status(resp, provider="openai")

    def test_provider_error_400(self):
        resp = self._mock_response(
            400,
            {"error": {"message": "max_tokens is required", "type": "invalid_request_error"}},
        )
        with pytest.raises(KaosLLMProviderError) as exc_info:
            raise_for_status(resp, provider="anthropic", model="claude-sonnet-4-6")
        assert exc_info.value.status_code == 400
        assert "max_tokens" in str(exc_info.value)

    def test_provider_error_429(self):
        resp = self._mock_response(429, {"error": {"message": "Rate limit exceeded"}})
        with pytest.raises(KaosLLMProviderError) as exc_info:
            raise_for_status(resp, provider="openai")
        assert exc_info.value.status_code == 429

    def test_provider_error_500(self):
        resp = self._mock_response(500)
        with pytest.raises(KaosLLMProviderError) as exc_info:
            raise_for_status(resp, provider="openai")
        assert exc_info.value.status_code == 500

    def test_anthropic_error_format(self):
        resp = self._mock_response(
            400,
            {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "max_tokens is required"},
            },
        )
        with pytest.raises(KaosLLMProviderError) as exc_info:
            raise_for_status(resp, provider="anthropic")
        assert "max_tokens" in str(exc_info.value)


class TestErrors:
    def test_provider_error_details(self):
        err = KaosLLMProviderError(
            "test error",
            provider="openai",
            model="gpt-5",
            status_code=400,
            raw_error={"error": {"message": "bad request"}},
            fix="Fix your request",
        )
        assert err.provider == "openai"
        assert err.model == "gpt-5"
        assert err.status_code == 400
        assert err.raw_error is not None
        assert err.fix == "Fix your request"
        assert err.details["provider"] == "openai"

    def test_provider_error_default_retry_after_is_none(self):
        err = KaosLLMProviderError("test", provider="openai", model="gpt-5", status_code=429)
        assert err.retry_after is None

    def test_provider_error_carries_retry_after(self):
        err = KaosLLMProviderError(
            "rate limited",
            provider="openai",
            status_code=429,
            retry_after=12.5,
        )
        assert err.retry_after == 12.5
        assert err.details["retry_after"] == 12.5


# ---------------------------------------------------------------------------
# Task 4 — Retry-After parsing (RFC 9110 §10.2.3)
# ---------------------------------------------------------------------------


class TestParseRetryAfter:
    """parse_retry_after() handles delta-seconds + HTTP-date per RFC 9110."""

    def test_delta_seconds_integer(self):
        assert parse_retry_after("30") == 30.0

    def test_delta_seconds_zero(self):
        assert parse_retry_after("0") == 0.0

    def test_delta_seconds_negative_clamped(self):
        # RFC says non-negative, but we clamp defensively.
        assert parse_retry_after("-5") == 0.0

    def test_delta_seconds_float(self):
        # Provider SDKs (OpenAI Python) tolerate floats here.
        assert parse_retry_after("1.5") == 1.5

    def test_http_date_future(self):
        now = datetime(2026, 10, 21, 7, 0, 0, tzinfo=UTC)
        # 28 minutes later
        target = "Wed, 21 Oct 2026 07:28:00 GMT"
        result = parse_retry_after(target, now=now)
        assert result is not None
        assert abs(result - 28 * 60) < 0.01

    def test_http_date_past_clamped(self):
        now = datetime(2026, 10, 21, 8, 0, 0, tzinfo=UTC)
        target = "Wed, 21 Oct 2026 07:28:00 GMT"  # in the past relative to now
        assert parse_retry_after(target, now=now) == 0.0

    def test_garbage_input(self):
        assert parse_retry_after("garbage") is None

    def test_none_input(self):
        assert parse_retry_after(None) is None

    def test_empty_input(self):
        assert parse_retry_after("") is None
        assert parse_retry_after("   ") is None

    def test_whitespace_around_seconds(self):
        # email.utils tolerates surrounding whitespace; we strip first.
        assert parse_retry_after("  30  ") == 30.0


class TestRaiseForStatusRetryAfter:
    """raise_for_status() must surface the Retry-After header."""

    def _resp(self, status: int, headers: dict[str, str] | None = None) -> httpx.Response:
        return httpx.Response(
            status_code=status,
            request=httpx.Request("POST", "https://api.test.com/v1"),
            headers=headers or {},
        )

    def test_retry_after_seconds_attached_to_provider_error(self):
        resp = self._resp(429, {"Retry-After": "42"})
        with pytest.raises(KaosLLMProviderError) as exc_info:
            raise_for_status(resp, provider="openai")
        assert exc_info.value.retry_after == 42.0

    def test_retry_after_ms_header_preferred(self):
        # OpenAI's non-standard millisecond header is finer-grained
        # and should be preferred when both are present.
        resp = self._resp(429, {"retry-after-ms": "1500", "Retry-After": "10"})
        with pytest.raises(KaosLLMProviderError) as exc_info:
            raise_for_status(resp, provider="openai")
        assert exc_info.value.retry_after == 1.5

    def test_no_retry_after_means_none(self):
        resp = self._resp(429)
        with pytest.raises(KaosLLMProviderError) as exc_info:
            raise_for_status(resp, provider="openai")
        assert exc_info.value.retry_after is None


class TestRetryAfterHonoured:
    """The retry loop must sleep at least Retry-After seconds when present."""

    async def test_429_with_retry_after_sleeps_at_least_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr("kaos_llm_client.transport.asyncio.sleep", fake_sleep)

        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(
                    429,
                    headers={"Retry-After": "5"},
                    json={"error": {"message": "slow down"}},
                )
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url=_MOCK_BASE) as client:
            resp = await execute_with_retry(
                client,
                _make_request(),
                # Backoff base small so jittered fallback would NOT reach 5s.
                retry_policy=RetryPolicy(max_retries=2, backoff_base=0.001, max_backoff=60.0),
                provider="test",
            )

        assert resp.status_code == 200
        assert len(sleeps) == 1
        # Must be exactly the header value (capped at max_backoff=60), not jittered.
        assert sleeps[0] == 5.0

    async def test_retry_after_capped_at_max_backoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A hostile server returning Retry-After: 999999 must be clamped to max_backoff."""
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr("kaos_llm_client.transport.asyncio.sleep", fake_sleep)

        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(
                    429,
                    headers={"Retry-After": "999999"},
                    json={"error": {"message": "rate limited"}},
                )
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url=_MOCK_BASE) as client:
            await execute_with_retry(
                client,
                _make_request(),
                retry_policy=RetryPolicy(max_retries=2, backoff_base=0.001, max_backoff=10.0),
                provider="test",
            )

        assert sleeps == [10.0]


# ---------------------------------------------------------------------------
# Task 5 — Full-jitter exponential backoff
# ---------------------------------------------------------------------------


class TestFullJitterBackoff:
    def test_jitter_distribution_in_window(self):
        """200 samples at attempt=2 must all land in [0, base*4]."""
        policy = RetryPolicy(backoff_base=1.0, max_backoff=60.0)
        samples = [policy.backoff_seconds(2) for _ in range(200)]
        assert all(0.0 <= s <= 4.0 for s in samples)

    def test_jitter_mean_in_25pct_of_expected(self):
        """Mean of 200 samples should be ~base*2 (i.e. expo/2) within ±25%."""
        policy = RetryPolicy(backoff_base=1.0, max_backoff=60.0)
        samples = [policy.backoff_seconds(2) for _ in range(200)]
        mean = sum(samples) / len(samples)
        expected = 2.0  # uniform on [0, 4] has mean 2
        assert 0.75 * expected <= mean <= 1.25 * expected, f"mean={mean}"

    def test_max_backoff_caps_high_attempts(self):
        """At attempt=20 the unclamped value would be base*2**20; the cap holds."""
        policy = RetryPolicy(backoff_base=1.0, max_backoff=60.0)
        samples = [policy.backoff_seconds(20) for _ in range(50)]
        assert all(0.0 <= s <= 60.0 for s in samples)

    def test_zero_base_returns_zero(self):
        policy = RetryPolicy(backoff_base=0.0, max_backoff=60.0)
        assert policy.backoff_seconds(3) == 0.0

    def test_default_max_backoff_is_60(self):
        policy = RetryPolicy()
        assert policy.max_backoff == 60.0


# ---------------------------------------------------------------------------
# Task 6 — Response-body size cap (non-streaming)
# ---------------------------------------------------------------------------


class TestResponseBodySizeCap:
    async def test_oversized_content_length_rejected(self):
        """Content-Length > cap must raise KaosLLMTransportError before parsing."""
        big = 50 * 1024 * 1024  # 50 MiB
        body = b'{"x":"' + (b"a" * 32) + b'"}'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"Content-Length": str(big)},
                content=body,
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url=_MOCK_BASE) as client:
            with pytest.raises(KaosLLMTransportError) as exc_info:
                await execute_with_retry(
                    client,
                    _make_request(),
                    retry_policy=RetryPolicy(max_retries=0, backoff_base=0.001),
                    provider="test",
                    max_response_bytes=32 * 1024 * 1024,
                )
        assert "too large" in str(exc_info.value).lower()
        # Either the declared size or the cap should be in the message.
        assert "32" in str(exc_info.value) or "50" in str(exc_info.value)

    async def test_oversized_buffered_body_without_content_length(self):
        """When Content-Length is absent, fall back to len(response.content)."""
        body = b"x" * (5 * 1024 * 1024)  # 5 MiB

        def handler(request: httpx.Request) -> httpx.Response:
            # httpx will compute Content-Length from `content` automatically;
            # we strip it to simulate a server that omits the header (chunked).
            resp = httpx.Response(200, content=body)
            # remove Content-Length to exercise the buffered-body branch
            if "content-length" in resp.headers:
                del resp.headers["content-length"]
            return resp

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url=_MOCK_BASE) as client:
            with pytest.raises(KaosLLMTransportError) as exc_info:
                await execute_with_retry(
                    client,
                    _make_request(),
                    retry_policy=RetryPolicy(max_retries=0, backoff_base=0.001),
                    provider="test",
                    max_response_bytes=1 * 1024 * 1024,  # 1 MiB cap
                )
        assert "too large" in str(exc_info.value).lower()

    async def test_under_cap_passes(self):
        """Bodies below the cap must succeed normally."""
        body = b'{"ok": true}'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url=_MOCK_BASE) as client:
            resp = await execute_with_retry(
                client,
                _make_request(),
                retry_policy=RetryPolicy(max_retries=0, backoff_base=0.001),
                provider="test",
                max_response_bytes=1024,
            )
        assert resp.status_code == 200

    async def test_no_cap_disables_check(self):
        """max_response_bytes=None disables the check (back-compat)."""
        body = b"x" * (10 * 1024)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url=_MOCK_BASE) as client:
            resp = await execute_with_retry(
                client,
                _make_request(),
                retry_policy=RetryPolicy(max_retries=0, backoff_base=0.001),
                provider="test",
                max_response_bytes=None,
            )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Task 6 — Stream wall-clock cap
# ---------------------------------------------------------------------------


class TestStreamWallClockCap:
    async def test_stream_exceeds_max_duration_raises(self):
        """A stream that yields one line and then sleeps must terminate with TransportError."""
        first_line_emitted = asyncio.Event()

        async def slow_aiter_text():
            yield 'data: {"hi": 1}\n\n'
            first_line_emitted.set()
            # Sleep "forever" — but bounded so the test eventually completes
            # if the cap doesn't fire (it should fire long before).
            await asyncio.sleep(5.0)
            yield 'data: {"never": 2}\n\n'

        response = MagicMock(spec=httpx.Response)
        response.aiter_text = slow_aiter_text

        chunks: list[dict] = []
        with pytest.raises(KaosLLMTransportError) as exc_info:
            async for chunk in parse_sse_stream(response, max_duration=0.2):
                chunks.append(chunk)
        assert "wall-clock" in str(exc_info.value).lower()
        # The first chunk MUST have been delivered before the timeout fired.
        assert chunks == [{"hi": 1}]

    async def test_stream_no_cap_completes_normally(self):
        """max_duration=None disables the cap (back-compat)."""

        async def fast_aiter_text():
            yield 'data: {"a": 1}\n\ndata: [DONE]\n\n'

        response = MagicMock(spec=httpx.Response)
        response.aiter_text = fast_aiter_text

        chunks: list[dict] = []
        async for chunk in parse_sse_stream(response, max_duration=None):
            chunks.append(chunk)
        assert chunks == [{"a": 1}]


# ---------------------------------------------------------------------------
# Settings defaults sanity check
# ---------------------------------------------------------------------------


class TestSettingsTransportDefaults:
    def test_max_response_bytes_default_is_32_mib(self):
        from kaos_llm_client.settings import KaosLLMSettings

        s = KaosLLMSettings()
        assert s.max_response_bytes == 32 * 1024 * 1024

    def test_stream_max_duration_default_is_600s(self):
        from kaos_llm_client.settings import KaosLLMSettings

        s = KaosLLMSettings()
        assert s.stream_max_duration == 600.0

    def test_request_options_defaults_to_none(self):
        from kaos_llm_client.types import RequestOptions

        opts = RequestOptions()
        assert opts.max_response_bytes is None
        assert opts.stream_max_duration is None
