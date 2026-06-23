"""Transport layer: httpx wrappers, retry policy, SSE stream parsing.

Async is the primary implementation. Sync methods wrap async via event loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import email.utils
import hashlib
import json
import random
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
from kaos_core.logging import get_logger

from kaos_llm_client.errors import (
    KaosLLMAuthError,
    KaosLLMProviderError,
    KaosLLMRetryExhaustedError,
    KaosLLMStreamInterruptedError,
    KaosLLMTransportError,
)
from kaos_llm_client.types import ProviderRequest

logger = get_logger("kaos_llm_client.transport")
egress_logger = get_logger("kaos_llm_client.transport.egress")


# ---------------------------------------------------------------------------
# Response header redaction
# ---------------------------------------------------------------------------

# Sensitive response headers that must be redacted before being echoed onto
# ``ProviderResponse.response_headers``. Hooks, instrumentation, and
# cassette recorders all read those headers — leaking auth/session material
# into a captured cassette or a log line would be a real-world incident.
#
# Header names are matched case-insensitively (HTTP/1.1 §3.2 says field
# names are case-insensitive, and httpx normalises but third-party
# transports may not).
#
# References:
#   - RFC 6265 §4.1 (Set-Cookie) / §4.2 (Cookie)
#   - RFC 7235 §4 (WWW-Authenticate / Proxy-Authenticate / Authorization /
#     Proxy-Authorization)
#   - OWASP CSRF cheat sheet (X-Csrf-Token / X-Xsrf-Token)
_REDACTED_RESPONSE_HEADERS: tuple[str, ...] = (
    "Set-Cookie",
    "Cookie",
    "Authorization",
    "Proxy-Authorization",
    "WWW-Authenticate",
    "Proxy-Authenticate",
    "X-Csrf-Token",
    "X-Xsrf-Token",
)

_REDACTED_RESPONSE_HEADERS_LOWER: frozenset[str] = frozenset(
    h.lower() for h in _REDACTED_RESPONSE_HEADERS
)


def redact_response_headers(headers: Any) -> dict[str, str]:
    """Return a plain ``dict`` copy of ``headers`` with sensitive values redacted.

    Accepts any mapping-like (``httpx.Headers``, ``dict[str, str]``, list of
    pairs). The returned dict has every value for a header in
    :data:`_REDACTED_RESPONSE_HEADERS` (case-insensitive) replaced with the
    string ``"<redacted>"``. Non-redacted entries are passed through untouched.
    """
    out: dict[str, str] = {}
    # ``httpx.Headers`` iterates only unique keys; using ``.items()`` instead
    # of ``dict(headers)`` keeps multi-valued ``Set-Cookie`` entries comma-
    # joined as httpx already does, which is what callers see today.
    items = headers.items() if hasattr(headers, "items") else headers
    for key, value in items:
        if key.lower() in _REDACTED_RESPONSE_HEADERS_LOWER:
            out[key] = "<redacted>"
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Retry-After parsing (RFC 9110 §10.2.3)
# ---------------------------------------------------------------------------


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Parse an HTTP ``Retry-After`` header value into seconds.

    Two formats are recognised per RFC 9110 §10.2.3:

    - ``delta-seconds``: a non-negative integer (or float — provider SDKs
      tolerate floats). ``"30"`` → ``30.0``, ``"-5"`` → ``0.0`` (clamped),
      ``"0"`` → ``0.0``.
    - ``HTTP-date``: ``"Wed, 21 Oct 2026 07:28:00 GMT"`` → seconds until
      that instant from ``now`` (default: ``datetime.now(UTC)``), clamped
      to ``>= 0``.

    Returns ``None`` for ``None`` input, an empty string, or any
    unparseable value — callers should fall back to their own backoff.

    The ``now`` parameter exists so tests can pin "current time" without
    monkeypatching ``datetime``.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None

    # delta-seconds first (much more common in practice)
    try:
        seconds = float(text)
    except ValueError:
        pass
    else:
        return max(0.0, seconds)

    # HTTP-date fallback
    try:
        dt = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        # email.utils.parsedate_to_datetime returns naive only for malformed
        # dates lacking a TZ; treat as UTC per RFC 9110 (HTTP-date is GMT).
        dt = dt.replace(tzinfo=UTC)
    reference = now if now is not None else datetime.now(UTC)
    delta = (dt - reference).total_seconds()
    return max(0.0, delta)


def _retry_after_from_response(response: httpx.Response) -> float | None:
    """Extract a Retry-After delay from an httpx response.

    Accepts both the standard ``Retry-After`` header and OpenAI's
    non-standard ``retry-after-ms`` (milliseconds, finer-grained — see
    OpenAI Python SDK ``_base_client.py:_parse_retry_after_header``).
    Headers are matched case-insensitively (httpx normalises).
    """
    # Non-standard millisecond variant — OpenAI emits this on rate limits.
    ms_header = response.headers.get("retry-after-ms")
    if ms_header is not None:
        try:
            return max(0.0, float(ms_header) / 1000.0)
        except ValueError:
            pass
    return parse_retry_after(response.headers.get("retry-after"))


# ---------------------------------------------------------------------------
# Vendor egress audit log (plan §Issue 4)
# ---------------------------------------------------------------------------


def _request_body_digest(body: Any) -> tuple[int, str]:
    """Compute (bytes, sha256-hex) for an outbound provider request body.

    ``json.dumps`` with sorted keys and a separator-tight encoding gives
    a stable hash regardless of dict iteration order — auditors who
    re-hash from a captured cassette get the same digest as production.
    Non-serialisable values (e.g. bytes inside a multipart body) fall
    through to ``default=str`` so the digest still represents the call
    even if it can't be byte-perfect.
    """
    try:
        serialised = json.dumps(
            body, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False
        )
    except (TypeError, ValueError):
        serialised = str(body)
    encoded = serialised.encode("utf-8", errors="replace")
    return len(encoded), hashlib.sha256(encoded).hexdigest()


def emit_vendor_egress_log(
    *,
    provider: str,
    model: str,
    body: Any,
    attempt: int = 0,
    scrub_patterns: tuple[str, ...] | None = None,
    scrubbed_chars: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit one structured ``vendor_egress`` log line per outbound LLM call.

    The plan (``2026-05-22-launch-blocker-top-10.md`` §Issue 4) requires
    one such line per provider call, recording: provider, model, bytes,
    request_hash (sha256 of the serialised body), scrub_patterns,
    scrubbed_chars, and timestamp. The hash + bytes pair lets an
    auditor diff what was billed vs what left the process without
    storing the prompt text itself.

    The log line is emitted at INFO level on the
    ``kaos_llm_client.transport.egress`` logger so operators can route
    it to a dedicated egress sink (Splunk/Datadog/audit JSONL) without
    drowning in retry/latency chatter. Failure here is swallowed —
    audit logging MUST NOT break a real LLM call.
    """
    try:
        size_bytes, request_hash = _request_body_digest(body)
        merged_extra: dict[str, Any] = {
            "event": "vendor_egress",
            "provider": provider,
            "model": model,
            "bytes": size_bytes,
            "request_hash": f"sha256:{request_hash}",
            "scrub_patterns": list(scrub_patterns or ()),
            "scrubbed_chars": int(scrubbed_chars),
            "attempt": attempt,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if extra:
            for k, v in extra.items():
                merged_extra.setdefault(k, v)
        egress_logger.info(
            "vendor_egress provider=%s model=%s bytes=%d request_hash=%s",
            provider,
            model,
            size_bytes,
            f"sha256:{request_hash[:12]}",
            extra=merged_extra,
        )
    except Exception:  # pragma: no cover - defensive
        egress_logger.debug("vendor_egress log emission failed", exc_info=True)


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Configurable retry with full-jitter exponential backoff.

    The backoff schedule follows the AWS Architecture Blog "full jitter"
    formulation (`Marc Brooker, "Exponential Backoff and Jitter"
    <https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/>`_),
    which remains the recommended pattern in 2026 for client retry
    storms: each delay is a uniform random pick in ``[0, expo]`` where
    ``expo = min(max_backoff, backoff_base * 2**attempt)``. This
    distributes load across the retry window and prevents synchronised
    retry waves from N clients all colliding on the same boundary.

    When a ``Retry-After`` header is honoured, the jitter is NOT applied
    on top — the server's instruction is treated as authoritative.
    """

    max_retries: int = 3
    backoff_base: float = 1.0
    max_backoff: float = 60.0
    """Cap on a single backoff sleep, in seconds. Without a cap,
    ``base * 2**attempt`` grows without bound on long retry chains and
    individual sleeps can drift into multi-hour territory."""
    retryable_status_codes: frozenset[int] = field(
        default_factory=lambda: frozenset({429, 500, 502, 503, 504})
    )

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine whether to retry the request."""
        if attempt >= self.max_retries:
            return False
        if isinstance(error, KaosLLMAuthError):
            return False  # never retry auth
        if isinstance(error, KaosLLMProviderError):
            return error.status_code in self.retryable_status_codes
        if isinstance(error, KaosLLMTransportError):
            return True
        return isinstance(error, httpx.ConnectError | httpx.ReadTimeout | httpx.ConnectTimeout)

    def backoff_seconds(self, attempt: int) -> float:
        """Full-jitter exponential backoff for the given attempt (0-indexed).

        Returns a value in ``[0, min(max_backoff, base * 2**attempt)]``.
        Callers that have a server-supplied ``Retry-After`` should prefer
        that value verbatim and skip jittered backoff entirely.
        """
        cap = self.max_backoff
        # ``2 ** attempt`` is well-defined for any non-negative attempt;
        # for very large attempts ``min`` clamps before the float
        # multiplication can overflow the cap.
        expo = min(cap, self.backoff_base * (2**attempt))
        if expo <= 0:
            return 0.0
        # ``random.uniform`` is intentional here: backoff jitter is a
        # load-balancing concern, not a cryptographic one. Switching to
        # ``secrets`` would burn entropy without improving outcomes.
        return random.uniform(0.0, expo)


# ---------------------------------------------------------------------------
# Event loop helpers
# ---------------------------------------------------------------------------


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get the running event loop or create a new one."""
    try:
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def run_sync(coro: Any) -> Any:
    """Run an async coroutine synchronously.

    Handles both cases: when an event loop is already running (uses
    asyncio.run in a new thread) and when none is running.
    """
    try:
        asyncio.get_running_loop()
        # Loop already running — run in a thread to avoid blocking
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# httpx client creation
# ---------------------------------------------------------------------------


def create_http_client(
    base_url: str,
    timeout: float = 120.0,
    headers: dict[str, str] | None = None,
    *,
    trust_env: bool = False,
) -> httpx.Client:
    """Create a configured sync httpx client.

    ``trust_env`` controls whether httpx honours ambient ``HTTP_PROXY``,
    ``HTTPS_PROXY``, ``NO_PROXY``, and ``SSL_CERT_FILE`` env vars. The
    default is ``False`` to prevent prompts and authorization headers
    from being routed through an ambient proxy unless the caller opts in.
    """
    return httpx.Client(
        base_url=base_url,
        http2=True,
        timeout=httpx.Timeout(timeout),
        headers=headers or {},
        trust_env=trust_env,
    )


def create_async_http_client(
    base_url: str,
    timeout: float = 120.0,
    headers: dict[str, str] | None = None,
    *,
    trust_env: bool = False,
) -> httpx.AsyncClient:
    """Create a configured async httpx client. See :func:`create_http_client`."""
    return httpx.AsyncClient(
        base_url=base_url,
        http2=True,
        timeout=httpx.Timeout(timeout),
        headers=headers or {},
        trust_env=trust_env,
    )


# ---------------------------------------------------------------------------
# Response error handling
# ---------------------------------------------------------------------------


def raise_for_status(
    response: httpx.Response,
    *,
    provider: str,
    model: str | None = None,
) -> None:
    """Raise appropriate ``KaosLLMError`` subclass for HTTP error responses."""
    if response.is_success:
        return

    status_code = response.status_code

    # Try to parse error body
    raw_error: dict[str, Any] | None = None
    with contextlib.suppress(json.JSONDecodeError, ValueError):
        raw_error = response.json()

    # Extract error message from common provider formats
    error_msg = _extract_error_message(raw_error, status_code)

    if status_code in (401, 403):
        raise KaosLLMAuthError(
            f"{provider} authentication failed ({status_code}): {error_msg}",
            provider=provider,
            model=model,
            status_code=status_code,
            fix=(
                f"Check your API key. Set KAOS_LLM_{provider.upper()}_API_KEY environment variable."
            ),
        )

    # Capture Retry-After (RFC 9110 §10.2.3 + OpenAI's non-standard
    # retry-after-ms) so the retry loop can honour the server's hint.
    retry_after = _retry_after_from_response(response)

    raise KaosLLMProviderError(
        f"{provider} returned {status_code}: {error_msg}",
        provider=provider,
        model=model,
        status_code=status_code,
        raw_error=raw_error,
        fix=_suggest_fix(status_code, provider),
        retry_after=retry_after,
    )


def _extract_error_message(raw_error: dict[str, Any] | None, status_code: int) -> str:
    """Extract a human-readable error message from provider error responses."""
    if raw_error is None:
        return f"HTTP {status_code}"

    # OpenAI format: {"error": {"message": "..."}}
    if "error" in raw_error:
        error = raw_error["error"]
        if isinstance(error, dict):
            return error.get("message", str(error))
        return str(error)

    # Anthropic format: {"type": "error", "error": {"type": "...", "message": "..."}}
    if raw_error.get("type") == "error" and "error" in raw_error:
        error = raw_error["error"]
        if isinstance(error, dict):
            return error.get("message", str(error))

    # Google format: {"error": {"message": "...", "status": "..."}}
    # Already covered by OpenAI format handler above

    return str(raw_error)


def _suggest_fix(status_code: int, provider: str) -> str:
    """Suggest a fix based on HTTP status code."""
    fixes: dict[int, str] = {
        400: "Check request parameters. Ensure required fields are present.",
        404: f"Verify the model name and {provider} API endpoint URL.",
        429: "Rate limited. Wait and retry, or reduce request frequency.",
        500: f"Internal {provider} server error. Retry after a brief delay.",
        502: f"{provider} gateway error. Retry after a brief delay.",
        503: f"{provider} service temporarily unavailable. Retry after a brief delay.",
    }
    return fixes.get(status_code, f"Check the {provider} API documentation.")


# ---------------------------------------------------------------------------
# SSE stream parsing
# ---------------------------------------------------------------------------


async def parse_sse_stream(
    response: httpx.Response,
    *,
    max_duration: float | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Parse Server-Sent Events from an httpx streaming response.

    Yields parsed JSON dicts from ``data:`` lines. Stops at ``[DONE]``.

    ``max_duration`` is a wall-clock cap (seconds) measured from the
    moment iteration starts. If the stream is still open past that
    point, the iterator raises :class:`KaosLLMTransportError`. This
    bounds memory and time spent on a server that forgot to send
    ``[DONE]`` or whose socket has stalled mid-stream. ``None`` (the
    default) disables the cap; pass the resolved
    ``KaosLLMSettings.stream_max_duration`` from the caller.

    B1.3 (broad-reliability roadmap #570): network failures that fire
    AFTER the first chunk are surfaced as
    :class:`KaosLLMStreamInterruptedError` carrying the raw bytes
    received so far. Pre-B1.3, an httpx ``ReadError`` /
    ``RemoteProtocolError`` mid-stream surfaced as an opaque
    ``KaosLLMTransportError`` with no partial-text payload, which let
    SPA consumers ship a half-message with no recovery signal. The
    typed error is raised only when ``bytes_received > 0`` so a
    completely-failed connection still surfaces as the standard
    transport error.
    """
    t_start = time.monotonic()
    buffer = ""
    bytes_received = 0
    try:
        async for chunk in response.aiter_text():
            if max_duration is not None and (time.monotonic() - t_start) > max_duration:
                raise KaosLLMTransportError(
                    f"Stream wall-clock exceeded: open for >{max_duration:.1f}s. "
                    "The provider may have stalled or forgotten to terminate the stream. "
                    "Increase KAOS_LLM_STREAM_MAX_DURATION (or pass "
                    "RequestOptions(stream_max_duration=...)) if longer streams are expected."
                )
            bytes_received += len(chunk)
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        return
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        logger.debug("Skipping unparseable SSE data: %s", data[:100])
                # Ignore event:, id:, retry: lines
    except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ProtocolError) as exc:
        # B1.3: mid-stream HTTP / network interruption. Surface a typed
        # error carrying the bytes received so the caller can decide
        # between retry-as-fresh-call and ship-partial-with-footer.
        # ``buffer`` carries the last partial line we couldn't fully
        # parse; the typed payload doesn't include it (callers track
        # user-visible text separately via provider deltas), but the
        # byte count tells them whether anything reached them.
        if bytes_received == 0:
            # Pre-first-byte failures are recoverable as a fresh call.
            # Keep the standard transport-error shape so existing
            # retry policies still apply.
            raise KaosLLMTransportError(
                "Streaming connection failed before any data was received."
            ) from exc
        raise KaosLLMStreamInterruptedError(
            f"Streaming connection interrupted after {bytes_received} bytes",
            partial_text="",  # transport doesn't know provider-delta text
            bytes_received=bytes_received,
            cause=exc,
        ) from exc


def parse_sse_stream_sync(
    response: httpx.Response,
    *,
    max_duration: float | None = None,
) -> Iterator[dict[str, Any]]:
    """Parse Server-Sent Events from a sync httpx streaming response.

    See :func:`parse_sse_stream` for the ``max_duration`` semantics.

    B1.3 (#570): mirrors the async variant's interruption handling —
    network failures after the first chunk raise
    :class:`KaosLLMStreamInterruptedError` instead of an opaque
    transport error.
    """
    t_start = time.monotonic()
    buffer = ""
    bytes_received = 0
    try:
        for chunk in response.iter_text():
            if max_duration is not None and (time.monotonic() - t_start) > max_duration:
                raise KaosLLMTransportError(
                    f"Stream wall-clock exceeded: open for >{max_duration:.1f}s. "
                    "The provider may have stalled or forgotten to terminate the stream. "
                    "Increase KAOS_LLM_STREAM_MAX_DURATION (or pass "
                    "RequestOptions(stream_max_duration=...)) if longer streams are expected."
                )
            bytes_received += len(chunk)
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        return
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        logger.debug("Skipping unparseable SSE data: %s", data[:100])
    except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ProtocolError) as exc:
        if bytes_received == 0:
            raise KaosLLMTransportError(
                "Streaming connection failed before any data was received."
            ) from exc
        raise KaosLLMStreamInterruptedError(
            f"Streaming connection interrupted after {bytes_received} bytes",
            partial_text="",
            bytes_received=bytes_received,
            cause=exc,
        ) from exc


# ---------------------------------------------------------------------------
# Response-body size cap
# ---------------------------------------------------------------------------


def _enforce_response_size(
    response: httpx.Response,
    *,
    max_response_bytes: int | None,
    provider: str,
) -> None:
    """Raise ``KaosLLMTransportError`` if the response body exceeds the cap.

    Two checks, in order:

    1. ``Content-Length`` header (if present): rejects oversized
       responses BEFORE the bytes are touched, even though httpx has
       typically already read them in non-streaming mode.
    2. ``len(response.content)`` (when no Content-Length): the buffered
       body length. ``response.content`` would force a read on a
       streaming response — we MUST NOT call this on streaming
       responses; ``request_stream_async`` uses ``client.stream()`` and
       relies on the wall-clock cap instead.

    ``max_response_bytes=None`` disables the check (used by callers that
    have already validated, or for streaming).
    """
    if max_response_bytes is None:
        return
    cl = response.headers.get("content-length")
    if cl is not None:
        try:
            declared = int(cl)
        except ValueError:
            declared = -1
        if declared > max_response_bytes:
            raise KaosLLMTransportError(
                f"Response too large from {provider}: "
                f"Content-Length={declared} bytes exceeds cap "
                f"({max_response_bytes} bytes). "
                "Increase KAOS_LLM_MAX_RESPONSE_BYTES (or pass "
                "RequestOptions(max_response_bytes=...)) if a larger response "
                "is expected; otherwise this likely indicates a misbehaving "
                "endpoint.",
                provider=provider,
            )
        # Content-Length within budget: trust it; httpx already read the body.
        return
    # No Content-Length — fall back to the actually-buffered body size.
    body_len = len(response.content)
    if body_len > max_response_bytes:
        raise KaosLLMTransportError(
            f"Response too large from {provider}: "
            f"{body_len} bytes exceeds cap ({max_response_bytes} bytes). "
            "Increase KAOS_LLM_MAX_RESPONSE_BYTES (or pass "
            "RequestOptions(max_response_bytes=...)) if a larger response "
            "is expected; otherwise this likely indicates a misbehaving "
            "endpoint.",
            provider=provider,
        )


# ---------------------------------------------------------------------------
# Retry sleep computation
# ---------------------------------------------------------------------------


def _compute_retry_sleep(
    last_error: Exception,
    attempt: int,
    retry_policy: RetryPolicy,
) -> tuple[float, float | None]:
    """Decide how long to sleep before the next retry.

    Returns ``(sleep_seconds, retry_after_used)``. When the captured
    error carries a server-supplied ``Retry-After`` (RFC 9110 §10.2.3)
    AND that delta is non-negative, we honour it verbatim — capped at
    ``retry_policy.max_backoff`` so a hostile server can't strand the
    client for hours. We deliberately do NOT add jitter on top of a
    Retry-After hint: the server's instruction is authoritative.

    Otherwise we fall back to ``retry_policy.backoff_seconds(attempt)``
    (full-jitter exponential backoff).
    """
    retry_after = getattr(last_error, "retry_after", None)
    if isinstance(retry_after, int | float) and retry_after >= 0:
        capped = min(float(retry_after), retry_policy.max_backoff)
        return capped, float(retry_after)
    return retry_policy.backoff_seconds(attempt), None


# ---------------------------------------------------------------------------
# Request execution with retry
# ---------------------------------------------------------------------------


async def execute_with_retry(
    client: httpx.AsyncClient,
    request: ProviderRequest,
    *,
    retry_policy: RetryPolicy,
    provider: str,
    timeout: float | None = None,
    on_retry: Any = None,
    log_extra: dict[str, Any] | None = None,
    max_response_bytes: int | None = None,
) -> httpx.Response:
    """Execute an HTTP request with retry policy.

    Returns the successful httpx.Response.
    Raises KaosLLMRetryExhaustedError if all retries fail.

    Args:
        on_retry: Optional callback ``(request, attempt, exception)`` fired before each retry.
        log_extra: Optional ``extra=`` payload (typically ``{"session_id": ..., "trace_id": ...}``)
            merged into the structured log records emitted by this function. Lets callers
            propagate ``KaosContext`` correlation IDs into the kaos-core ``ContextFilter``.
        max_response_bytes: Optional hard cap on the response body size. ``None`` (the
            default) disables the check; callers that own a settings object should pass
            ``settings.max_response_bytes`` or the per-request override from
            ``RequestOptions``. The check is non-streaming only — streaming responses
            (``request_stream_async``) are bounded by ``stream_max_duration`` instead.
    """
    base_extra: dict[str, Any] = dict(log_extra) if log_extra else {}
    last_error: Exception | None = None

    for attempt in range(retry_policy.max_retries + 1):
        success_response: httpx.Response | None = None
        success_latency_ms: float | None = None
        # Egress audit (Issue 4) — emit one structured ``vendor_egress``
        # line per outbound attempt. Re-attempts ship the same body, so
        # they get logged each time; the ``attempt`` field disambiguates.
        emit_vendor_egress_log(
            provider=provider,
            model=request.model,
            body=request.body,
            attempt=attempt,
            extra=base_extra,
        )
        try:
            t0 = time.monotonic()
            response = await client.request(
                "POST",
                request.endpoint,
                json=request.body,
                headers=request.headers,
                timeout=timeout,
            )
            latency_ms = (time.monotonic() - t0) * 1000

            raise_for_status(response, provider=provider, model=request.model)

            # Capture for post-handler size check below. We deliberately
            # do NOT run ``_enforce_response_size`` inside this try block
            # because its ``KaosLLMTransportError`` would otherwise be
            # swallowed by the generic transport-error catch and become
            # spuriously retryable. Size-cap rejections are terminal.
            success_response = response
            success_latency_ms = latency_ms

        except KaosLLMAuthError:
            raise  # never retry auth errors

        except (KaosLLMProviderError, KaosLLMTransportError) as exc:
            last_error = exc
            if retry_policy.should_retry(exc, attempt):
                backoff, retry_after_used = _compute_retry_sleep(exc, attempt, retry_policy)
                if on_retry:
                    on_retry(request, attempt, exc)
                # Log line includes the actual error message so operators
                # tailing stderr (CI runs, benchmark scripts, agent
                # observability) see WHAT we're retrying — not just "LLM
                # retry". This is the difference between catching a flex-
                # tier 500 storm in real time versus discovering it in
                # post-mortem benchmark numbers.
                tier = request.body.get("service_tier")
                logger.warning(
                    "LLM retry %d/%d for %s on %s (tier=%s, backoff=%.1fs, retry_after=%s): %s",
                    attempt + 1,
                    retry_policy.max_retries,
                    request.model,
                    provider,
                    tier or "default",
                    backoff,
                    retry_after_used,
                    exc,
                    extra={
                        **base_extra,
                        "provider": provider,
                        "attempt": attempt + 1,
                        "max_retries": retry_policy.max_retries,
                        "backoff_s": backoff,
                        "retry_after_s": retry_after_used,
                        "service_tier": tier,
                        "error": str(exc),
                    },
                )
                await asyncio.sleep(backoff)
                continue
            # Non-retryable: break out of the loop so the
            # service_tier-fallback path below has a chance to run.
            # Auth errors (401/403) are caught above and re-raised,
            # bypassing this fall-through entirely.
            break

        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
            last_error = KaosLLMTransportError(
                f"Connection error to {provider}: {exc}",
                provider=provider,
            )
            if retry_policy.should_retry(exc, attempt):
                # Connection-level errors don't carry headers, so there
                # is no Retry-After to honour — pure jittered backoff.
                backoff, retry_after_used = _compute_retry_sleep(last_error, attempt, retry_policy)
                if on_retry:
                    on_retry(request, attempt, last_error)
                tier = request.body.get("service_tier")
                logger.warning(
                    "LLM transport retry %d/%d for %s on %s (tier=%s, backoff=%.1fs, "
                    "retry_after=%s): %s",
                    attempt + 1,
                    retry_policy.max_retries,
                    request.model,
                    provider,
                    tier or "default",
                    backoff,
                    retry_after_used,
                    exc,
                    extra={
                        **base_extra,
                        "provider": provider,
                        "attempt": attempt + 1,
                        "backoff_s": backoff,
                        "retry_after_s": retry_after_used,
                        "service_tier": tier,
                        "error": str(exc),
                    },
                )
                await asyncio.sleep(backoff)
                continue
            break

        # Reached only on a successful response. The size cap is enforced
        # AFTER the try/except so that any error it raises bypasses the
        # retry catch (a "Response too large" failure is terminal — there
        # is no point retrying the same endpoint hoping it returns a
        # smaller payload).
        if success_response is not None:
            _enforce_response_size(
                success_response,
                max_response_bytes=max_response_bytes,
                provider=provider,
            )
            if success_latency_ms is not None:
                success_response.extensions["latency_ms"] = success_latency_ms  # type: ignore[index]
            return success_response

    # Graceful flex fallback: if all retries failed AND the request body
    # included service_tier (e.g. "flex"), retry ONE more time without it.
    # Flex is in beta and intermittently returns 5xx / 429 on some models
    # and with structured output. This fallback ensures the request
    # succeeds at standard pricing rather than failing entirely.
    #
    # Gating is intentionally broad: ANY retryable provider error (429,
    # 500, 502, 503, 504) OR transport-level failure triggers the
    # fallback — the empirical pattern with gpt-5.4 / gpt-5.4-mini on
    # flex tier has been a mix of 500s and 503/504s and read-timeouts.
    # Restricting to 500 only (the previous behavior) caused 4 of 5 docs
    # to fail in the WS-TR.PR-6f.6 CUAD benchmark.
    #
    # References:
    #   - https://community.openai.com/t/critical-bug-using-flex-model-for-gpt5-since-last-monday-500-internal-server-error/1364470
    #   - https://community.openai.com/t/flex-service-tier-500-error/1362451
    # Gate the fallback on max_retries > 0: a caller passing
    # max_retries=0 has explicitly opted out of all recovery (e.g., a
    # unit test asserting exactly-one-call). Firing the fallback in
    # that case would double the call count.
    fallback_eligible = (
        retry_policy.max_retries > 0
        and "service_tier" in request.body
        and (
            (
                isinstance(last_error, KaosLLMProviderError)
                and getattr(last_error, "status_code", None) in retry_policy.retryable_status_codes
            )
            or isinstance(last_error, KaosLLMTransportError)
        )
    )
    fallback_error: Exception | None = None
    if fallback_eligible:
        original_tier = request.body.get("service_tier")
        fallback_body = {k: v for k, v in request.body.items() if k != "service_tier"}
        logger.warning(
            "service_tier=%r failed after %d retries (last_error=%s) — "
            "falling back to default tier",
            original_tier,
            retry_policy.max_retries + 1,
            last_error,
            extra={**base_extra, "provider": provider, "model": request.model},
        )
        try:
            t0 = time.monotonic()
            response = await client.request(
                "POST",
                request.endpoint,
                json=fallback_body,
                headers=request.headers,
                timeout=timeout,
            )
            latency_ms = (time.monotonic() - t0) * 1000
            raise_for_status(response, provider=provider, model=request.model)
            _enforce_response_size(
                response,
                max_response_bytes=max_response_bytes,
                provider=provider,
            )
            response.extensions["latency_ms"] = latency_ms  # type: ignore[index]
            logger.debug(
                "service_tier=%r fallback succeeded for %s on %s",
                original_tier,
                request.model,
                provider,
                extra={**base_extra, "provider": provider, "model": request.model},
            )
            return response
        except Exception as exc:
            fallback_error = exc
            logger.warning(
                "Flex fallback to default tier ALSO failed: %s",
                exc,
                extra={**base_extra, "provider": provider, "model": request.model},
            )

    final_error = fallback_error or last_error

    # If we never had a chance to retry (max_retries=0) AND the error
    # was non-retryable in the first place, raise the underlying provider
    # error directly — KaosLLMRetryExhaustedError is misleading for the
    # zero-retry case. Otherwise wrap in RetryExhausted so callers can
    # distinguish "exhausted retries" from "first-attempt failure".
    if isinstance(last_error, KaosLLMProviderError) and not retry_policy.should_retry(
        last_error, 0
    ):
        # Non-retryable status code (e.g. 400, 422) — re-raise verbatim.
        # Authentication errors are handled above and never reach here.
        raise last_error

    raise KaosLLMRetryExhaustedError(
        f"All {retry_policy.max_retries} retry attempts exhausted for {provider}"
        + (" (incl. service_tier fallback)" if fallback_eligible else ""),
        attempts=retry_policy.max_retries,
        last_error=final_error,
        provider=provider,
    )
