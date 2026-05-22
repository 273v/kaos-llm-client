"""Exception hierarchy for kaos-llm-client.

All exceptions inherit ``KaosCoreError(message, **details)`` so that structured
context flows through for agent-friendly error messages.
"""

from __future__ import annotations

from typing import Any

from kaos_core.exceptions import KaosCoreError


class KaosLLMError(KaosCoreError):
    """Base for all kaos-llm-client errors."""


class KaosLLMAuthError(KaosLLMError):
    """Authentication failed. Never retried.

    Raised when:
    - API key is missing or empty
    - Provider returns 401/403
    """


class KaosLLMTransportError(KaosLLMError):
    """Network or connection failure. May be retried.

    Raised when:
    - Connection refused / timeout
    - DNS resolution failure
    - HTTP/2 protocol error
    """


class KaosLLMProviderError(KaosLLMError):
    """Provider returned an error response (4xx/5xx).

    Carries status_code, raw error body, and provider name in details.

    ``retry_after`` carries the number of seconds the server asked the
    client to wait before retrying — parsed from the ``Retry-After`` HTTP
    header (RFC 9110 §10.2.3) or the non-standard ``retry-after-ms``
    header used by OpenAI. ``None`` means the server did not advise a
    delay; callers should fall back to their normal backoff policy.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str | None = None,
        status_code: int,
        raw_error: dict[str, Any] | None = None,
        fix: str | None = None,
        retry_after: float | None = None,
        **details: Any,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            model=model,
            status_code=status_code,
            raw_error=raw_error,
            fix=fix,
            retry_after=retry_after,
            **details,
        )
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.raw_error = raw_error
        self.fix = fix
        self.retry_after = retry_after


class KaosLLMRetryExhaustedError(KaosLLMTransportError):
    """All retry attempts exhausted.

    Carries the last exception as ``__cause__`` and attempt count in details.
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        last_error: Exception | None = None,
        **details: Any,
    ) -> None:
        super().__init__(message, attempts=attempts, **details)
        self.attempts = attempts
        # Expose as both an attribute (callers don't need to know about
        # __cause__ chaining) AND as __cause__ (Python's standard
        # exception-chain inspection still works).
        self.last_error = last_error
        if last_error is not None:
            self.__cause__ = last_error


class KaosLLMStreamInterruptedError(KaosLLMTransportError):
    """Streaming response was interrupted after the first chunk.

    B1.3 (broad-reliability roadmap #570): a network failure or
    provider 5xx that fires AFTER the streaming response has begun
    cannot be retried transparently — the partial bytes already
    reached the client. This error carries them so the caller can
    decide between (a) retry-as-fresh-call (acceptable when no
    partial text was emitted, e.g. the disconnect happened during
    initial connection setup), and (b) ship-partial-with-footer
    (the user has already seen N bytes; tell them honestly that
    streaming stopped at that point).

    ``partial_text`` carries everything successfully received before
    the interrupt — the concatenated content payload (NOT the raw
    SSE wire bytes). When the underlying provider exposes structured
    deltas, the producer should accumulate the user-visible text and
    pass it here.

    ``cause`` is the underlying httpx / network exception. Available
    as ``__cause__`` AND as a typed attribute for callers that
    inspect by attribute name.
    """

    def __init__(
        self,
        message: str,
        *,
        partial_text: str = "",
        bytes_received: int = 0,
        cause: Exception | None = None,
        **details: Any,
    ) -> None:
        super().__init__(
            message,
            partial_text=partial_text,
            bytes_received=bytes_received,
            **details,
        )
        self.partial_text = partial_text
        self.bytes_received = bytes_received
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause


class KaosLLMValidationError(KaosLLMError):
    """Response failed Pydantic validation in the ``pydantic()`` helper.

    Carries the raw text and validation errors in details.
    """


class KaosLLMProviderPolicyError(KaosLLMError):
    """Tenant-policy violation — the requested provider does not satisfy a
    declared tenant compliance constraint (e.g. ``hipaa_required=True``
    against a non-BAA provider).

    Carries ``provider``, ``model``, and the unmet constraint name so
    callers can surface a remediation hint without re-deriving it.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        constraint: str | None = None,
        **details: Any,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            model=model,
            constraint=constraint,
            **details,
        )
        self.provider = provider
        self.model = model
        self.constraint = constraint
