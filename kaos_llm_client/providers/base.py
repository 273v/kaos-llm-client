"""Abstract base class for all LLM provider clients.

The base class owns transport (httpx), retry logic, caching, and the
convenience API. Subclasses implement provider-specific request building,
response parsing, and auth header injection.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, TypeVar, cast

import httpx
from kaos_core.logging import get_logger

from kaos_llm_client.cache import (
    CacheBackend,
    FileCache,
    NullCache,
    auth_scope_digest,
    cache_key,
)
from kaos_llm_client.cost import estimate_call_cost
from kaos_llm_client.errors import KaosLLMValidationError
from kaos_llm_client.json_utils import extract_json
from kaos_llm_client.profiles import (
    ModelProfile,
    StructuredOutputMode,
    resolve_profile,
)
from kaos_llm_client.settings import KaosLLMSettings
from kaos_llm_client.transport import (
    RetryPolicy,
    create_async_http_client,
    create_http_client,
    execute_with_retry,
    parse_sse_stream,
    raise_for_status,
    redact_response_headers,
    run_sync,
)
from kaos_llm_client.types import (
    CachePolicy,
    EmbeddingResponse,
    ProviderRequest,
    ProviderResponse,
    RequestHooks,
    RequestOptions,
    StreamAccumulator,
    StreamChunk,
    ToolChoice,
    ToolDefinition,
)

logger = get_logger("kaos_llm_client.providers.base")

T = TypeVar("T")


class BaseProviderClient(ABC):
    """Abstract base for all LLM provider clients.

    Subclasses must implement:
    - ``_provider_name`` — string identifier (e.g., "openai", "anthropic")
    - ``_build_request()`` — construct a ProviderRequest from messages + kwargs
    - ``_parse_response()`` — parse raw JSON dict into ProviderResponse
    - ``_parse_stream_chunk()`` — parse a single SSE data dict into StreamChunk
    - ``_build_headers()`` — return auth + provider-specific headers
    - ``_default_endpoint()`` — return the default API endpoint path
    - ``_base_url`` property — return the provider's base URL
    """

    _provider_name: str = ""

    def __init__(
        self,
        model: str,
        *,
        settings: KaosLLMSettings | None = None,
        context: Any = None,
        profile: ModelProfile | None = None,
        cache: CacheBackend | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        hooks: RequestHooks | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        # Settings resolution honours the kaos-core Configuration Hierarchy:
        # an explicit ``settings=`` kwarg wins outright; otherwise we route
        # through ``KaosLLMSettings.from_context(context)`` so per-request
        # ``KaosContext._config`` overrides beat env vars. ``from_context``
        # accepts ``context=None`` and degrades to ``cls()``.
        if settings is not None:
            self._settings = settings
        else:
            self._settings = KaosLLMSettings.from_context(context)
        self._context = context
        self._api_key_override = api_key
        self._base_url_override = base_url
        self._hooks = hooks
        self._extra_kwargs = kwargs

        # Resolve profile
        self.profile = profile or resolve_profile(self._provider_name, model)

        # Transport
        self._timeout = timeout if timeout is not None else self._settings.default_timeout
        self._retry_policy = RetryPolicy(
            max_retries=max_retries
            if max_retries is not None
            else self._settings.default_max_retries,
            backoff_base=self._settings.retry_backoff_base,
        )

        # Cache
        if cache is not None:
            self._cache = cache
        elif self._settings.cache_enabled:
            cache_path = self._settings.cache_path or str(Path.home() / ".cache" / "kaos" / "llm")
            self._cache = FileCache(cache_path)
        else:
            self._cache = NullCache()

        # httpx clients (lazy init)
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

        # Per-instance metrics — incremented on every cache hit/miss in
        # ``request_async`` / ``request_stream_async``. Exposed via
        # :meth:`metrics` so callers can build dashboards without
        # parsing log lines. Streaming and embedding paths bump the
        # same counters as plain chat.
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    # --- Properties ---

    @property
    def _base_url(self) -> str:
        """Return the provider's base URL."""
        if self._base_url_override:
            return self._base_url_override
        return self._get_default_base_url()

    @abstractmethod
    def _get_default_base_url(self) -> str:
        """Return the default base URL from settings."""
        ...

    # --- httpx client lifecycle ---

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = create_http_client(
                base_url=self._base_url,
                timeout=self._timeout,
                trust_env=self._settings.trust_env,
            )
        return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = create_async_http_client(
                base_url=self._base_url,
                timeout=self._timeout,
                trust_env=self._settings.trust_env,
            )
        return self._async_client

    def close(self) -> None:
        """Close underlying HTTP clients."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None
        if self._async_client is not None:
            run_sync(self._async_client.aclose())
            self._async_client = None

    async def aclose(self) -> None:
        """Async close underlying HTTP clients."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def __enter__(self) -> BaseProviderClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    async def __aenter__(self) -> BaseProviderClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    # --- Abstract methods for subclasses ---

    @abstractmethod
    def _build_request(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ProviderRequest:
        """Build a provider-native request from messages."""
        ...

    @abstractmethod
    def _parse_response(self, raw: dict[str, Any], request: ProviderRequest) -> ProviderResponse:
        """Parse a raw provider JSON response into a ProviderResponse."""
        ...

    @abstractmethod
    def _parse_stream_chunk(self, data: dict[str, Any]) -> StreamChunk | list[StreamChunk]:
        """Parse a single SSE data dict into one or more StreamChunks.

        Returns a single StreamChunk or a list when one SSE event produces
        multiple logical chunks (e.g., OpenAI parallel tool call deltas).
        """
        ...

    @abstractmethod
    def _build_headers(self) -> dict[str, str]:
        """Return auth and provider-specific headers."""
        ...

    @abstractmethod
    def _default_endpoint(self) -> str:
        """Return the default API endpoint path."""
        ...

    def _resolve_api_key(self) -> str:
        """Resolve the API key from override, settings, or raise."""
        if self._api_key_override:
            return self._api_key_override
        return self._get_api_key_from_settings()

    @abstractmethod
    def _get_api_key_from_settings(self) -> str:
        """Get the API key from settings. Raise if not available."""
        ...

    def _hook_request(self, request: ProviderRequest) -> ProviderRequest:
        """Return the request to pass to lifecycle hooks.

        Auth headers (``Authorization``, ``api-key``) are stripped by
        default to prevent accidental key leakage when hooks log the
        request. Pass ``RequestHooks(..., include_auth_headers=True)``
        to opt out of redaction (e.g. for transport debugging).
        """
        if not self._hooks or self._hooks.include_auth_headers:
            return request
        if not request.headers:
            return request
        redacted_headers = {
            k: ("<redacted>" if k.lower() in ("authorization", "api-key") else v)
            for k, v in request.headers.items()
        }
        # Avoid mutating the live request — model_copy yields a shallow
        # copy with header replacement; the underlying httpx call still
        # sees the original headers.
        return request.model_copy(update={"headers": redacted_headers})

    def _cache_auth_scope(self) -> str | None:
        """Return a short digest of the credential for cache-key namespacing.

        Used by :func:`cache_key` to prevent two principals on the same
        host from sharing cached responses (multi-tenant isolation). The
        digest is a one-way BLAKE2b hash of the resolved api-key — never
        the raw key — so it is safe to embed in cache filenames.

        Returns ``None`` if no credential can be resolved (e.g. AAD-only
        Azure clients before the async token has been resolved); in that
        case the cache key falls back to base-url-only namespacing,
        which is still per-host but not per-principal.
        """
        try:
            api_key = self._resolve_api_key()
        except Exception:
            # Auth missing → no per-principal scoping; cache key falls
            # back to base-url-only namespacing (per-host but not
            # per-principal).
            return None
        return auth_scope_digest(api_key)

    # --- Logging context ---

    # Canonical structured-log keys emitted by kaos-llm-client. Any new
    # log call site SHOULD pull from this set so a single grep finds
    # them everywhere — Splunk/Datadog/OTel exporters can index them
    # without re-parsing the message string. Keys NOT in this list
    # should be added here first (and to the README key reference)
    # before being emitted in production code.
    #
    # OpenTelemetry ``gen_ai.*`` mapping (loose, as documented in
    # ``kaos_llm_client.cost``):
    #
    #   provider          -> gen_ai.system
    #   model             -> gen_ai.request.model
    #   request_id        -> gen_ai.response.id (when no response_id)
    #   response_id       -> gen_ai.response.id
    #   input_tokens      -> gen_ai.usage.input_tokens
    #   output_tokens     -> gen_ai.usage.output_tokens
    #   total_tokens      -> gen_ai.usage.total_tokens
    #   reasoning_tokens  -> (no canonical OTel mapping; pass-through)
    #   latency_ms        -> (no canonical OTel mapping; pass-through)
    #   cache_hit         -> (no canonical OTel mapping; pass-through)
    #   estimated_usd     -> (no canonical OTel mapping; pass-through)
    #
    # Canonical keys: provider, model, request_id, response_id,
    # session_id, trace_id, tool_name, attempt, latency_ms, cache_hit,
    # error, retry_after_s, input_tokens, output_tokens, total_tokens,
    # reasoning_tokens, estimated_usd, embedding_count,
    # embedding_dimensions, total_providers.
    def _log_extra(
        self,
        *,
        request: ProviderRequest | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Build the ``extra=`` payload for a structured log record.

        Populates ``session_id`` and ``trace_id`` from ``self._context`` so
        kaos-core's ``ContextFilter`` can attach them to the emitted log
        record. Falls back to a 16-char prefix of ``request.request_id``
        for ``trace_id`` when the context has no trace of its own.

        Missing IDs render as the literal string ``"-"`` (matching
        kaos-core's ``StructuredFormatter`` placeholder for absent
        context) rather than ``None`` — otherwise the formatter would
        emit ``[session=None trace=None]`` instead of the intended
        ``[session=- trace=-]``.

        See the module-level comment immediately above this method for
        the canonical structured-log key set used across kaos-llm-client.
        """
        ctx = self._context
        session_id: str | None = None
        trace_id: str | None = None
        if ctx is not None:
            session_id = getattr(ctx, "session_id", None)
            trace_id = getattr(ctx, "trace_id", None)
        if trace_id is None and request is not None:
            req_id = request.request_id or ""
            trace_id = req_id[:16] if req_id else None

        payload: dict[str, Any] = {
            "session_id": session_id if session_id else "-",
            "trace_id": trace_id if trace_id else "-",
        }
        payload.update(extra)
        return payload

    # --- Metrics ---

    def metrics(self) -> dict[str, Any]:
        """Return per-instance cache-hit / cache-miss counters.

        Returns a dict with ``cache_hits``, ``cache_misses``, and
        ``cache_hit_rate``. ``cache_hit_rate`` is ``0.0`` when no calls
        have been made (no division by zero). Call sites that want to
        report aggregate cache effectiveness can sample this method
        between batches without parsing log lines.
        """
        total = self._cache_hits + self._cache_misses
        rate = (self._cache_hits / total) if total > 0 else 0.0
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": rate,
        }

    # --- Per-call cost / completion log ---

    def _emit_call_complete_log(
        self,
        request: ProviderRequest,
        result: ProviderResponse | None,
        *,
        cache_hit: bool,
    ) -> None:
        """Emit one structured ``LLM call complete`` info-log per call.

        This is the foundation the live-tier $50/run cost ceiling
        documented in ``docs/oss/40-ci-cd/live-tier.yml.md`` builds on.
        Wrapped in try/except so a logging-side failure (e.g. stale
        pricing-table dict raising ``KeyError``) never breaks a real
        provider call. The cost helper itself returns ``None`` on
        unknown models — that's a normal pass-through, not an error.
        """
        try:
            usage = result.usage if result is not None else None
            estimated_usd: float | None
            try:
                estimated_usd = estimate_call_cost(usage, self.model)
            except Exception:
                estimated_usd = None

            input_tokens = usage.input_tokens if usage is not None else 0
            output_tokens = usage.output_tokens if usage is not None else 0
            total_tokens = usage.total_tokens if usage is not None else 0
            reasoning_tokens = (
                getattr(usage, "reasoning_tokens", None) if usage is not None else None
            )
            response_id = result.response_id if result is not None else None
            latency_ms = getattr(result, "latency_ms", None) if result is not None else None

            logger.info(
                "LLM call complete",
                extra=self._log_extra(
                    request=request,
                    provider=self._provider_name,
                    model=self.model,
                    request_id=(request.request_id or "")[:16] or None,
                    response_id=response_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cache_hit=cache_hit,
                    latency_ms=latency_ms,
                    estimated_usd=estimated_usd,
                ),
            )
        except Exception:  # pragma: no cover - log-side defensive guard
            # Cost / log emission MUST NOT break a real LLM call. If the
            # pricing helper or kaos-core logging chokes (e.g. a
            # ContextFilter raising on a pathological extra value), we
            # fall through silently with a debug breadcrumb.
            logger.debug("LLM call-complete log emission failed", exc_info=True)

    # --- Message preprocessing ---

    def _preprocess_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip CachePoint messages for providers that don't support them.

        Subclasses may override to convert cache markers into provider-specific
        cache control annotations (e.g., Anthropic's ``cache_control`` blocks).
        """
        return [m for m in messages if m.get("role") != "cache_point"]

    # --- Raw request API ---

    async def request_async(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: RequestOptions | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Make a single async request to the provider."""
        messages = self._preprocess_messages(messages)
        request = self._build_request(
            messages, tools=tools, tool_choice=tool_choice, stream=False, **kwargs
        )
        request.headers.update(self._build_headers())

        # Lifecycle hook: on_request
        if self._hooks and self._hooks.on_request:
            self._hooks.on_request(self._hook_request(request))

        # Check cache
        effective_policy = self._resolve_cache_policy(options)
        if effective_policy != CachePolicy.SKIP:
            key = cache_key(
                request,
                base_url=self._base_url,
                auth_scope=self._cache_auth_scope(),
            )
            cached = self._cache.get(key)
            if cached is not None:
                self._cache_hits += 1
                logger.debug(
                    "Cache hit for %s:%s",
                    self._provider_name,
                    self.model,
                    extra=self._log_extra(
                        request=request,
                        provider=self._provider_name,
                        model=self.model,
                        cache_hit=True,
                    ),
                )
                # Emit the per-call completion log on cache-hit too so
                # dashboards can count cached calls separately. Cost is
                # zero on cache-hit (the call is free), but tokens are
                # whatever the cached response held.
                self._emit_call_complete_log(request, cached, cache_hit=True)
                return cached
            else:
                self._cache_misses += 1

        # Apply per-request options
        timeout = options.timeout if options and options.timeout else None
        retry_policy = self._retry_policy
        if options:
            if options.max_retries is not None or options.retry_backoff_base is not None:
                retry_policy = RetryPolicy(
                    max_retries=options.max_retries
                    if options.max_retries is not None
                    else retry_policy.max_retries,
                    backoff_base=options.retry_backoff_base
                    if options.retry_backoff_base is not None
                    else retry_policy.backoff_base,
                    max_backoff=retry_policy.max_backoff,
                )
            if options.extra_headers:
                request.headers.update(options.extra_headers)

        # Per-request response-size cap takes priority over the settings default.
        max_response_bytes = (
            options.max_response_bytes
            if options and options.max_response_bytes is not None
            else self._settings.max_response_bytes
        )

        # Execute request
        client = self._get_async_client()
        try:
            response = await execute_with_retry(
                client,
                request,
                retry_policy=retry_policy,
                provider=self._provider_name,
                timeout=timeout,
                on_retry=self._hooks.on_retry if self._hooks else None,
                log_extra=self._log_extra(request=request),
                max_response_bytes=max_response_bytes,
            )
        except Exception as exc:
            # Lifecycle hook: on_error
            if self._hooks and self._hooks.on_error:
                self._hooks.on_error(self._hook_request(request), exc)
            raise

        raw = response.json()
        latency_ms = response.extensions.get("latency_ms")
        result = self._parse_response(raw, request)
        # Carry transport metadata onto the result. Sensitive headers
        # (Set-Cookie, Authorization, etc.) are redacted before exposure
        # — see ``redact_response_headers`` in transport.py for the list.
        updates: dict[str, Any] = {
            "status_code": response.status_code,
            "response_headers": redact_response_headers(response.headers),
        }
        if latency_ms is not None:
            updates["latency_ms"] = latency_ms
        result = result.model_copy(update=updates)

        # Lifecycle hook: on_response
        if self._hooks and self._hooks.on_response:
            self._hooks.on_response(self._hook_request(request), result)

        # Per-call completion log — runs AFTER on_response so any
        # downstream observability hooks see the result first. Wrapped
        # in its own try/except inside the helper; never breaks a real
        # call.
        self._emit_call_complete_log(request, result, cache_hit=False)

        # Store in cache
        if effective_policy != CachePolicy.SKIP:
            key = cache_key(
                request,
                base_url=self._base_url,
                auth_scope=self._cache_auth_scope(),
            )
            self._cache.put(key, result)

        return result

    def request(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: RequestOptions | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Make a single sync request to the provider."""
        return run_sync(
            self.request_async(
                messages, tools=tools, tool_choice=tool_choice, options=options, **kwargs
            )
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
        """Make a streaming async request. Yields StreamChunks.

        Includes retry on stream setup failure (429/5xx), lifecycle hooks,
        and per-request options — same contract as ``request_async()``.
        """
        import asyncio

        messages = self._preprocess_messages(messages)
        request = self._build_request(
            messages, tools=tools, tool_choice=tool_choice, stream=True, **kwargs
        )
        request.headers.update(self._build_headers())

        # Per-request options
        timeout = options.timeout if options and options.timeout else None
        retry_policy = self._retry_policy
        if options:
            if options.max_retries is not None or options.retry_backoff_base is not None:
                retry_policy = RetryPolicy(
                    max_retries=options.max_retries
                    if options.max_retries is not None
                    else retry_policy.max_retries,
                    backoff_base=options.retry_backoff_base
                    if options.retry_backoff_base is not None
                    else retry_policy.backoff_base,
                    max_backoff=retry_policy.max_backoff,
                )
            if options.extra_headers:
                request.headers.update(options.extra_headers)

        # Per-request stream wall-clock cap takes priority over the settings default.
        stream_max_duration = (
            options.stream_max_duration
            if options and options.stream_max_duration is not None
            else self._settings.stream_max_duration
        )

        # Lifecycle hook: on_request
        if self._hooks and self._hooks.on_request:
            self._hooks.on_request(self._hook_request(request))

        client = self._get_async_client()

        # Retry loop for stream setup (connection + initial response)
        last_error: Exception | None = None
        yielded_any = False
        for attempt in range(retry_policy.max_retries + 1):
            try:
                async with client.stream(
                    "POST",
                    request.endpoint,
                    json=request.body,
                    headers=request.headers,
                    timeout=timeout,
                ) as response:
                    # Streaming responses don't read the body by default.
                    # If the server returned an error, raise_for_status()
                    # needs the JSON body to extract the provider's error
                    # message — without aread() it would crash with
                    # httpx.ResponseNotRead and mask the real failure
                    # (notably blocking the flex-tier 500 fallback).
                    if not response.is_success:
                        await response.aread()
                    raise_for_status(response, provider=self._provider_name, model=self.model)

                    accumulator = StreamAccumulator(
                        provider=self._provider_name,
                        model=self.model,
                        request_id=request.request_id,
                        strip_leading_whitespace=self.profile.strip_leading_whitespace,
                    )

                    yielded_any = False
                    async for data in parse_sse_stream(response, max_duration=stream_max_duration):
                        result = self._parse_stream_chunk(data)
                        chunks = result if isinstance(result, list) else [result]
                        for chunk in chunks:
                            accumulator.feed(chunk)
                            yielded_any = True
                            yield chunk

                    # Yield final done chunk
                    done_chunk = StreamChunk(type="done")
                    accumulator.feed(done_chunk)
                    yield done_chunk

                    # Lifecycle hook: on_response
                    if self._hooks and self._hooks.on_response:
                        self._hooks.on_response(
                            self._hook_request(request), accumulator.accumulated
                        )

                    # Per-call completion log for streaming. Usage may
                    # be missing on streamed responses for providers
                    # that don't emit a final usage chunk; the helper
                    # handles ``usage=None`` gracefully and emits zeros.
                    self._emit_call_complete_log(request, accumulator.accumulated, cache_hit=False)

                    # Cache the accumulated response
                    effective_policy = self._resolve_cache_policy(options)
                    if effective_policy != CachePolicy.SKIP:
                        key = cache_key(
                            request,
                            base_url=self._base_url,
                            auth_scope=self._cache_auth_scope(),
                        )
                        self._cache.put(key, accumulator.accumulated)

                    return  # success — exit retry loop

            except Exception as exc:
                # Once chunks have been yielded, retrying would duplicate
                # output. Re-raise immediately instead of falling back.
                if yielded_any:
                    if self._hooks and self._hooks.on_error:
                        self._hooks.on_error(self._hook_request(request), exc)
                    raise

                last_error = exc
                if retry_policy.should_retry(exc, attempt):
                    backoff = retry_policy.backoff_seconds(attempt)
                    if self._hooks and self._hooks.on_retry:
                        self._hooks.on_retry(request, attempt, exc)
                    logger.warning(
                        "Stream retry",
                        extra=self._log_extra(
                            request=request,
                            provider=self._provider_name,
                            attempt=attempt + 1,
                            backoff_s=backoff,
                        ),
                    )
                    await asyncio.sleep(backoff)
                    continue

                # Not retryable — fire error hook and raise
                if self._hooks and self._hooks.on_error:
                    self._hooks.on_error(self._hook_request(request), exc)
                raise

        # All retries exhausted
        if self._hooks and self._hooks.on_error and last_error:
            self._hooks.on_error(self._hook_request(request), last_error)
        if last_error:
            raise last_error

    # --- Convenience API ---

    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Convenience async chat call."""
        return await self.request_async(messages, tools=tools, tool_choice=tool_choice, **kwargs)

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Convenience sync chat call."""
        return run_sync(self.chat_async(messages, tools=tools, tool_choice=tool_choice, **kwargs))

    async def chat_stream_async(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Convenience async streaming chat call."""
        async for chunk in self.request_stream_async(
            messages, tools=tools, tool_choice=tool_choice, **kwargs
        ):
            yield chunk

    async def json_async(
        self,
        messages: list[dict[str, Any]],
        *,
        schema: dict[str, Any] | None = None,
        output_mode: StructuredOutputMode | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Request JSON output from the model.

        Selects the output strategy based on the model profile:
        - ``native``: uses provider's built-in JSON schema enforcement
        - ``tool``: defines a return_output tool with the schema
        - ``prompted``: adds schema instructions to the system prompt
        """
        mode = output_mode or self.profile.default_structured_output_mode

        if mode == StructuredOutputMode.NATIVE:
            kwargs = self._apply_native_json_mode(kwargs, schema)
        elif mode == StructuredOutputMode.TOOL and schema:
            kwargs = self._apply_tool_json_mode(messages, kwargs, schema)
        elif mode == StructuredOutputMode.PROMPTED and schema:
            messages = self._apply_prompted_json_mode(messages, schema)

        return await self.request_async(messages, **kwargs)

    def json(
        self,
        messages: list[dict[str, Any]],
        *,
        schema: dict[str, Any] | None = None,
        output_mode: StructuredOutputMode | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Sync version of ``json_async``."""
        return run_sync(self.json_async(messages, schema=schema, output_mode=output_mode, **kwargs))

    async def pydantic_async(
        self,
        messages: list[dict[str, Any]],
        *,
        output_type: type[T],
        output_mode: StructuredOutputMode | None = None,
        output_validator: Any = None,
        max_validation_retries: int = 0,
        **kwargs: Any,
    ) -> T:
        """Request structured output and validate as a Pydantic model.

        Args:
            messages: Conversation messages.
            output_type: Pydantic BaseModel subclass to validate against.
            output_mode: Override the structured output strategy.
            output_validator: Optional ``Callable[[T], T]`` that validates/transforms
                the parsed model instance. Raise ``ValueError`` or
                ``ValidationError`` to trigger a retry.
            max_validation_retries: How many times to retry on validator failure.
                Each retry appends the error message to the conversation so the
                model can self-correct.
            **kwargs: Additional arguments forwarded to ``json_async``.

        Returns:
            A validated Pydantic model instance.

        Raises:
            KaosLLMValidationError: If the response cannot be parsed/validated
                after all retries are exhausted.
        """
        from pydantic import BaseModel, ValidationError

        if not (isinstance(output_type, type) and issubclass(output_type, BaseModel)):
            raise KaosLLMValidationError(
                f"output_type must be a Pydantic BaseModel subclass, got {output_type}",
                output_type=str(output_type),
            )

        schema = output_type.model_json_schema()
        current_messages = list(messages)
        last_error: Exception | None = None

        for attempt in range(max_validation_retries + 1):
            response = await self.json_async(
                current_messages, schema=schema, output_mode=output_mode, **kwargs
            )

            # Try to parse from output_json first, then from text
            data = response.output_json
            if data is None:
                text = response.text
                data = extract_json(text)

            if data is None:
                last_error = KaosLLMValidationError(
                    "Could not extract JSON from model response",
                    raw_text=response.text[:500],
                    fix="Ensure the model returns valid JSON. Try a different output_mode.",
                )
                if attempt < max_validation_retries:
                    current_messages = [
                        *list(messages),
                        {"role": "assistant", "content": response.text},
                        {
                            "role": "user",
                            "content": f"Your response was not valid JSON. Error: {last_error}. "
                            "Please try again and return valid JSON.",
                        },
                    ]
                    continue
                raise last_error

            try:
                result = cast(T, output_type.model_validate(data))
            except (ValueError, ValidationError) as exc:
                last_error = KaosLLMValidationError(
                    f"Pydantic validation failed: {exc}",
                    raw_data=str(data)[:500],
                    validation_error=str(exc),
                    fix="Check that the model output matches the schema. "
                    "Try adding more specific instructions in the prompt.",
                )
                if attempt < max_validation_retries:
                    current_messages = [
                        *list(messages),
                        {"role": "assistant", "content": response.text},
                        {
                            "role": "user",
                            "content": f"Validation error: {exc}. Please fix and try again.",
                        },
                    ]
                    continue
                raise last_error from exc

            # Run output_validator if provided
            if output_validator is not None:
                try:
                    result = output_validator(result)
                except (ValueError, ValidationError) as exc:
                    last_error = KaosLLMValidationError(
                        f"Output validator failed: {exc}",
                        raw_data=str(data)[:500],
                        validation_error=str(exc),
                        fix="The model output passed schema validation but failed "
                        "the custom validator. Check validator logic or prompt.",
                    )
                    if attempt < max_validation_retries:
                        current_messages = [
                            *list(messages),
                            {"role": "assistant", "content": response.text},
                            {
                                "role": "user",
                                "content": f"Validation error: {exc}. Please fix and try again.",
                            },
                        ]
                        continue
                    raise last_error from exc

            # Attach the full response for introspection
            object.__setattr__(result, "_response", response)
            return result

        # Should not reach here, but just in case. Use an explicit raise
        # rather than ``assert`` — asserts are stripped under ``python -O``
        # and we want the guard to remain in optimised builds.
        if last_error is None:
            raise RuntimeError("json_async exhausted retries without producing a response or error")
        raise last_error

    def pydantic(
        self,
        messages: list[dict[str, Any]],
        *,
        output_type: type[T],
        output_mode: StructuredOutputMode | None = None,
        output_validator: Any = None,
        max_validation_retries: int = 0,
        **kwargs: Any,
    ) -> T:
        """Sync version of ``pydantic_async``."""
        return run_sync(
            self.pydantic_async(
                messages,
                output_type=output_type,
                output_mode=output_mode,
                output_validator=output_validator,
                max_validation_retries=max_validation_retries,
                **kwargs,
            )
        )

    # --- Embeddings ---

    async def embed_async(
        self,
        input: str | list[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        options: RequestOptions | None = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Create embeddings for the given input text(s).

        Not all providers support embeddings. The default implementation raises
        ``NotImplementedError``. Providers that support embeddings (e.g., OpenAI)
        override this method.

        Args:
            input: A single string or list of strings to embed.
            model: Optional model override (defaults to ``self.model``).
            dimensions: Optional output dimensionality (provider-dependent).
            options: Transport-level options (timeout, retries).
            **kwargs: Additional provider-specific parameters.

        Returns:
            An ``EmbeddingResponse`` with the embedding vectors.

        Raises:
            NotImplementedError: If the provider does not support embeddings.
        """
        raise NotImplementedError(
            f"{self._provider_name} does not support embeddings. "
            "Use an embedding-capable provider such as openai or mistral."
        )

    def embed(
        self,
        input: str | list[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        options: RequestOptions | None = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Sync version of ``embed_async``."""
        return run_sync(
            self.embed_async(input, model=model, dimensions=dimensions, options=options, **kwargs)
        )

    # --- Structured output mode helpers ---

    def _apply_native_json_mode(
        self, kwargs: dict[str, Any], schema: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Apply native JSON mode parameters. Subclasses may override."""
        return kwargs

    def _apply_tool_json_mode(
        self,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply tool-based structured output mode."""
        tool = ToolDefinition(
            name="return_output",
            description="Return structured output matching the specified schema.",
            parameters=schema,
        )
        kwargs["tools"] = [tool]
        kwargs["tool_choice"] = ToolChoice(type="specific", name="return_output")
        return kwargs

    def _apply_prompted_json_mode(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Apply prompted structured output mode by modifying messages."""
        schema_str = json.dumps(schema, indent=2)
        instruction = (
            f"\n\nIMPORTANT: Return your response as valid JSON matching this schema:\n"
            f"```json\n{schema_str}\n```\n"
            f"Return ONLY the JSON, no other text."
        )

        messages = list(messages)  # copy
        # Append to last user message or add new one
        if messages and messages[-1].get("role") == "user":
            content = messages[-1].get("content", "")
            if isinstance(content, str):
                messages[-1] = {**messages[-1], "content": content + instruction}
            else:
                messages.append({"role": "user", "content": instruction})
        else:
            messages.append({"role": "user", "content": instruction})

        return messages

    def _resolve_cache_policy(self, options: RequestOptions | None) -> CachePolicy:
        """Determine effective cache policy for a request."""
        if options and options.cache_policy != CachePolicy.DEFAULT:
            return options.cache_policy
        return CachePolicy.FORCE if self._settings.cache_enabled else CachePolicy.SKIP
