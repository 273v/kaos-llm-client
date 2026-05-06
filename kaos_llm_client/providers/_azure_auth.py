"""Shared Azure-auth and URL helpers for the Azure-hosted OpenAI clients.

Used by both ``AzureOpenAIClient`` (chat completions) and
``AzureOpenAIResponsesClient`` (Responses API). Provides:

- AAD token resolution (static token, sync/async provider) with caching.
- Azure-specific ``_get_default_base_url`` / ``_get_api_key_from_settings`` /
  ``_build_headers`` overrides.
- Async entry-point wrappers (``request_async``, ``request_stream_async``)
  that pre-resolve AAD tokens before the sync ``_build_headers`` runs.

Designed as a cooperative mixin: ``class AzureXxx(_AzureAuthMixin, BaseXxx)``.
The mixin's overrides shadow the base provider's via MRO.

Azure-side gotchas this code accommodates
-----------------------------------------

1. **AAD requires a custom-subdomain endpoint.** Regional endpoints like
   ``https://eastus2.api.cognitive.microsoft.com/`` reject Bearer tokens with
   ``400 BadRequest: "Please provide a custom subdomain for token
   authentication"`` regardless of the API path. Use the resource's custom
   subdomain (e.g. ``https://my-resource.openai.azure.com/`` or
   ``https://my-resource.cognitiveservices.azure.com/``) for AAD. api-key
   auth works on either form.

2. **Two header schemes.** Azure uses ``api-key: <KEY>`` for resource keys,
   NOT ``Authorization: Bearer <KEY>``. AAD tokens go in
   ``Authorization: Bearer <TOKEN>``. The OpenAI public API uses
   ``Authorization: Bearer <KEY>`` for the api-key — a copy-paste of the
   OpenAI auth code into an Azure context will silently fail.

3. **AAD RBAC takes time to propagate.** After ``az role assignment create``,
   data-plane RBAC (``Microsoft.CognitiveServices/accounts/OpenAI/...``) often
   takes 5-15 minutes (occasionally longer) to take effect. Errors during
   that window alternate between ``lacks the required data action`` and
   ``Principal does not have access to API/Operation``. Wait, then retry —
   not a code bug.

4. **The required role is "Cognitive Services OpenAI User".** "Contributor"
   on the management plane is *not* sufficient for inference calls.

5. **Async token providers must be awaited.** ``azure.identity.aio`` returns
   coroutines from ``get_bearer_token_provider``. Sync ``_build_headers``
   can't await; this mixin's async entry points pre-resolve the token into
   ``self._resolved_ad_token`` before the sync header construction runs.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

from kaos_llm_client.errors import KaosLLMAuthError, KaosLLMError

if TYPE_CHECKING:
    from kaos_llm_client.settings import KaosLLMSettings
    from kaos_llm_client.types import (
        ProviderResponse,
        RequestOptions,
        StreamChunk,
        ToolChoice,
        ToolDefinition,
    )

logger = get_logger("kaos_llm_client.providers._azure_auth")

# Type alias: AAD token provider may be sync (``get_bearer_token_provider``
# from ``azure.identity``) or async (``azure.identity.aio`` credentials wrapped
# via ``get_bearer_token_provider``).
AzureADTokenProvider = Callable[[], "str | Awaitable[str]"]


class _AzureAuthMixin:
    """Mixin providing AAD + api-key auth and ``{endpoint}/openai`` URL.

    Subclass must define ``_default_endpoint()`` (and any API-specific
    request building); this mixin handles auth and base URL only.
    """

    _provider_name: str = "azure"

    # --- Attributes provided by the eventual base provider via cooperative MRO.
    # Declared here as class-level annotations so the type checker can resolve
    # uses inside the mixin without creating runtime descriptors.
    if TYPE_CHECKING:
        model: str
        _settings: KaosLLMSettings

        def _resolve_api_key(self) -> str: ...

        async def request_async(self, *args: Any, **kwargs: Any) -> ProviderResponse: ...

        def request_stream_async(
            self,
            messages: list[dict[str, Any]],
            *,
            tools: list[ToolDefinition] | None = None,
            tool_choice: ToolChoice | None = None,
            options: RequestOptions | None = None,
            **kwargs: Any,
        ) -> AsyncIterator[StreamChunk]: ...

    def __init__(
        self,
        *,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AzureADTokenProvider | None = None,
        **kwargs: Any,
    ) -> None:
        """Construct an Azure-flavoured client.

        Auth precedence (highest to lowest):

        1. ``azure_ad_token_provider`` — callable returning a fresh Bearer
           token. Sync or async. Called once per request.
        2. ``azure_ad_token`` — static Bearer token (constructor or
           ``KAOS_LLM_AZURE_OPENAI_AD_TOKEN`` / ``AZURE_OPENAI_AD_TOKEN`` env).
        3. ``api_key`` — static subscription key (``api-key`` header).
        """
        super().__init__(**kwargs)
        self._explicit_ad_token = azure_ad_token
        self._ad_token_provider = azure_ad_token_provider
        # Transient cache populated by async entry points before they call
        # the sync ``_build_headers``. Cleared after each request.
        self._resolved_ad_token: str | None = None

    # --- Azure deployment + api-version (read from settings) ---

    @property
    def _deployment(self) -> str:
        return self.model

    @property
    def _api_version(self) -> str:
        return self._settings.azure_openai_api_version

    # --- AAD token resolution ---

    def _has_aad_auth(self) -> bool:
        if self._explicit_ad_token or self._ad_token_provider is not None:
            return True
        return self._settings.azure_openai_ad_token is not None

    def _resolve_ad_token_sync(self) -> str | None:
        """Resolve a Bearer token synchronously, or ``None`` if unavailable.

        Precedence: cached → provider (sync) → constructor static token →
        settings env token. Returns ``None`` when the only configured source
        is an async provider — the async entry points pre-populate
        ``_resolved_ad_token`` in that case.
        """
        if self._resolved_ad_token:
            return self._resolved_ad_token

        provider = self._ad_token_provider
        if provider is not None and not inspect.iscoroutinefunction(provider):
            try:
                result = provider()
            except Exception as exc:
                raise KaosLLMAuthError(
                    f"azure_ad_token_provider raised: {exc}",
                    provider=self._provider_name,
                    fix="Check the credential configuration "
                    "(e.g. `az login`, managed identity, env vars).",
                ) from exc
            if inspect.isawaitable(result):
                # Async result from a sync-typed callable — defer to the
                # async resolver. Don't await synchronously here.
                return None
            if not isinstance(result, str) or not result:
                raise KaosLLMAuthError(
                    "azure_ad_token_provider must return a non-empty string.",
                    provider=self._provider_name,
                    fix="Use ``azure.identity.get_bearer_token_provider(...)`` "
                    "or an equivalent callable returning a bearer token.",
                )
            return result

        if self._explicit_ad_token:
            return self._explicit_ad_token

        settings_token = self._settings.azure_openai_ad_token
        if settings_token is not None:
            secret = settings_token.get_secret_value()
            if secret:
                return secret

        return None

    async def _resolve_ad_token_async(self) -> str | None:
        provider = self._ad_token_provider
        if provider is not None:
            try:
                result = provider()
                if inspect.isawaitable(result):
                    result = await result
            except Exception as exc:
                raise KaosLLMAuthError(
                    f"azure_ad_token_provider raised: {exc}",
                    provider=self._provider_name,
                    fix="Check the credential configuration "
                    "(e.g. `az login`, managed identity, env vars).",
                ) from exc
            if not isinstance(result, str) or not result:
                raise KaosLLMAuthError(
                    "azure_ad_token_provider must return a non-empty string.",
                    provider=self._provider_name,
                )
            return result

        return self._resolve_ad_token_sync()

    # --- Base URL + auth + headers (override base provider) ---

    def _get_default_base_url(self) -> str:
        endpoint = self._settings.azure_openai_endpoint
        if not endpoint:
            raise KaosLLMError(
                "Azure OpenAI endpoint is not configured.",
                provider=self._provider_name,
                fix="Set KAOS_LLM_AZURE_OPENAI_ENDPOINT (or AZURE_OPENAI_ENDPOINT) to "
                "your resource endpoint, e.g. "
                "'https://my-resource.openai.azure.com/' or "
                "'https://my-region.api.cognitive.microsoft.com/'.",
            )
        return endpoint.rstrip("/") + "/openai"

    def _get_api_key_from_settings(self) -> str:
        key = self._settings.azure_openai_api_key
        if key is None:
            raise KaosLLMAuthError(
                "Azure OpenAI API key is not configured.",
                provider=self._provider_name,
                fix="Set KAOS_LLM_AZURE_OPENAI_API_KEY (or AZURE_OPENAI_API_KEY), "
                "supply azure_ad_token / azure_ad_token_provider, or pass "
                "api_key= to the client constructor.",
            )
        secret = key.get_secret_value()
        if not secret:
            raise KaosLLMAuthError(
                "Azure OpenAI API key is empty.",
                provider=self._provider_name,
                fix="Set KAOS_LLM_AZURE_OPENAI_API_KEY to a valid resource key.",
            )
        return secret

    def _build_headers(self) -> dict[str, str]:
        ad_token = self._resolve_ad_token_sync()
        if ad_token:
            return {
                "Authorization": f"Bearer {ad_token}",
                "Content-Type": "application/json",
            }
        api_key = self._resolve_api_key()
        return {
            "api-key": api_key,
            "Content-Type": "application/json",
        }

    # --- Async entry-point wrappers (handle async AAD providers) ---

    async def request_async(self, *args: Any, **kwargs: Any) -> Any:
        if self._has_aad_auth():
            self._resolved_ad_token = await self._resolve_ad_token_async()
        try:
            return await super().request_async(*args, **kwargs)  # ty: ignore[unresolved-attribute]
        finally:
            self._resolved_ad_token = None

    async def request_stream_async(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: RequestOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        if self._has_aad_auth():
            self._resolved_ad_token = await self._resolve_ad_token_async()
        try:
            async for chunk in super().request_stream_async(  # ty: ignore[unresolved-attribute]
                messages,
                tools=tools,
                tool_choice=tool_choice,
                options=options,
                **kwargs,
            ):
                yield chunk
        finally:
            self._resolved_ad_token = None
