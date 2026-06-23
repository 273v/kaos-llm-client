"""Azure OpenAI / Azure AI Foundry chat-completions client.

Azure-hosted OpenAI deployments speak the same chat-completions JSON wire
format as ``api.openai.com`` but differ in three places:

- **URL path**: ``{endpoint}/openai/deployments/{deployment}/chat/completions``
  (the deployment name appears in the path; the body still includes ``model``
  set to the deployment name).
- **Auth header**: ``api-key: <KEY>`` (not ``Authorization: Bearer <KEY>``).
  Azure AD tokens use ``Authorization: Bearer <TOKEN>`` instead.
- **Required query param**: ``?api-version=YYYY-MM-DD-preview``.

Reference: ``openai.lib.azure.AzureOpenAI`` in the upstream openai-python SDK
(``openai/lib/azure.py`` — see ``_deployments_endpoints`` set for the list of
paths that get the ``/deployments/{name}/`` segment injected).

When NOT to use this client
---------------------------

For ``gpt-5.4+`` chat models, prefer ``AzureOpenAIResponsesClient`` — per
Azure docs, **tool calling on chat-completions with ``reasoning: none`` is
unsupported starting at GPT-5.4**. A no-tools chat with this client will
return 200 OK on gpt-5.4-mini, but as soon as you pass tools the request
will silently misbehave (tool definitions ignored, or refused at the
provider level). The Responses API path handles tool calling correctly on
those models without forcing reasoning to be enabled. Reasoning models
(``o1`` / ``o3`` / ``o4`` / ``gpt-5.5``) likewise get richer features (stream
reasoning summaries, ``previous_response_id`` continuations) on Responses.

Wire quirks specific to chat completions on Azure
-------------------------------------------------

- **``max_completion_tokens`` is mandatory** for current chat models.
  ``max_tokens`` (the legacy field still accepted by ``api.openai.com``)
  returns 400 ``Unsupported parameter: 'max_tokens' is not supported with
  this model. Use 'max_completion_tokens' instead.`` ``AZURE_OPENAI_DEFAULT``
  in ``profiles.py`` enforces the rename via ``max_tokens_field``.
- **No ``service_tier`` (flex)** — Azure does not honour OpenAI's flex
  pricing tier; the profile sets ``supports_service_tier=False`` so the
  field is not sent.
- **``content_filter_results`` extras** appear on each ``choices[i]`` —
  parsed-but-ignored by the OpenAI-compatible parser; no special handling.
"""

from __future__ import annotations

from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_client.providers._azure_auth import (
    AzureADTokenProvider,  # noqa: F401  (re-exported for back-compat)
    _AzureAuthMixin,
)
from kaos_llm_client.providers.openai import OpenAIClient
from kaos_llm_client.types import (
    EmbeddingResponse,
    ProviderRequest,
    RequestOptions,
    UsageInfo,
)

logger = get_logger("kaos_llm_client.providers.azure_openai")


class AzureOpenAIClient(_AzureAuthMixin, OpenAIClient):
    """Client for Azure OpenAI / Azure AI Foundry chat-completions deployments.

    The ``model`` argument is interpreted as the **Azure deployment name**.
    For typical setups where a deployment is named after its underlying
    OpenAI model (e.g., a ``gpt-5.4-mini`` deployment), passing the model
    name works unchanged: ``create_client("azure:gpt-5.4-mini")``.

    For deployments with custom names, pass the deployment name:
    ``create_client("azure:my-prod-gpt5")``.

    Azure-specific configuration (via ``KaosLLMSettings`` or env vars):

    - ``azure_openai_endpoint`` / ``KAOS_LLM_AZURE_OPENAI_ENDPOINT`` /
      legacy ``AZURE_OPENAI_ENDPOINT``
    - ``azure_openai_api_key`` / ``KAOS_LLM_AZURE_OPENAI_API_KEY`` /
      legacy ``AZURE_OPENAI_API_KEY``
    - ``azure_openai_ad_token`` / ``KAOS_LLM_AZURE_OPENAI_AD_TOKEN`` /
      legacy ``AZURE_OPENAI_AD_TOKEN`` — static AAD bearer
    - ``azure_openai_api_version`` / ``KAOS_LLM_AZURE_OPENAI_API_VERSION`` /
      legacy ``AZURE_OPENAI_API_VERSION`` / ``OPENAI_API_VERSION`` — defaults
      to ``2024-12-01-preview``.

    Constructor kwargs ``azure_ad_token`` and ``azure_ad_token_provider``
    accept Azure-Identity tokens. See ``_AzureAuthMixin`` for details.
    """

    _provider_name: str = "azure-openai"

    # --- URL routing ---

    def _default_endpoint(self) -> str:
        return f"/deployments/{self._deployment}/chat/completions?api-version={self._api_version}"

    def _embeddings_endpoint(self) -> str:
        return f"/deployments/{self._deployment}/embeddings?api-version={self._api_version}"

    # --- Embeddings (Azure path differs from /v1/embeddings) ---

    async def embed_async(
        self,
        input: str | list[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        options: RequestOptions | None = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        from kaos_llm_client.cost import estimate_call_cost
        from kaos_llm_client.transport import execute_with_retry

        # Pre-resolve AAD token (handles async providers); release after.
        owns_token_cache = False
        if self._has_aad_auth() and self._resolved_ad_token is None:
            self._resolved_ad_token = await self._resolve_ad_token_async()
            owns_token_cache = True

        try:
            if isinstance(input, str):
                input = [input]

            deployment = model or self._deployment
            body: dict[str, Any] = {
                "model": deployment,
                "input": input,
            }
            if dimensions is not None:
                body["dimensions"] = dimensions
            body.update(kwargs)

            endpoint = f"/deployments/{deployment}/embeddings?api-version={self._api_version}"
            request = ProviderRequest(
                provider=self._provider_name,
                model=deployment,
                endpoint=endpoint,
                body=body,
            )
            request.headers.update(self._build_headers())

            timeout = options.timeout if options and options.timeout else None
            client = self._get_async_client()
            response = await execute_with_retry(
                client,
                request,
                retry_policy=self._retry_policy,
                provider=self._provider_name,
                timeout=timeout,
                log_extra=self._log_extra(request=request),
            )

            raw = response.json()

            embeddings: list[list[float]] = []
            for item in raw.get("data", []):
                embeddings.append(item.get("embedding", []))

            usage_raw = raw.get("usage", {})
            usage = UsageInfo(
                input_tokens=usage_raw.get("prompt_tokens", 0),
                total_tokens=usage_raw.get("total_tokens", 0),
            )

            # Per-call completion log for Azure embeddings — same shape
            # as the OpenAI-compat embed log.
            try:
                estimated_usd = estimate_call_cost(usage, deployment)
            except Exception:
                estimated_usd = None
            try:
                logger.debug(
                    "LLM call complete",
                    extra=self._log_extra(
                        request=request,
                        provider=self._provider_name,
                        model=deployment,
                        request_id=(request.request_id or "")[:16] or None,
                        input_tokens=usage.input_tokens,
                        output_tokens=0,
                        total_tokens=usage.total_tokens,
                        cache_hit=False,
                        embedding_count=len(embeddings),
                        estimated_usd=estimated_usd,
                    ),
                )
            except Exception:  # pragma: no cover - log-side defensive guard
                logger.debug("LLM embed call-complete log emission failed", exc_info=True)

            return EmbeddingResponse(
                provider=self._provider_name,
                model=raw.get("model", deployment),
                embeddings=embeddings,
                usage=usage,
                raw=raw,
                request_id=request.request_id,
            )
        finally:
            if owns_token_cache:
                self._resolved_ad_token = None
