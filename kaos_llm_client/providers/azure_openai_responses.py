"""Azure OpenAI / Azure AI Foundry Responses API client.

Required for ``gpt-5.4+`` chat models on Azure. Per Azure docs, tool calling
on the chat-completions endpoint with ``reasoning: none`` is unsupported
starting at GPT-5.4 — meaning a chat-completions ``tools=[...]`` request
will return without a tool call even when one is obviously warranted.
Reasoning models (``o1`` / ``o3`` / ``o4`` / ``gpt-5.5``) get richer
reasoning summaries and stateful conversations on the Responses API as
well.

Wire format (verified live)
---------------------------

- **URL**: ``{endpoint}/openai/responses?api-version={ver}``.

  Critical: there is **no** ``/deployments/{name}/`` segment in this path.
  Per the upstream openai-python SDK
  (``openai.lib.azure._deployments_endpoints``), ``/responses`` is NOT in
  the set of paths that get the deployment segment injected — the deployment
  name goes into the request body's ``model`` field instead. This differs
  from chat completions, embeddings, and audio paths, all of which DO get
  the ``/deployments/{name}/`` segment.

- **Auth**: ``api-key: <KEY>`` or ``Authorization: Bearer <AAD_TOKEN>``.
  AAD requires the resource's custom-subdomain endpoint
  (``https://my-resource.openai.azure.com/``); regional endpoints
  (``https://eastus2.api.cognitive.microsoft.com/``) accept api-key only —
  see ``_azure_auth.py`` module docstring for the full Azure-side gotcha
  list.

- **Body**: standard Responses API JSON — ``input`` (an array of input
  items, NOT ``messages``), ``model`` (the deployment name), ``tools``,
  ``reasoning``, ``previous_response_id``, ``max_output_tokens``, etc. The
  request shape comes from ``OpenAIResponsesClient`` unchanged; only the
  URL/auth differ from the OpenAI direct path.

- **Response**: also Responses API shape — ``output[]`` with items typed as
  ``message`` (text), ``function_call`` (tool use), or ``reasoning``
  (thinking summary). IDs prefixed with ``resp_`` and ``msg_`` /
  ``fc_`` / ``rs_``.

API version
-----------

Default is set by ``KaosLLMSettings.azure_openai_api_version``
(``2024-12-01-preview``). For Responses API it is recommended to bump to
``2025-04-01-preview`` or newer — newer features (e.g. richer tool-call
streaming events) require the newer ``api-version``. Override via
``KAOS_LLM_AZURE_OPENAI_API_VERSION`` or the per-client setting.

Deploying ``gpt-5.4-mini`` to a custom-subdomain resource (for AAD)
-------------------------------------------------------------------

Note: ``gpt-5.4-mini`` (and other current-generation models) only accept
``GlobalStandard`` / ``DataZoneStandard`` SKUs — the older ``Standard`` SKU
returns ``InvalidResourceProperties: SKU 'Standard' of account deployment
is not supported by the model 'gpt-5.4-mini'``::

    az cognitiveservices account deployment create \\
      -n <resource> -g <rg> \\
      --deployment-name gpt-5.4-mini \\
      --model-name gpt-5.4-mini --model-version 2026-03-17 \\
      --model-format OpenAI \\
      --sku-name GlobalStandard --sku-capacity 50

Use ``az cognitiveservices model list --location <region>`` to enumerate
supported SKUs for any model+version pair before deploying.
"""

from __future__ import annotations

from kaos_core.logging import get_logger

from kaos_llm_client.providers._azure_auth import _AzureAuthMixin
from kaos_llm_client.providers.openai_responses import OpenAIResponsesClient

logger = get_logger("kaos_llm_client.providers.azure_openai_responses")


class AzureOpenAIResponsesClient(_AzureAuthMixin, OpenAIResponsesClient):
    """Client for the Azure Responses API (``/openai/responses``).

    The ``model`` argument is the Azure deployment name; it is sent in the
    request body's ``model`` field (not the URL path).

    Use this for any ``gpt-5.4+`` chat model on Azure where tool calling is
    required, and for reasoning models (``o1`` / ``o3`` / ``o4`` /
    ``gpt-5.5``) for the richer Responses-API features.

    Construct via factory::

        client = create_client("azure-responses:gpt-5.4-mini")
        # or, identical:
        client = create_client("azure-foundry:gpt-5.4-mini")

    AAD auth (DefaultAzureCredential)::

        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        client = create_client(
            "azure-responses:gpt-5.4-mini",
            azure_ad_token_provider=get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default",
            ),
        )
    """

    _provider_name: str = "azure-openai-responses"

    # --- URL routing ---
    #
    # /responses is NOT a deployments-prefixed endpoint (matches openai-python
    # SDK behaviour). The deployment name goes into the JSON body, not the
    # URL path.

    def _default_endpoint(self) -> str:
        return f"/responses?api-version={self._api_version}"
