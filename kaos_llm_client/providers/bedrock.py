"""AWS Bedrock OpenAI-compatible Responses API client.

Bedrock exposes an OpenAI-compatible Responses surface at
``bedrock-mantle.<region>.api.aws/v1/responses``. The wire format matches
``api.openai.com/v1/responses`` (Responses API JSON shape:
``input``/``output``/``model``); the only differences are the host and the
auth-bearer format.

Auth
----

Bearer token issued via AWS — typically via ``aws bedrock
create-bearer-token`` or vendor scripts. The token format is
``bedrock-api-key-<base64-encoded-presigned-URL>`` and embeds a SigV4
signature with a typical 12-hour TTL. Sent as
``Authorization: Bearer <token>`` (standard OpenAI header form, NOT the
Azure ``api-key`` header).

Settings precedence (highest → lowest)
--------------------------------------

1. Constructor ``api_key=`` / ``base_url=``.
2. ``KaosContext._config`` per-request overrides.
3. ``KAOS_LLM_BEDROCK_API_KEY`` / ``KAOS_LLM_BEDROCK_BASE_URL`` env.
4. Legacy ``AWS_BEARER_TOKEN_BEDROCK`` env.
5. Field defaults (``us-east-2`` base URL).

Why not just use ``openai-responses:`` with overrides?
------------------------------------------------------

You can — but the user has to remember the URL gotcha (drop ``/v1``)
and feed Bedrock-specific env vars into OpenAI-named slots. A dedicated
``bedrock:`` prefix routes deployment-name → model body field cleanly,
reads the right env var, and keeps the OpenAI direct-API settings
uncontaminated.
"""

from __future__ import annotations

from kaos_core.logging import get_logger

from kaos_llm_client.errors import KaosLLMAuthError
from kaos_llm_client.providers.openai_responses import OpenAIResponsesClient

logger = get_logger("kaos_llm_client.providers.bedrock")


class BedrockClient(OpenAIResponsesClient):
    """Client for AWS Bedrock's OpenAI-compatible Responses API.

    The ``model`` argument is the Bedrock model identifier — typically a
    namespaced form like ``openai.gpt-oss-120b`` or ``anthropic.claude-...``.
    It is sent in the request body's ``model`` field; there are no path
    deployments segments.

    Construct via factory::

        client = create_client("bedrock:openai.gpt-oss-120b")

    Or with explicit credentials::

        client = create_client(
            "bedrock:openai.gpt-oss-120b",
            api_key="bedrock-api-key-...",
            base_url="https://bedrock-mantle.us-east-2.api.aws",
        )

    Override region by setting ``KAOS_LLM_BEDROCK_BASE_URL`` to the
    region-specific endpoint (e.g.
    ``https://bedrock-mantle.us-west-2.api.aws``).
    """

    _provider_name: str = "bedrock"

    # --- Settings overrides ---

    def _get_default_base_url(self) -> str:
        return self._settings.bedrock_base_url

    def _get_api_key_from_settings(self) -> str:
        key = self._settings.bedrock_api_key
        if key is None:
            raise KaosLLMAuthError(
                "Bedrock bearer token is not configured.",
                provider=self._provider_name,
                fix="Set KAOS_LLM_BEDROCK_API_KEY (or AWS_BEARER_TOKEN_BEDROCK) "
                "to a Bedrock bearer token, or pass api_key= to the client. "
                "Tokens are typically generated via the AWS console or "
                "`aws bedrock create-bearer-token`.",
            )
        secret = key.get_secret_value()
        if not secret:
            raise KaosLLMAuthError(
                "Bedrock bearer token is empty.",
                provider=self._provider_name,
                fix="Set KAOS_LLM_BEDROCK_API_KEY to a non-empty bearer token.",
            )
        return secret
