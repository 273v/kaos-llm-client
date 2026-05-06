"""Typed settings for kaos-llm-client.

Resolution priority:
1. Explicit overrides via ``from_context()``
2. ``KaosContext._config`` entries
3. Environment variables (``KAOS_LLM_`` prefix)
4. ``.env`` file
5. Field defaults

Security note (base-URL validation)
-----------------------------------

Provider base URLs (``openai_base_url``, ``anthropic_base_url`` …,
``azure_openai_endpoint``, ``bedrock_base_url``) are sent prompts plus
authentication headers. An attacker who controls the corresponding env
vars can therefore redirect those requests anywhere — an SSRF-adjacent
escalation path on misconfigured machines.

To defend against that, ``KaosLLMSettings`` runs a post-validation pass
that rejects any non-``https://`` URL and any URL whose hostname is a
private / loopback / link-local IP literal (RFC 1918, RFC 4193, RFC
3927, RFC 6890) or a well-known local hostname (``localhost``).

Local-development escape hatch: set
``KAOS_LLM_ALLOW_INSECURE_BASE_URL=1`` (or pass
``allow_insecure_base_url=True``) when intentionally pointing at
``http://localhost:11434`` for Ollama, ``http://127.0.0.1:8000`` for
vLLM, etc. The default is ``False`` — opt-in only.
"""

from __future__ import annotations

import ipaddress
import os
from typing import Any
from urllib.parse import urlparse

from kaos_core.config import ModuleSettings
from pydantic import SecretStr, model_validator
from pydantic_settings import SettingsConfigDict

# Settings field names that hold a provider base-URL or endpoint and
# therefore receive https-only / private-address validation. Kept as a
# tuple constant so the validator and any future tooling share the same
# authoritative list.
_BASE_URL_FIELDS: tuple[str, ...] = (
    "openai_base_url",
    "anthropic_base_url",
    "google_base_url",
    "xai_base_url",
    "groq_base_url",
    "mistral_base_url",
    "openrouter_base_url",
    "azure_openai_endpoint",
    "bedrock_base_url",
)

# Hostnames that are unambiguously local even though they are not raw
# IP literals. ``ipaddress.ip_address`` would raise ``ValueError`` on
# these — we have to reject them by name explicitly.
_LOCAL_HOSTNAMES: frozenset[str] = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
    }
)


class KaosLLMSettings(ModuleSettings):
    """Configuration for kaos-llm-client.

    Env vars use the ``KAOS_LLM_`` prefix:
    - ``KAOS_LLM_OPENAI_API_KEY``, ``KAOS_LLM_ANTHROPIC_API_KEY``, etc.
    - ``KAOS_LLM_OPENAI_BASE_URL`` — override for proxies/local models
    - ``KAOS_LLM_DEFAULT_TIMEOUT`` — request timeout in seconds

    Legacy/standard env vars (backward compatible):
    - ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``GOOGLE_API_KEY``,
      ``GOOGLE_GENERATIVE_AI_API_KEY``, ``XAI_API_KEY``

    This follows the same ``mode="before"`` fallback pattern used by
    ``KaosWebSettings`` and ``KaosSourceGovInfoSettings``.
    """

    # Provider API keys
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    xai_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    mistral_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None

    # Provider endpoints (overridable for proxies, local models)
    openai_base_url: str = "https://api.openai.com"
    anthropic_base_url: str = "https://api.anthropic.com"
    google_base_url: str = "https://generativelanguage.googleapis.com"
    xai_base_url: str = "https://api.x.ai"
    groq_base_url: str = "https://api.groq.com/openai"
    mistral_base_url: str = "https://api.mistral.ai"
    openrouter_base_url: str = "https://openrouter.ai/api"
    openrouter_site_url: str | None = None

    allow_insecure_base_url: bool = False
    """Opt-in escape hatch for the base-URL security validator.

    By default (``False``) every provider base-URL — ``openai_base_url``,
    ``anthropic_base_url``, ``azure_openai_endpoint``, ``bedrock_base_url``,
    etc. — must be ``https://`` and must NOT resolve to a private /
    loopback / link-local address. That's an SSRF-style defence: prompts
    and bearer headers are too sensitive to be shipped over plaintext or
    redirected to ``192.168.0.0/16`` by a hostile env var.

    Set to ``True`` (env: ``KAOS_LLM_ALLOW_INSECURE_BASE_URL=1``) for
    legitimate local-development workflows — Ollama at
    ``http://localhost:11434``, vLLM at ``http://127.0.0.1:8000``,
    LiteLLM behind a non-TLS dev proxy, etc. Production deployments
    should always leave this ``False``.

    See the module docstring for the full security note.

    Env: ``KAOS_LLM_ALLOW_INSECURE_BASE_URL``."""

    # ----- Azure OpenAI / Azure AI Foundry -----
    #
    # Used by both ``AzureOpenAIClient`` (chat completions, ``azure:``) and
    # ``AzureOpenAIResponsesClient`` (Responses API, ``azure-responses:`` /
    # ``azure-foundry:``). See ``providers/_azure_auth.py`` for the
    # auth-flow rules and Azure-side gotchas.
    #
    # Env: ``KAOS_LLM_AZURE_OPENAI_*`` (primary), with legacy
    # ``AZURE_OPENAI_*`` / ``OPENAI_API_VERSION`` fallbacks (see
    # ``_legacy_env_fallback`` below).

    azure_openai_api_key: SecretStr | None = None
    """Resource subscription key. Sent as the ``api-key`` header. Works on
    both regional (``*.api.cognitive.microsoft.com``) and custom-subdomain
    (``*.openai.azure.com``) endpoints. Mutually exclusive with AAD when
    both are set, AAD wins (see ``_AzureAuthMixin``)."""

    azure_openai_ad_token: SecretStr | None = None
    """Static AAD bearer token. Sent as ``Authorization: Bearer <token>``.
    Only works on **custom-subdomain endpoints** — Azure rejects AAD on
    regional endpoints with ``400 BadRequest: Please provide a custom
    subdomain for token authentication``. For dynamic tokens with refresh,
    pass ``azure_ad_token_provider=...`` to the client constructor instead
    (e.g. ``azure.identity.get_bearer_token_provider(...)``)."""

    azure_openai_endpoint: str | None = None
    """Resource endpoint URL. Two valid forms:

    - Custom-subdomain (required for AAD):
      ``https://my-resource.openai.azure.com/`` or
      ``https://my-resource.cognitiveservices.azure.com/``
    - Regional (api-key only): ``https://eastus2.api.cognitive.microsoft.com/``

    Trailing slash is stripped; the client appends ``/openai`` to form the
    base URL."""

    azure_openai_api_version: str = "2024-12-01-preview"
    """Azure OpenAI API version (sent as ``?api-version=...`` query param on
    every request). Default ``2024-12-01-preview`` works for chat
    completions on current models. For the Responses API path, bumping to
    ``2025-04-01-preview`` (or newer) is recommended — newer features
    (richer tool-call streaming, certain reasoning settings) require the
    newer api-version."""

    # ----- AWS Bedrock (OpenAI-compatible Responses API) -----
    #
    # Bedrock exposes an OpenAI-compatible Responses surface at
    # ``bedrock-mantle.<region>.api.aws``. Auth is a single SigV4-presigned
    # bearer token (issued via ``aws bedrock create-bearer-token`` or scripts
    # that wrap that). Sent as ``Authorization: Bearer <token>`` — same
    # header scheme as direct OpenAI.
    #
    # The base URL must NOT include ``/v1`` — our Responses path is
    # ``/v1/responses``, so concatenation gives the right result only when
    # the base URL is bare-host. The OpenAI Python SDK example shows
    # ``OPENAI_BASE_URL=.../v1`` because that SDK uses ``/responses``
    # (without ``/v1``) as the relative path.

    bedrock_api_key: SecretStr | None = None
    """Bedrock bearer token. Sent as ``Authorization: Bearer <token>``.
    Tokens are SigV4-presigned and short-lived (typically 12 hours); rotate
    via your AWS auth flow before expiry. Legacy env: ``AWS_BEARER_TOKEN_BEDROCK``."""

    bedrock_base_url: str = "https://bedrock-mantle.us-east-2.api.aws"
    """Bedrock Responses-API endpoint. Defaults to ``us-east-2``. Override
    for other regions (e.g. ``https://bedrock-mantle.us-west-2.api.aws``).
    Do NOT include ``/v1`` — the client's endpoint path already has it."""

    # Google Vertex AI (used when google_base_url contains aiplatform.googleapis.com)
    google_project: str | None = None
    google_location: str = "us-central1"

    # Transport defaults
    default_timeout: float = 120.0
    default_max_retries: int = 3
    retry_backoff_base: float = 1.0

    max_response_bytes: int = 32 * 1024 * 1024
    """Hard cap on the size of a non-streaming response body, in bytes.

    The transport layer rejects any response whose ``Content-Length``
    exceeds this value (or whose buffered body exceeds it when no
    Content-Length header is present). 32 MiB comfortably accommodates
    today's largest legitimate provider responses (e.g. base64-encoded
    images and embeddings batches) while preventing a hostile or
    misbehaving endpoint from exhausting client memory.

    Streaming responses are bounded by ``stream_max_duration`` instead.

    Env: ``KAOS_LLM_MAX_RESPONSE_BYTES``."""

    stream_max_duration: float = 600.0
    """Wall-clock cap on a single SSE / streaming response, in seconds.

    Once the stream has been open this long, the transport raises
    ``KaosLLMTransportError`` even if data is still arriving. Defaults to
    10 minutes, which is well above the longest realistic generation
    (Anthropic 200K-token completions, OpenAI ``o1`` deep-reasoning
    runs) but below the kind of duration that would indicate a stuck
    socket or a server that forgot to send ``[DONE]``.

    Env: ``KAOS_LLM_STREAM_MAX_DURATION``."""

    trust_env: bool = False
    """Whether httpx should honour ambient ``HTTP_PROXY`` / ``HTTPS_PROXY``
    / ``NO_PROXY`` / ``SSL_CERT_FILE`` env vars when issuing requests.

    Defaults to ``False`` so prompts and bearer headers are not routed
    through ambient proxies by surprise. Set to ``True`` (env:
    ``KAOS_LLM_TRUST_ENV=1``) when an enterprise deployment intentionally
    relies on corporate proxy settings. Documented in the module docstring
    of ``transport.py``.

    Env: ``KAOS_LLM_TRUST_ENV``."""

    # Cost management
    default_service_tier: str | None = "flex"
    """Default service tier for OpenAI requests. ``"flex"`` saves ~50% on
    supported models. If the server returns a 500 with flex, the transport
    layer automatically retries once without ``service_tier`` (graceful
    fallback — see ``transport.py:execute_with_retry``).

    Flex is in beta and has known reliability issues on some models.
    The fallback ensures requests succeed even when flex is unstable,
    at the cost of one extra round-trip on failure.

    References:
      - https://community.openai.com/t/critical-bug-using-flex-model-for-gpt5-since-last-monday-500-internal-server-error/1364470
      - https://community.openai.com/t/flex-service-tier-500-error/1362451
      - https://developers.openai.com/api/docs/guides/flex-processing

    Env: ``KAOS_LLM_DEFAULT_SERVICE_TIER``."""

    # Cache
    cache_enabled: bool = False
    cache_path: str | None = None  # default: ~/.cache/kaos/llm

    model_config = SettingsConfigDict(
        env_prefix="KAOS_LLM_",
        env_file=".env",
        extra="ignore",
    )

    @model_validator(mode="before")
    @classmethod
    def _legacy_env_fallback(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Support standard env var names for backward compatibility.

        Same pattern as ``KaosWebSettings._legacy_env_fallback`` and
        ``KaosSourceGovInfoSettings._legacy_env_fallback``.
        """
        _LEGACY_MAP: dict[str, list[str]] = {
            "openai_api_key": ["OPENAI_API_KEY"],
            "anthropic_api_key": ["ANTHROPIC_API_KEY"],
            "google_api_key": ["GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"],
            "xai_api_key": ["XAI_API_KEY"],
            "groq_api_key": ["GROQ_API_KEY"],
            "mistral_api_key": ["MISTRAL_API_KEY"],
            "openrouter_api_key": ["OPENROUTER_API_KEY"],
            "azure_openai_api_key": ["AZURE_OPENAI_API_KEY"],
            "azure_openai_ad_token": ["AZURE_OPENAI_AD_TOKEN"],
            "azure_openai_endpoint": ["AZURE_OPENAI_ENDPOINT"],
            "azure_openai_api_version": ["AZURE_OPENAI_API_VERSION", "OPENAI_API_VERSION"],
            "bedrock_api_key": ["AWS_BEARER_TOKEN_BEDROCK"],
        }
        for field_name, env_vars in _LEGACY_MAP.items():
            if not values.get(field_name):
                for env_var in env_vars:
                    legacy = os.environ.get(env_var)
                    if legacy:
                        values[field_name] = legacy
                        break
        return values

    @model_validator(mode="after")
    def _validate_base_urls(self) -> KaosLLMSettings:
        """Reject insecure or private-address base URLs unless opted-in.

        Runs in ``mode="after"`` so values are already coerced to their
        target type (``str`` / ``str | None``). Each entry of
        :data:`_BASE_URL_FIELDS` is validated against three rules:

        1. The URL must use the ``https`` scheme.
        2. The hostname must not be a private / loopback / link-local
           IP literal (``ipaddress.ip_address(hostname)`` →
           ``is_private`` / ``is_loopback`` / ``is_link_local``).
        3. The hostname must not be one of the well-known local
           hostnames (``localhost`` and friends — see
           :data:`_LOCAL_HOSTNAMES`).

        All three checks are bypassed when
        ``self.allow_insecure_base_url`` is ``True`` — that is the
        documented escape hatch for local model servers.

        Raises ``ValueError`` (which pydantic wraps into a
        ``ValidationError``) with a message that includes:

        - the offending field name
        - the offending URL
        - the env var to flip if the user genuinely wants this URL
        """
        if self.allow_insecure_base_url:
            return self

        for field_name in _BASE_URL_FIELDS:
            value = getattr(self, field_name, None)
            if value is None or value == "":
                continue
            self._check_base_url(field_name, value)
        return self

    @staticmethod
    def _check_base_url(field_name: str, url: str) -> None:
        """Raise ``ValueError`` if *url* fails the security rules.

        Helper extracted so the validator stays readable. See
        :meth:`_validate_base_urls` for the rule list.
        """
        parsed = urlparse(url)
        scheme = (parsed.scheme or "").lower()
        hostname = (parsed.hostname or "").lower()
        env_hint = (
            "Set KAOS_LLM_ALLOW_INSECURE_BASE_URL=1 (or pass "
            "allow_insecure_base_url=True) ONLY for trusted local-dev "
            "endpoints (Ollama, vLLM)."
        )

        if scheme != "https":
            raise ValueError(
                f"{field_name}={url!r} is not https. Provider base URLs "
                "carry prompts plus auth tokens and must use TLS. "
                f"{env_hint}"
            )

        if not hostname:
            raise ValueError(
                f"{field_name}={url!r} has no hostname. Provide a full "
                f"https://host[:port] URL. {env_hint}"
            )

        if hostname in _LOCAL_HOSTNAMES:
            raise ValueError(
                f"{field_name}={url!r} points at a local hostname "
                f"({hostname!r}). Local addresses are blocked to prevent "
                f"prompts/tokens from being redirected. {env_hint}"
            )

        try:
            ip = ipaddress.ip_address(hostname)
        except ValueError:
            # Not an IP literal — assume a public DNS name and accept.
            return

        if ip.is_private or ip.is_loopback or ip.is_link_local:
            kind = (
                "loopback" if ip.is_loopback else ("link-local" if ip.is_link_local else "private")
            )
            raise ValueError(
                f"{field_name}={url!r} resolves to a {kind} IP "
                f"({ip.compressed}). Local/private addresses are blocked "
                f"to prevent prompts/tokens from being redirected. "
                f"{env_hint}"
            )
