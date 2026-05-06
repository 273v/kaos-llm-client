"""Tests for the base-URL security validator on ``KaosLLMSettings``.

Rationale
---------

Provider base URLs ride sensitive payloads (full prompts + bearer
auth headers). An attacker who can flip the corresponding env var
gets an SSRF-adjacent escalation path. The settings layer therefore
rejects:

- non-``https://`` base URLs
- URLs whose hostname is a private / loopback / link-local IP literal
  (RFC 1918, RFC 4193, RFC 3927, RFC 6890)
- URLs whose hostname is one of the well-known local hostnames
  (``localhost`` and friends)

…unless ``allow_insecure_base_url=True`` is passed (or
``KAOS_LLM_ALLOW_INSECURE_BASE_URL=1`` is set), which is the documented
escape hatch for local-dev model servers.

These tests pin all of that behaviour, including the public defaults,
the Bedrock default, and Azure custom-subdomain endpoints — none of
which should ever trip the validator.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kaos_llm_client.settings import KaosLLMSettings

# ---------------------------------------------------------------------------
# Public defaults — must always validate cleanly.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip any base-URL / allow-insecure env vars from the dev shell.

    Without this, a developer who has ``KAOS_LLM_OPENAI_BASE_URL`` set
    in their ``.env`` would see surprising failures here.
    """
    for name in (
        "KAOS_LLM_ALLOW_INSECURE_BASE_URL",
        "KAOS_LLM_OPENAI_BASE_URL",
        "KAOS_LLM_ANTHROPIC_BASE_URL",
        "KAOS_LLM_GOOGLE_BASE_URL",
        "KAOS_LLM_XAI_BASE_URL",
        "KAOS_LLM_GROQ_BASE_URL",
        "KAOS_LLM_MISTRAL_BASE_URL",
        "KAOS_LLM_OPENROUTER_BASE_URL",
        "KAOS_LLM_AZURE_OPENAI_ENDPOINT",
        "KAOS_LLM_BEDROCK_BASE_URL",
        "AZURE_OPENAI_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)


class TestPublicDefaultsValidate:
    """Every public-default URL must pass the validator unchanged."""

    def test_openai_default(self) -> None:
        s = KaosLLMSettings()
        assert s.openai_base_url == "https://api.openai.com"

    def test_anthropic_default(self) -> None:
        s = KaosLLMSettings()
        assert s.anthropic_base_url == "https://api.anthropic.com"

    def test_google_default(self) -> None:
        s = KaosLLMSettings()
        assert s.google_base_url == "https://generativelanguage.googleapis.com"

    def test_xai_default(self) -> None:
        s = KaosLLMSettings()
        assert s.xai_base_url == "https://api.x.ai"

    def test_groq_default(self) -> None:
        s = KaosLLMSettings()
        assert s.groq_base_url == "https://api.groq.com/openai"

    def test_mistral_default(self) -> None:
        s = KaosLLMSettings()
        assert s.mistral_base_url == "https://api.mistral.ai"

    def test_openrouter_default(self) -> None:
        s = KaosLLMSettings()
        assert s.openrouter_base_url == "https://openrouter.ai/api"

    def test_bedrock_default(self) -> None:
        s = KaosLLMSettings()
        assert s.bedrock_base_url == "https://bedrock-mantle.us-east-2.api.aws"

    def test_azure_endpoint_none_is_allowed(self) -> None:
        """``azure_openai_endpoint`` is ``None`` by default — must not trip."""
        s = KaosLLMSettings(azure_openai_endpoint=None)
        assert s.azure_openai_endpoint is None

    def test_azure_custom_subdomain_validates(self) -> None:
        """The custom-subdomain Azure endpoint pattern is HTTPS public DNS."""
        s = KaosLLMSettings(azure_openai_endpoint="https://test-us-east2-273-vio.openai.azure.com/")
        assert s.azure_openai_endpoint == "https://test-us-east2-273-vio.openai.azure.com/"

    def test_azure_regional_endpoint_validates(self) -> None:
        s = KaosLLMSettings(azure_openai_endpoint="https://eastus2.api.cognitive.microsoft.com/")
        assert s.azure_openai_endpoint == "https://eastus2.api.cognitive.microsoft.com/"


# ---------------------------------------------------------------------------
# HTTP scheme rejection.
# ---------------------------------------------------------------------------


class TestSchemeRejection:
    def test_http_rejected_with_clear_message(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            KaosLLMSettings(openai_base_url="http://api.openai.com")
        msg = str(excinfo.value)
        assert "openai_base_url" in msg
        assert "http://api.openai.com" in msg
        assert "https" in msg.lower()
        assert "KAOS_LLM_ALLOW_INSECURE_BASE_URL" in msg

    def test_http_rejected_for_anthropic(self) -> None:
        with pytest.raises(ValidationError):
            KaosLLMSettings(anthropic_base_url="http://api.anthropic.com")

    def test_ftp_rejected(self) -> None:
        with pytest.raises(ValidationError):
            KaosLLMSettings(openai_base_url="ftp://example.com")


# ---------------------------------------------------------------------------
# Local / private address rejection (even on HTTPS).
# ---------------------------------------------------------------------------


class TestPrivateAddressRejection:
    def test_loopback_https_rejected(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            KaosLLMSettings(openai_base_url="https://127.0.0.1:8000")
        assert "loopback" in str(excinfo.value).lower()

    def test_private_ip_https_rejected(self) -> None:
        """Private IP rejected EVEN ON HTTPS — this is the SSRF rule."""
        with pytest.raises(ValidationError) as excinfo:
            KaosLLMSettings(openai_base_url="https://192.168.1.5")
        assert "private" in str(excinfo.value).lower()

    def test_link_local_rejected(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            KaosLLMSettings(openai_base_url="https://169.254.169.254")
        assert "link-local" in str(excinfo.value).lower()

    def test_localhost_hostname_rejected(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            KaosLLMSettings(openai_base_url="https://localhost:8000")
        assert "localhost" in str(excinfo.value).lower()

    def test_ipv6_loopback_rejected(self) -> None:
        with pytest.raises(ValidationError):
            KaosLLMSettings(openai_base_url="https://[::1]:8000")

    def test_ipv6_private_rejected(self) -> None:
        # fc00::/7 is RFC 4193 ULA = private.
        with pytest.raises(ValidationError):
            KaosLLMSettings(openai_base_url="https://[fc00::1]")


# ---------------------------------------------------------------------------
# allow_insecure_base_url escape hatch.
# ---------------------------------------------------------------------------


class TestAllowInsecureBaseUrl:
    def test_localhost_http_with_flag(self) -> None:
        s = KaosLLMSettings(
            openai_base_url="http://localhost:11434",
            allow_insecure_base_url=True,
        )
        assert s.openai_base_url == "http://localhost:11434"

    def test_private_ip_http_with_flag(self) -> None:
        s = KaosLLMSettings(
            openai_base_url="http://192.168.1.5",
            allow_insecure_base_url=True,
        )
        assert s.openai_base_url == "http://192.168.1.5"

    def test_private_ip_https_with_flag(self) -> None:
        s = KaosLLMSettings(
            openai_base_url="https://192.168.1.5",
            allow_insecure_base_url=True,
        )
        assert s.openai_base_url == "https://192.168.1.5"

    def test_loopback_with_flag(self) -> None:
        s = KaosLLMSettings(
            openai_base_url="http://127.0.0.1:8000",
            allow_insecure_base_url=True,
        )
        assert s.openai_base_url == "http://127.0.0.1:8000"

    def test_env_var_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``KAOS_LLM_ALLOW_INSECURE_BASE_URL=1`` + insecure base URL works."""
        monkeypatch.setenv("KAOS_LLM_ALLOW_INSECURE_BASE_URL", "1")
        monkeypatch.setenv("KAOS_LLM_OPENAI_BASE_URL", "http://localhost:8000")
        s = KaosLLMSettings()
        assert s.allow_insecure_base_url is True
        assert s.openai_base_url == "http://localhost:8000"

    def test_env_var_only_does_not_bypass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting the insecure URL via env without the flag still fails."""
        monkeypatch.setenv("KAOS_LLM_OPENAI_BASE_URL", "http://localhost:8000")
        with pytest.raises(ValidationError):
            KaosLLMSettings()


# ---------------------------------------------------------------------------
# Field-by-field coverage of the protected list.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "openai_base_url",
        "anthropic_base_url",
        "google_base_url",
        "xai_base_url",
        "groq_base_url",
        "mistral_base_url",
        "openrouter_base_url",
        "azure_openai_endpoint",
        "bedrock_base_url",
    ],
)
def test_each_protected_field_rejects_http(field: str) -> None:
    """Every URL in ``_BASE_URL_FIELDS`` is policed."""
    # We construct via ``**kwargs`` so a single test parametrizes over
    # every protected field. ``ty`` can't see through ``**dict[str, str]``
    # and complains about each field's individual type — the values are
    # all ``str | None`` URLs at runtime, so this is safe.
    kwargs: dict[str, str] = {field: "http://attacker.example.com"}
    with pytest.raises(ValidationError):
        KaosLLMSettings(**kwargs)  # ty: ignore[invalid-argument-type]
