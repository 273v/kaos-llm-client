"""Wire-format verification for AzureOpenAIClient.

Uses ``httpx.MockTransport`` to capture the actual request and assert:
- URL path includes ``/openai/deployments/{deployment}/chat/completions``
- ``api-version`` query parameter is present
- ``api-key`` auth header is set (NOT ``Authorization: Bearer``)
- Body shape matches OpenAI chat-completions format
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from kaos_llm_client import create_client
from kaos_llm_client.errors import KaosLLMAuthError, KaosLLMError
from kaos_llm_client.providers.azure_openai import AzureOpenAIClient
from kaos_llm_client.settings import KaosLLMSettings


def _make_settings(
    *,
    endpoint: str | None = "https://eastus2.api.cognitive.microsoft.com/",
    api_key: str | None = "test-key",
    api_version: str = "2024-12-01-preview",
) -> KaosLLMSettings:
    return KaosLLMSettings(
        azure_openai_endpoint=endpoint,
        azure_openai_api_key=SecretStr(api_key) if api_key else None,
        azure_openai_api_version=api_version,
        # Disable flex tier to make assertions deterministic
        default_service_tier=None,
    )


def _fake_chat_response(model: str = "gpt-5.4-mini-2026-03-17") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "READY"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }


def test_factory_routes_azure_prefix() -> None:
    settings = _make_settings()
    client = create_client("azure:gpt-5.4-mini", settings=settings)
    assert isinstance(client, AzureOpenAIClient)
    assert client.model == "gpt-5.4-mini"
    # Azure-default profile uses ``max_completion_tokens`` (Azure rejects
    # ``max_tokens`` on current chat models) and disables flex tier.
    from kaos_llm_client.profiles import AZURE_OPENAI_DEFAULT

    assert client.profile is AZURE_OPENAI_DEFAULT
    assert client.profile.max_tokens_field == "max_completion_tokens"


def test_factory_routes_azure_openai_prefix() -> None:
    settings = _make_settings()
    client = create_client("azure-openai:gpt-5.4-mini", settings=settings)
    assert isinstance(client, AzureOpenAIClient)


def test_base_url_constructs_from_endpoint() -> None:
    client = AzureOpenAIClient(model="gpt-5.4-mini", settings=_make_settings())
    # base_url is the resource endpoint + /openai
    assert client._base_url == "https://eastus2.api.cognitive.microsoft.com/openai"


def test_base_url_strips_trailing_slash() -> None:
    settings = _make_settings(endpoint="https://r.example.com//")
    client = AzureOpenAIClient(model="gpt-5.4-mini", settings=settings)
    assert client._base_url == "https://r.example.com/openai"


def test_default_endpoint_uses_deployment_and_api_version() -> None:
    client = AzureOpenAIClient(model="gpt-5.4-mini", settings=_make_settings())
    endpoint = client._default_endpoint()
    assert endpoint == ("/deployments/gpt-5.4-mini/chat/completions?api-version=2024-12-01-preview")


def test_auth_header_uses_api_key_not_bearer() -> None:
    client = AzureOpenAIClient(model="gpt-5.4-mini", settings=_make_settings())
    headers = client._build_headers()
    assert headers["api-key"] == "test-key"
    assert "Authorization" not in headers
    assert headers["Content-Type"] == "application/json"


def test_missing_endpoint_raises_with_fix() -> None:
    settings = _make_settings(endpoint=None)
    client = AzureOpenAIClient(model="gpt-5.4-mini", settings=settings)
    with pytest.raises(KaosLLMError) as exc:
        _ = client._base_url
    msg = str(exc.value)
    assert "endpoint" in msg.lower()
    assert "AZURE_OPENAI_ENDPOINT" in msg


def test_missing_api_key_raises_with_fix() -> None:
    settings = _make_settings(api_key=None)
    client = AzureOpenAIClient(model="gpt-5.4-mini", settings=settings)
    with pytest.raises(KaosLLMAuthError) as exc:
        client._build_headers()
    msg = str(exc.value)
    assert "AZURE_OPENAI_API_KEY" in msg


def test_chat_request_wire_format() -> None:
    """Capture the actual httpx request and verify URL + headers + body."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["path"] = request.url.path
        captured["query"] = dict(request.url.params)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=_fake_chat_response())

    transport = httpx.MockTransport(handler)
    client = AzureOpenAIClient(model="gpt-5.4-mini", settings=_make_settings())
    # Inject a mock-transport async client; the provider uses _async_client cache
    client._async_client = httpx.AsyncClient(base_url=client._base_url, transport=transport)

    import asyncio

    async def go() -> None:
        await client.chat_async([{"role": "user", "content": "hello"}], max_tokens=16)

    asyncio.run(go())

    # URL: full reconstruction
    assert captured["path"] == "/openai/deployments/gpt-5.4-mini/chat/completions"
    assert captured["query"] == {"api-version": "2024-12-01-preview"}

    # Auth: api-key, not Bearer
    headers_lower = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers_lower.get("api-key") == "test-key"
    assert "authorization" not in headers_lower

    # Body: standard OpenAI chat shape; model = deployment
    body = captured["body"]
    assert body["model"] == "gpt-5.4-mini"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    # Azure profile maps max_tokens kwarg to ``max_completion_tokens``
    # (Azure rejects the legacy field on current chat models).
    assert body["max_completion_tokens"] == 16
    assert "max_tokens" not in body
    # ``service_tier`` must be absent — Azure does not honour flex.
    assert "service_tier" not in body


def test_chat_response_parses_normally() -> None:
    """Azure responses parse through the standard OpenAI parser."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_fake_chat_response())

    transport = httpx.MockTransport(handler)
    client = AzureOpenAIClient(model="gpt-5.4-mini", settings=_make_settings())
    client._async_client = httpx.AsyncClient(base_url=client._base_url, transport=transport)

    import asyncio

    async def go() -> str | None:
        r = await client.chat_async([{"role": "user", "content": "hello"}], max_tokens=16)
        return r.text

    text = asyncio.run(go())
    assert text == "READY"


def test_embeddings_endpoint_uses_deployment_path() -> None:
    """Embeddings hit ``/deployments/{name}/embeddings`` (not ``/v1/embeddings``)."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["query"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={
                "data": [{"embedding": [0.1, 0.2]}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    transport = httpx.MockTransport(handler)
    client = AzureOpenAIClient(model="text-embedding-3-small", settings=_make_settings())
    client._async_client = httpx.AsyncClient(base_url=client._base_url, transport=transport)

    import asyncio

    async def go() -> None:
        await client.embed_async("hello")

    asyncio.run(go())

    assert captured["path"] == "/openai/deployments/text-embedding-3-small/embeddings"
    assert captured["query"] == {"api-version": "2024-12-01-preview"}


def test_legacy_env_fallback_picks_up_AZURE_OPENAI_ENDPOINT(monkeypatch) -> None:
    """``AZURE_OPENAI_ENDPOINT`` env var feeds ``azure_openai_endpoint``."""
    monkeypatch.delenv("KAOS_LLM_AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("KAOS_LLM_AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://legacy.example.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "legacy-key")
    settings = KaosLLMSettings()
    assert settings.azure_openai_endpoint == "https://legacy.example.com/"
    assert settings.azure_openai_api_key is not None
    assert settings.azure_openai_api_key.get_secret_value() == "legacy-key"


def test_kaos_llm_prefix_takes_priority_over_legacy(monkeypatch) -> None:
    monkeypatch.setenv("KAOS_LLM_AZURE_OPENAI_ENDPOINT", "https://primary.example.com/")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://legacy.example.com/")
    settings = KaosLLMSettings()
    assert settings.azure_openai_endpoint == "https://primary.example.com/"


def test_api_version_legacy_fallback(monkeypatch) -> None:
    monkeypatch.delenv("KAOS_LLM_AZURE_OPENAI_API_VERSION", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    monkeypatch.setenv("OPENAI_API_VERSION", "2025-04-01-preview")
    settings = KaosLLMSettings()
    assert settings.azure_openai_api_version == "2025-04-01-preview"


# ---------------------------------------------------------------------------
# AAD / azure_ad_token / token_provider flows
# ---------------------------------------------------------------------------


def _capture_handler() -> tuple[dict, httpx.MockTransport]:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json=_fake_chat_response())

    return captured, httpx.MockTransport(handler)


def test_static_aad_token_via_constructor() -> None:
    """``azure_ad_token=...`` produces ``Authorization: Bearer <TOKEN>``."""
    captured, transport = _capture_handler()
    client = AzureOpenAIClient(
        model="gpt-5.4-mini",
        settings=_make_settings(api_key=None),
        azure_ad_token="aad-token-static",
    )
    client._async_client = httpx.AsyncClient(base_url=client._base_url, transport=transport)

    import asyncio

    asyncio.run(client.chat_async([{"role": "user", "content": "hi"}], max_tokens=8))

    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers["authorization"] == "Bearer aad-token-static"
    assert "api-key" not in headers


def test_aad_token_via_settings_env(monkeypatch) -> None:
    """``KAOS_LLM_AZURE_OPENAI_AD_TOKEN`` env populates the setting."""
    monkeypatch.delenv("KAOS_LLM_AZURE_OPENAI_AD_TOKEN", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_AD_TOKEN", raising=False)
    monkeypatch.setenv("KAOS_LLM_AZURE_OPENAI_AD_TOKEN", "env-aad-token")
    settings = KaosLLMSettings(
        azure_openai_endpoint="https://r.example.com/",
    )
    assert settings.azure_openai_ad_token is not None
    assert settings.azure_openai_ad_token.get_secret_value() == "env-aad-token"

    client = AzureOpenAIClient(model="gpt-5.4-mini", settings=settings)
    headers = client._build_headers()
    assert headers["Authorization"] == "Bearer env-aad-token"


def test_aad_token_legacy_env_fallback(monkeypatch) -> None:
    monkeypatch.delenv("KAOS_LLM_AZURE_OPENAI_AD_TOKEN", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", "legacy-aad-token")
    settings = KaosLLMSettings()
    assert settings.azure_openai_ad_token is not None
    assert settings.azure_openai_ad_token.get_secret_value() == "legacy-aad-token"


def test_provider_token_takes_priority_over_static() -> None:
    """A registered provider supersedes both api_key and the static AD token."""
    captured, transport = _capture_handler()

    calls = {"count": 0}

    def provider() -> str:
        calls["count"] += 1
        return f"provider-token-{calls['count']}"

    client = AzureOpenAIClient(
        model="gpt-5.4-mini",
        settings=_make_settings(),  # has api_key="test-key"
        azure_ad_token="ignored-static-token",
        azure_ad_token_provider=provider,
    )
    client._async_client = httpx.AsyncClient(base_url=client._base_url, transport=transport)

    import asyncio

    asyncio.run(client.chat_async([{"role": "user", "content": "hi"}], max_tokens=8))

    headers = {k.lower(): v for k, v in captured["headers"].items()}
    # Provider was called and its token was used, not the static one or api-key
    assert calls["count"] == 1
    assert headers["authorization"] == "Bearer provider-token-1"
    assert "api-key" not in headers


def test_async_provider_resolved_per_request() -> None:
    """Async ``azure_ad_token_provider`` (e.g. azure.identity.aio) is awaited."""
    captured, transport = _capture_handler()

    async def async_provider() -> str:
        return "async-aad-token"

    client = AzureOpenAIClient(
        model="gpt-5.4-mini",
        settings=_make_settings(api_key=None),
        azure_ad_token_provider=async_provider,
    )
    client._async_client = httpx.AsyncClient(base_url=client._base_url, transport=transport)

    import asyncio

    asyncio.run(client.chat_async([{"role": "user", "content": "hi"}], max_tokens=8))

    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers["authorization"] == "Bearer async-aad-token"


def test_token_cache_cleared_between_requests() -> None:
    """Each request resolves a fresh token (no stale cache leak)."""
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(dict(request.headers))
        return httpx.Response(200, json=_fake_chat_response())

    transport = httpx.MockTransport(handler)

    counter = {"n": 0}

    def provider() -> str:
        counter["n"] += 1
        return f"token-{counter['n']}"

    client = AzureOpenAIClient(
        model="gpt-5.4-mini",
        settings=_make_settings(api_key=None),
        azure_ad_token_provider=provider,
    )
    client._async_client = httpx.AsyncClient(base_url=client._base_url, transport=transport)

    import asyncio

    async def go() -> None:
        await client.chat_async([{"role": "user", "content": "hi"}], max_tokens=8)
        await client.chat_async([{"role": "user", "content": "hi"}], max_tokens=8)

    asyncio.run(go())

    headers0 = {k.lower(): v for k, v in captured[0].items()}
    headers1 = {k.lower(): v for k, v in captured[1].items()}
    assert headers0["authorization"] == "Bearer token-1"
    assert headers1["authorization"] == "Bearer token-2"
    # Cache cleared after each call
    assert client._resolved_ad_token is None


def test_provider_returning_empty_string_raises() -> None:
    def bad_provider() -> str:
        return ""

    client = AzureOpenAIClient(
        model="gpt-5.4-mini",
        settings=_make_settings(api_key=None),
        azure_ad_token_provider=bad_provider,
    )
    with pytest.raises(KaosLLMAuthError) as exc:
        client._build_headers()
    assert "non-empty" in str(exc.value)


def test_provider_raising_wraps_in_auth_error() -> None:
    class CredentialError(RuntimeError):
        pass

    def failing_provider() -> str:
        raise CredentialError("az login required")

    client = AzureOpenAIClient(
        model="gpt-5.4-mini",
        settings=_make_settings(api_key=None),
        azure_ad_token_provider=failing_provider,
    )
    with pytest.raises(KaosLLMAuthError) as exc:
        client._build_headers()
    msg = str(exc.value)
    assert "az login required" in msg


def test_falls_back_to_api_key_when_no_aad_configured() -> None:
    """No AD token / provider → use ``api-key`` header."""
    captured, transport = _capture_handler()
    client = AzureOpenAIClient(model="gpt-5.4-mini", settings=_make_settings())
    client._async_client = httpx.AsyncClient(base_url=client._base_url, transport=transport)

    import asyncio

    asyncio.run(client.chat_async([{"role": "user", "content": "hi"}], max_tokens=8))

    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers.get("api-key") == "test-key"
    assert "authorization" not in headers


def test_factory_passes_aad_kwargs_through() -> None:
    """``create_client(..., azure_ad_token=...)`` reaches the constructor."""
    settings = _make_settings(api_key=None)
    client = create_client(
        "azure:gpt-5.4-mini",
        settings=settings,
        azure_ad_token="kw-token",
    )
    assert isinstance(client, AzureOpenAIClient)
    headers = client._build_headers()
    assert headers["Authorization"] == "Bearer kw-token"
