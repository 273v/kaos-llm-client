"""Wire-format verification for the AWS Bedrock OpenAI-compatible client.

Bedrock speaks the same Responses-API JSON shape as ``api.openai.com`` —
only base URL and bearer token differ. These tests verify settings
threading, URL construction, auth header, and that the response parser
inherits unchanged from ``OpenAIResponsesClient``.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr

from kaos_llm_client import create_client
from kaos_llm_client.errors import KaosLLMAuthError
from kaos_llm_client.providers.bedrock import BedrockClient
from kaos_llm_client.settings import KaosLLMSettings


def _make_settings(
    *,
    base_url: str = "https://bedrock-mantle.us-east-2.api.aws",
    api_key: str | None = "bedrock-api-key-test",
) -> KaosLLMSettings:
    return KaosLLMSettings(
        bedrock_base_url=base_url,
        bedrock_api_key=SecretStr(api_key) if api_key else None,
    )


@pytest.fixture(autouse=True)
def _clear_credential_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bedrock settings honour both ``KAOS_LLM_BEDROCK_API_KEY`` and the
    legacy ``AWS_BEARER_TOKEN_BEDROCK`` env var. Tests that construct
    ``KaosLLMSettings()`` without explicit kwargs rely on neither being
    set in the dev shell — clear both for every test in this file so
    results are deterministic regardless of ``~/.bashrc`` contents.
    """
    for var in ("KAOS_LLM_BEDROCK_API_KEY", "AWS_BEARER_TOKEN_BEDROCK"):
        monkeypatch.delenv(var, raising=False)


def _fake_responses_payload(text: str = "OK") -> dict:
    return {
        "id": "resp_bedrock_test",
        "object": "response",
        "model": "openai.gpt-oss-120b",
        "status": "completed",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 5,
            "output_tokens": 2,
            "total_tokens": 7,
        },
    }


def test_factory_routes_bedrock_prefix() -> None:
    client = create_client("bedrock:openai.gpt-oss-120b", settings=_make_settings())
    assert isinstance(client, BedrockClient)
    assert client.model == "openai.gpt-oss-120b"


def test_base_url_from_settings() -> None:
    client = BedrockClient(model="openai.gpt-oss-120b", settings=_make_settings())
    assert client._base_url == "https://bedrock-mantle.us-east-2.api.aws"


def test_default_endpoint_is_responses_with_v1() -> None:
    """Endpoint is ``/v1/responses`` so the bare host base URL joins correctly."""
    client = BedrockClient(model="openai.gpt-oss-120b", settings=_make_settings())
    assert client._default_endpoint() == "/v1/responses"


def test_auth_uses_bearer_header() -> None:
    """Bedrock uses the standard ``Authorization: Bearer`` header (NOT ``api-key``)."""
    client = BedrockClient(model="openai.gpt-oss-120b", settings=_make_settings())
    headers = client._build_headers()
    assert headers["Authorization"] == "Bearer bedrock-api-key-test"
    assert "api-key" not in headers


def test_missing_api_key_raises_with_actionable_fix() -> None:
    client = BedrockClient(model="openai.gpt-oss-120b", settings=_make_settings(api_key=None))
    with pytest.raises(KaosLLMAuthError) as exc:
        client._build_headers()
    msg = str(exc.value)
    assert "KAOS_LLM_BEDROCK_API_KEY" in msg
    assert "AWS_BEARER_TOKEN_BEDROCK" in msg


def test_legacy_aws_bearer_token_env_fallback(monkeypatch) -> None:
    """``AWS_BEARER_TOKEN_BEDROCK`` env var feeds ``bedrock_api_key``."""
    monkeypatch.delenv("KAOS_LLM_BEDROCK_API_KEY", raising=False)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key-from-env")
    settings = KaosLLMSettings()
    assert settings.bedrock_api_key is not None
    assert settings.bedrock_api_key.get_secret_value() == "bedrock-api-key-from-env"


def test_kaos_prefix_takes_priority_over_legacy(monkeypatch) -> None:
    monkeypatch.setenv("KAOS_LLM_BEDROCK_API_KEY", "primary")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "legacy")
    settings = KaosLLMSettings()
    assert settings.bedrock_api_key is not None
    assert settings.bedrock_api_key.get_secret_value() == "primary"


def test_request_wire_format() -> None:
    """Capture the actual httpx request and verify URL + headers + body shape."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["path"] = request.url.path
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=_fake_responses_payload("BEDROCK_OK"))

    transport = httpx.MockTransport(handler)
    client = BedrockClient(model="openai.gpt-oss-120b", settings=_make_settings())
    client._async_client = httpx.AsyncClient(base_url=client._base_url, transport=transport)

    async def go() -> str | None:
        r = await client.chat_async([{"role": "user", "content": "hi"}], max_tokens=64)
        return r.text

    text = asyncio.run(go())

    # URL: bare-host base + ``/v1/responses`` (no double v1, no deployments
    # segment, no api-version query).
    assert captured["path"] == "/v1/responses"
    assert "/v1/v1" not in captured["url"]
    assert "deployments" not in captured["path"]

    # Auth: standard OpenAI Bearer header (NOT api-key).
    headers_lower = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers_lower["authorization"] == "Bearer bedrock-api-key-test"
    assert "api-key" not in headers_lower

    # Body: Responses API shape; model is the Bedrock model id.
    body = captured["body"]
    assert body["model"] == "openai.gpt-oss-120b"
    assert "input" in body
    assert body["max_output_tokens"] == 64

    # Response parsed via inherited Responses-API parser.
    assert text == "BEDROCK_OK"


def test_region_override_via_base_url() -> None:
    """Different region → different base URL via setting override."""
    client = BedrockClient(
        model="openai.gpt-oss-120b",
        settings=_make_settings(base_url="https://bedrock-mantle.us-west-2.api.aws"),
    )
    assert client._base_url == "https://bedrock-mantle.us-west-2.api.aws"


def test_factory_passes_explicit_base_url_through() -> None:
    """``create_client(..., base_url=...)`` overrides the settings default."""
    client = create_client(
        "bedrock:openai.gpt-oss-120b",
        settings=_make_settings(),
        base_url="https://bedrock-mantle.eu-west-1.api.aws",
    )
    assert isinstance(client, BedrockClient)
    assert client._base_url == "https://bedrock-mantle.eu-west-1.api.aws"
