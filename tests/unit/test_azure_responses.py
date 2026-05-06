"""Wire-format verification for AzureOpenAIResponsesClient.

The Responses API on Azure differs from chat completions:
- URL: ``/openai/responses?api-version=...`` (NO ``/deployments/{name}/`` segment).
- Body: Responses API shape (``input`` array, ``model`` = deployment name,
  ``max_output_tokens``, etc.).
- Auth: api-key OR Authorization: Bearer (AAD).
"""

from __future__ import annotations

import asyncio
import json

import httpx
from pydantic import SecretStr

from kaos_llm_client import create_client
from kaos_llm_client.providers.azure_openai_responses import (
    AzureOpenAIResponsesClient,
)
from kaos_llm_client.settings import KaosLLMSettings


def _make_settings(
    *,
    endpoint: str | None = "https://test-resource.openai.azure.com/",
    api_key: str | None = "test-key",
    api_version: str = "2025-04-01-preview",
) -> KaosLLMSettings:
    return KaosLLMSettings(
        azure_openai_endpoint=endpoint,
        azure_openai_api_key=SecretStr(api_key) if api_key else None,
        azure_openai_api_version=api_version,
    )


def _fake_responses_payload(text: str = "OK") -> dict:
    return {
        "id": "resp_test",
        "object": "response",
        "model": "gpt-5.4-mini",
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


def test_factory_routes_azure_responses_prefix() -> None:
    client = create_client("azure-responses:gpt-5.4-mini", settings=_make_settings())
    assert isinstance(client, AzureOpenAIResponsesClient)
    assert client.model == "gpt-5.4-mini"


def test_factory_routes_azure_foundry_prefix() -> None:
    client = create_client("azure-foundry:gpt-5.4-mini", settings=_make_settings())
    assert isinstance(client, AzureOpenAIResponsesClient)


def test_default_endpoint_is_responses_not_deployments() -> None:
    """``/responses`` is NOT a deployments-prefixed Azure endpoint."""
    client = AzureOpenAIResponsesClient(model="gpt-5.4-mini", settings=_make_settings())
    endpoint = client._default_endpoint()
    assert endpoint == "/responses?api-version=2025-04-01-preview"
    # Crucial: must NOT include /deployments/ segment
    assert "deployments" not in endpoint


def test_base_url_construction() -> None:
    client = AzureOpenAIResponsesClient(model="gpt-5.4-mini", settings=_make_settings())
    assert client._base_url == "https://test-resource.openai.azure.com/openai"


def test_api_key_auth_header() -> None:
    client = AzureOpenAIResponsesClient(model="gpt-5.4-mini", settings=_make_settings())
    headers = client._build_headers()
    assert headers["api-key"] == "test-key"
    assert "Authorization" not in headers


def test_aad_auth_header_overrides_api_key() -> None:
    client = AzureOpenAIResponsesClient(
        model="gpt-5.4-mini",
        settings=_make_settings(),  # has api-key="test-key"
        azure_ad_token="bearer-token-here",
    )
    headers = client._build_headers()
    assert headers["Authorization"] == "Bearer bearer-token-here"
    assert "api-key" not in headers


def test_request_wire_format() -> None:
    """Capture the actual httpx request and verify URL + body shape."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["query"] = dict(request.url.params)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=_fake_responses_payload("RESPONSES_OK"))

    transport = httpx.MockTransport(handler)
    client = AzureOpenAIResponsesClient(model="gpt-5.4-mini", settings=_make_settings())
    client._async_client = httpx.AsyncClient(base_url=client._base_url, transport=transport)

    async def go() -> str | None:
        r = await client.chat_async(
            [{"role": "user", "content": "hello"}],
            max_tokens=128,
        )
        return r.text

    text = asyncio.run(go())

    # URL: NO /deployments/ in the path; api-version in query.
    assert captured["path"] == "/openai/responses"
    assert "deployments" not in captured["path"]
    assert captured["query"] == {"api-version": "2025-04-01-preview"}

    # Auth: api-key
    headers_lower = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers_lower.get("api-key") == "test-key"
    assert "authorization" not in headers_lower

    # Body: Responses API shape — `input` (not `messages`), `model` = deployment,
    # `max_output_tokens` (not `max_tokens` / `max_completion_tokens`).
    body = captured["body"]
    assert body["model"] == "gpt-5.4-mini"
    assert "input" in body
    assert "messages" not in body
    assert isinstance(body["input"], list)
    assert body["input"][0]["type"] == "message"
    assert body["input"][0]["role"] == "user"
    assert body["max_output_tokens"] == 128

    # Response parsed back to text
    assert text == "RESPONSES_OK"


def test_async_aad_provider_resolved_for_responses() -> None:
    """Async AAD provider works for the Responses path the same way."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json=_fake_responses_payload())

    transport = httpx.MockTransport(handler)

    async def async_provider() -> str:
        return "async-bearer"

    client = AzureOpenAIResponsesClient(
        model="gpt-5.4-mini",
        settings=_make_settings(api_key=None),
        azure_ad_token_provider=async_provider,
    )
    client._async_client = httpx.AsyncClient(base_url=client._base_url, transport=transport)

    async def go() -> None:
        await client.chat_async([{"role": "user", "content": "hi"}], max_tokens=8)

    asyncio.run(go())

    headers_lower = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers_lower["authorization"] == "Bearer async-bearer"
    # Cache cleared after the request
    assert client._resolved_ad_token is None


def test_tool_call_wire_format() -> None:
    """Tool definitions are sent in the Responses-API tool format."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "id": "resp_tool",
                "object": "response",
                "model": "gpt-5.4-mini",
                "status": "completed",
                "output": [
                    {
                        "id": "fc_1",
                        "type": "function_call",
                        "call_id": "call_abc",
                        "name": "multiply",
                        "arguments": '{"a": 17, "b": 23}',
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    transport = httpx.MockTransport(handler)
    client = AzureOpenAIResponsesClient(model="gpt-5.4-mini", settings=_make_settings())
    client._async_client = httpx.AsyncClient(base_url=client._base_url, transport=transport)

    from kaos_llm_client.types import ToolDefinition

    async def go():
        return await client.chat_async(
            [{"role": "user", "content": "17 * 23 ?"}],
            tools=[
                ToolDefinition(
                    name="multiply",
                    description="multiply two ints",
                    parameters={
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                )
            ],
            max_tokens=128,
        )

    response = asyncio.run(go())

    # Body has Responses-API tool format
    body = captured["body"]
    assert "tools" in body
    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["name"] == "multiply"

    # Response parsed back into a tool_use part
    tool_parts = [p for p in response.parts if p.type == "tool_use"]
    assert len(tool_parts) == 1
    assert tool_parts[0].tool_call is not None
    assert tool_parts[0].tool_call.name == "multiply"
    assert tool_parts[0].tool_call.arguments == {"a": 17, "b": 23}


def test_chat_client_and_responses_client_are_distinct() -> None:
    """``azure:`` (chat completions) and ``azure-responses:`` route differently."""
    from kaos_llm_client.providers.azure_openai import AzureOpenAIClient

    chat_client = create_client("azure:gpt-5.4-mini", settings=_make_settings())
    resp_client = create_client("azure-responses:gpt-5.4-mini", settings=_make_settings())

    assert isinstance(chat_client, AzureOpenAIClient)
    assert isinstance(resp_client, AzureOpenAIResponsesClient)

    chat_endpoint = chat_client._default_endpoint()
    resp_endpoint = resp_client._default_endpoint()

    assert "deployments" in chat_endpoint
    assert "chat/completions" in chat_endpoint
    assert "deployments" not in resp_endpoint
    assert resp_endpoint.startswith("/responses")
