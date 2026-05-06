"""Live integration tests for the Azure-hosted OpenAI clients.

Two flavours are exercised:

- **Chat completions** (``AzureOpenAIClient``, prefix ``azure:``) — the
  legacy/wide-compat path. Adequate for plain prompts on any deployment.
- **Responses API** (``AzureOpenAIResponsesClient``, prefix ``azure-responses:``)
  — the modern path. **Required for tool calling on gpt-5.4+** (per Azure
  docs, chat-completions tool calling with ``reasoning: none`` is unsupported
  on those models).

Each test skips cleanly unless its required env vars are set. Two auth modes
are demonstrated in the Responses test: api-key (works on any endpoint) and
``DefaultAzureCredential`` via ``azure-identity`` (custom-subdomain
endpoint required — see ``providers/_azure_auth.py`` for details).

Run locally::

    # Chat completions test
    KAOS_LLM_AZURE_OPENAI_ENDPOINT=https://eastus2.api.cognitive.microsoft.com/ \\
    KAOS_LLM_AZURE_OPENAI_API_KEY=<key> \\
    KAOS_LLM_AZURE_OPENAI_TEST_DEPLOYMENT=gpt-5.4-mini \\
        uv run pytest tests/integration/test_azure_live.py::test_azure_chat_live -v

    # Responses API + tool calling (api-key)
    KAOS_LLM_AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com/ \\
    KAOS_LLM_AZURE_OPENAI_API_KEY=<key> \\
    KAOS_LLM_AZURE_OPENAI_TEST_DEPLOYMENT=gpt-5.4-mini \\
    KAOS_LLM_AZURE_OPENAI_API_VERSION=2025-04-01-preview \\
        uv run pytest tests/integration/test_azure_live.py::test_azure_responses_live_with_tools -v

    # AAD via DefaultAzureCredential (requires azure-identity, custom-subdomain)
    KAOS_LLM_AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com/ \\
    KAOS_LLM_AZURE_OPENAI_USE_AAD=1 \\
    KAOS_LLM_AZURE_OPENAI_TEST_DEPLOYMENT=gpt-5.4-mini \\
        uv run pytest tests/integration/test_azure_live.py::test_azure_responses_live_aad -v

Never commit keys or tokens — they are read from the environment.
"""

from __future__ import annotations

import os

import pytest

from kaos_llm_client import create_client

# Module-level marker — every test in this file hits a real Azure OpenAI
# resource and is therefore part of the live ``integration`` tier. CI uses
# ``pytest -m "not integration"`` to skip live tests; without this marker
# the runner would still collect (and skip on missing env) these tests on
# the unit-only path, which violates the marker contract documented in
# CLAUDE.md.
pytestmark = pytest.mark.integration

_REQUIRED_BASE = (
    "KAOS_LLM_AZURE_OPENAI_ENDPOINT",
    "KAOS_LLM_AZURE_OPENAI_TEST_DEPLOYMENT",
)


def _skip_unless(*env_vars: str) -> None:
    missing = [v for v in env_vars if not os.environ.get(v)]
    if missing:
        pytest.skip(f"Azure live test skipped — set: {', '.join(missing)}")


# ---------------------------------------------------------------------------
# Chat completions path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_azure_chat_live() -> None:
    """``azure:`` (chat completions) end-to-end with api-key auth."""
    _skip_unless(*_REQUIRED_BASE, "KAOS_LLM_AZURE_OPENAI_API_KEY")
    deployment = os.environ["KAOS_LLM_AZURE_OPENAI_TEST_DEPLOYMENT"]

    client = create_client(f"azure:{deployment}")
    try:
        response = await client.chat_async(
            [{"role": "user", "content": "Reply with the single word READY."}],
            max_tokens=64,
        )
    finally:
        await client.aclose()

    assert response.text is not None
    text = response.text.strip().upper()
    assert "READY" in text, f"Expected READY in Azure response, got: {response.text!r}"
    assert response.model
    assert response.usage is not None
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0


# ---------------------------------------------------------------------------
# Responses API path — required for gpt-5.4+ tool calling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_azure_responses_live() -> None:
    """``azure-responses:`` end-to-end with api-key auth — plain chat."""
    _skip_unless(*_REQUIRED_BASE, "KAOS_LLM_AZURE_OPENAI_API_KEY")
    deployment = os.environ["KAOS_LLM_AZURE_OPENAI_TEST_DEPLOYMENT"]

    client = create_client(f"azure-responses:{deployment}")
    try:
        response = await client.chat_async(
            [{"role": "user", "content": "Reply with the single word READY."}],
            max_tokens=128,
        )
    finally:
        await client.aclose()

    assert response.text is not None
    assert "READY" in response.text.strip().upper(), f"Expected READY, got: {response.text!r}"
    # Responses API IDs start with ``resp_``
    assert response.response_id is not None
    assert response.response_id.startswith("resp_")
    assert response.usage is not None
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0


@pytest.mark.asyncio
async def test_azure_responses_live_with_tools() -> None:
    """The architectural reason this client exists: tool calling on gpt-5.4+.

    Chat-completions ``tools=[...]`` with ``reasoning: none`` is silently
    broken on gpt-5.4+ per Azure docs. The Responses-API path makes it
    work — this test asserts the tool actually gets called.
    """
    _skip_unless(*_REQUIRED_BASE, "KAOS_LLM_AZURE_OPENAI_API_KEY")
    deployment = os.environ["KAOS_LLM_AZURE_OPENAI_TEST_DEPLOYMENT"]

    from kaos_llm_client.types import ToolDefinition

    client = create_client(f"azure-responses:{deployment}")
    try:
        response = await client.chat_async(
            [
                {
                    "role": "user",
                    "content": "What is 17 times 23? Use the multiply tool.",
                }
            ],
            tools=[
                ToolDefinition(
                    name="multiply",
                    description="Multiply two integers.",
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
            max_tokens=512,
        )
    finally:
        await client.aclose()

    tool_parts = [p for p in response.parts if p.type == "tool_use"]
    assert tool_parts, (
        f"Expected at least one tool_use part on gpt-5.4+ Responses API, "
        f"got parts={[p.type for p in response.parts]}; text={response.text!r}"
    )
    tool_call = tool_parts[0].tool_call
    assert tool_call is not None
    assert tool_call.name == "multiply"
    assert tool_call.arguments == {"a": 17, "b": 23}, tool_call.arguments


@pytest.mark.asyncio
async def test_azure_responses_live_aad() -> None:
    """AAD via ``DefaultAzureCredential`` against a custom-subdomain resource.

    Skipped unless ``KAOS_LLM_AZURE_OPENAI_USE_AAD=1`` is set AND
    ``azure-identity`` is installed. Requires the resource principal to have
    the ``Cognitive Services OpenAI User`` role; RBAC propagation can take
    5-15 minutes after assignment.
    """
    _skip_unless(*_REQUIRED_BASE, "KAOS_LLM_AZURE_OPENAI_USE_AAD")
    if os.environ.get("KAOS_LLM_AZURE_OPENAI_USE_AAD") not in ("1", "true", "yes"):
        pytest.skip("Set KAOS_LLM_AZURE_OPENAI_USE_AAD=1 to enable")

    try:
        # `azure-identity` is provided by the optional `[azure]` extra:
        #   uv add 'kaos-llm-client[azure]'
        # The test gracefully skips when the extra isn't installed.
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    except ImportError:
        pytest.skip(
            "azure-identity not installed; install the [azure] extra: "
            "`uv add 'kaos-llm-client[azure]'`"
        )

    deployment = os.environ["KAOS_LLM_AZURE_OPENAI_TEST_DEPLOYMENT"]
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )

    client = create_client(
        f"azure-responses:{deployment}",
        azure_ad_token_provider=token_provider,
    )
    try:
        response = await client.chat_async(
            [{"role": "user", "content": "Reply with the single word READY."}],
            max_tokens=128,
        )
    finally:
        await client.aclose()

    assert response.text is not None
    assert "READY" in response.text.strip().upper()
    assert response.response_id is not None
    assert response.response_id.startswith("resp_")
