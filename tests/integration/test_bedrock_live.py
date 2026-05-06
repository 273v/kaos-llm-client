"""Live integration test for the AWS Bedrock client.

Skipped unless ``KAOS_LLM_BEDROCK_API_KEY`` (or legacy
``AWS_BEARER_TOKEN_BEDROCK``) is set in the environment. Optional
``KAOS_LLM_BEDROCK_TEST_MODEL`` overrides the default model id
(``openai.gpt-oss-120b``).

Run locally with the bashrc-managed token::

    bash -ic 'KAOS_LLM_BEDROCK_API_KEY="$AWS_BEARER_TOKEN_BEDROCK" \\
              uv run pytest tests/integration/test_bedrock_live.py -v'

Bedrock bearer tokens are SigV4-presigned and short-lived (~12 h);
rotate via your AWS auth flow when expired. Never commit them.
"""

from __future__ import annotations

import os

import pytest

from kaos_llm_client import create_client

# Module-level marker — every test in this file hits the real AWS Bedrock
# Responses API and is therefore part of the live ``integration`` tier.
# Without this, ``pytest -m "not integration"`` (CI's unit-only selector)
# still collects these tests, violating the marker contract in CLAUDE.md.
pytestmark = pytest.mark.integration


def _resolve_api_key() -> str | None:
    """Read either KAOS_LLM_BEDROCK_API_KEY or the legacy AWS env var."""
    return os.environ.get("KAOS_LLM_BEDROCK_API_KEY") or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")


def _skip_if_unconfigured() -> None:
    if not _resolve_api_key():
        pytest.skip(
            "Bedrock live test skipped — set KAOS_LLM_BEDROCK_API_KEY "
            "(or AWS_BEARER_TOKEN_BEDROCK) to a valid Bedrock bearer token."
        )


@pytest.mark.asyncio
async def test_bedrock_responses_live() -> None:
    _skip_if_unconfigured()

    model = os.environ.get("KAOS_LLM_BEDROCK_TEST_MODEL", "openai.gpt-oss-120b")

    # Settings pick up the legacy ``AWS_BEARER_TOKEN_BEDROCK`` automatically;
    # no explicit kwargs needed if env vars are set.
    client = create_client(f"bedrock:{model}")
    try:
        response = await client.chat_async(
            [{"role": "user", "content": "Reply with the single word READY."}],
            max_tokens=512,
        )
    finally:
        await client.aclose()

    assert response.text is not None
    assert "READY" in response.text.strip().upper(), (
        f"Expected READY in Bedrock response, got: {response.text!r}"
    )
    # Bedrock returns the model id we passed in
    assert response.model
    # Response IDs are ``resp_...`` for the Responses API
    assert response.response_id is not None
    assert response.response_id.startswith("resp_")
    # Usage should be populated
    assert response.usage is not None
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
