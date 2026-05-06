"""Verify sensitive response headers are redacted before reaching consumers.

Provider responses echo their headers onto ``ProviderResponse.response_headers``
so hooks, instrumentation, cassettes, and logs can see transport metadata.
Some headers carry credentials (``Set-Cookie``, ``Authorization``,
``Proxy-Authorization``, ``WWW-Authenticate``) and must NOT be exposed
unredacted — leaking those into a captured cassette or a log line would be
a real-world incident.

These tests assert:

- ``redact_response_headers`` masks the documented sensitive headers
  case-insensitively while preserving everything else verbatim.
- The full ``BaseProviderClient.request_async`` round-trip applies the
  redaction by the time ``ProviderResponse.response_headers`` is observed.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from kaos_llm_client.providers.openai import OpenAIClient
from kaos_llm_client.transport import (
    _REDACTED_RESPONSE_HEADERS,
    redact_response_headers,
)

OPENAI_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-redact-1",
    "object": "chat.completion",
    "model": "gpt-5",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}


SENSITIVE_RESPONSE_HEADERS: dict[str, str] = {
    "Set-Cookie": "session=foo; Path=/; HttpOnly",
    "Cookie": "tracker=abc",
    "Authorization": "Bearer leaked-secret",
    "Proxy-Authorization": "Basic leaked",
    "WWW-Authenticate": 'Bearer realm="api"',
    "Proxy-Authenticate": 'Basic realm="proxy"',
    "X-Csrf-Token": "csrf-leak",
    "X-Xsrf-Token": "xsrf-leak",
}


class TestRedactResponseHeadersHelper:
    def test_redacts_documented_set(self) -> None:
        redacted = redact_response_headers(SENSITIVE_RESPONSE_HEADERS)
        for key in SENSITIVE_RESPONSE_HEADERS:
            assert redacted[key] == "<redacted>", (
                f"expected {key} to be redacted; got {redacted[key]!r}"
            )

    def test_preserves_non_sensitive(self) -> None:
        headers = {"X-Request-Id": "abc-123", "Content-Type": "application/json"}
        redacted = redact_response_headers(headers)
        assert redacted == headers

    def test_case_insensitive(self) -> None:
        headers = {
            "set-cookie": "x=y",
            "AUTHORIZATION": "Bearer leak",
            "x-CSRF-token": "tok",
        }
        redacted = redact_response_headers(headers)
        assert redacted["set-cookie"] == "<redacted>"
        assert redacted["AUTHORIZATION"] == "<redacted>"
        assert redacted["x-CSRF-token"] == "<redacted>"

    def test_accepts_httpx_headers_mapping(self) -> None:
        """``httpx.Headers`` is mapping-like; redaction must work on it directly."""
        h = httpx.Headers(
            [
                ("Set-Cookie", "a=b"),
                ("X-Other", "ok"),
            ]
        )
        redacted = redact_response_headers(h)
        # httpx normalises to lowercase keys
        assert any(v == "<redacted>" for v in redacted.values())
        assert any(v == "ok" for v in redacted.values())

    def test_documented_set_matches_module_tuple(self) -> None:
        """The advertised redaction set is exactly what the helper masks."""
        for header in _REDACTED_RESPONSE_HEADERS:
            redacted = redact_response_headers({header: "secret"})
            assert redacted[header] == "<redacted>"


class TestRedactionInRequestPipeline:
    def test_provider_response_strips_sensitive_headers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: an OpenAI client call returns redacted response_headers."""

        # Build a mock transport that echoes sensitive headers back.
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=OPENAI_RESPONSE,
                headers={
                    "Set-Cookie": "session=leaked; Path=/",
                    "Authorization": "Bearer leaked",
                    "X-Request-Id": "req-keep-me",
                    "Content-Type": "application/json",
                },
            )

        client = OpenAIClient(model="gpt-5", api_key="test-key")
        client._async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url=client._base_url,
        )

        response = client.chat([{"role": "user", "content": "Hi"}])

        assert response.response_headers, "expected response_headers to be populated"
        # httpx lowercases header keys, but redact_response_headers is case-
        # insensitive, so we can lookup by either casing in the result.
        rh_lower = {k.lower(): v for k, v in response.response_headers.items()}
        assert rh_lower.get("set-cookie") == "<redacted>"
        assert rh_lower.get("authorization") == "<redacted>"
        # Non-sensitive headers passed through.
        assert rh_lower.get("x-request-id") == "req-keep-me"
        assert rh_lower.get("content-type", "").startswith("application/json")
