"""Tests for FunctionClient — deterministic test double for LLM providers."""

from __future__ import annotations

from typing import Any

import pytest

from kaos_llm_client.errors import KaosLLMProviderError
from kaos_llm_client.profiles import ModelProfile
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart, ProviderResponse, UsageInfo


def _make_response(text: str = "test") -> ProviderResponse:
    return ProviderResponse(
        provider="function",
        model="test",
        raw={},
        parts=[ContentPart(type="text", text=text)],
        usage=UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15),
    )


class TestFunctionClient:
    def test_basic_chat(self) -> None:
        """Create FunctionClient with a sync function, call chat(), verify response.text."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response("hello from function")

        client = FunctionClient(function=handler)
        response = client.chat([{"role": "user", "content": "Hi"}])
        assert response.text == "hello from function"

    async def test_async_chat(self) -> None:
        """Async version of basic_chat using chat_async."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response("async hello")

        client = FunctionClient(function=handler)
        response = await client.chat_async([{"role": "user", "content": "Hi"}])
        assert response.text == "async hello"

    def test_call_history_tracking(self) -> None:
        """Verify self.call_history records messages and kwargs."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response()

        client = FunctionClient(function=handler)
        msgs = [{"role": "user", "content": "test message"}]
        client.chat(msgs)

        assert len(client.call_history) == 1
        recorded_msgs, _recorded_kwargs = client.call_history[0]
        assert recorded_msgs == msgs

    async def test_async_function(self) -> None:
        """Pass an async function and verify it works."""

        async def handler(
            messages: list[dict[str, Any]], profile: ModelProfile
        ) -> ProviderResponse:
            return _make_response("from async fn")

        client = FunctionClient(function=handler)
        response = await client.chat_async([{"role": "user", "content": "Hi"}])
        assert response.text == "from async fn"

    def test_multiple_calls(self) -> None:
        """Make multiple calls and verify history has all of them."""
        call_count = 0

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            nonlocal call_count
            call_count += 1
            return _make_response(f"response {call_count}")

        client = FunctionClient(function=handler)
        r1 = client.chat([{"role": "user", "content": "first"}])
        r2 = client.chat([{"role": "user", "content": "second"}])
        r3 = client.chat([{"role": "user", "content": "third"}])

        assert r1.text == "response 1"
        assert r2.text == "response 2"
        assert r3.text == "response 3"
        assert len(client.call_history) == 3
        assert client.call_history[0][0] == [{"role": "user", "content": "first"}]
        assert client.call_history[2][0] == [{"role": "user", "content": "third"}]

    def test_error_simulation(self) -> None:
        """Function raises KaosLLMProviderError and it propagates."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            raise KaosLLMProviderError(
                "Simulated provider error",
                provider="function",
                model="test",
                status_code=500,
            )

        client = FunctionClient(function=handler)
        with pytest.raises(KaosLLMProviderError, match="Simulated provider error"):
            client.chat([{"role": "user", "content": "Hi"}])

    def test_default_model_name(self) -> None:
        """Default model is 'function-test'."""
        client = FunctionClient()
        assert client.model == "function-test"

    def test_custom_model_name(self) -> None:
        """Can override model name."""
        client = FunctionClient(model="my-custom-model")
        assert client.model == "my-custom-model"
