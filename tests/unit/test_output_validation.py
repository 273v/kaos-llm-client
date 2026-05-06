"""Tests for output validation in pydantic_async — validator param and retry logic."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from kaos_llm_client.errors import KaosLLMValidationError
from kaos_llm_client.profiles import ModelProfile
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart, ProviderResponse, UsageInfo


class UserInfo(BaseModel):
    name: str
    age: int


def _make_response(data: dict[str, Any]) -> ProviderResponse:
    """Helper to build a ProviderResponse with JSON text content."""
    return ProviderResponse(
        provider="function",
        model="test",
        raw={},
        parts=[ContentPart(type="text", text=json.dumps(data))],
        usage=UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15),
    )


class TestValidatorPasses:
    """Validator accepts on first try, no retry needed."""

    def test_validator_passes(self) -> None:
        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response({"name": "Alice", "age": 30})

        def validator(result: UserInfo) -> UserInfo:
            # Accept as-is
            return result

        client = FunctionClient(function=handler)
        result = client.pydantic(
            [{"role": "user", "content": "Get user info"}],
            output_type=UserInfo,
            output_validator=validator,
            max_validation_retries=2,
        )
        assert result.name == "Alice"
        assert result.age == 30
        # Only one call was made (no retries)
        assert len(client.call_history) == 1


class TestValidatorFailsAndRetries:
    """Validator raises ValueError, retries with error in messages, succeeds on retry."""

    def test_validator_fails_and_retries(self) -> None:
        call_count = 0

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First attempt: age is negative (will fail validator)
                return _make_response({"name": "Bob", "age": -5})
            # Second attempt: corrected
            return _make_response({"name": "Bob", "age": 25})

        def validator(result: UserInfo) -> UserInfo:
            if result.age < 0:
                raise ValueError("Age must be non-negative")
            return result

        client = FunctionClient(function=handler)
        result = client.pydantic(
            [{"role": "user", "content": "Get user info"}],
            output_type=UserInfo,
            output_validator=validator,
            max_validation_retries=2,
        )
        assert result.name == "Bob"
        assert result.age == 25
        # Two calls: first failed validator, second succeeded
        assert len(client.call_history) == 2

    def test_retry_messages_contain_error(self) -> None:
        """On retry, the messages sent should contain the validation error."""
        call_count = 0

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_response({"name": "Bob", "age": -1})
            # On retry, check that error feedback was included
            last_msg = messages[-1]["content"]
            assert "Age must be positive" in last_msg
            return _make_response({"name": "Bob", "age": 25})

        def validator(result: UserInfo) -> UserInfo:
            if result.age < 0:
                raise ValueError("Age must be positive")
            return result

        client = FunctionClient(function=handler)
        result = client.pydantic(
            [{"role": "user", "content": "Get user info"}],
            output_type=UserInfo,
            output_validator=validator,
            max_validation_retries=1,
        )
        assert result.age == 25


class TestValidatorExhausted:
    """max_validation_retries=1, fails twice, raises KaosLLMValidationError."""

    def test_validator_exhausted(self) -> None:
        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            # Always return invalid data
            return _make_response({"name": "Charlie", "age": -1})

        def validator(result: UserInfo) -> UserInfo:
            if result.age < 0:
                raise ValueError("Age must be non-negative")
            return result

        client = FunctionClient(function=handler)
        with pytest.raises(KaosLLMValidationError, match="Output validator failed"):
            client.pydantic(
                [{"role": "user", "content": "Get user info"}],
                output_type=UserInfo,
                output_validator=validator,
                max_validation_retries=1,
            )
        # Two calls: initial + 1 retry
        assert len(client.call_history) == 2


class TestNoValidatorDefault:
    """No validator param, works as before."""

    def test_no_validator_default(self) -> None:
        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response({"name": "Diana", "age": 28})

        client = FunctionClient(function=handler)
        result = client.pydantic(
            [{"role": "user", "content": "Get user info"}],
            output_type=UserInfo,
        )
        assert result.name == "Diana"
        assert result.age == 28
        assert len(client.call_history) == 1

    def test_no_validator_pydantic_validation_still_works(self) -> None:
        """Without a custom validator, Pydantic schema validation still applies."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            # Return invalid data (age is a string, not int)
            return ProviderResponse(
                provider="function",
                model="test",
                raw={},
                parts=[ContentPart(type="text", text='{"name": "Eve", "age": "not-a-number"}')],
                usage=UsageInfo(),
            )

        client = FunctionClient(function=handler)
        # Pydantic will coerce "not-a-number" to int which fails
        with pytest.raises(KaosLLMValidationError):
            client.pydantic(
                [{"role": "user", "content": "Get user info"}],
                output_type=UserInfo,
            )
