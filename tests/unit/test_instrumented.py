"""Tests for InstrumentedClient — timing, token counting, and cost tracking."""

from __future__ import annotations

from typing import Any

from kaos_llm_client.profiles import ModelProfile
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.providers.instrumented import InstrumentedClient
from kaos_llm_client.types import ContentPart, ProviderResponse, UsageInfo


def _make_response(
    text: str = "test",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> ProviderResponse:
    return ProviderResponse(
        provider="function",
        model="test",
        raw={},
        parts=[ContentPart(type="text", text=text)],
        usage=UsageInfo(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


class TestRecordsTimingAndTokens:
    def test_records_timing_and_tokens(self) -> None:
        """Make a request and verify total_requests and total_input_tokens are updated."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response(input_tokens=100, output_tokens=50)

        inner = FunctionClient(function=handler)
        client = InstrumentedClient(inner)

        assert client.total_requests == 0
        assert client.total_input_tokens == 0
        assert client.total_output_tokens == 0

        response = client.chat([{"role": "user", "content": "Hello"}])
        assert response.text == "test"
        assert client.total_requests == 1
        assert client.total_input_tokens == 100
        assert client.total_output_tokens == 50

    def test_accumulates_across_multiple_requests(self) -> None:
        """Multiple requests accumulate token counts."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response(input_tokens=50, output_tokens=25)

        inner = FunctionClient(function=handler)
        client = InstrumentedClient(inner)

        client.chat([{"role": "user", "content": "first"}])
        client.chat([{"role": "user", "content": "second"}])
        client.chat([{"role": "user", "content": "third"}])

        assert client.total_requests == 3
        assert client.total_input_tokens == 150
        assert client.total_output_tokens == 75


class TestCostCalculation:
    def test_cost_calculation(self) -> None:
        """Set cost_per_input/output, verify total_cost is calculated correctly."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response(input_tokens=1000, output_tokens=500)

        inner = FunctionClient(function=handler)
        client = InstrumentedClient(
            inner,
            cost_per_input_token=0.000003,  # $3/M input
            cost_per_output_token=0.000015,  # $15/M output
        )

        client.chat([{"role": "user", "content": "Hello"}])

        expected_cost = (1000 * 0.000003) + (500 * 0.000015)
        assert abs(client.total_cost - expected_cost) < 1e-10

    def test_cost_accumulates(self) -> None:
        """Cost accumulates across multiple requests."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response(input_tokens=100, output_tokens=50)

        inner = FunctionClient(function=handler)
        client = InstrumentedClient(
            inner,
            cost_per_input_token=0.00001,
            cost_per_output_token=0.00002,
        )

        client.chat([{"role": "user", "content": "first"}])
        client.chat([{"role": "user", "content": "second"}])

        expected_cost = 2 * ((100 * 0.00001) + (50 * 0.00002))
        assert abs(client.total_cost - expected_cost) < 1e-10


class TestNoCostWhenNotConfigured:
    def test_no_cost_when_not_configured(self) -> None:
        """total_cost stays 0 when cost rates are not set."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response(input_tokens=500, output_tokens=200)

        inner = FunctionClient(function=handler)
        client = InstrumentedClient(inner)

        client.chat([{"role": "user", "content": "Hello"}])

        assert client.total_cost == 0.0
        # Tokens are still tracked
        assert client.total_input_tokens == 500
        assert client.total_output_tokens == 200


class TestResetCounters:
    def test_reset_counters(self) -> None:
        """reset_counters() clears all accumulated state."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response(input_tokens=100, output_tokens=50)

        inner = FunctionClient(function=handler)
        client = InstrumentedClient(
            inner,
            cost_per_input_token=0.00001,
            cost_per_output_token=0.00002,
        )

        client.chat([{"role": "user", "content": "Hello"}])
        assert client.total_requests == 1

        client.reset_counters()
        assert client.total_requests == 0
        assert client.total_input_tokens == 0
        assert client.total_output_tokens == 0
        assert client.total_cost == 0.0
