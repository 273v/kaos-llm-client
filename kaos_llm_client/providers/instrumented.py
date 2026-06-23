"""Instrumented wrapper that logs request/response with timing, tokens, and cost.

Wraps any ``BaseProviderClient`` via delegation (``WrapperClient``) and
accumulates per-request metrics: latency, token counts, and estimated cost.
"""

from __future__ import annotations

import time
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_client.providers.wrapper import WrapperClient
from kaos_llm_client.types import (
    ProviderResponse,
    RequestOptions,
    ToolChoice,
    ToolDefinition,
)

logger = get_logger("kaos_llm_client.providers.instrumented")


class InstrumentedClient(WrapperClient):
    """Wrapper that logs request/response with timing, tokens, and cost.

    Usage::

        from kaos_llm_client.providers.instrumented import InstrumentedClient

        inner = create_client("openai:gpt-5")
        client = InstrumentedClient(inner, cost_per_input_token=0.000003)
        response = client.chat([{"role": "user", "content": "Hello"}])
        print(f"Requests: {client.total_requests}, Tokens: {client.total_input_tokens}")
    """

    def __init__(
        self,
        wrapped: Any,
        *,
        cost_per_input_token: float | None = None,
        cost_per_output_token: float | None = None,
    ) -> None:
        super().__init__(wrapped)
        self.total_requests: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost: float = 0.0
        self._cost_per_input = cost_per_input_token
        self._cost_per_output = cost_per_output_token

    def _record(self, response: ProviderResponse, elapsed_s: float) -> None:
        """Record metrics from a completed response."""
        self.total_requests += 1
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        if self._cost_per_input is not None:
            self.total_cost += response.usage.input_tokens * self._cost_per_input
        if self._cost_per_output is not None:
            self.total_cost += response.usage.output_tokens * self._cost_per_output

        # Pull session_id / trace_id from the wrapped client's _log_extra
        # so InstrumentedClient logs are correlatable with the provider
        # client's logs (request_id, cache-hit, etc.).
        wrapped = self._wrapped
        log_extra_fn = getattr(wrapped, "_log_extra", None)
        base_extra: dict[str, Any] = {}
        if callable(log_extra_fn):
            base_extra = log_extra_fn()
        logger.debug(
            "LLM request completed",
            extra={
                **base_extra,
                "provider": response.provider,
                "model": response.model,
                "request_id": (response.request_id or "")[:16] or None,
                "response_id": response.response_id,
                "elapsed_s": round(elapsed_s, 3),
                "latency_ms": response.latency_ms,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
                "total_cost": round(self.total_cost, 6),
            },
        )

    async def request_async(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: RequestOptions | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Make an async request with timing instrumentation."""
        t0 = time.monotonic()
        response = await self._wrapped.request_async(
            messages, tools=tools, tool_choice=tool_choice, options=options, **kwargs
        )
        self._record(response, time.monotonic() - t0)
        return response

    def request(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: RequestOptions | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Make a sync request with timing instrumentation."""
        t0 = time.monotonic()
        response = self._wrapped.request(
            messages, tools=tools, tool_choice=tool_choice, options=options, **kwargs
        )
        self._record(response, time.monotonic() - t0)
        return response

    def reset_counters(self) -> None:
        """Reset all accumulated counters to zero."""
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
