"""OpenAI provider client.

Extends ``OpenAICompatibleClient`` with OpenAI-specific features:
- ``response_format`` with ``json_schema`` for structured outputs
- ``reasoning`` parameter for o-series models (o1, o3, o4)
"""

from __future__ import annotations

from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_client.errors import KaosLLMAuthError
from kaos_llm_client.providers.openai_compat import OpenAICompatibleClient
from kaos_llm_client.types import (
    ProviderRequest,
    ToolChoice,
    ToolDefinition,
)

logger = get_logger("kaos_llm_client.providers.openai")


class OpenAIClient(OpenAICompatibleClient):
    """Client for the OpenAI API.

    Adds OpenAI-specific features on top of the compatible base:

    - **Structured outputs**: ``response_format`` with ``json_schema`` for
      native schema enforcement (strict mode).
    - **Reasoning models**: ``reasoning`` parameter for o-series models
      that support extended thinking (``reasoning.effort``).
    """

    _provider_name: str = "openai"

    # --- Auth overrides ---

    def _get_api_key_from_settings(self) -> str:
        key = self._settings.openai_api_key
        if key is None:
            raise KaosLLMAuthError(
                "OpenAI API key is not configured.",
                provider=self._provider_name,
                fix="Set KAOS_LLM_OPENAI_API_KEY environment variable or pass api_key= "
                "to the client constructor.",
            )
        secret = key.get_secret_value()
        if not secret:
            raise KaosLLMAuthError(
                "OpenAI API key is empty.",
                provider=self._provider_name,
                fix="Set KAOS_LLM_OPENAI_API_KEY to a valid API key.",
            )
        return secret

    # --- Request building ---

    def _build_request(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ProviderRequest:
        """Build an OpenAI chat completions request with OpenAI-specific features.

        Handles:
        - ``reasoning`` parameter for o-series models (popped from kwargs and
          applied as a top-level body field).
        - All base OpenAI-compatible features (tools, tool_choice, streaming).
        """
        # Extract OpenAI-specific kwargs before passing to the base builder
        reasoning = kwargs.pop("reasoning", None)

        # Build the base request
        request = super()._build_request(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
            **kwargs,
        )

        # Apply reasoning_effort for o-series models.
        # Accepts: reasoning={"effort": "high"} or reasoning_effort="high" (from kwargs)
        if reasoning is not None:
            if isinstance(reasoning, dict):
                effort = reasoning.get("effort", "medium")
            else:
                effort = str(reasoning)
            request.body["reasoning_effort"] = effort

        return request

    # --- Structured output ---

    def _apply_native_json_mode(
        self, kwargs: dict[str, Any], schema: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Apply OpenAI native structured output via ``response_format``.

        When a schema is provided, uses ``json_schema`` response format with
        strict mode. Applies the profile's ``json_schema_transformer`` to ensure
        the schema meets OpenAI's strict-mode constraints (all properties
        required, ``additionalProperties: false``, no unsupported keywords).

        When no schema is provided, falls back to ``json_object`` mode.
        """
        kwargs = dict(kwargs)

        if schema is not None:
            # Apply schema transformer if the profile defines one
            transformed_schema = schema
            if self.profile.json_schema_transformer is not None:
                transformer = self.profile.json_schema_transformer(schema, strict=True)
                transformed_schema = transformer.transform()

            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "strict": True,
                    "schema": transformed_schema,
                },
            }
        else:
            # No schema — use basic JSON object mode
            kwargs["response_format"] = {"type": "json_object"}

        return kwargs
