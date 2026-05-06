"""Mistral provider client.

Thin subclass of ``OpenAICompatibleClient`` -- Mistral's API follows the OpenAI
chat completions contract. Only the base URL and API key source differ.
"""

from __future__ import annotations

from kaos_core.logging import get_logger

from kaos_llm_client.errors import KaosLLMAuthError
from kaos_llm_client.providers.openai_compat import OpenAICompatibleClient

logger = get_logger("kaos_llm_client.providers.mistral")


class MistralClient(OpenAICompatibleClient):
    """Mistral API client.

    Extends ``OpenAICompatibleClient`` with Mistral-specific base URL and API key
    resolution. All request building, response parsing, and streaming logic
    is inherited from the OpenAI-compatible base.
    """

    _provider_name: str = "mistral"

    def _get_default_base_url(self) -> str:
        return self._settings.mistral_base_url

    def _get_api_key_from_settings(self) -> str:
        key = self._settings.mistral_api_key
        if key is None:
            raise KaosLLMAuthError(
                "Mistral API key is not configured.",
                provider=self._provider_name,
                fix="Set KAOS_LLM_MISTRAL_API_KEY environment variable or pass api_key= "
                "to the client constructor.",
            )
        secret = key.get_secret_value()
        if not secret:
            raise KaosLLMAuthError(
                "Mistral API key is empty.",
                provider=self._provider_name,
                fix="Set KAOS_LLM_MISTRAL_API_KEY to a valid API key.",
            )
        return secret
