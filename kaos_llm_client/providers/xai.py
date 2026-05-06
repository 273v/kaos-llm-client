"""xAI/Grok provider client.

Thin subclass of ``OpenAICompatibleClient`` — xAI's API follows the OpenAI
chat completions contract. Only the base URL and API key source differ.
"""

from __future__ import annotations

from kaos_core.logging import get_logger

from kaos_llm_client.errors import KaosLLMAuthError
from kaos_llm_client.providers.openai_compat import OpenAICompatibleClient

logger = get_logger("kaos_llm_client.providers.xai")


class XAIClient(OpenAICompatibleClient):
    """xAI/Grok API client.

    Extends ``OpenAICompatibleClient`` with xAI-specific base URL and API key
    resolution. All request building, response parsing, and streaming logic
    is inherited from the OpenAI-compatible base.
    """

    _provider_name: str = "xai"

    def _get_default_base_url(self) -> str:
        return self._settings.xai_base_url

    def _get_api_key_from_settings(self) -> str:
        key = self._settings.xai_api_key
        if key is None:
            raise KaosLLMAuthError(
                "xAI API key is not configured.",
                provider=self._provider_name,
                fix="Set KAOS_LLM_XAI_API_KEY environment variable or pass api_key= "
                "to the client constructor.",
            )
        secret = key.get_secret_value()
        if not secret:
            raise KaosLLMAuthError(
                "xAI API key is empty.",
                provider=self._provider_name,
                fix="Set KAOS_LLM_XAI_API_KEY to a valid API key.",
            )
        return secret
