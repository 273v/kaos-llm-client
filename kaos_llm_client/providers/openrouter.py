"""OpenRouter provider client.

Thin subclass of ``OpenAICompatibleClient`` -- OpenRouter's API follows the OpenAI
chat completions contract. Adds the ``HTTP-Referer`` header from settings.
"""

from __future__ import annotations

from kaos_core.logging import get_logger

from kaos_llm_client.errors import KaosLLMAuthError
from kaos_llm_client.providers.openai_compat import OpenAICompatibleClient

logger = get_logger("kaos_llm_client.providers.openrouter")


class OpenRouterClient(OpenAICompatibleClient):
    """OpenRouter API client.

    Extends ``OpenAICompatibleClient`` with OpenRouter-specific base URL, API key
    resolution, and ``HTTP-Referer`` header injection. All request building,
    response parsing, and streaming logic is inherited from the OpenAI-compatible base.
    """

    _provider_name: str = "openrouter"

    def _get_default_base_url(self) -> str:
        return self._settings.openrouter_base_url

    def _get_api_key_from_settings(self) -> str:
        key = self._settings.openrouter_api_key
        if key is None:
            raise KaosLLMAuthError(
                "OpenRouter API key is not configured.",
                provider=self._provider_name,
                fix="Set KAOS_LLM_OPENROUTER_API_KEY environment variable or pass api_key= "
                "to the client constructor.",
            )
        secret = key.get_secret_value()
        if not secret:
            raise KaosLLMAuthError(
                "OpenRouter API key is empty.",
                provider=self._provider_name,
                fix="Set KAOS_LLM_OPENROUTER_API_KEY to a valid API key.",
            )
        return secret

    def _build_headers(self) -> dict[str, str]:
        headers = super()._build_headers()
        site_url = self._settings.openrouter_site_url
        if site_url:
            headers["HTTP-Referer"] = site_url
        return headers
