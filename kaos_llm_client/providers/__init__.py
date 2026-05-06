"""Provider registry and factory function."""

from __future__ import annotations

from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_client.errors import KaosLLMError
from kaos_llm_client.profiles import infer_provider
from kaos_llm_client.providers.base import BaseProviderClient
from kaos_llm_client.settings import KaosLLMSettings

logger = get_logger("kaos_llm_client.providers")

# Lazy provider class registry — maps provider name to (module, class_name)
_PROVIDER_REGISTRY: dict[str, tuple[str, str]] = {
    "openai": ("kaos_llm_client.providers.openai", "OpenAIClient"),
    "anthropic": ("kaos_llm_client.providers.anthropic", "AnthropicClient"),
    "google": ("kaos_llm_client.providers.google", "GoogleClient"),
    "xai": ("kaos_llm_client.providers.xai", "XAIClient"),
    "groq": ("kaos_llm_client.providers.groq", "GroqClient"),
    "mistral": ("kaos_llm_client.providers.mistral", "MistralClient"),
    "openrouter": ("kaos_llm_client.providers.openrouter", "OpenRouterClient"),
    "openai-responses": ("kaos_llm_client.providers.openai_responses", "OpenAIResponsesClient"),
    "openai-compatible": ("kaos_llm_client.providers.openai_compat", "OpenAICompatibleClient"),
    # Azure-hosted OpenAI deployments. Two transports:
    # - ``azure:`` / ``azure-openai:`` → chat completions (legacy path; for
    #   gpt-5.4+ tool calling is broken on this path per Azure docs).
    # - ``azure-responses:`` / ``azure-foundry:`` → Responses API (correct
    #   path for gpt-5.4+ tool calling and reasoning models).
    # ``azure-foundry`` is an alias matching Microsoft's "Azure AI Foundry"
    # branding for the Responses API surface; behaviour is identical.
    "azure": ("kaos_llm_client.providers.azure_openai", "AzureOpenAIClient"),
    "azure-openai": ("kaos_llm_client.providers.azure_openai", "AzureOpenAIClient"),
    "azure-responses": (
        "kaos_llm_client.providers.azure_openai_responses",
        "AzureOpenAIResponsesClient",
    ),
    "azure-foundry": (
        "kaos_llm_client.providers.azure_openai_responses",
        "AzureOpenAIResponsesClient",
    ),
    # AWS Bedrock OpenAI-compatible Responses API. See ``providers/bedrock.py``.
    "bedrock": ("kaos_llm_client.providers.bedrock", "BedrockClient"),
    "function": ("kaos_llm_client.providers.function", "FunctionClient"),
}


def _load_provider_class(provider: str) -> type[BaseProviderClient]:
    """Lazily import and return a provider class."""
    if provider not in _PROVIDER_REGISTRY:
        raise KaosLLMError(
            f"Unknown provider: {provider!r}",
            provider=provider,
            known_providers=sorted(_PROVIDER_REGISTRY.keys()),
            fix=f"Use one of: {', '.join(sorted(_PROVIDER_REGISTRY.keys()))}",
        )

    module_path, class_name = _PROVIDER_REGISTRY[provider]
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_client(
    model: str,
    *,
    settings: KaosLLMSettings | None = None,
    context: Any = None,
    **kwargs: Any,
) -> BaseProviderClient:
    """Create a client for the given model string.

    Model strings use ``provider:model`` format::

        "openai:gpt-5"
        "anthropic:claude-sonnet-4-6"
        "google:gemini-2.5-pro"
        "xai:grok-3"
        "openai-compatible:my-model"  (requires base_url in settings or kwargs)

    If no provider prefix, infers from known model name patterns.

    Args:
        model: Model string, optionally prefixed with provider.
        settings: LLM settings. Created from env vars if not provided.
        context: Optional KaosContext for session/trace correlation.
        **kwargs: Additional arguments passed to the provider client constructor.

    Returns:
        A configured provider client.

    Raises:
        KaosLLMError: If the provider cannot be determined.
    """
    # Parse provider:model
    provider: str | None = None
    model_name = model

    if ":" in model:
        parts = model.split(":", 1)
        provider = parts[0]
        model_name = parts[1]

    # Infer provider from model name patterns
    if provider is None:
        provider = infer_provider(model_name)

    if provider is None:
        raise KaosLLMError(
            f"Cannot determine provider for model: {model!r}. "
            f"Use 'provider:model' format (e.g., 'openai:gpt-5') or a known model name.",
            model=model,
            fix="Prefix the model with a provider name: openai:, anthropic:, google:, xai:, "
            "openai-compatible:",
        )

    cls = _load_provider_class(provider)
    return cls(
        model=model_name,
        settings=settings,
        context=context,
        **kwargs,
    )


__all__ = [
    "BaseProviderClient",
    "create_client",
]
