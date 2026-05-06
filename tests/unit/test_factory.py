"""Tests for create_client factory — provider resolution and instantiation."""

from __future__ import annotations

import pytest

from kaos_llm_client.errors import KaosLLMError
from kaos_llm_client.providers import create_client
from kaos_llm_client.providers.anthropic import AnthropicClient
from kaos_llm_client.providers.google import GoogleClient
from kaos_llm_client.providers.openai import OpenAIClient
from kaos_llm_client.providers.xai import XAIClient
from kaos_llm_client.settings import KaosLLMSettings


class TestCreateClientWithPrefix:
    """Test create_client with explicit 'provider:model' format."""

    def test_create_client_with_provider_prefix(self) -> None:
        """'openai:gpt-5' creates an OpenAIClient."""
        client = create_client("openai:gpt-5", api_key="test-key")
        assert isinstance(client, OpenAIClient)
        assert client.model == "gpt-5"

    def test_create_client_anthropic_prefix(self) -> None:
        """'anthropic:claude-sonnet-4-6' creates an AnthropicClient."""
        client = create_client("anthropic:claude-sonnet-4-6", api_key="test-key")
        assert isinstance(client, AnthropicClient)
        assert client.model == "claude-sonnet-4-6"

    def test_create_client_google_prefix(self) -> None:
        """'google:gemini-2.5-pro' creates a GoogleClient."""
        client = create_client("google:gemini-2.5-pro", api_key="test-key")
        assert isinstance(client, GoogleClient)
        assert client.model == "gemini-2.5-pro"

    def test_create_client_xai_prefix(self) -> None:
        """'xai:grok-3' creates an XAIClient."""
        client = create_client("xai:grok-3", api_key="test-key")
        assert isinstance(client, XAIClient)
        assert client.model == "grok-3"


class TestCreateClientInference:
    """Test create_client with model-name-only inference (no prefix)."""

    def test_create_client_infers_openai(self) -> None:
        """'gpt-5' (no prefix) infers openai."""
        client = create_client("gpt-5", api_key="test-key")
        assert isinstance(client, OpenAIClient)
        assert client.model == "gpt-5"

    def test_create_client_infers_anthropic(self) -> None:
        """'claude-sonnet-4-6' infers anthropic."""
        client = create_client("claude-sonnet-4-6", api_key="test-key")
        assert isinstance(client, AnthropicClient)
        assert client.model == "claude-sonnet-4-6"

    def test_create_client_infers_google(self) -> None:
        """'gemini-2.5-pro' infers google."""
        client = create_client("gemini-2.5-pro", api_key="test-key")
        assert isinstance(client, GoogleClient)
        assert client.model == "gemini-2.5-pro"

    def test_create_client_unknown_raises(self) -> None:
        """'unknown-model' raises KaosLLMError."""
        with pytest.raises(KaosLLMError, match="Cannot determine provider"):
            create_client("unknown-model", api_key="test-key")


class TestCreateClientPassthrough:
    """Test that settings and kwargs flow through to the provider client."""

    def test_create_client_passes_settings(self) -> None:
        """settings parameter flows through to the client."""
        settings = KaosLLMSettings(default_timeout=42.0)
        client = create_client("openai:gpt-5", settings=settings, api_key="test-key")
        assert isinstance(client, OpenAIClient)
        assert client._settings.default_timeout == 42.0

    def test_create_client_passes_api_key(self) -> None:
        """api_key kwarg flows through to the client."""
        client = create_client("openai:gpt-5", api_key="test-key")
        assert isinstance(client, OpenAIClient)
        assert client._api_key_override == "test-key"
