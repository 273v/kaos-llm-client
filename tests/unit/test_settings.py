"""Tests for kaos_llm_client.settings."""

from __future__ import annotations

import pytest

from kaos_llm_client.settings import KaosLLMSettings


class TestKaosLLMSettings:
    def test_defaults(self):
        settings = KaosLLMSettings()
        # Base URLs are always set to defaults
        assert settings.openai_base_url == "https://api.openai.com"
        assert settings.anthropic_base_url == "https://api.anthropic.com"
        assert settings.google_base_url == "https://generativelanguage.googleapis.com"
        assert settings.xai_base_url == "https://api.x.ai"
        # Google Vertex AI defaults
        assert settings.google_project is None
        assert settings.google_location == "us-central1"
        # Transport defaults
        assert settings.default_timeout == 120.0
        assert settings.default_max_retries == 3
        assert settings.retry_backoff_base == 1.0
        assert settings.trust_env is False
        assert settings.cache_enabled is False
        assert settings.cache_path is None

    def test_trust_env_opt_in(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("KAOS_LLM_TRUST_ENV", "true")
        settings = KaosLLMSettings()
        assert settings.trust_env is True

    def test_api_keys_none_without_env(self, monkeypatch: pytest.MonkeyPatch):
        """Without any env vars, API keys are None."""
        monkeypatch.delenv("KAOS_LLM_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("KAOS_LLM_ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("KAOS_LLM_GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_GENERATIVE_AI_API_KEY", raising=False)
        monkeypatch.delenv("KAOS_LLM_XAI_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        settings = KaosLLMSettings()
        assert settings.openai_api_key is None
        assert settings.anthropic_api_key is None
        assert settings.google_api_key is None
        assert settings.xai_api_key is None

    def test_env_prefix(self):
        # The env prefix should be KAOS_LLM_
        assert settings_env_prefix() == "KAOS_LLM_"

    def test_from_context_overrides(self):
        # ModuleSettings.from_context should work
        settings = KaosLLMSettings.from_context(None, default_timeout=30.0)
        assert settings.default_timeout == 30.0

    def test_google_vertex_env_vars(self, monkeypatch: pytest.MonkeyPatch):
        """Vertex AI settings from KAOS_LLM_ prefixed env vars."""
        monkeypatch.setenv("KAOS_LLM_GOOGLE_PROJECT", "my-gcp-project")
        monkeypatch.setenv("KAOS_LLM_GOOGLE_LOCATION", "europe-west4")
        settings = KaosLLMSettings()
        assert settings.google_project == "my-gcp-project"
        assert settings.google_location == "europe-west4"


def settings_env_prefix() -> str:
    """Extract the env prefix from model_config."""
    config = KaosLLMSettings.model_config
    return config.get("env_prefix", "")
