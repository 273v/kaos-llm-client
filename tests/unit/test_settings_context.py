"""Verify ``BaseProviderClient`` honours the kaos-core Configuration Hierarchy.

Resolution order (CLAUDE.md → ``ModuleSettings``):

1. Explicit ``settings=`` kwarg passed to the client
2. ``KaosContext._config`` entries (per-request override)
3. ``KAOS_LLM_*`` env vars
4. Legacy env vars (``OPENAI_API_KEY``, etc.)
5. ``.env`` / field defaults

Before this test existed, ``BaseProviderClient.__init__`` did
``settings or KaosLLMSettings()`` which silently dropped the per-request
override path — a quiet regression of the documented contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from kaos_llm_client.providers.openai import OpenAIClient
from kaos_llm_client.settings import KaosLLMSettings


@dataclass
class _FakeContext:
    """Minimal stand-in for ``kaos_core.context.KaosContext``.

    ``ModuleSettings.from_context`` only reads ``getattr(context, "_config",
    None)`` so we don't need the real type for these assertions.
    """

    _config: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    trace_id: str | None = None


class TestSettingsContextResolution:
    def test_context_config_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``KaosContext._config`` beats env vars (level 2 > level 3)."""
        # Env says trust_env=true; context says false. Context wins.
        monkeypatch.setenv("KAOS_LLM_TRUST_ENV", "true")
        ctx = _FakeContext(_config={"trust_env": False})

        client = OpenAIClient(model="gpt-5", api_key="test-key", context=ctx)

        assert client._settings.trust_env is False

    def test_explicit_settings_wins_over_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit ``settings=`` kwarg beats ``context._config`` (level 1 > 2)."""
        ctx = _FakeContext(_config={"trust_env": False, "default_timeout": 5.0})
        explicit = KaosLLMSettings(trust_env=True, default_timeout=42.0)

        client = OpenAIClient(
            model="gpt-5",
            api_key="test-key",
            settings=explicit,
            context=ctx,
        )

        # Explicit settings flow through verbatim — context is ignored
        # entirely when an explicit ``settings=`` is provided.
        assert client._settings is explicit
        assert client._settings.trust_env is True
        assert client._settings.default_timeout == 42.0

    def test_env_var_used_when_no_context_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env vars still resolve when context has no matching _config key."""
        monkeypatch.setenv("KAOS_LLM_TRUST_ENV", "true")
        # Context exists but supplies no override for trust_env.
        ctx = _FakeContext(_config={"default_timeout": 10.0})

        client = OpenAIClient(model="gpt-5", api_key="test-key", context=ctx)

        # trust_env came from env; default_timeout came from context.
        assert client._settings.trust_env is True
        assert client._settings.default_timeout == 10.0

    def test_no_context_no_settings_uses_env_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``context=None`` degrades to ``KaosLLMSettings()`` (env + defaults)."""
        monkeypatch.setenv("KAOS_LLM_DEFAULT_TIMEOUT", "77.0")

        client = OpenAIClient(model="gpt-5", api_key="test-key")

        assert client._settings.default_timeout == 77.0
