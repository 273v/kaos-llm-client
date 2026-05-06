"""Credential-gating fixtures for live integration tests.

Tests are skipped by default unless the corresponding API key environment
variable is set. Checks both KAOS_LLM_* and standard env var names.
Mark tests with the appropriate ``requires_*`` marker.
"""

from __future__ import annotations

import os

import pytest


def _has_key(*env_vars: str) -> bool:
    """Return True if any of the given env vars is set and non-empty."""
    return any(os.getenv(v) for v in env_vars)


# Skip markers for each provider — check both KAOS_LLM_ and standard names
requires_openai = pytest.mark.skipif(
    not _has_key("KAOS_LLM_OPENAI_API_KEY", "OPENAI_API_KEY"),
    reason="No OpenAI API key",
)
requires_anthropic = pytest.mark.skipif(
    not _has_key("KAOS_LLM_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    reason="No Anthropic API key",
)
requires_google = pytest.mark.skipif(
    not _has_key("KAOS_LLM_GOOGLE_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"),
    reason="No Google API key",
)

# Secondary providers — README L108 advertises these, so each gets one
# cheap chat smoke to prove the wire format and auth still work. Heavy
# matrix coverage stays on OpenAI/Anthropic/Google to keep the routine
# live gate fast.
requires_xai = pytest.mark.skipif(
    not _has_key("KAOS_LLM_XAI_API_KEY", "XAI_API_KEY"),
    reason="No xAI API key",
)
requires_groq = pytest.mark.skipif(
    not _has_key("KAOS_LLM_GROQ_API_KEY", "GROQ_API_KEY"),
    reason="No Groq API key",
)
requires_mistral = pytest.mark.skipif(
    not _has_key("KAOS_LLM_MISTRAL_API_KEY", "MISTRAL_API_KEY"),
    reason="No Mistral API key",
)
requires_openrouter = pytest.mark.skipif(
    not _has_key("KAOS_LLM_OPENROUTER_API_KEY", "OPENROUTER_API_KEY"),
    reason="No OpenRouter API key",
)
