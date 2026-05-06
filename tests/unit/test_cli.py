"""Tests for kaos_llm_client.cli — all CLI commands."""

from __future__ import annotations

import json

import pytest

from kaos_llm_client.cli import main


def _clear_all_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all API key env vars so tests run clean."""
    for var in (
        "KAOS_LLM_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "KAOS_LLM_ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY",
        "KAOS_LLM_GOOGLE_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENERATIVE_AI_API_KEY",
        "KAOS_LLM_XAI_API_KEY",
        "XAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


class TestCliCheck:
    def test_check_no_keys_exits(self, monkeypatch: pytest.MonkeyPatch):
        """check with no API keys configured exits with error."""
        _clear_all_api_keys(monkeypatch)
        with pytest.raises(SystemExit) as exc_info:
            main(["check"])
        assert exc_info.value.code == 1

    def test_check_json_no_keys(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ):
        """check --json with no keys returns JSON with no_providers status."""
        _clear_all_api_keys(monkeypatch)
        with pytest.raises(SystemExit):
            main(["check", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["command"] == "check"
        assert data["status"] == "no_providers"

    def test_check_with_openai_key(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ):
        """check with KAOS_LLM_OPENAI_API_KEY set shows configured."""
        monkeypatch.setenv("KAOS_LLM_OPENAI_API_KEY", "sk-test-key-12345678")  # gitleaks:allow
        main(["check", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["command"] == "check"
        providers = data["providers"]
        assert len(providers) >= 1
        openai_result = next(p for p in providers if p["provider"] == "openai")
        assert openai_result["has_key"] is True
        assert openai_result["status"] == "configured"

    def test_check_specific_provider(self, capsys: pytest.CaptureFixture[str]):
        """check --provider openai checks only that provider."""
        main(["check", "--provider", "openai", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        providers = data["providers"]
        assert len(providers) == 1
        assert providers[0]["provider"] == "openai"


class TestCliProfiles:
    def test_profiles_json(self, capsys: pytest.CaptureFixture[str]):
        """profiles --json returns all known profiles."""
        main(["profiles", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["command"] == "profiles"
        profiles = data["profiles"]
        assert "openai" in profiles
        assert "anthropic" in profiles
        assert "google" in profiles
        assert "xai" in profiles
        assert "openai-compatible" in profiles

    def test_profiles_text(self, capsys: pytest.CaptureFixture[str]):
        """profiles without --json outputs readable text."""
        main(["profiles"])
        captured = capsys.readouterr()
        assert "openai" in captured.out
        assert "anthropic" in captured.out

    def test_profiles_contains_expected_fields(self, capsys: pytest.CaptureFixture[str]):
        """profiles JSON contains profile fields."""
        main(["profiles", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        openai_profile = data["profiles"]["openai"]
        assert "supports_tools" in openai_profile
        assert "supports_streaming" in openai_profile
        assert "supports_vision" in openai_profile
        assert "max_tokens_field" in openai_profile


class TestCliConfig:
    def test_config_json(self, capsys: pytest.CaptureFixture[str]):
        """config --json returns settings with secrets redacted."""
        main(["config", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["command"] == "config"
        settings = data["settings"]
        assert "openai_base_url" in settings
        assert settings["openai_base_url"] == "https://api.openai.com"
        assert settings["default_timeout"] == 120.0

    def test_config_redacts_keys(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ):
        """config redacts API keys in output."""
        monkeypatch.setenv("KAOS_LLM_OPENAI_API_KEY", "sk-very-secret-key-that-should-not-show")
        main(["config", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        key_value = data["settings"]["openai_api_key"]
        # Should be redacted — NOT the full key
        assert "very-secret" not in str(key_value)

    def test_config_text(self, capsys: pytest.CaptureFixture[str]):
        """config without --json outputs readable text."""
        main(["config"])
        captured = capsys.readouterr()
        assert "openai_base_url" in captured.out
        assert "default_timeout" in captured.out


class TestCliChat:
    def test_chat_requires_model(self):
        """chat without --model fails."""
        with pytest.raises(SystemExit) as exc_info:
            main(["chat", "--message", "hello"])
        assert exc_info.value.code != 0

    def test_chat_requires_message(self):
        """chat without --message fails."""
        with pytest.raises(SystemExit) as exc_info:
            main(["chat", "--model", "openai:gpt-5"])
        assert exc_info.value.code != 0


class TestCliEntryPoint:
    def test_no_command_prints_help(self, capsys: pytest.CaptureFixture[str]):
        """No command prints help and exits."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1
