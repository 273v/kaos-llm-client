"""Live integration tests for the cassette record/replay system.

Records a real LLM call to a cassette file, then replays it deterministically
without network access. Proves the full round-trip: live → record → replay.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kaos_llm_client.cassette import (
    Cassette,
    CassetteMode,
    CassetteRecorder,
    CassetteReplayClient,
    cassette_key,
    use_cassette,
)
from kaos_llm_client.providers import create_client


def _has_anthropic_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("KAOS_LLM_ANTHROPIC_API_KEY"))


def _has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("KAOS_LLM_OPENAI_API_KEY"))


@pytest.mark.integration
class TestCassetteRecordReplay:
    """Record a live LLM call, then replay it deterministically."""

    def test_record_and_replay_anthropic(self, tmp_path: Path) -> None:
        if not _has_anthropic_key():
            pytest.skip("ANTHROPIC_API_KEY not set")

        cassette_path = tmp_path / "anthropic.jsonl"
        messages = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]

        # Phase 1: Record
        recorder = CassetteRecorder(cassette_path)
        client = create_client("anthropic:claude-haiku-4-5", hooks=recorder.hooks)
        live_response = client.chat(messages)
        recorder.save()

        assert recorder.entry_count == 1
        assert cassette_path.exists()
        assert "4" in live_response.text

        # Phase 2: Replay
        cassette = Cassette.load(cassette_path)
        replay = CassetteReplayClient(cassette)

        replay_request = client._build_request(messages, stream=False)
        replayed = replay.lookup(replay_request)

        assert replayed is not None
        assert replayed.text == live_response.text
        assert replayed.usage.input_tokens == live_response.usage.input_tokens
        assert replayed.usage.output_tokens == live_response.usage.output_tokens

    def test_record_and_replay_openai(self, tmp_path: Path) -> None:
        if not _has_openai_key():
            pytest.skip("OPENAI_API_KEY not set")

        cassette_path = tmp_path / "openai.jsonl"
        messages = [{"role": "user", "content": "What is the capital of France? One word."}]

        # Phase 1: Record
        recorder = CassetteRecorder(cassette_path)
        client = create_client("openai:gpt-5.4-nano", hooks=recorder.hooks)
        live_response = client.chat(messages)
        recorder.save()

        assert recorder.entry_count == 1
        assert "paris" in live_response.text.lower()

        # Phase 2: Replay
        cassette = Cassette.load(cassette_path)
        replay = CassetteReplayClient(cassette)

        replay_request = client._build_request(messages, stream=False)
        replayed = replay.lookup(replay_request)

        assert replayed is not None
        assert replayed.text == live_response.text

    @pytest.mark.asyncio
    async def test_multi_turn_record_replay(self, tmp_path: Path) -> None:
        if not _has_anthropic_key():
            pytest.skip("ANTHROPIC_API_KEY not set")

        cassette_path = tmp_path / "multi_turn.jsonl"
        questions = ["What is 2+2?", "What is 3+3?", "What is 4+4?"]

        # Phase 1: Record 3 turns
        recorder = CassetteRecorder(cassette_path)
        client = create_client("anthropic:claude-haiku-4-5", hooks=recorder.hooks)

        responses = []
        for q in questions:
            resp = await client.chat_async(
                [{"role": "user", "content": f"{q} Answer with just the number."}]
            )
            responses.append(resp)
        recorder.save()
        await client.aclose()

        assert recorder.entry_count == 3

        # Phase 2: Replay all 3 turns
        cassette = Cassette.load(cassette_path)
        replay = CassetteReplayClient(cassette)

        replay_client = create_client("anthropic:claude-haiku-4-5")
        for i, q in enumerate(questions):
            req = replay_client._build_request(
                [{"role": "user", "content": f"{q} Answer with just the number."}],
                stream=False,
            )
            replayed = replay.lookup(req)
            assert replayed is not None, f"Turn {i} replay miss"
            assert replayed.text == responses[i].text
        await replay_client.aclose()

    def test_use_cassette_auto_mode(self, tmp_path: Path) -> None:
        if not _has_anthropic_key():
            pytest.skip("ANTHROPIC_API_KEY not set")

        cassette_path = tmp_path / "auto.jsonl"
        messages = [{"role": "user", "content": "Say 'hello' and nothing else."}]

        # First run: records to cassette (file doesn't exist yet)
        with use_cassette(cassette_path, mode=CassetteMode.AUTO) as ctx:
            client = create_client("anthropic:claude-haiku-4-5", hooks=ctx.hooks)
            live_response = client.chat(messages)

        assert cassette_path.exists()
        cassette = Cassette.load(cassette_path)
        assert cassette.size == 1

        # Second run: replays from cassette (no network needed)
        with use_cassette(cassette_path, mode=CassetteMode.REPLAY) as ctx:
            req = client._build_request(messages, stream=False)
            replayed = ctx.lookup(req)
            assert replayed is not None
            assert replayed.text == live_response.text

    def test_cassette_preserves_tool_calls(self, tmp_path: Path) -> None:
        if not _has_openai_key():
            pytest.skip("OPENAI_API_KEY not set")

        from kaos_llm_client.types import ToolChoice, ToolDefinition

        cassette_path = tmp_path / "tools.jsonl"
        messages = [{"role": "user", "content": "What is the weather in Paris?"}]
        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get weather for a city",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        ]

        # Phase 1: Record
        recorder = CassetteRecorder(cassette_path)
        client = create_client("openai:gpt-5.4-nano", hooks=recorder.hooks)
        live_response = client.chat(messages, tools=tools, tool_choice=ToolChoice(type="required"))
        recorder.save()

        assert len(live_response.tool_calls) >= 1

        # Phase 2: Replay
        cassette = Cassette.load(cassette_path)
        replay = CassetteReplayClient(cassette)
        req = client._build_request(
            messages, tools=tools, tool_choice=ToolChoice(type="required"), stream=False
        )
        replayed = replay.lookup(req)

        assert replayed is not None
        assert len(replayed.tool_calls) == len(live_response.tool_calls)
        assert replayed.tool_calls[0].name == live_response.tool_calls[0].name
        assert replayed.tool_calls[0].arguments == live_response.tool_calls[0].arguments


@pytest.mark.integration
class TestCassetteKeyStability:
    """Verify that cassette keys are stable across sessions."""

    def test_key_stable_across_client_instances(self) -> None:
        if not _has_anthropic_key():
            pytest.skip("ANTHROPIC_API_KEY not set")

        messages = [{"role": "user", "content": "test stability"}]

        client1 = create_client("anthropic:claude-haiku-4-5")
        client2 = create_client("anthropic:claude-haiku-4-5")

        req1 = client1._build_request(messages, stream=False)
        req2 = client2._build_request(messages, stream=False)

        assert cassette_key(req1) == cassette_key(req2)
