"""Record/replay cassette system for deterministic LLM testing.

Records LLM request/response pairs to JSONL files during live calls,
then replays them without network access for deterministic regression tests.

Three modes:
- ``RECORD``: live calls + write every request/response pair to cassette
- ``REPLAY``: read-only, raise on cache miss (strict determinism)
- ``AUTO``: replay on hit, fall through to live + record on miss (VCR-style)

Security notes
--------------

A cassette is a near-verbatim transcript of every prompt sent and every
response received during recording — it routinely contains the same
sensitive content the user typed into the model (PII, source code,
internal documents, etc.). Even though :func:`_sanitize_request` strips
``Authorization`` / ``x-api-key`` headers, the body itself is preserved.

To match the on-disk cache (see ``cache.py``), every cassette write
goes to disk with **owner-only** permissions:

- The parent directory is created with mode ``0o700`` and force-chmod'd
  to ``0o700`` afterward (``mkdir(mode=...)`` honours the umask, so the
  explicit chmod is required).
- The JSONL file itself is opened with
  ``os.O_WRONLY | os.O_CREAT | os.O_TRUNC`` and mode ``0o600``, then
  force-chmod'd in case ``os.open`` honoured umask.

This is the same pattern used by ``FileCache.put`` and is non-negotiable
on shared / multi-tenant hosts.

Integration patterns:

1. **Context manager** (recommended for tests)::

    async with use_cassette("tests/cassettes/my_test.jsonl", mode=CassetteMode.AUTO) as csc:
        client = create_client("openai:gpt-5.4-nano", hooks=csc.hooks)
        response = await client.chat_async([...])
        # First run: live call, recorded to cassette
        # Subsequent runs: replayed from cassette

2. **Explicit recorder** (for recording only)::

    recorder = CassetteRecorder("cassettes/session.jsonl")
    client = create_client("openai:gpt-5.4-nano", hooks=recorder.hooks)
    ...
    recorder.close()

3. **Explicit replay client** (for replay only)::

    cassette = Cassette.load("cassettes/session.jsonl")
    client = CassetteReplayClient(wrapped_client, cassette=cassette)
    response = await client.request_async(messages)  # replayed
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from kaos_core.logging import get_logger

from kaos_llm_client.errors import KaosLLMError
from kaos_llm_client.types import (
    ProviderRequest,
    ProviderResponse,
    RequestHooks,
)

logger = get_logger("kaos_llm_client.cassette")

_CASSETTE_KEY_DIGEST_SIZE = 16  # BLAKE2b digest bytes → 32-char hex key
_LOG_KEY_PREFIX_LEN = 12  # Hex chars of cassette key shown in debug logs

# Cassette files contain prompts + provider responses. Owner-only
# permissions are non-negotiable on shared / multi-tenant hosts. Same
# values as ``cache.py``.
_DIR_MODE = 0o700
_FILE_MODE = 0o600

# Fields to strip from responses for deterministic matching
_NONDETERMINISTIC_RESPONSE_FIELDS = frozenset(
    {
        "response_headers",
        "latency_ms",
        "status_code",
    }
)

# Fields to strip from request headers (auth tokens)
_SENSITIVE_HEADER_PREFIXES = (
    "authorization",
    "x-api-key",
    "api-key",
)


class CassetteMode(StrEnum):
    """Cassette operation mode."""

    RECORD = "record"
    REPLAY = "replay"
    AUTO = "auto"


class CassetteMissError(KaosLLMError):
    """Raised in REPLAY mode when no recorded response matches the request."""

    def __init__(self, key: str, *, provider: str, model: str) -> None:
        self.key = key
        self.provider = provider
        self.model = model
        super().__init__(
            f"No cassette entry for key={key} (provider={provider}, model={model}). "
            "Re-record the cassette with CassetteMode.RECORD or CassetteMode.AUTO. "
            "Alternative: use CassetteMode.AUTO to fall through to live calls on miss.",
            key=key,
            provider=provider,
            model=model,
            fix="Re-run the test with CassetteMode.RECORD to capture new interactions.",
        )


def cassette_key(request: ProviderRequest) -> str:
    """Compute a deterministic key from request content.

    Hashes (provider, model, endpoint, body) with BLAKE2b-16.
    Excludes headers (contain auth), request_id (random UUID),
    and stream flag (same logical request).
    """
    canonical = json.dumps(
        {
            "provider": request.provider,
            "model": request.model,
            "endpoint": request.endpoint,
            "body": request.body,
        },
        sort_keys=True,
    )
    return hashlib.blake2b(canonical.encode(), digest_size=_CASSETTE_KEY_DIGEST_SIZE).hexdigest()


def _sanitize_request(request: ProviderRequest) -> dict[str, Any]:
    """Serialize a request, stripping sensitive headers."""
    data = request.model_dump()
    headers = data.get("headers", {})
    data["headers"] = {
        k: v
        for k, v in headers.items()
        if not any(k.lower().startswith(p) for p in _SENSITIVE_HEADER_PREFIXES)
    }
    return data


def _sanitize_response(response: ProviderResponse) -> dict[str, Any]:
    """Serialize a response, stripping non-deterministic transport metadata."""
    data = response.model_dump()
    for field_name in _NONDETERMINISTIC_RESPONSE_FIELDS:
        data.pop(field_name, None)
    return data


@dataclass(frozen=True, slots=True)
class CassetteEntry:
    """One recorded request/response pair."""

    key: str
    sequence: int
    request: dict[str, Any]
    response: dict[str, Any]
    metadata: dict[str, Any]

    def to_json_line(self) -> str:
        """Serialize to a single JSON line."""
        return json.dumps(
            {
                "key": self.key,
                "sequence": self.sequence,
                "request": self.request,
                "response": self.response,
                "metadata": self.metadata,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json_line(cls, line: str) -> CassetteEntry:
        """Deserialize from a single JSON line."""
        data = json.loads(line)
        return cls(
            key=data["key"],
            sequence=data["sequence"],
            request=data["request"],
            response=data["response"],
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)  # Mutable accumulator: entries added during recording
class Cassette:
    """In-memory cassette holding recorded entries.

    Supports two matching strategies:
    - **hash-based**: look up by request content hash (default)
    - **sequential**: replay entries in order, ignoring content
    """

    entries: list[CassetteEntry] = field(default_factory=list)
    _by_key: dict[str, list[CassetteEntry]] = field(default_factory=dict)
    _key_cursors: dict[str, int] = field(default_factory=dict)
    _seq_cursor: int = field(default=0)

    @classmethod
    def load(cls, path: str | Path) -> Cassette:
        """Load a cassette from a JSONL file."""
        path = Path(path)
        cassette = cls()
        if not path.exists():
            return cassette
        for line in path.read_text().strip().splitlines():
            line = line.strip()
            if not line:
                continue
            entry = CassetteEntry.from_json_line(line)
            cassette.entries.append(entry)
            cassette._by_key.setdefault(entry.key, []).append(entry)
        logger.debug("Loaded cassette with %d entries from %s", len(cassette.entries), path)
        return cassette

    def save(self, path: str | Path) -> None:
        """Save all entries to a JSONL file.

        Permissions: parent directory ``0o700``, file ``0o600``.
        Cassettes contain full prompts and provider responses — owner-only
        is the floor on multi-tenant / shared hosts. Mirrors the
        ``FileCache.put`` pattern in ``cache.py``.
        """
        path = Path(path)
        # ``mkdir(mode=...)`` is masked by the process umask; force-chmod
        # the parent afterward so we get a true ``0o700`` even when the
        # umask is permissive (e.g. ``0o022``).
        path.parent.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        with contextlib.suppress(OSError):
            path.parent.chmod(_DIR_MODE)

        body = "".join(f"{entry.to_json_line()}\n" for entry in self.entries)
        # Atomic-ish write: open with restricted mode, write, close.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _FILE_MODE)
        with os.fdopen(fd, "w") as f:
            f.write(body)
        # Force-chmod in case ``os.open`` honoured umask.
        with contextlib.suppress(OSError):
            path.chmod(_FILE_MODE)
        logger.debug("Saved cassette with %d entries to %s", len(self.entries), path)

    def add(self, entry: CassetteEntry) -> None:
        """Add a new entry to the cassette."""
        self.entries.append(entry)
        self._by_key.setdefault(entry.key, []).append(entry)

    def lookup(self, key: str) -> CassetteEntry | None:
        """Look up an entry by content hash.

        When multiple entries share the same key (e.g., identical requests
        made multiple times in a session), returns them in recording order
        using a per-key cursor.
        """
        entries = self._by_key.get(key)
        if not entries:
            return None
        cursor = self._key_cursors.get(key, 0)
        if cursor >= len(entries):
            return None
        self._key_cursors[key] = cursor + 1
        return entries[cursor]

    def next_sequential(self) -> CassetteEntry | None:
        """Return the next entry in sequence order."""
        if self._seq_cursor >= len(self.entries):
            return None
        entry = self.entries[self._seq_cursor]
        self._seq_cursor += 1
        return entry

    def reset_cursors(self) -> None:
        """Reset all lookup cursors to the beginning."""
        self._key_cursors.clear()
        self._seq_cursor = 0

    @property
    def size(self) -> int:
        return len(self.entries)


class CassetteRecorder:
    """Records LLM request/response pairs to a cassette via RequestHooks.

    Attach ``recorder.hooks`` to any client to record interactions::

        recorder = CassetteRecorder("tests/cassettes/my_test.jsonl")
        client = create_client("openai:gpt-5.4-nano", hooks=recorder.hooks)
        response = client.chat([...])
        recorder.save()  # flush to disk
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._cassette = Cassette()
        self._sequence = 0
        self._pending: dict[str, ProviderRequest] = {}

        self.hooks = RequestHooks(
            on_request=self._on_request,
            on_response=self._on_response,
        )

    def _on_request(self, request: ProviderRequest) -> None:
        """Stash the request for pairing with the response."""
        self._pending[request.request_id] = request

    def _on_response(self, request: ProviderRequest, response: ProviderResponse) -> None:
        """Record the request/response pair."""
        self._pending.pop(request.request_id, None)
        key = cassette_key(request)
        entry = CassetteEntry(
            key=key,
            sequence=self._sequence,
            request=_sanitize_request(request),
            response=_sanitize_response(response),
            metadata={
                "recorded_at": time.time(),
                "provider": request.provider,
                "model": request.model,
                "latency_ms": response.latency_ms,
            },
        )
        self._cassette.add(entry)
        self._sequence += 1
        logger.debug(
            "Recorded entry #%d: key=%s provider=%s model=%s",
            entry.sequence,
            key[:_LOG_KEY_PREFIX_LEN],
            request.provider,
            request.model,
        )

    def save(self) -> None:
        """Flush recorded entries to disk."""
        self._cassette.save(self.path)

    def close(self) -> None:
        """Flush and release resources."""
        self.save()

    @property
    def cassette(self) -> Cassette:
        return self._cassette

    @property
    def entry_count(self) -> int:
        return self._cassette.size


def _response_from_entry(entry: CassetteEntry) -> ProviderResponse:
    """Reconstruct a ProviderResponse from a cassette entry."""
    data = dict(entry.response)
    data.setdefault("status_code", 200)
    data.setdefault("response_headers", {})
    data.setdefault("latency_ms", 0.0)
    return ProviderResponse.model_validate(data)


class CassetteReplayClient:
    """Replays responses from a pre-recorded cassette.

    Not a full WrapperClient — instead, provides a ``lookup()`` method
    that callers use to check for a cached response before making a live call.
    This keeps the replay logic composable with any client pattern.

    For the full auto/replay/record lifecycle, use ``use_cassette()``.
    """

    def __init__(
        self,
        cassette: Cassette,
        *,
        sequential: bool = False,
    ) -> None:
        self._cassette = cassette
        self._sequential = sequential

    def lookup(self, request: ProviderRequest) -> ProviderResponse | None:
        """Try to find a recorded response for this request.

        Returns None if no match is found.
        """
        if self._sequential:
            entry = self._cassette.next_sequential()
        else:
            key = cassette_key(request)
            entry = self._cassette.lookup(key)

        if entry is None:
            return None

        return _response_from_entry(entry)


@dataclass(slots=True)  # Mutable: holds recorder/replay state for context duration
class CassetteContext:
    """Combined record + replay context returned by ``use_cassette()``.

    Provides:
    - ``hooks``: attach to a client for recording
    - ``lookup(request)``: check for a replay hit
    - ``mode``: the active cassette mode
    """

    mode: CassetteMode
    path: Path
    hooks: RequestHooks
    _recorder: CassetteRecorder | None
    _replay: CassetteReplayClient | None
    _cassette: Cassette

    def lookup(self, request: ProviderRequest) -> ProviderResponse | None:
        """Look up a recorded response for the request."""
        if self._replay is None:
            return None
        return self._replay.lookup(request)

    @property
    def entry_count(self) -> int:
        return self._cassette.size


@contextmanager
def use_cassette(
    path: str | Path,
    *,
    mode: CassetteMode = CassetteMode.AUTO,
    sequential: bool = False,
) -> Generator[CassetteContext]:
    """Context manager for cassette-based testing.

    Usage::

        with use_cassette("tests/cassettes/test.jsonl") as ctx:
            client = create_client("openai:gpt-5.4-nano", hooks=ctx.hooks)
            response = client.chat([...])

    In AUTO mode: replays recorded responses on match, records new ones on miss.
    In RECORD mode: always makes live calls and records.
    In REPLAY mode: never makes live calls, raises CassetteMissError on miss.
    """
    path = Path(path)

    # Load existing cassette for replay modes
    if mode in (CassetteMode.REPLAY, CassetteMode.AUTO):
        cassette = Cassette.load(path)
    else:
        cassette = Cassette()

    recorder: CassetteRecorder | None = None
    replay: CassetteReplayClient | None = None

    if mode in (CassetteMode.RECORD, CassetteMode.AUTO):
        recorder = CassetteRecorder(path)
        recorder._cassette = cassette
        recorder._sequence = cassette.size

    if mode in (CassetteMode.REPLAY, CassetteMode.AUTO):
        replay = CassetteReplayClient(cassette, sequential=sequential)

    hooks = recorder.hooks if recorder else RequestHooks()

    ctx = CassetteContext(
        mode=mode,
        path=path,
        hooks=hooks,
        _recorder=recorder,
        _replay=replay,
        _cassette=cassette,
    )

    try:
        yield ctx
    finally:
        if recorder:
            recorder.save()


@asynccontextmanager
async def use_cassette_async(
    path: str | Path,
    *,
    mode: CassetteMode = CassetteMode.AUTO,
    sequential: bool = False,
) -> AsyncGenerator[CassetteContext]:
    """Async version of ``use_cassette()``."""
    with use_cassette(path, mode=mode, sequential=sequential) as ctx:
        yield ctx
