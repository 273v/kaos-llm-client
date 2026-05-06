"""Pluggable response cache with BLAKE2b keying and gzipped JSON storage.

Security notes
--------------

The on-disk cache stores full provider responses (text, tool calls,
embeddings) — equivalent in sensitivity to whatever the user prompted
the model with. The default ``cache_enabled=False`` keeps this
contained; when enabled, this module:

- Creates the cache root and per-key subdirectories with mode ``0o700``
  (owner-only) so other local users can't read entries on shared hosts.
- Writes individual cache files with mode ``0o600``.
- Includes a BLAKE2b digest of the api-key into the cache key
  (``cache_key(..., auth_scope=...)``) so two principals using the same
  prompt+model do not share cache entries.
- Refuses to ``clear()`` if ``self.path`` falls outside an allowlist of
  safe roots, defending against a misconfigured ``cache_path`` wiping
  unrelated directories.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

from kaos_core.logging import get_logger
from pydantic import ValidationError

from kaos_llm_client.types import ProviderRequest, ProviderResponse

logger = get_logger("kaos_llm_client.cache")

# Cache files contain prompts + full provider responses. Owner-only
# permissions are non-negotiable on shared / multi-tenant hosts.
_DIR_MODE = 0o700
_FILE_MODE = 0o600

# Allowlist of path roots ``FileCache.clear()`` will accept. The default
# location (~/.cache/kaos/llm) and conventional system caches are
# permitted; arbitrary paths (like ``/`` or ``$HOME``) are rejected.
#
# The literal ``/tmp`` / ``/var/tmp`` / ``/var/folders`` strings below
# trigger bandit B108 / ruff S108 (insecure temp-file use), but the use
# here is the *opposite* of insecure: these paths form a defensive
# allowlist that REJECTS clears outside of them. We are not creating
# temp files; we are restricting where ``rmtree`` can run.
_CLEAR_ALLOWLIST_ROOTS: tuple[Path, ...] = (
    Path.home() / ".cache",
    Path("/tmp"),  # nosec B108  — allowlist entry, not temp-file use
    Path("/var/tmp"),  # nosec B108  — allowlist entry, not temp-file use
    Path("/var/folders"),  # nosec B108  — macOS per-user temp allowlist entry
)


class CacheBackend(ABC):
    """Pluggable response cache interface."""

    @abstractmethod
    def get(self, key: str) -> ProviderResponse | None:
        """Retrieve a cached response by key, or None."""
        ...

    @abstractmethod
    def put(self, key: str, response: ProviderResponse) -> None:
        """Store a response under the given key."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all cached entries."""
        ...


class NullCache(CacheBackend):
    """No-op cache. Default when caching is disabled."""

    def get(self, key: str) -> ProviderResponse | None:
        return None

    def put(self, key: str, response: ProviderResponse) -> None:
        pass

    def clear(self) -> None:
        pass


class FileCache(CacheBackend):
    """File-backed cache with gzipped JSON storage.

    Cache files are stored as ``{path}/{key[:2]}/{key}.json.gz``.
    Two-character directory prefix prevents filesystem issues with
    many files in one directory.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        # ``mkdir(mode=...)`` only honours the bits permitted by the
        # process umask. Force the mode explicitly for the cache root.
        with contextlib.suppress(OSError):
            self.path.chmod(_DIR_MODE)

    def _key_path(self, key: str) -> Path:
        """Build the file path for a cache key."""
        return self.path / key[:2] / f"{key}.json.gz"

    def get(self, key: str) -> ProviderResponse | None:
        """Read and decompress a cached response."""
        file_path = self._key_path(key)
        if not file_path.exists():
            return None

        try:
            data = gzip.decompress(file_path.read_bytes())
            parsed = json.loads(data)
            return ProviderResponse.model_validate(parsed)
        except (OSError, gzip.BadGzipFile, json.JSONDecodeError, ValidationError, ValueError):
            logger.debug(
                "Cache read failed for key %s",
                key,
                exc_info=True,
                extra={"cache_op": "read", "key": key[:8]},
            )
            return None

    def put(self, key: str, response: ProviderResponse) -> None:
        """Compress and write a response to the cache.

        Permissions: parent directory ``0o700``, file ``0o600``. Cache
        contents include full prompts and provider responses — owner-only
        is the floor on multi-tenant / shared hosts.
        """
        file_path = self._key_path(key)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
            with contextlib.suppress(OSError):
                file_path.parent.chmod(_DIR_MODE)
            data = response.model_dump_json().encode()
            # Atomic-ish write: open with restricted mode, write, close.
            fd = os.open(file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _FILE_MODE)
            with os.fdopen(fd, "wb") as fh:
                fh.write(gzip.compress(data))
            # Force-chmod in case ``os.open`` honoured umask.
            with contextlib.suppress(OSError):
                file_path.chmod(_FILE_MODE)
        except (OSError, TypeError, ValueError):
            logger.debug(
                "Cache write failed for key %s",
                key,
                exc_info=True,
                extra={"cache_op": "write", "key": key[:8]},
            )

    def clear(self) -> None:
        """Remove all cached files.

        Refuses to delete a path outside the allowlist
        (``~/.cache``, ``/tmp``, ``/var/tmp``, ``/var/folders``). A
        misconfigured ``cache_path`` (e.g. set to ``/``) MUST NOT cause
        ``shutil.rmtree`` to wipe unrelated directories. If the path is
        outside the allowlist, raises ``RuntimeError``.
        """
        import shutil

        resolved = self.path.resolve()
        allowed_roots = tuple(root.expanduser().resolve() for root in _CLEAR_ALLOWLIST_ROOTS)
        if not any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots):
            raise RuntimeError(
                f"FileCache.clear() refused: {resolved!s} is outside the "
                f"safe-clear allowlist {tuple(str(root) for root in allowed_roots)}. "
                "Set cache_path to a location under one of those prefixes, "
                "or remove the directory manually."
            )
        if resolved.exists():
            shutil.rmtree(resolved)
            self.path.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
            with contextlib.suppress(OSError):
                self.path.chmod(_DIR_MODE)


def cache_key(
    request: ProviderRequest,
    *,
    base_url: str = "",
    auth_scope: str | None = None,
) -> str:
    """Compute a deterministic cache key from request content.

    Included: provider, model, endpoint, body, base_url, ``auth_scope``.
    Excluded: timeout, retry config, raw api_key, transient headers,
    cache_policy.

    ``base_url`` prevents cache bleeding across different hosts/proxies
    when using OpenAI-compatible clients pointed at different servers.

    ``auth_scope`` is a short opaque digest of the credential the
    request was made with (typically a BLAKE2b hash of the api-key,
    NOT the key itself). Including it in the key namespace prevents
    two principals on the same machine from sharing cache entries
    when they happen to make identical requests but should not see
    each other's cached responses (multi-tenant data isolation).
    Pass ``None`` (default) only when the cache is single-tenant.
    """
    canonical = json.dumps(
        {
            "base_url": base_url,
            "provider": request.provider,
            "model": request.model,
            "endpoint": request.endpoint,
            "body": request.body,
            "auth_scope": auth_scope,
        },
        sort_keys=True,
    )
    return hashlib.blake2b(canonical.encode(), digest_size=16).hexdigest()


def auth_scope_digest(api_key: str | None) -> str | None:
    """Return a short opaque digest of an api-key, suitable for use as
    ``auth_scope`` in :func:`cache_key`. ``None`` yields ``None``.

    The digest is a 16-hex-char BLAKE2b hash; it is one-way and short
    enough that even logging it does not meaningfully help an attacker
    recover the key.
    """
    if not api_key:
        return None
    return hashlib.blake2b(api_key.encode(), digest_size=8).hexdigest()
