"""Schema canonicalization for provider cache stability.

Providers cache compiled structured-output schemas to amortize first-use
compilation. OpenAI keeps a global schema cache keyed by exact schema bytes
(`response_format.json_schema.schema`); Anthropic caches per-conversation for
24 hours. Two callers sending semantically identical schemas with different
key ordering miss the cache, paying the compile cost on every call.

This module provides:

- :func:`canonicalize` — produces a schema dict with deterministic key order.
  Object keys are sorted recursively; array element order is preserved
  (arrays encode semantic order in JSON Schema, e.g., ``required``,
  ``prefixItems``).
- :func:`schema_hash` — SHA-256 hex digest of the canonical JSON bytes.
  Stable across Python runs and machines. Suitable for cache keys, trace
  correlation, and batch resume fingerprints.

Usage::

    from kaos_llm_client.schema_cache import canonicalize, schema_hash

    schema = {"type": "object", "properties": {"b": {"type": "int"}, "a": {"type": "str"}}}
    canonical = canonicalize(schema)
    # → {"properties": {"a": {"type": "str"}, "b": {"type": "int"}}, "type": "object"}
    fingerprint = schema_hash(schema)  # 64-char SHA-256 hex

Both functions are pure — they never mutate the input schema.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonicalize(schema: Any) -> Any:
    """Return a deeply-copied schema with deterministic key order.

    Dicts have keys sorted lexicographically at every level. Lists preserve
    their original order (lists in JSON Schema are semantically ordered —
    e.g., ``required``, ``prefixItems``, ``allOf`` — so reordering would
    change meaning).

    Non-container values (str / int / float / bool / None) are passed through
    by reference.

    Args:
        schema: A JSON-Schema-like dict, list, or primitive.

    Returns:
        A new structure with sorted keys throughout. Safe to mutate without
        affecting the input.
    """
    if isinstance(schema, dict):
        return {key: canonicalize(schema[key]) for key in sorted(schema)}
    if isinstance(schema, list):
        return [canonicalize(item) for item in schema]
    return schema


def schema_hash(schema: Any) -> str:
    """Return the SHA-256 hex digest of the canonical schema JSON bytes.

    The digest is stable across Python runs, machines, and dict insertion
    order. Use it as a cache key, as the ``program_hash`` for a resumable
    :func:`batch_run` whose work depends on a schema, or as a fingerprint
    in trace correlation.

    Encoding: compact JSON (``separators=(",", ":")``), ASCII-safe (non-ASCII
    characters are escaped). This guarantees byte-exact reproducibility.

    Args:
        schema: A JSON-Schema-like value (dict / list / primitive).

    Returns:
        A 64-character lowercase hex string.
    """
    canonical = canonicalize(schema)
    payload = json.dumps(canonical, separators=(",", ":"), ensure_ascii=True, sort_keys=False)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()
