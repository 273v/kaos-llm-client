"""JSON extraction and normalization from model text output.

Handles common model quirks: markdown code fences, preamble text, trailing
text, partial JSON, and JSONL. Uses ``pydantic_core.from_json`` with
``allow_partial=True`` for recovering truncated JSON (adopted from
alea-llm-client).
"""

from __future__ import annotations

import json
import re
from typing import Any

# Markdown code fence pattern: ```json ... ``` or ``` ... ```
_CODE_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)


def extract_json(text: str) -> Any | None:
    """Extract JSON from model text output.

    Strategy:
    1. Strip markdown code blocks (``\\`\\`\\`json ... \\`\\`\\```)
    2. Direct parse with ``json.loads``
    3. Try ``pydantic_core.from_json(allow_partial=True)`` for truncated JSON
    4. Find first ``{`` or ``[`` / last ``}`` or ``]`` bracket matching
    5. JSONL fallback: try parsing each line as independent JSON

    Returns:
        Parsed JSON value, or None if no valid JSON found.
    """
    if not text or not text.strip():
        return None

    # 1. Try extracting from code fences first
    fence_match = _CODE_FENCE_RE.search(text)
    if fence_match:
        inner = fence_match.group(1).strip()
        try:
            return json.loads(inner)
        except (json.JSONDecodeError, ValueError):
            # Try partial recovery on fenced content
            result = _try_partial_json(inner)
            if result is not None:
                return result

    stripped = text.strip()

    # 2. Direct parse (model returned clean JSON)
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. Find first { or [ and matching last } or ]
    result = _extract_by_brackets(stripped)
    if result is not None:
        return result

    # 4. JSONL fallback: try parsing each non-empty line
    result = _try_jsonl(stripped)
    if result is not None:
        return result

    # 5. Try pydantic_core partial JSON recovery (handles truncated output)
    # This is last because it may return partial results
    result = _try_partial_json(stripped)
    if result is not None:
        return result

    return None


def _try_partial_json(text: str) -> Any | None:
    """Try parsing with pydantic_core's partial JSON support.

    This handles truncated model output where JSON was cut off mid-stream.
    Adopted from alea-llm-client's use of ``pydantic_core.from_json``.
    """
    try:
        from pydantic_core import from_json

        result = from_json(text.encode(), allow_partial="trailing-strings")
        if isinstance(result, dict | list):
            return result
    except (ValueError, ImportError):
        pass
    return None


def _extract_by_brackets(text: str) -> Any | None:
    """Find the outermost JSON object or array by bracket matching."""
    # Find first opening bracket
    obj_start = text.find("{")
    arr_start = text.find("[")

    if obj_start == -1 and arr_start == -1:
        return None

    # Pick the first one
    if obj_start == -1:
        start = arr_start
        close_char = "]"
    elif arr_start == -1 or obj_start <= arr_start:
        start = obj_start
        close_char = "}"
    else:
        start = arr_start
        close_char = "]"

    # Find matching closing bracket from the end
    last_close = text.rfind(close_char)
    if last_close <= start:
        return None

    candidate = text[start : last_close + 1]
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None


def _try_jsonl(text: str) -> list[Any] | None:
    """Try parsing text as JSONL (one JSON value per line)."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    results: list[Any] = []
    for line in lines:
        try:
            results.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            return None

    return results
