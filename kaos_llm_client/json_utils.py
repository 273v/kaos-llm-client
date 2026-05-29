"""JSON extraction and normalization from model text output.

Handles common model quirks: markdown code fences, preamble text, trailing
text, partial JSON, and JSONL. Uses ``pydantic_core.from_json`` with
``allow_partial=True`` for recovering truncated JSON (adopted from
alea-llm-client).

Partial/lenient recovery is a *last resort*. A complete-but-malformed object
(e.g. a string field that contains an unescaped inline ``"``) is salvaged via
:func:`_repair_inline_quotes` BEFORE partial recovery, so that complete output
round-trips without dropping trailing fields. The ``allow_partial`` path is
only meant for genuinely truncated streams, and it must never win over a
more-complete repair of the same text.
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


def extract_json(text: str, *, allow_partial: bool = True) -> Any | None:
    """Extract JSON from model text output.

    Strategy:
    1. Strip markdown code blocks (``\\`\\`\\`json ... \\`\\`\\```)
    2. Direct parse with ``json.loads``
    3. Find first ``{`` or ``[`` / last ``}`` or ``]`` bracket matching
    4. JSONL fallback: try parsing each line as independent JSON
    5. Repair inline unescaped double-quotes inside string values
    6. ``pydantic_core.from_json(allow_partial=...)`` for truncated JSON

    Args:
        text: Raw model output.
        allow_partial: When ``False``, the lenient ``pydantic_core`` partial
            recovery in step 6 is skipped. Callers that know the response was
            NOT truncated (e.g. ``stop_reason in {"end_turn", "stop"}``) should
            pass ``False`` so a complete-but-malformed object fails loudly
            instead of being silently truncated to its first field.

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
            # Repair inline unescaped quotes BEFORE any partial recovery so a
            # complete fenced object is not truncated to its first field.
            result = _try_repair_inline_quotes(inner)
            if result is not None:
                return result
            if allow_partial:
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

    # 5. Repair inline unescaped double-quotes inside string values.
    # Models that quote document text verbatim (e.g. a memo containing
    # ``...provisions "shall remain in full force..."``) emit a COMPLETE
    # object whose string field has an unescaped ``"``. Strict json.loads
    # breaks at that quote. This salvage re-escapes such quotes so the whole
    # object parses, which preserves trailing fields. It runs BEFORE partial
    # recovery so a complete object is never silently truncated to its first
    # field.
    result = _try_repair_inline_quotes(stripped)
    if result is not None:
        return result

    # 6. Try pydantic_core partial JSON recovery (handles truncated output).
    # This is the LAST resort because it may return partial results: it is
    # only appropriate for genuinely truncated streams, not for complete
    # output that merely contains an inline quote (handled in step 5).
    if allow_partial:
        result = _try_partial_json(stripped)
        if result is not None:
            return result

    return None


def _try_repair_inline_quotes(text: str) -> Any | None:
    """Re-escape inline unescaped double-quotes inside JSON string values.

    Returns the parsed object on success, or ``None`` if no repair makes the
    text parse. Only operates on a single top-level object/array candidate.
    """
    obj_start = text.find("{")
    arr_start = text.find("[")
    if obj_start == -1 and arr_start == -1:
        return None
    if obj_start == -1:
        start, close_char = arr_start, "]"
    elif arr_start == -1 or obj_start <= arr_start:
        start, close_char = obj_start, "}"
    else:
        start, close_char = arr_start, "]"
    last_close = text.rfind(close_char)
    if last_close <= start:
        return None
    candidate = text[start : last_close + 1]

    repaired = _repair_inline_quotes(candidate)
    if repaired is None or repaired == candidate:
        return None
    try:
        result = json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(result, dict | list):
        return result
    return None


def _repair_inline_quotes(text: str) -> str | None:
    """Escape unescaped ``"`` that appear *inside* JSON string values.

    Single forward pass. A ``"`` encountered while inside a string is treated
    as the string's closing quote only when the next non-whitespace character
    is structural (``,``, ``}``, ``]``, or ``:``) or end-of-input — i.e. a
    position where a string token legitimately ends. Any other ``"`` inside a
    string is an inline quote and is escaped to ``\\"``.

    Returns the repaired text, or ``None`` if the input was already balanced
    (so callers can skip a redundant reparse).
    """
    out: list[str] = []
    in_string = False
    escape = False
    changed = False
    n = len(text)
    for i, ch in enumerate(text):
        if escape:
            out.append(ch)
            escape = False
            continue
        if in_string:
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                # Look ahead: is this a legitimate string terminator?
                j = i + 1
                while j < n and text[j] in " \t\r\n":
                    j += 1
                if j >= n or text[j] in ",}]:":
                    out.append(ch)
                    in_string = False
                else:
                    # Inline quote inside the value — escape it.
                    out.append('\\"')
                    changed = True
                continue
            out.append(ch)
            continue
        # Outside a string.
        if ch == '"':
            in_string = True
        out.append(ch)
    if not changed:
        return None
    return "".join(out)


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
