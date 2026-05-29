"""Tests for kaos_llm_client.json_utils — JSON extraction edge cases."""

from __future__ import annotations

from kaos_llm_client.json_utils import extract_json


class TestExtractJson:
    def test_clean_json_object(self):
        assert extract_json('{"key": "value"}') == {"key": "value"}

    def test_clean_json_array(self):
        assert extract_json("[1, 2, 3]") == [1, 2, 3]

    def test_empty_input(self):
        assert extract_json("") is None
        assert extract_json("   ") is None

    def test_none_like(self):
        assert extract_json("not json at all") is None

    def test_code_fence_json(self):
        text = '```json\n{"name": "test"}\n```'
        assert extract_json(text) == {"name": "test"}

    def test_code_fence_no_lang(self):
        text = '```\n{"name": "test"}\n```'
        # Without "json" label, falls through to bracket matching
        result = extract_json(text)
        assert result == {"name": "test"}

    def test_preamble_text(self):
        text = 'Here is the JSON:\n{"key": "value"}'
        assert extract_json(text) == {"key": "value"}

    def test_epilogue_text(self):
        text = '{"key": "value"}\nHope that helps!'
        assert extract_json(text) == {"key": "value"}

    def test_preamble_and_epilogue(self):
        text = 'Sure!\n{"key": "value"}\nLet me know!'
        assert extract_json(text) == {"key": "value"}

    def test_nested_objects(self):
        text = '{"outer": {"inner": [1, 2, 3]}}'
        result = extract_json(text)
        assert result == {"outer": {"inner": [1, 2, 3]}}

    def test_array_with_preamble(self):
        text = 'The results are:\n[{"id": 1}, {"id": 2}]'
        result = extract_json(text)
        assert result == [{"id": 1}, {"id": 2}]

    def test_jsonl(self):
        text = '{"a": 1}\n{"b": 2}\n{"c": 3}'
        result = extract_json(text)
        assert result == [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_jsonl_single_line_not_triggered(self):
        # Single line should not trigger JSONL mode
        text = '{"a": 1}'
        result = extract_json(text)
        assert result == {"a": 1}

    def test_whitespace_handling(self):
        text = '  \n  {"key": "value"}  \n  '
        assert extract_json(text) == {"key": "value"}

    def test_code_fence_with_extra_whitespace(self):
        text = '```json\n  {"key": "value"}  \n```'
        assert extract_json(text) == {"key": "value"}

    def test_multiple_code_fences_first_wins(self):
        text = '```json\n{"first": true}\n```\n```json\n{"second": true}\n```'
        assert extract_json(text) == {"first": True}

    def test_boolean_values(self):
        assert extract_json('{"flag": true}') == {"flag": True}
        assert extract_json('{"flag": false}') == {"flag": False}

    def test_null_values(self):
        assert extract_json('{"value": null}') == {"value": None}

    def test_numeric_types(self):
        result = extract_json('{"int": 42, "float": 3.14}')
        assert result == {"int": 42, "float": 3.14}


class TestInlineUnescapedQuoteSalvage:
    """A COMPLETE object whose string field contains an unescaped inline ``"``
    must round-trip without dropping trailing fields.

    Regression: ``pydantic_core.from_json(allow_partial="trailing-strings")``
    silently truncated such objects to their first field, so trailing fields
    (``score``/``needs_more_extraction``/...) were lost and iterative loops
    falsely "converged" on a fragment.
    """

    # Complete object: the ``memo`` value quotes document text verbatim with
    # an unescaped double-quote, and trailing fields follow.
    BAD = (
        '{"memo": "The clause says "shall remain in full force" and effect.", '
        '"score": 5, "needs_more_extraction": true}'
    )

    def test_inline_quote_preserves_all_fields(self):
        result = extract_json(self.BAD)
        assert result is not None
        assert set(result) == {"memo", "score", "needs_more_extraction"}
        assert result["score"] == 5
        assert result["needs_more_extraction"] is True
        assert "shall remain in full force" in result["memo"]

    def test_inline_quote_in_code_fence(self):
        text = f"```json\n{self.BAD}\n```"
        result = extract_json(text)
        assert result is not None
        assert set(result) == {"memo", "score", "needs_more_extraction"}
        assert result["needs_more_extraction"] is True

    def test_no_partial_truncation_to_first_field(self):
        # The historical bug returned exactly {"memo": "<prefix>"} and nothing
        # else. Guard against that specific regression.
        result = extract_json(self.BAD)
        assert result != {"memo": "The clause says "}

    def test_allow_partial_false_does_not_truncate_complete_object(self):
        # Even with partial recovery disabled, the repair salvage must recover
        # the complete object.
        result = extract_json(self.BAD, allow_partial=False)
        assert result is not None
        assert set(result) == {"memo", "score", "needs_more_extraction"}

    def test_genuine_truncation_still_recovered_when_allowed(self):
        # A genuinely truncated stream (cut off mid-string, no closing quote /
        # braces) must still be recovered by partial recovery.
        truncated = '{"memo": "The clause says shall remain in full force and ef'
        result = extract_json(truncated, allow_partial=True)
        assert result is not None
        assert "memo" in result

    def test_genuine_truncation_suppressed_when_partial_disabled(self):
        truncated = '{"memo": "The clause says shall remain in full force and ef'
        # With partial disabled and no structural close, nothing parses.
        assert extract_json(truncated, allow_partial=False) is None
