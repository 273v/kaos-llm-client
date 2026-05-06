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
