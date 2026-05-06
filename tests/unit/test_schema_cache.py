"""Unit tests for kaos_llm_client.schema_cache — canonicalize + schema_hash."""

from __future__ import annotations

from kaos_llm_client.schema_cache import canonicalize, schema_hash


class TestCanonicalize:
    def test_sorts_dict_keys(self) -> None:
        inp = {"b": 1, "a": 2, "c": 3}
        out = canonicalize(inp)
        assert list(out.keys()) == ["a", "b", "c"]
        assert out == {"a": 2, "b": 1, "c": 3}

    def test_sorts_nested_dicts(self) -> None:
        inp = {"outer": {"z": 1, "y": 2, "x": 3}}
        out = canonicalize(inp)
        assert list(out["outer"].keys()) == ["x", "y", "z"]

    def test_preserves_list_order(self) -> None:
        """Arrays are semantic in JSON Schema — must NOT be reordered."""
        inp = {"required": ["name", "age", "email"]}
        out = canonicalize(inp)
        assert out["required"] == ["name", "age", "email"]

    def test_preserves_list_order_of_sub_schemas(self) -> None:
        inp = {"allOf": [{"type": "string"}, {"type": "object"}]}
        out = canonicalize(inp)
        assert out["allOf"][0] == {"type": "string"}
        assert out["allOf"][1] == {"type": "object"}

    def test_canonicalizes_items_in_list(self) -> None:
        inp = {"anyOf": [{"b": 1, "a": 2}, {"y": 1, "x": 2}]}
        out = canonicalize(inp)
        assert list(out["anyOf"][0].keys()) == ["a", "b"]
        assert list(out["anyOf"][1].keys()) == ["x", "y"]

    def test_does_not_mutate_input(self) -> None:
        inp = {"b": 1, "a": 2}
        canonicalize(inp)
        # Input still has insertion order
        assert list(inp.keys()) == ["b", "a"]

    def test_passes_through_primitives(self) -> None:
        assert canonicalize("hello") == "hello"
        assert canonicalize(42) == 42
        assert canonicalize(3.14) == 3.14
        assert canonicalize(True) is True
        assert canonicalize(None) is None

    def test_full_json_schema_shape(self) -> None:
        inp = {
            "type": "object",
            "properties": {
                "effective_date": {"type": "string", "format": "date"},
                "parties": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["effective_date", "parties"],
        }
        out = canonicalize(inp)
        # Top-level keys sorted.
        assert list(out.keys()) == ["properties", "required", "type"]
        # Properties sorted.
        assert list(out["properties"].keys()) == ["effective_date", "parties"]
        # required array unchanged.
        assert out["required"] == ["effective_date", "parties"]


class TestSchemaHash:
    def test_stable_across_key_orderings(self) -> None:
        a = {"type": "object", "properties": {"x": 1, "y": 2}, "required": ["x", "y"]}
        b = {"required": ["x", "y"], "properties": {"y": 2, "x": 1}, "type": "object"}
        assert schema_hash(a) == schema_hash(b)

    def test_different_required_order_differs(self) -> None:
        """Array order changes meaning → different hash."""
        a = {"required": ["x", "y"]}
        b = {"required": ["y", "x"]}
        assert schema_hash(a) != schema_hash(b)

    def test_different_types_differ(self) -> None:
        a = {"type": "object"}
        b = {"type": "array"}
        assert schema_hash(a) != schema_hash(b)

    def test_hex_length_is_64(self) -> None:
        h = schema_hash({"type": "object"})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_works_on_primitives(self) -> None:
        # Edge case: schema is just a boolean (valid in JSON Schema).
        assert len(schema_hash(True)) == 64
        assert schema_hash(True) != schema_hash(False)

    def test_unicode_safe(self) -> None:
        schema = {"description": "café déjà vu"}
        h = schema_hash(schema)
        assert len(h) == 64
        # Same schema always hashes same.
        assert h == schema_hash({"description": "café déjà vu"})
