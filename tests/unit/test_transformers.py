"""Unit tests for JSON Schema transformers.

Covers the WS-TR.PR-1 additions:
- Extended `OpenAIJsonSchemaTransformer._STRIPPED_KEYS` (format, default,
  minItems, maxItems, plus the legacy numeric/string constraints)
- Canonicalization applied in transform()
- Recursion into anyOf/allOf/oneOf + $defs
- New `AnthropicJsonSchemaTransformer` with the same strip surface + strict
  semantics (`additionalProperties: false`, all-required)
- Google transformer canonicalization
"""

from __future__ import annotations

from kaos_llm_client.profiles import (
    AnthropicJsonSchemaTransformer,
    GoogleJsonSchemaTransformer,
    OpenAIJsonSchemaTransformer,
)


class TestOpenAIStripList:
    def test_strip_list_superset(self) -> None:
        """The WS-TR.PR-1 strip list must include the 4 new keys."""
        keys = OpenAIJsonSchemaTransformer._STRIPPED_KEYS
        # New (PR-1) additions:
        assert "format" in keys
        assert "default" in keys
        assert "minItems" in keys
        assert "maxItems" in keys
        # Existing (pre-PR-1) entries:
        assert "pattern" in keys
        assert "minLength" in keys
        assert "maximum" in keys
        assert "exclusiveMaximum" in keys

    def test_strips_all_unsupported(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": 40},
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "email": {"type": "string", "format": "email", "pattern": r"^\S+@\S+$"},
                "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "status": {"type": "string", "default": "pending"},
            },
        }
        out = OpenAIJsonSchemaTransformer(schema, strict=True).transform()
        props = out["properties"]
        assert "minLength" not in props["name"]
        assert "maxLength" not in props["name"]
        assert "minimum" not in props["age"]
        assert "maximum" not in props["age"]
        assert "format" not in props["email"]
        assert "pattern" not in props["email"]
        assert "minItems" not in props["tags"]
        assert "default" not in props["status"]

    def test_preserves_required_type_properties(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
            },
        }
        out = OpenAIJsonSchemaTransformer(schema, strict=True).transform()
        assert out["additionalProperties"] is False
        assert out["required"] == ["name"]
        assert out["properties"]["name"]["type"] == "string"

    def test_recurses_into_anyof(self) -> None:
        schema = {
            "anyOf": [
                {"type": "string", "pattern": "x.*"},
                {"type": "object", "properties": {"k": {"type": "string", "maxLength": 5}}},
            ]
        }
        out = OpenAIJsonSchemaTransformer(schema, strict=True).transform()
        # pattern stripped in first branch
        assert "pattern" not in out["anyOf"][0]
        # nested maxLength stripped
        assert "maxLength" not in out["anyOf"][1]["properties"]["k"]

    def test_recurses_into_defs(self) -> None:
        schema = {
            "type": "object",
            "properties": {"x": {"$ref": "#/$defs/Sub"}},
            "$defs": {
                "Sub": {"type": "object", "properties": {"y": {"type": "string", "pattern": ".*"}}},
            },
        }
        out = OpenAIJsonSchemaTransformer(schema, strict=True).transform()
        sub = out["$defs"]["Sub"]
        assert "pattern" not in sub["properties"]["y"]
        assert sub["additionalProperties"] is False
        assert sub["required"] == ["y"]

    def test_output_is_canonical(self) -> None:
        """Transform output has sorted keys at every level."""
        schema = {"type": "object", "properties": {"z": {"type": "string"}, "a": {"type": "int"}}}
        out = OpenAIJsonSchemaTransformer(schema, strict=True).transform()
        # Top-level keys sorted
        assert list(out.keys()) == sorted(out.keys())
        # Property keys sorted (lexicographic: a, z)
        assert list(out["properties"].keys()) == ["a", "z"]

    def test_non_strict_pass_through(self) -> None:
        schema = {"type": "string", "pattern": "x"}
        out = OpenAIJsonSchemaTransformer(schema, strict=False).transform()
        # Non-strict returns input as-is (pattern preserved)
        assert out == schema


class TestAnthropicTransformer:
    def test_strict_applies_strip_list(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1, "format": "email"},
            },
        }
        out = AnthropicJsonSchemaTransformer(schema, strict=True).transform()
        assert "minLength" not in out["properties"]["name"]
        assert "format" not in out["properties"]["name"]

    def test_strict_sets_additional_properties_false(self) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        out = AnthropicJsonSchemaTransformer(schema, strict=True).transform()
        assert out["additionalProperties"] is False

    def test_strict_makes_all_required(self) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "string"}, "y": {"type": "int"}}}
        out = AnthropicJsonSchemaTransformer(schema, strict=True).transform()
        assert set(out["required"]) == {"x", "y"}

    def test_output_is_canonical(self) -> None:
        schema = {"type": "object", "properties": {"z": {"type": "string"}, "a": {"type": "int"}}}
        out = AnthropicJsonSchemaTransformer(schema, strict=True).transform()
        assert list(out.keys()) == sorted(out.keys())

    def test_non_strict_pass_through(self) -> None:
        schema = {"type": "string", "pattern": "x"}
        out = AnthropicJsonSchemaTransformer(schema, strict=False).transform()
        assert out == schema


class TestGoogleTransformerCanonicalization:
    def test_output_is_canonical(self) -> None:
        """Google transformer now canonicalizes output for trace stability."""
        schema = {
            "type": "object",
            "properties": {"z": {"type": "string"}, "a": {"type": "integer"}},
        }
        out = GoogleJsonSchemaTransformer(schema).transform()
        assert list(out.keys()) == sorted(out.keys())
        assert list(out["properties"].keys()) == ["a", "z"]

    def test_still_inlines_refs(self) -> None:
        """Canonicalization doesn't break the ref-inlining behavior."""
        schema = {
            "type": "object",
            "properties": {"sub": {"$ref": "#/$defs/Inner"}},
            "$defs": {"Inner": {"type": "object", "properties": {"v": {"type": "string"}}}},
        }
        out = GoogleJsonSchemaTransformer(schema).transform()
        # $defs removed, ref resolved to an inline object.
        assert "$defs" not in out
        assert out["properties"]["sub"]["type"] == "object"
