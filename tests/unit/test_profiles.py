"""Tests for kaos_llm_client.profiles — profile resolution and schema transformers."""

from __future__ import annotations

import pytest

from kaos_llm_client.profiles import (
    ANTHROPIC_DEFAULT,
    GOOGLE_DEFAULT,
    OPENAI_COMPATIBLE_DEFAULT,
    OPENAI_DEFAULT,
    OPENAI_REASONING,
    XAI_DEFAULT,
    AnthropicModelProfile,
    GoogleJsonSchemaTransformer,
    GoogleModelProfile,
    JsonSchemaTransformer,
    ModelProfile,
    OpenAIJsonSchemaTransformer,
    OpenAIModelProfile,
    StructuredOutputMode,
    _resolve_anthropic_profile,
    _resolve_google_profile,
    _resolve_openai_profile,
    _resolve_xai_profile,
    infer_provider,
    resolve_profile,
)


class TestStructuredOutputMode:
    def test_values(self):
        assert StructuredOutputMode.TOOL == "tool"
        assert StructuredOutputMode.NATIVE == "native"
        assert StructuredOutputMode.PROMPTED == "prompted"


class TestModelProfile:
    def test_frozen(self):
        import dataclasses

        profile = ModelProfile()
        assert dataclasses.is_dataclass(profile)
        # frozen=True means direct attribute assignment raises FrozenInstanceError
        attr = "supports_tools"
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(profile, attr, False)

    def test_defaults(self):
        profile = ModelProfile()
        assert profile.supports_tools is True
        assert profile.supports_streaming is True
        assert profile.supports_vision is False
        assert profile.supports_thinking is False
        assert profile.max_tokens_field == "max_tokens"
        assert profile.system_prompt_location == "messages"
        assert profile.requires_max_tokens is False


class TestProviderProfiles:
    def test_openai_default(self):
        assert OPENAI_DEFAULT.supports_vision is True
        assert OPENAI_DEFAULT.supports_native_structured_output is True
        assert OPENAI_DEFAULT.default_structured_output_mode == StructuredOutputMode.NATIVE
        assert OPENAI_DEFAULT.max_tokens_field == "max_tokens"
        assert OPENAI_DEFAULT.json_schema_transformer is OpenAIJsonSchemaTransformer

    def test_openai_reasoning(self):
        assert OPENAI_REASONING.supports_thinking is True
        assert OPENAI_REASONING.thinking_parameter == "reasoning"
        assert OPENAI_REASONING.max_tokens_field == "max_completion_tokens"

    def test_anthropic_default(self):
        assert ANTHROPIC_DEFAULT.supports_vision is True
        assert ANTHROPIC_DEFAULT.supports_thinking is True
        assert ANTHROPIC_DEFAULT.system_prompt_location == "top_level"
        assert ANTHROPIC_DEFAULT.requires_max_tokens is True
        assert ANTHROPIC_DEFAULT.thinking_parameter == "thinking"
        assert ANTHROPIC_DEFAULT.stream_format == "anthropic_sse"
        # April 2026: Anthropic GA'd native `output_config.format` — default
        # flipped from TOOL fallback to NATIVE structured outputs.
        assert ANTHROPIC_DEFAULT.supports_native_structured_output is True
        assert ANTHROPIC_DEFAULT.default_structured_output_mode == StructuredOutputMode.NATIVE

    def test_google_default(self):
        assert GOOGLE_DEFAULT.supports_vision is True
        assert GOOGLE_DEFAULT.max_tokens_field == "maxOutputTokens"
        assert GOOGLE_DEFAULT.system_prompt_location == "system_instruction"
        assert GOOGLE_DEFAULT.stream_format == "google_sse"
        assert GOOGLE_DEFAULT.json_schema_transformer is GoogleJsonSchemaTransformer

    def test_openai_compatible_default(self):
        assert OPENAI_COMPATIBLE_DEFAULT.supports_vision is False
        assert OPENAI_COMPATIBLE_DEFAULT.supports_native_structured_output is False
        assert (
            OPENAI_COMPATIBLE_DEFAULT.default_structured_output_mode
            == StructuredOutputMode.PROMPTED
        )

    def test_xai_default(self):
        assert XAI_DEFAULT.supports_vision is True
        assert XAI_DEFAULT.provider_name == "xai"


class TestResolveProfile:
    def test_openai_provider(self):
        profile = resolve_profile("openai", "gpt-5")
        assert profile is OPENAI_DEFAULT

    def test_anthropic_provider(self):
        profile = resolve_profile("anthropic", "claude-sonnet-4-6")
        assert profile is ANTHROPIC_DEFAULT

    def test_google_provider(self):
        profile = resolve_profile("google", "gemini-2.0-flash")
        # Resolver returns a per-model variant (Gemini 2.0 caps at 8K
        # output, while GOOGLE_DEFAULT defaults to 65K for unknown models).
        # Identity check would fail since `.update()` produces a fresh
        # instance — assert the structural fields instead.
        assert profile.provider_name == "google"
        assert profile.default_max_tokens == 8_192

    def test_google_25_gets_thinking_profile(self):
        from kaos_llm_client.profiles import GOOGLE_THINKING

        profile = resolve_profile("google", "gemini-2.5-pro")
        assert profile is GOOGLE_THINKING
        assert profile.supports_thinking is True

    def test_xai_provider(self):
        profile = resolve_profile("xai", "grok-3")
        # Resolver returns a per-model variant (grok-3 caps at 16K
        # output). Identity check would fail since `.update()` produces
        # a fresh instance.
        assert profile.provider_name == "xai"
        assert profile.default_max_tokens == 16_384

    def test_unknown_provider_gets_compatible(self):
        profile = resolve_profile("unknown", "my-model")
        assert profile is OPENAI_COMPATIBLE_DEFAULT

    def test_o3_model_gets_reasoning_profile(self):
        profile = resolve_profile("openai", "o3-mini")
        assert profile is OPENAI_REASONING

    def test_o4_model_gets_reasoning_profile(self):
        profile = resolve_profile("openai", "o4-mini")
        assert profile is OPENAI_REASONING


class TestInferProvider:
    def test_gpt_prefix(self):
        assert infer_provider("gpt-5") == "openai"
        assert infer_provider("gpt-4.1-nano") == "openai"

    def test_claude_prefix(self):
        assert infer_provider("claude-sonnet-4-6") == "anthropic"

    def test_gemini_prefix(self):
        assert infer_provider("gemini-2.5-pro") == "google"

    def test_grok_prefix(self):
        assert infer_provider("grok-3") == "xai"

    def test_o_series(self):
        assert infer_provider("o3-mini") == "openai"
        assert infer_provider("o4-mini") == "openai"

    def test_unknown(self):
        assert infer_provider("my-custom-model") is None


class TestOpenAIModelProfile:
    def test_openai_model_profile_extra_fields(self):
        """OpenAIModelProfile has supports_reasoning_effort and other OpenAI-specific fields."""
        assert isinstance(OPENAI_DEFAULT, OpenAIModelProfile)
        assert OPENAI_DEFAULT.supports_reasoning_effort is False
        assert OPENAI_DEFAULT.supports_strict_mode is True
        assert OPENAI_DEFAULT.supports_response_format is True
        assert OPENAI_DEFAULT.supports_audio_output is False

    def test_openai_reasoning_has_reasoning_effort(self):
        assert isinstance(OPENAI_REASONING, OpenAIModelProfile)
        assert OPENAI_REASONING.supports_reasoning_effort is True

    def test_is_subclass_of_model_profile(self):
        assert isinstance(OPENAI_DEFAULT, ModelProfile)


class TestAnthropicModelProfile:
    def test_anthropic_model_profile_extra_fields(self):
        """AnthropicModelProfile has supports_prompt_caching and other Anthropic-specific fields."""
        assert isinstance(ANTHROPIC_DEFAULT, AnthropicModelProfile)
        assert ANTHROPIC_DEFAULT.supports_prompt_caching is True
        assert ANTHROPIC_DEFAULT.supports_extended_thinking is True
        assert ANTHROPIC_DEFAULT.max_thinking_budget == 32768
        assert ANTHROPIC_DEFAULT.supports_pdf_input is True
        assert ANTHROPIC_DEFAULT.anthropic_version == "2023-06-01"

    def test_is_subclass_of_model_profile(self):
        assert isinstance(ANTHROPIC_DEFAULT, ModelProfile)


class TestGoogleModelProfile:
    def test_google_model_profile_extra_fields(self):
        """GoogleModelProfile has Google-specific fields."""
        assert isinstance(GOOGLE_DEFAULT, GoogleModelProfile)
        assert GOOGLE_DEFAULT.supports_grounding is False
        assert GOOGLE_DEFAULT.supports_code_execution is False
        assert GOOGLE_DEFAULT.google_api_version == "v1beta"

    def test_is_subclass_of_model_profile(self):
        assert isinstance(GOOGLE_DEFAULT, ModelProfile)


class TestProfileUpdate:
    def test_profile_update(self):
        """update() returns new instance with replaced fields."""
        original = ModelProfile(supports_vision=False, provider_name="test")
        updated = original.update(supports_vision=True, default_max_tokens=8192)
        assert updated.supports_vision is True
        assert updated.default_max_tokens == 8192
        assert updated.provider_name == "test"
        # Original is unchanged (frozen). Default max_tokens follows the
        # 2026 frontier-model floor; bumped from 4096 alongside the
        # provider-specific profiles so a deliverable doesn't truncate
        # at a Claude-2-era ceiling.
        assert original.supports_vision is False
        assert original.default_max_tokens == 100_000

    def test_update_returns_same_type(self):
        """update() on a subclass returns the same subclass type."""
        original = OpenAIModelProfile(provider_name="openai")
        updated = original.update(supports_reasoning_effort=True)
        assert isinstance(updated, OpenAIModelProfile)
        assert updated.supports_reasoning_effort is True


class TestResolverFunctions:
    def test_resolve_openai_gpt(self):
        """OpenAI resolver returns OPENAI_DEFAULT for gpt models."""
        assert _resolve_openai_profile("gpt-5") is OPENAI_DEFAULT

    def test_resolve_openai_reasoning(self):
        """OpenAI resolver returns OPENAI_REASONING for o-series models."""
        assert _resolve_openai_profile("o3-mini") is OPENAI_REASONING
        assert _resolve_openai_profile("o4-mini") is OPENAI_REASONING
        assert _resolve_openai_profile("o1-preview") is OPENAI_REASONING

    def test_resolve_anthropic(self):
        """Anthropic resolver always returns ANTHROPIC_DEFAULT."""
        assert _resolve_anthropic_profile("claude-sonnet-4-6") is ANTHROPIC_DEFAULT

    def test_resolve_google_default(self):
        """Google resolver returns Gemini-2.0-tuned profile for non-thinking models."""
        # 2.0 ceilings at 8K output; resolver returns
        # GOOGLE_DEFAULT.update(default_max_tokens=8192).
        profile = _resolve_google_profile("gemini-2.0-flash")
        assert profile.provider_name == "google"
        assert profile.default_max_tokens == 8_192

    def test_resolve_google_thinking(self):
        """Google resolver returns GOOGLE_THINKING for 2.5+ models."""
        from kaos_llm_client.profiles import GOOGLE_THINKING

        assert _resolve_google_profile("gemini-2.5-pro") is GOOGLE_THINKING
        assert _resolve_google_profile("gemini-3-ultra") is GOOGLE_THINKING

    def test_resolve_xai_default(self):
        """xAI resolver returns grok-3-tuned profile for non-grok-4 models."""
        profile = _resolve_xai_profile("grok-3")
        assert profile.provider_name == "xai"
        assert profile.default_max_tokens == 16_384

    def test_resolve_xai_grok4(self):
        """xAI resolver returns XAI_GROK4 for grok-4 models."""
        from kaos_llm_client.profiles import XAI_GROK4

        assert _resolve_xai_profile("grok-4") is XAI_GROK4


class TestJsonSchemaTransformer:
    def test_base_passthrough(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        t = JsonSchemaTransformer(schema)
        assert t.transform() == schema

    def test_openai_non_strict_passthrough(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        t = OpenAIJsonSchemaTransformer(schema, strict=False)
        assert t.transform() == schema

    def test_openai_strict_adds_required(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        t = OpenAIJsonSchemaTransformer(schema, strict=True)
        result = t.transform()
        assert result["additionalProperties"] is False
        assert set(result["required"]) == {"name", "age"}

    def test_openai_strict_removes_unsupported(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": 100},
            },
        }
        t = OpenAIJsonSchemaTransformer(schema, strict=True)
        result = t.transform()
        name_schema = result["properties"]["name"]
        assert "minLength" not in name_schema
        assert "maxLength" not in name_schema

    def test_openai_strict_nested_objects(self):
        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
            },
        }
        t = OpenAIJsonSchemaTransformer(schema, strict=True)
        result = t.transform()
        nested = result["properties"]["nested"]
        assert nested["additionalProperties"] is False
        assert nested["required"] == ["x"]

    def test_openai_strict_array_items(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                    },
                },
            },
        }
        t = OpenAIJsonSchemaTransformer(schema, strict=True)
        result = t.transform()
        item_schema = result["properties"]["items"]["items"]
        assert item_schema["additionalProperties"] is False


class TestGoogleJsonSchemaTransformer:
    """Tests for GoogleJsonSchemaTransformer."""

    def test_const_to_enum(self):
        schema = {
            "type": "object",
            "properties": {
                "status": {"const": "active"},
            },
        }
        t = GoogleJsonSchemaTransformer(schema)
        result = t.transform()
        prop = result["properties"]["status"]
        assert "const" not in prop
        assert prop["enum"] == ["active"]

    def test_title_stripped(self):
        schema = {
            "type": "object",
            "title": "MyModel",
            "properties": {
                "name": {"type": "string", "title": "Full Name"},
            },
        }
        t = GoogleJsonSchemaTransformer(schema)
        result = t.transform()
        assert "title" not in result
        assert "title" not in result["properties"]["name"]

    def test_format_to_description(self):
        schema = {
            "type": "object",
            "properties": {
                "email": {"type": "string", "format": "email"},
            },
        }
        t = GoogleJsonSchemaTransformer(schema)
        result = t.transform()
        prop = result["properties"]["email"]
        assert "format" not in prop
        assert "Format: email" in prop["description"]

    def test_format_appends_to_existing_description(self):
        schema = {
            "type": "object",
            "properties": {
                "ts": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Timestamp of the event",
                },
            },
        }
        t = GoogleJsonSchemaTransformer(schema)
        result = t.transform()
        desc = result["properties"]["ts"]["description"]
        assert "Timestamp of the event" in desc
        assert "Format: date-time" in desc

    def test_default_stripped(self):
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "default": 10},
            },
        }
        t = GoogleJsonSchemaTransformer(schema)
        result = t.transform()
        assert "default" not in result["properties"]["count"]

    def test_ref_inlined(self):
        schema = {
            "type": "object",
            "$defs": {
                "Address": {
                    "type": "object",
                    "title": "Address",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                },
            },
            "properties": {
                "home": {"$ref": "#/$defs/Address"},
                "work": {"$ref": "#/$defs/Address"},
            },
        }
        t = GoogleJsonSchemaTransformer(schema)
        result = t.transform()

        # $defs should be gone
        assert "$defs" not in result
        assert "definitions" not in result

        # Both properties should be inlined objects
        home = result["properties"]["home"]
        assert home["type"] == "object"
        assert "street" in home["properties"]
        # title should be stripped from inlined defs too
        assert "title" not in home

        work = result["properties"]["work"]
        assert work["type"] == "object"
        assert "city" in work["properties"]

    def test_ref_with_sibling_description_override(self):
        schema = {
            "type": "object",
            "$defs": {
                "Color": {
                    "type": "string",
                    "enum": ["red", "green", "blue"],
                    "description": "Original description",
                },
            },
            "properties": {
                "favorite": {
                    "$ref": "#/$defs/Color",
                    "description": "Overridden description",
                },
            },
        }
        t = GoogleJsonSchemaTransformer(schema)
        result = t.transform()
        prop = result["properties"]["favorite"]
        assert prop["description"] == "Overridden description"
        assert prop["enum"] == ["red", "green", "blue"]

    def test_nested_ref_inlined(self):
        schema = {
            "type": "object",
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                },
                "Outer": {
                    "type": "object",
                    "properties": {"inner": {"$ref": "#/$defs/Inner"}},
                },
            },
            "properties": {
                "data": {"$ref": "#/$defs/Outer"},
            },
        }
        t = GoogleJsonSchemaTransformer(schema)
        result = t.transform()
        data = result["properties"]["data"]
        assert data["type"] == "object"
        inner = data["properties"]["inner"]
        assert inner["type"] == "object"
        assert "value" in inner["properties"]

    def test_does_not_mutate_input(self):
        schema = {
            "type": "object",
            "title": "Test",
            "properties": {
                "x": {"type": "string", "const": "fixed", "title": "X"},
            },
        }
        import copy

        original = copy.deepcopy(schema)
        GoogleJsonSchemaTransformer(schema).transform()
        assert schema == original

    def test_array_items_cleaned(self):
        schema = {
            "type": "array",
            "title": "Items",
            "items": {
                "type": "object",
                "title": "Item",
                "properties": {
                    "id": {"type": "integer", "format": "int64"},
                },
            },
        }
        t = GoogleJsonSchemaTransformer(schema)
        result = t.transform()
        assert "title" not in result
        assert "title" not in result["items"]
        assert "format" not in result["items"]["properties"]["id"]
