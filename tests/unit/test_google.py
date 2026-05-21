"""Provider contract tests for the Google Gemini client.

Tests ``_build_request()`` and ``_parse_response()`` directly — no HTTP calls.
Covers both AI Studio and Vertex AI modes.
"""

from __future__ import annotations

import pytest

from kaos_llm_client.errors import KaosLLMAuthError
from kaos_llm_client.profiles import GoogleJsonSchemaTransformer
from kaos_llm_client.providers.google import GoogleClient, _tool_def_to_google
from kaos_llm_client.settings import KaosLLMSettings
from kaos_llm_client.types import (
    ProviderRequest,
    ProviderResponse,
    ToolDefinition,
)

_VERTEX_BASE_URL = "https://us-central1-aiplatform.googleapis.com"


def _make_client(model: str = "gemini-2.5-pro") -> GoogleClient:
    """Create a Google client with a test key (no settings resolution)."""
    return GoogleClient(model=model, api_key="test-key")


def _make_vertex_client(
    model: str = "gemini-2.5-pro",
    *,
    project: str = "my-project",
    location: str = "us-central1",
) -> GoogleClient:
    """Create a Vertex AI client with a test token."""
    settings = KaosLLMSettings(google_project=project, google_location=location)
    return GoogleClient(
        model=model,
        api_key="ya29.test-token",
        base_url=_VERTEX_BASE_URL,
        settings=settings,
    )


def _make_request(request_id: str = "req-test") -> ProviderRequest:
    """Create a minimal ProviderRequest for parse tests."""
    return ProviderRequest(
        provider="google",
        model="gemini-2.5-pro",
        endpoint="/v1beta/models/gemini-2.5-pro:generateContent",
        body={},
        request_id=request_id,
    )


class TestGoogleBuildRequest:
    """Tests for GoogleClient._build_request()."""

    def test_build_request_basic(self):
        client = _make_client()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you?"},
        ]
        req = client._build_request(messages)

        assert req.provider == "google"
        assert "contents" in req.body

        contents = req.body["contents"]
        assert len(contents) == 3

        # "assistant" mapped to "model"
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"] == [{"text": "hello"}]
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"] == [{"text": "hi there"}]
        assert contents[2]["role"] == "user"
        assert contents[2]["parts"] == [{"text": "how are you?"}]

    def test_system_instruction_extraction(self):
        client = _make_client()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]
        req = client._build_request(messages)

        # System message extracted to top-level systemInstruction
        assert "systemInstruction" in req.body
        si = req.body["systemInstruction"]
        assert si["parts"] == [{"text": "You are helpful."}]

        # Only the user message remains in contents
        assert len(req.body["contents"]) == 1
        assert req.body["contents"][0]["role"] == "user"

    def test_generation_config(self):
        client = _make_client()
        messages = [{"role": "user", "content": "hi"}]
        req = client._build_request(messages)

        # generationConfig should have maxOutputTokens
        assert "generationConfig" in req.body
        gc = req.body["generationConfig"]
        assert gc["maxOutputTokens"] == client.profile.default_max_tokens

        # Explicit override via max_tokens kwarg
        req2 = client._build_request(messages, max_tokens=2048)
        assert req2.body["generationConfig"]["maxOutputTokens"] == 2048


class TestGoogleParseResponse:
    """Tests for GoogleClient._parse_response()."""

    def test_parse_response_text(self):
        client = _make_client()
        raw = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "hello"}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 1,
                "totalTokenCount": 6,
            },
        }
        request = _make_request()
        resp = client._parse_response(raw, request)

        assert isinstance(resp, ProviderResponse)
        assert resp.text == "hello"
        assert resp.stop_reason == "STOP"
        assert resp.usage.input_tokens == 5
        assert resp.usage.output_tokens == 1
        assert resp.usage.total_tokens == 6
        assert resp.request_id == "req-test"


class TestGoogleHeaders:
    """Tests for GoogleClient._build_headers()."""

    def test_headers_x_goog_api_key(self):
        client = _make_client()
        headers = client._build_headers()

        assert headers["x-goog-api-key"] == "test-key"
        assert headers["Content-Type"] == "application/json"
        # Google does NOT use Authorization Bearer
        assert "Authorization" not in headers


class TestGoogleEndpoint:
    """Tests for GoogleClient endpoint construction."""

    def test_endpoint_includes_model(self):
        client = _make_client(model="gemini-2.5-pro")
        endpoint = client._default_endpoint()

        assert "gemini-2.5-pro" in endpoint
        assert endpoint == "/v1beta/models/gemini-2.5-pro:generateContent"


# ---------------------------------------------------------------------------
# Vertex AI mode tests
# ---------------------------------------------------------------------------


class TestVertexDetection:
    """Tests for _is_vertex property."""

    def test_ai_studio_not_vertex(self):
        client = _make_client()
        assert client._is_vertex is False

    def test_vertex_base_url_detected(self):
        client = _make_vertex_client()
        assert client._is_vertex is True

    def test_vertex_regional_url_detected(self):
        settings = KaosLLMSettings(google_project="p", google_location="europe-west4")
        client = GoogleClient(
            model="gemini-2.5-pro",
            api_key="tok",
            base_url="https://europe-west4-aiplatform.googleapis.com",
            settings=settings,
        )
        assert client._is_vertex is True


class TestVertexEndpoint:
    """Tests for Vertex AI endpoint construction."""

    def test_default_endpoint(self):
        client = _make_vertex_client(project="my-gcp-project", location="us-central1")
        endpoint = client._default_endpoint()

        assert endpoint == (
            "/v1/projects/my-gcp-project/locations/us-central1"
            "/publishers/google/models/gemini-2.5-pro:generateContent"
        )

    def test_stream_endpoint(self):
        client = _make_vertex_client(project="proj", location="europe-west4")
        endpoint = client._stream_endpoint()

        assert endpoint == (
            "/v1/projects/proj/locations/europe-west4"
            "/publishers/google/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
        )

    def test_missing_project_raises(self):
        settings = KaosLLMSettings(google_project=None)
        client = GoogleClient(
            model="gemini-2.5-pro",
            api_key="tok",
            base_url=_VERTEX_BASE_URL,
            settings=settings,
        )
        with pytest.raises(KaosLLMAuthError, match="project ID"):
            client._default_endpoint()


class TestVertexHeaders:
    """Tests for Vertex AI header construction."""

    def test_bearer_token(self):
        client = _make_vertex_client()
        headers = client._build_headers()

        assert headers["Authorization"] == "Bearer ya29.test-token"
        assert headers["Content-Type"] == "application/json"
        # Vertex does NOT use x-goog-api-key
        assert "x-goog-api-key" not in headers

    def test_ai_studio_does_not_use_bearer(self):
        client = _make_client()
        headers = client._build_headers()

        assert "Authorization" not in headers
        assert "x-goog-api-key" in headers


class TestVertexBuildRequest:
    """Tests that Vertex request building uses the correct endpoint."""

    def test_non_streaming_request_uses_vertex_endpoint(self):
        client = _make_vertex_client(project="p1", location="us-east1")
        messages = [{"role": "user", "content": "hello"}]
        req = client._build_request(messages)

        assert req.endpoint == (
            "/v1/projects/p1/locations/us-east1"
            "/publishers/google/models/gemini-2.5-pro:generateContent"
        )
        # Body structure should be identical to AI Studio
        assert "contents" in req.body

    def test_streaming_request_uses_vertex_endpoint(self):
        client = _make_vertex_client(project="p1", location="us-east1")
        messages = [{"role": "user", "content": "hello"}]
        req = client._build_request(messages, stream=True)

        assert "streamGenerateContent?alt=sse" in req.endpoint
        assert "/v1/projects/p1/locations/us-east1/" in req.endpoint


# ---------------------------------------------------------------------------
# Tool dispatch tests (R0.3 — kaos-llm-client 0.1.1 / reliability roadmap #560)
#
# Without the schema transformer applied to tool parameter blocks, Google
# Gemini's ``generateContent`` returns HTTP 400 on every tool turn whenever a
# tool declares its parameters with ``$ref``/``$defs`` (the default Pydantic /
# JSONSchema shape for any nested model) or with ``const``/``default``/``title``
# keywords (also common). This was the root cause behind both Gemini Pro and
# Flash being unusable for tool-using legal research in the SPA — confirmed by
# the worker-honesty audit (kaos-modules/docs/audits/2026-05-21-worker-honesty.md).
# ---------------------------------------------------------------------------


class TestGoogleToolDispatch:
    """Tests that tool parameter blocks are sanitized for Gemini's strict JSONSchema subset."""

    def test_tool_def_to_google_without_transformer_leaves_schema_untouched(self):
        """When no transformer is provided, the legacy (raw) shape passes through."""
        tool = ToolDefinition(
            name="search",
            description="Search the corpus.",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string", "title": "Query"}},
            },
        )

        decl = _tool_def_to_google(tool)

        assert decl["name"] == "search"
        assert decl["description"] == "Search the corpus."
        # Without a transformer, ``title`` (which Gemini rejects) survives.
        assert decl["parameters"]["properties"]["q"]["title"] == "Query"

    def test_tool_def_to_google_inlines_refs_and_defs(self):
        """``$ref`` + ``$defs`` are inlined so Gemini can validate the schema."""
        tool = ToolDefinition(
            name="ingest_entity",
            description="Ingest a structured legal entity.",
            parameters={
                "type": "object",
                "$defs": {
                    "Address": {
                        "type": "object",
                        "title": "Address",
                        "properties": {
                            "street": {"type": "string"},
                            "city": {"type": "string"},
                        },
                    }
                },
                "properties": {
                    "name": {"type": "string"},
                    "address": {"$ref": "#/$defs/Address"},
                },
            },
        )

        decl = _tool_def_to_google(tool, schema_transformer=GoogleJsonSchemaTransformer)

        params = decl["parameters"]
        # $ref and $defs must be inlined / stripped.
        assert "$defs" not in params
        assert "$ref" not in params["properties"]["address"]
        # The referenced object's structure must be present in-place.
        address = params["properties"]["address"]
        assert address["type"] == "object"
        assert "street" in address["properties"]
        assert "city" in address["properties"]
        # ``title`` (Gemini-rejected) must be stripped on both root and nested.
        assert "title" not in address

    def test_tool_def_to_google_strips_gemini_unsupported_keywords(self):
        """``title``, ``const``, ``default`` are stripped / rewritten for Gemini."""
        tool = ToolDefinition(
            name="set_mode",
            description="Switch operational mode.",
            parameters={
                "type": "object",
                "title": "SetModeParams",
                "properties": {
                    "mode": {
                        "type": "string",
                        "const": "active",  # Gemini rejects const; transformer rewrites to enum.
                        "title": "Mode",
                        "default": "active",  # Gemini rejects default; transformer strips it.
                    }
                },
            },
        )

        decl = _tool_def_to_google(tool, schema_transformer=GoogleJsonSchemaTransformer)
        params = decl["parameters"]
        mode = params["properties"]["mode"]

        # const -> enum: [value]
        assert "const" not in mode
        assert mode.get("enum") == ["active"]
        # default stripped
        assert "default" not in mode
        # title stripped (both root and nested)
        assert "title" not in mode
        assert "title" not in params

    def test_build_request_applies_transformer_to_tool_parameters(self):
        """End-to-end: ``GoogleClient._build_request`` sanitizes tool schemas via the profile."""
        client = _make_client(model="gemini-2.5-pro")
        # Sanity-check the profile actually configures the Gemini transformer.
        assert client.profile.json_schema_transformer is GoogleJsonSchemaTransformer

        tool = ToolDefinition(
            name="lookup",
            description="Lookup a record.",
            parameters={
                "type": "object",
                "$defs": {
                    "Filter": {
                        "type": "object",
                        "title": "Filter",
                        "properties": {"k": {"type": "string", "default": "id"}},
                    }
                },
                "properties": {
                    "filter": {"$ref": "#/$defs/Filter"},
                    "limit": {"type": "integer", "default": 10},
                },
            },
        )

        req = client._build_request(
            [{"role": "user", "content": "find x"}],
            tools=[tool],
        )

        assert "tools" in req.body
        decls = req.body["tools"][0]["functionDeclarations"]
        assert len(decls) == 1
        params = decls[0]["parameters"]

        # The transformer must have flowed all the way through.
        assert "$defs" not in params
        assert "$ref" not in params["properties"]["filter"]
        assert "default" not in params["properties"]["limit"]
        # Nested keywords cleaned too.
        nested = params["properties"]["filter"]
        assert "title" not in nested
        assert "default" not in nested["properties"]["k"]

    def test_build_request_without_tools_omits_tools_block(self):
        """Tools block must not appear when no tools are passed."""
        client = _make_client()
        req = client._build_request([{"role": "user", "content": "hi"}])
        assert "tools" not in req.body

    def test_build_request_with_non_gemini_profile_skips_transformer(self):
        """Profiles without a transformer (None) must not crash and must pass schema verbatim."""
        # We pass schema_transformer=None directly to the helper — this is
        # the path a profile without ``json_schema_transformer`` would take.
        tool = ToolDefinition(
            name="noop",
            description="No-op tool.",
            parameters={
                "type": "object",
                "title": "NoopParams",  # survives because no transformer
                "properties": {"x": {"type": "string", "title": "X"}},
            },
        )

        decl = _tool_def_to_google(tool, schema_transformer=None)
        # The raw schema (with ``title``) must survive untouched.
        assert decl["parameters"]["title"] == "NoopParams"
        assert decl["parameters"]["properties"]["x"]["title"] == "X"
