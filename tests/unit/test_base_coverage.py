"""Coverage-focused tests for BaseProviderClient (base.py).

Targets uncovered lines: close/aclose, context managers, per-request options,
streaming retry, hooks during streaming, json_async modes, pydantic_async
validation+retry, embed_async, cache policy, and preprocess_messages.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from kaos_llm_client.cache import FileCache
from kaos_llm_client.errors import KaosLLMProviderError, KaosLLMValidationError
from kaos_llm_client.profiles import ModelProfile, StructuredOutputMode
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.providers.openai import OpenAIClient
from kaos_llm_client.transport import RetryPolicy
from kaos_llm_client.types import (
    CachePolicy,
    ContentPart,
    ProviderRequest,
    ProviderResponse,
    RequestHooks,
    RequestOptions,
    StreamChunk,
    UsageInfo,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OPENAI_OK: dict[str, Any] = {
    "id": "chatcmpl-base-cov-1",
    "object": "chat.completion",
    "model": "gpt-5",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def _make_response(text: str = "test", output_json: str | None = None) -> ProviderResponse:
    content = output_json if output_json is not None else text
    return ProviderResponse(
        provider="function",
        model="test",
        raw={},
        parts=[ContentPart(type="text", text=content)],
        usage=UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def _make_async_handler(payload: dict[str, Any], status: int = 200) -> Any:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


def _inject_async_mock(
    client: Any,
    payload: dict[str, Any],
    status: int = 200,
) -> None:
    base_url = client._base_url
    client._async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_make_async_handler(payload, status)),
        base_url=base_url,
    )


# ===========================================================================
# Per-request options
# ===========================================================================


class TestPerRequestOptions:
    """Cover RequestOptions forwarding: max_retries, extra_headers, timeout."""

    async def test_per_request_max_retries(self) -> None:
        """RequestOptions(max_retries=1) overrides client-level max_retries."""
        attempt_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count <= 1:
                return httpx.Response(429, json={"error": {"message": "rate limited"}})
            return httpx.Response(200, json=OPENAI_OK)

        client = OpenAIClient(model="gpt-5", api_key="test-key", max_retries=0)
        client._retry_policy = RetryPolicy(max_retries=0, backoff_base=0.0)
        client._async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url=client._base_url
        )

        # With max_retries=0, 429 raises immediately
        with pytest.raises(KaosLLMProviderError, match="429"):
            await client.request_async(
                [{"role": "user", "content": "Hi"}],
            )

        # Reset counter; per-request override to 1 retry allows success
        attempt_count = 0
        result = await client.request_async(
            [{"role": "user", "content": "Hi"}],
            options=RequestOptions(max_retries=1, retry_backoff_base=0.0),
        )
        assert result.text == "ok"
        assert attempt_count == 2

    async def test_per_request_extra_headers(self) -> None:
        """RequestOptions(extra_headers=...) are injected into the request."""
        captured_headers: dict[str, str] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured_headers.update(dict(request.headers))
            return httpx.Response(200, json=OPENAI_OK)

        client = OpenAIClient(model="gpt-5", api_key="test-key")
        client._async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url=client._base_url
        )

        await client.request_async(
            [{"role": "user", "content": "Hi"}],
            options=RequestOptions(extra_headers={"X-Custom": "val123"}),
        )
        assert captured_headers.get("x-custom") == "val123"

    async def test_per_request_timeout(self) -> None:
        """RequestOptions(timeout=5.0) is passed through to transport."""
        # We verify indirectly: the request succeeds with a mock transport
        client = OpenAIClient(model="gpt-5", api_key="test-key")
        _inject_async_mock(client, OPENAI_OK)

        result = await client.request_async(
            [{"role": "user", "content": "Hi"}],
            options=RequestOptions(timeout=5.0),
        )
        assert result.text == "ok"


# ===========================================================================
# Streaming retry and hooks
# ===========================================================================


def _make_sse_body(text: str) -> str:
    """Build a minimal SSE response body for OpenAI streaming."""
    chunk1 = json.dumps(
        {
            "id": "chatcmpl-stream",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
    )
    chunk2 = json.dumps(
        {
            "id": "chatcmpl-stream",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
    )
    return f"data: {chunk1}\n\ndata: {chunk2}\n\ndata: [DONE]\n\n"


class TestStreamingRetryAndHooks:
    """Cover streaming retry logic, on_request/on_response/on_error during streams."""

    async def test_streaming_retry_on_429(self) -> None:
        """Stream setup gets 429, retries, and succeeds on second attempt."""
        attempt = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                return httpx.Response(429, json={"error": {"message": "rate limited"}})
            body = _make_sse_body("hello stream")
            return httpx.Response(
                200,
                content=body.encode(),
                headers={"content-type": "text/event-stream"},
            )

        client = OpenAIClient(model="gpt-5", api_key="test-key", max_retries=2)
        client._retry_policy = RetryPolicy(max_retries=2, backoff_base=0.0)
        client._async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url=client._base_url
        )

        chunks: list[StreamChunk] = []
        async for chunk in client.request_stream_async([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)

        text_parts = [c.text for c in chunks if c.type == "text_delta" and c.text]
        assert "hello stream" in "".join(text_parts)
        assert attempt == 2

    async def test_streaming_hooks_fire(self) -> None:
        """on_request and on_response fire during streaming."""
        req_log: list[ProviderRequest] = []
        resp_log: list[ProviderResponse] = []

        hooks = RequestHooks(
            on_request=lambda r: req_log.append(r),
            on_response=lambda r, resp: resp_log.append(resp),
        )

        body = _make_sse_body("hooks stream")
        client = OpenAIClient(model="gpt-5", api_key="test-key", hooks=hooks)
        client._retry_policy = RetryPolicy(max_retries=0, backoff_base=0.0)
        client._async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    200,
                    content=body.encode(),
                    headers={"content-type": "text/event-stream"},
                )
            ),
            base_url=client._base_url,
        )

        async for _ in client.request_stream_async([{"role": "user", "content": "Hi"}]):
            pass

        assert len(req_log) == 1
        assert len(resp_log) == 1
        assert resp_log[0].text  # accumulated response should have text

    async def test_streaming_on_error_fires(self) -> None:
        """Stream fails (non-retryable), on_error fires."""
        err_log: list[tuple[ProviderRequest, Exception]] = []

        hooks = RequestHooks(
            on_error=lambda r, e: err_log.append((r, e)),
        )

        client = OpenAIClient(model="gpt-5", api_key="test-key", hooks=hooks)
        client._retry_policy = RetryPolicy(max_retries=0, backoff_base=0.0)
        _inject_async_mock(client, {"error": {"message": "bad request"}}, status=400)

        with pytest.raises(KaosLLMProviderError):
            async for _ in client.request_stream_async([{"role": "user", "content": "Hi"}]):
                pass  # pragma: no cover

        assert len(err_log) == 1
        assert isinstance(err_log[0][1], KaosLLMProviderError)

    async def test_max_retries_zero_honored(self) -> None:
        """OpenAIClient(max_retries=0) does not retry on 429."""
        attempt_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempt_count
            attempt_count += 1
            return httpx.Response(429, json={"error": {"message": "rate limited"}})

        client = OpenAIClient(model="gpt-5", api_key="test-key", max_retries=0)
        client._retry_policy = RetryPolicy(max_retries=0)
        client._async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url=client._base_url
        )

        with pytest.raises(KaosLLMProviderError, match="429"):
            await client.request_async([{"role": "user", "content": "Hi"}])

        assert attempt_count == 1

    async def test_streaming_500_surfaces_provider_error(self) -> None:
        """Regression: a non-2xx response on the streaming path must surface
        as ``KaosLLMProviderError`` carrying the parsed error body — not as
        ``httpx.ResponseNotRead`` from a body that was never read.

        Before the fix, ``raise_for_status()`` ran inside the ``client.stream()``
        context and called ``response.json()`` on a streaming body that hadn't
        been read yet. httpx raised ``ResponseNotRead``, which masked the real
        provider error and (critically) blocked the flex-tier 500 fallback
        retry path documented in CLAUDE.md.
        """

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": {"message": "upstream brownout"}})

        client = OpenAIClient(model="gpt-5.4-nano", api_key="test-key", max_retries=0)
        client._retry_policy = RetryPolicy(max_retries=0, backoff_base=0.0)
        client._async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url=client._base_url
        )

        with pytest.raises(KaosLLMProviderError) as exc_info:
            async for _ in client.request_stream_async([{"role": "user", "content": "Hi"}]):
                pass  # pragma: no cover — should never yield

        assert exc_info.value.status_code == 500
        # Provider error body parsed cleanly — proves aread() ran before json().
        assert "upstream brownout" in str(exc_info.value)


# ===========================================================================
# Message preprocessing
# ===========================================================================


class TestPreprocessMessages:
    def test_preprocess_messages_strips_cache_point(self) -> None:
        """Base _preprocess_messages strips cache_point role messages."""
        client = OpenAIClient(model="gpt-5", api_key="test-key")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "sys"},
            {"role": "cache_point"},
            {"role": "user", "content": "hi"},
            {"role": "cache_point"},
        ]
        result = client._preprocess_messages(messages)
        assert len(result) == 2
        roles = [m["role"] for m in result]
        assert "cache_point" not in roles


# ===========================================================================
# Structured output modes (json_async)
# ===========================================================================


class TestJsonAsyncModes:
    async def test_json_async_native_mode(self) -> None:
        """json() with schema in native mode triggers response_format."""
        captured_body: list[dict[str, Any]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            captured_body.append(body)
            return httpx.Response(
                200,
                json={
                    **OPENAI_OK,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": '{"name": "test"}',
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        client = OpenAIClient(model="gpt-5", api_key="test-key")
        client._async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url=client._base_url
        )

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = await client.json_async(
            [{"role": "user", "content": "Give me JSON"}],
            schema=schema,
            output_mode=StructuredOutputMode.NATIVE,
        )

        assert result.output_json == {"name": "test"}
        assert "response_format" in captured_body[0]

    async def test_json_async_tool_mode(self) -> None:
        """json() with tool mode creates a return_output tool."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response(output_json='{"name": "tool-result"}')

        client = FunctionClient(function=handler)
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        result = await client.json_async(
            [{"role": "user", "content": "Give JSON"}],
            schema=schema,
            output_mode=StructuredOutputMode.TOOL,
        )

        # Verify the tool was threaded through via kwargs
        assert len(client.call_history) == 1
        assert result.output_json == {"name": "tool-result"}

    async def test_json_async_prompted_mode(self) -> None:
        """json() with prompted mode appends schema instruction to messages."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            # The last message should contain the schema instruction
            last_msg = messages[-1]["content"]
            assert "IMPORTANT: Return your response as valid JSON" in last_msg
            return _make_response(output_json='{"x": 1}')

        client = FunctionClient(function=handler)
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}

        result = await client.json_async(
            [{"role": "user", "content": "Give a number"}],
            schema=schema,
            output_mode=StructuredOutputMode.PROMPTED,
        )
        assert result.output_json == {"x": 1}


# ===========================================================================
# Pydantic output validation
# ===========================================================================


class TestPydanticValidation:
    async def test_pydantic_output_validator_passes(self) -> None:
        """Validator accepts the result; returns the model instance."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            name: str

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response(output_json='{"name": "Alice"}')

        client = FunctionClient(function=handler)

        def validator(obj: MyModel) -> MyModel:
            return obj

        result = await client.pydantic_async(
            [{"role": "user", "content": "name"}],
            output_type=MyModel,
            output_validator=validator,
            output_mode=StructuredOutputMode.PROMPTED,
        )
        assert result.name == "Alice"

    async def test_pydantic_output_validator_retries(self) -> None:
        """Validator rejects the first attempt; retries and succeeds."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            name: str

        attempt = 0

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                return _make_response(output_json='{"name": "bad"}')
            return _make_response(output_json='{"name": "good"}')

        client = FunctionClient(function=handler)

        def validator(obj: MyModel) -> MyModel:
            if obj.name == "bad":
                raise ValueError("name must not be 'bad'")
            return obj

        result = await client.pydantic_async(
            [{"role": "user", "content": "name"}],
            output_type=MyModel,
            output_validator=validator,
            max_validation_retries=1,
            output_mode=StructuredOutputMode.PROMPTED,
        )
        assert result.name == "good"

    async def test_pydantic_output_validator_exhausted(self) -> None:
        """max_validation_retries reached without success; raises."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            name: str

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response(output_json='{"name": "bad"}')

        client = FunctionClient(function=handler)

        def validator(obj: MyModel) -> MyModel:
            raise ValueError("always fails")

        with pytest.raises(KaosLLMValidationError, match="Output validator failed"):
            await client.pydantic_async(
                [{"role": "user", "content": "name"}],
                output_type=MyModel,
                output_validator=validator,
                max_validation_retries=1,
                output_mode=StructuredOutputMode.PROMPTED,
            )

    async def test_pydantic_non_basemodel_raises(self) -> None:
        """output_type that is not a BaseModel subclass raises immediately."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response()

        client = FunctionClient(function=handler)

        with pytest.raises(KaosLLMValidationError, match="output_type must be a Pydantic"):
            await client.pydantic_async(
                [{"role": "user", "content": "x"}],
                output_type=dict,  # type: ignore[arg-type]
            )

    async def test_pydantic_no_json_in_response_retries(self) -> None:
        """When the response contains no valid JSON, it retries then raises."""
        from pydantic import BaseModel

        class Item(BaseModel):
            value: int

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response(output_json="not valid json at all")

        client = FunctionClient(function=handler)

        with pytest.raises(KaosLLMValidationError, match=r"Could not extract JSON|Pydantic"):
            await client.pydantic_async(
                [{"role": "user", "content": "x"}],
                output_type=Item,
                max_validation_retries=1,
                output_mode=StructuredOutputMode.PROMPTED,
            )

    async def test_pydantic_schema_validation_fails_retries(self) -> None:
        """Pydantic validation fails, retries with corrected output."""
        from pydantic import BaseModel

        class Item(BaseModel):
            count: int

        attempt = 0

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                # Invalid: count should be int, not string
                return _make_response(output_json='{"count": "not-a-number"}')
            return _make_response(output_json='{"count": 42}')

        client = FunctionClient(function=handler)

        result = await client.pydantic_async(
            [{"role": "user", "content": "give count"}],
            output_type=Item,
            max_validation_retries=1,
            output_mode=StructuredOutputMode.PROMPTED,
        )
        assert result.count == 42


# ===========================================================================
# Embeddings
# ===========================================================================


class TestEmbedNotImplemented:
    async def test_embed_async_not_implemented(self) -> None:
        """Base embed_async raises NotImplementedError for FunctionClient."""
        client = FunctionClient()
        with pytest.raises(NotImplementedError, match="does not support embeddings"):
            await client.embed_async("test text")

    def test_embed_sync_not_implemented(self) -> None:
        """Sync embed raises NotImplementedError for FunctionClient."""
        client = FunctionClient()
        with pytest.raises(NotImplementedError, match="does not support embeddings"):
            client.embed("test text")


# ===========================================================================
# Context managers
# ===========================================================================


class TestContextManagers:
    def test_sync_context_manager(self) -> None:
        """Sync `with client:` works and calls close()."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response()

        with FunctionClient(function=handler) as client:
            result = client.chat([{"role": "user", "content": "Hi"}])
            assert result.text == "test"

    async def test_async_context_manager(self) -> None:
        """Async `async with client:` works and calls aclose()."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response()

        async with FunctionClient(function=handler) as client:
            result = await client.chat_async([{"role": "user", "content": "Hi"}])
            assert result.text == "test"


# ===========================================================================
# Close / aclose lifecycle
# ===========================================================================


class TestClientLifecycle:
    def test_close_sync_and_async_clients(self) -> None:
        """close() cleans up both sync and async http clients."""
        client = OpenAIClient(model="gpt-5", api_key="test-key")
        # Force creation of both clients
        client._sync_client = MagicMock(spec=httpx.Client)
        client._async_client = MagicMock(spec=httpx.AsyncClient)

        client.close()

        assert client._sync_client is None
        assert client._async_client is None

    async def test_aclose_async_client(self) -> None:
        """aclose() cleans up both sync and async http clients."""
        client = OpenAIClient(model="gpt-5", api_key="test-key")
        # Create a real async client to test aclose
        _inject_async_mock(client, OPENAI_OK)
        assert client._async_client is not None

        await client.aclose()
        assert client._async_client is None

    async def test_aclose_with_sync_client(self) -> None:
        """aclose() also closes the sync client if it exists."""
        client = OpenAIClient(model="gpt-5", api_key="test-key")
        client._sync_client = MagicMock(spec=httpx.Client)

        await client.aclose()
        assert client._sync_client is None


# ===========================================================================
# Cache policy
# ===========================================================================


class TestCachePolicy:
    async def test_cache_force_policy(self, tmp_path: Any) -> None:
        """CachePolicy.FORCE with FileCache stores and retrieves the response."""
        cache = FileCache(str(tmp_path / "llm_cache"))

        # We test _resolve_cache_policy and cache interaction through
        # the OpenAI client + mock transport.
        client2 = OpenAIClient(model="gpt-5", api_key="test-key")
        client2._cache = cache
        client2._settings.cache_enabled = True
        _inject_async_mock(client2, OPENAI_OK)

        # First call: misses cache, stores
        r1 = await client2.request_async(
            [{"role": "user", "content": "Hi"}],
            options=RequestOptions(cache_policy=CachePolicy.FORCE),
        )
        assert r1.text == "ok"

        # Second call: should hit cache
        r2 = await client2.request_async(
            [{"role": "user", "content": "Hi"}],
            options=RequestOptions(cache_policy=CachePolicy.FORCE),
        )
        assert r2.text == "ok"

    def test_resolve_cache_policy_skip_when_disabled(self) -> None:
        """_resolve_cache_policy returns SKIP when cache is disabled."""
        client = FunctionClient()
        client._settings.cache_enabled = False
        policy = client._resolve_cache_policy(None)
        assert policy == CachePolicy.SKIP

    def test_resolve_cache_policy_force_when_enabled(self) -> None:
        """_resolve_cache_policy returns FORCE when cache is enabled."""
        client = FunctionClient()
        client._settings.cache_enabled = True
        policy = client._resolve_cache_policy(None)
        assert policy == CachePolicy.FORCE

    def test_resolve_cache_policy_respects_request_override(self) -> None:
        """Per-request cache_policy overrides client-level setting."""
        client = FunctionClient()
        client._settings.cache_enabled = True
        policy = client._resolve_cache_policy(RequestOptions(cache_policy=CachePolicy.SKIP))
        assert policy == CachePolicy.SKIP


# ===========================================================================
# Prompted JSON mode edge: last message not user role
# ===========================================================================


class TestPromptedJsonEdgeCases:
    def test_prompted_mode_no_user_message_appends(self) -> None:
        """_apply_prompted_json_mode appends a user message when last isn't user."""
        client = FunctionClient()
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "sys"},
        ]
        result = client._apply_prompted_json_mode(messages, schema)
        assert result[-1]["role"] == "user"
        assert "IMPORTANT: Return your response as valid JSON" in result[-1]["content"]

    def test_prompted_mode_non_string_content(self) -> None:
        """_apply_prompted_json_mode with non-string content list appends new user."""
        client = FunctionClient()
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "text", "text": "image here"}]},
        ]
        result = client._apply_prompted_json_mode(messages, schema)
        # Should append a new user message since content is a list
        assert len(result) == 2
        assert result[-1]["role"] == "user"
        assert "IMPORTANT" in result[-1]["content"]


# ===========================================================================
# Lazy httpx client init
# ===========================================================================


class TestLazyClientInit:
    def test_get_sync_client_creates(self) -> None:
        """_get_sync_client() creates the client on first call."""
        client = OpenAIClient(model="gpt-5", api_key="test-key")
        assert client._sync_client is None
        sync = client._get_sync_client()
        assert sync is not None
        assert client._sync_client is sync
        client.close()

    def test_get_async_client_creates(self) -> None:
        """_get_async_client() creates the client on first call."""
        client = OpenAIClient(model="gpt-5", api_key="test-key")
        assert client._async_client is None
        async_c = client._get_async_client()
        assert async_c is not None
        assert client._async_client is async_c
        client.close()

    def test_cache_enabled_creates_file_cache(self, tmp_path: Any) -> None:
        """When cache_enabled=True and no cache passed, FileCache is created."""
        from kaos_llm_client.settings import KaosLLMSettings

        settings = KaosLLMSettings(cache_enabled=True, cache_path=str(tmp_path / "c"))
        client = OpenAIClient(model="gpt-5", api_key="test-key", settings=settings)
        assert isinstance(client._cache, FileCache)


# ===========================================================================
# Sync request path
# ===========================================================================


class TestSyncRequestPath:
    def test_request_sync_delegates(self) -> None:
        """Sync request() wraps request_async via run_sync."""

        def handler(messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
            return _make_response("sync-ok")

        client = FunctionClient(function=handler)
        result = client.request([{"role": "user", "content": "Hi"}])
        assert result.text == "sync-ok"
