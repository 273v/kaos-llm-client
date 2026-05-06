"""kaos-llm-client — Thin, provider-native LLM client for KAOS.

Usage::

    from kaos_llm_client import create_client

    client = create_client("openai:gpt-5")
    response = client.chat(messages=[{"role": "user", "content": "Hello!"}])
    print(response.text)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_llm_client._version import __version__

if TYPE_CHECKING:
    from kaos_core.registry.container import KaosRuntime
from kaos_llm_client.cache import CacheBackend, FileCache, NullCache
from kaos_llm_client.cassette import (
    Cassette,
    CassetteContext,
    CassetteEntry,
    CassetteMissError,
    CassetteMode,
    CassetteRecorder,
    CassetteReplayClient,
    cassette_key,
    use_cassette,
    use_cassette_async,
)
from kaos_llm_client.errors import (
    KaosLLMAuthError,
    KaosLLMError,
    KaosLLMProviderError,
    KaosLLMRetryExhaustedError,
    KaosLLMTransportError,
    KaosLLMValidationError,
)
from kaos_llm_client.messages import (
    AssistantMessage,
    CachePoint,
    ChatMessages,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from kaos_llm_client.multimodal import (
    audio_from_path,
    audio_input,
    document_from_bytes,
    document_from_path,
    document_url,
    image_from_bytes,
    image_from_path,
    image_url,
)
from kaos_llm_client.profiles import (
    ANTHROPIC_DEFAULT,
    ANTHROPIC_TOOL_FALLBACK,
    GOOGLE_DEFAULT,
    OPENAI_COMPATIBLE_DEFAULT,
    OPENAI_DEFAULT,
    OPENAI_REASONING,
    XAI_DEFAULT,
    AnthropicJsonSchemaTransformer,
    AnthropicModelProfile,
    GoogleJsonSchemaTransformer,
    GoogleModelProfile,
    JsonSchemaTransformer,
    ModelProfile,
    OpenAIJsonSchemaTransformer,
    OpenAIModelProfile,
    StructuredOutputMode,
    infer_provider,
    resolve_profile,
)
from kaos_llm_client.providers import BaseProviderClient, create_client
from kaos_llm_client.schema_cache import canonicalize as canonicalize_schema
from kaos_llm_client.schema_cache import schema_hash
from kaos_llm_client.settings import KaosLLMSettings
from kaos_llm_client.transport import RetryPolicy
from kaos_llm_client.types import (
    BinaryData,
    CachePolicy,
    ContentPart,
    EmbeddingResponse,
    ProviderRequest,
    ProviderResponse,
    RequestHooks,
    RequestOptions,
    StreamAccumulator,
    StreamChunk,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    UsageInfo,
)


def register_llm_client_tools(runtime: KaosRuntime) -> int:
    """Register LLM tools with a KaosRuntime.

    Used by ``kaos-mcp serve --module llm_client`` for auto-discovery.
    """
    from kaos_llm_client.tools import register_llm_tools

    return register_llm_tools(runtime)


__all__ = [
    "ANTHROPIC_DEFAULT",
    "ANTHROPIC_TOOL_FALLBACK",
    "GOOGLE_DEFAULT",
    "OPENAI_COMPATIBLE_DEFAULT",
    "OPENAI_DEFAULT",
    "OPENAI_REASONING",
    "XAI_DEFAULT",
    "AnthropicJsonSchemaTransformer",
    "AnthropicModelProfile",
    "AssistantMessage",
    "BaseProviderClient",
    "BinaryData",
    "CacheBackend",
    "CachePoint",
    "CachePolicy",
    "Cassette",
    "CassetteContext",
    "CassetteEntry",
    "CassetteMissError",
    "CassetteMode",
    "CassetteRecorder",
    "CassetteReplayClient",
    "ChatMessages",
    "ContentPart",
    "EmbeddingResponse",
    "FileCache",
    "GoogleJsonSchemaTransformer",
    "GoogleModelProfile",
    "JsonSchemaTransformer",
    "KaosLLMAuthError",
    "KaosLLMError",
    "KaosLLMProviderError",
    "KaosLLMRetryExhaustedError",
    "KaosLLMSettings",
    "KaosLLMTransportError",
    "KaosLLMValidationError",
    "ModelProfile",
    "NullCache",
    "OpenAIJsonSchemaTransformer",
    "OpenAIModelProfile",
    "ProviderRequest",
    "ProviderResponse",
    "RequestHooks",
    "RequestOptions",
    "RetryPolicy",
    "StreamAccumulator",
    "StreamChunk",
    "StructuredOutputMode",
    "SystemMessage",
    "ToolCall",
    "ToolChoice",
    "ToolDefinition",
    "ToolResultMessage",
    "UsageInfo",
    "UserMessage",
    "__version__",
    "audio_from_path",
    "audio_input",
    "canonicalize_schema",
    "cassette_key",
    "create_client",
    "document_from_bytes",
    "document_from_path",
    "document_url",
    "image_from_bytes",
    "image_from_path",
    "image_url",
    "infer_provider",
    "register_llm_client_tools",
    "resolve_profile",
    "schema_hash",
    "use_cassette",
    "use_cassette_async",
]
