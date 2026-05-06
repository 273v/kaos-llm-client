# Changelog

All notable changes to `kaos-llm-client` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0a1] — 2026-05-05

First public alpha release. Thin, provider-native LLM client for direct
model calls within the KAOS (Kelvin Agentic Operating System) platform.

### Added

- **Eleven provider clients** — OpenAI (Chat Completions + Responses API),
  Anthropic Messages, Google Gemini, xAI Grok, Groq, Mistral, OpenRouter,
  Azure OpenAI (chat completions via ``azure:``/``azure-openai:`` and
  Responses API via ``azure-responses:``/``azure-foundry:``), AWS Bedrock
  (OpenAI-compatible Responses API via ``bedrock:``), and a generic
  OpenAI-compatible client (VLLM, Ollama, LiteLLM, custom).
- **`[azure]` optional extra** — installs `azure-identity>=1.25.3`
  (Microsoft's official Azure SDK for Python identity package, MIT
  licensed). Required only when passing
  `azure_ad_token_provider=` (e.g., wrapping `DefaultAzureCredential`,
  `ManagedIdentityCredential`, `WorkloadIdentityCredential`) to
  `AzureOpenAIClient` or `AzureOpenAIResponsesClient`. Without the
  extra, api-key auth still works on every Azure endpoint
  (regional + custom-subdomain). Footprint: ~16 MB transitive
  (mostly `cryptography`, required for MSAL JWT signing + Azure CLI
  shared-cache decryption). Install with
  `uv add 'kaos-llm-client[azure]'`.
- **Convenience API** on every client — `chat()`, `json()`, `pydantic()`,
  `embed()` plus async variants. Sync wraps async via `run_sync()` with
  thread-pool fallback when an event loop is already running.
- **Multimodal input** helpers (`image_url`, `image_from_path`,
  `image_from_bytes`, `audio_input`, `audio_from_path`,
  `document_from_path`, `document_url`) and SSE **streaming** on every
  provider with `StreamAccumulator` reconstructing full responses from
  `StreamChunk` deltas.
- **Structured output** — `json()` and `pydantic()` with native, tool,
  and prompted modes plus validation retries; mode selected per-model
  via `ModelProfile`. Tool calling via `ToolDefinition` / `ToolChoice`
  with multi-turn workflows; provider-native schemas converted
  automatically.
- **Typed messages (optional)** — `SystemMessage`, `UserMessage`,
  `AssistantMessage`, `ToolResultMessage`, `CachePoint` subclass `dict`
  so they pass through to providers without conversion.
- **Profile-driven behavior** — `ModelProfile` encodes provider/model
  differences (no `if provider == "anthropic"` branches). Bare model
  strings infer the provider (e.g., `claude-sonnet-4-6` → anthropic).
- **Composition wrappers** — `FallbackClient` (provider order),
  `ConcurrencyLimitedClient` (asyncio.Semaphore cap),
  `InstrumentedClient` (timing, token counts, cost). Pluggable
  **response caching** via `CacheBackend` ABC with BLAKE2b-keyed
  `FileCache` and `NullCache`; `client.metrics()` returns hit-rate.
- **Lifecycle hooks** (`RequestHooks(on_request, on_response, on_error,
  on_retry)`), **cassette record/replay** (`CassetteRecorder`,
  `CassetteReplayClient`, `use_cassette()`), and **`FunctionClient`**
  for deterministic unit tests without HTTP mocks.
- **Per-call observability** — every successful call emits one structured
  `LLM call complete` info-log with `provider`, `model`, `request_id`,
  `response_id`, token counts, and `estimated_usd` from
  `cost.MODEL_PRICING`; cached calls emit `cache_hit=True`.
- **CLI** (`kaos-llm-client check`, `chat`, `profiles`, `config`) and
  **MCP server** (`kaos-llm-serve` exposing `kaos-llm-chat`,
  `kaos-llm-json`, `kaos-llm-embed` over stdio or streamable HTTP).
- **Typed settings** — `KaosLLMSettings` (`ModuleSettings` subclass) with
  `KAOS_LLM_` env prefix and legacy fallbacks (`OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_GENERATIVE_AI_API_KEY`,
  `XAI_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`, `OPENROUTER_API_KEY`)
  resolved via `@model_validator(mode="before")`.
- **Error hierarchy** — `KaosLLMError`, `KaosLLMAuthError` (never
  retried), `KaosLLMTransportError` (retryable), `KaosLLMProviderError`
  (carries `status_code`, `raw_error`, `provider`, `fix`),
  `KaosLLMRetryExhaustedError`, `KaosLLMValidationError`. All inherit
  `KaosCoreError`. Python 3.13 + 3.14 support.

### Security

- API keys stored as `pydantic.SecretStr` end-to-end and redacted in
  logs, error payloads, and `kaos-llm-client config` output.
- `kaos-llm-serve --http` defaults to `--host 127.0.0.1` (loopback);
  emits a startup warning when bound to a non-loopback interface and
  documents the expectation of an authenticated reverse proxy (mTLS,
  OAuth, IP allowlist) for any remote exposure.
- Hard payload caps — `KAOS_LLM_MAX_RESPONSE_BYTES` (default 32 MiB)
  bounds non-streaming bodies; `KAOS_LLM_STREAM_MAX_DURATION` (default
  600 s) caps wall-clock on streaming responses, preventing slowloris
  and runaway-stream resource exhaustion.
- Cost / log emission is wrapped in `try/except` so a logging-side
  failure cannot break a real LLM call.
- `service_tier="flex"` retry fallback — transport retries once without
  `service_tier` after standard retries exhaust on 500 errors, falling
  back to standard pricing rather than failing the request.

### License

- This release is the first to ship under the Apache License 2.0. Earlier
  internal versions were proprietary.

[Unreleased]: https://github.com/273v/kaos-llm-client/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/273v/kaos-llm-client/releases/tag/v0.1.0a1
