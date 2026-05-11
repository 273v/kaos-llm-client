# Changelog

All notable changes to `kaos-llm-client` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **`FileCache.clear()` safe-clear allowlist now works on Windows.**
  The allowlist previously hardcoded ``/tmp`` / ``/var/tmp`` /
  ``/var/folders`` plus ``~/.cache``. On Windows those POSIX paths
  resolve to ``D:\tmp`` etc. (the drive root), none of which contain
  pytest's tmpdir at ``C:\Users\...\AppData\Local\Temp\...``, so
  every cache-clear under the new Windows-x64 CI leg raised
  ``RuntimeError: FileCache.clear() refused: ... is outside the
  safe-clear allowlist``. Added ``Path(tempfile.gettempdir())`` to
  the allowlist (resolved at module-load) so the platform tempdir
  is always permitted on every OS. The original POSIX-flavored
  entries are kept so explicit ``/tmp/...`` cache paths still
  resolve as before. Files: ``kaos_llm_client/cache.py``.
- **Tests: cassette permission asserts gated to POSIX.**
  ``test_cassette_permissions.TestCassetteSavePermissions`` asserts
  ``_file_mode(out) == 0o600`` / ``0o700``. On Windows these report
  ``0o666`` / ``0o777`` because ``Path.chmod`` is essentially a no-op
  there (NTFS uses ACLs, not POSIX bits). The cassette permission
  contract is documented as POSIX-only — Windows relies on per-user
  profile / NTFS ACL isolation instead. The class now carries a
  ``sys.platform == "win32"`` skip; POSIX behavior is still covered
  on Linux + macOS. Files:
  ``tests/unit/test_cassette_permissions.py``.

## [0.1.0a2] — 2026-05-07

### Changed

- **`SamplingRequest`-style provider profiles bumped to 2026 frontier-model
  output budgets.** The 2023-era `default_max_tokens=4096` floor truncated
  multi-page deliverables — the Harvey CoC bench
  (`docs/benchmarks/harvey-coc-2026-05-06.json`) cut off mid-sentence at
  exactly the 4096-token boundary. New per-provider defaults:
  - OpenAI gpt-5.x: 128K (was 4K)
  - OpenAI reasoning (o3/o4, gpt-5.5): 200K (was 16K) — covers hidden
    chain-of-thought + visible answer
  - Azure OpenAI: 128K (tracks the underlying OpenAI model ceiling)
  - Anthropic Sonnet 4.5+ / Opus 4.7+: 100K (was 4K) — header-free,
    matches the published 2026-05-06 Anthropic ceiling
  - Anthropic 3.x: 8K — Claude 3.5/3.7 supported 8K, 3.0 supported 4K;
    8K is the safe value across the 3.x line
  - Google Gemini 2.5+ thinking models: 100K (was 4K)
  - Google Gemini 2.0: 8K (matches the published 2.0 ceiling)
  - xAI grok-4: 128K (was 4K)
  - xAI grok-3: 16K (matches the published grok-3 ceiling)
  - Per-provider resolvers now clamp Anthropic Haiku 4.5 down to 64K
    (header-free) since Haiku does NOT support the 100K Sonnet/Opus
    ceiling. Same pattern applies to Google's non-thinking 2.0 models
    (returns the 8K-tuned profile instead of the 100K default).
  Aligned with the same bump in `kaos-core` and `kaos-agents`.

- **`maintainers` field added to `pyproject.toml`** (cross-package
  metadata consistency). The next published wheel + sdist carry
  `Maintainer-email: Michael Bommarito <mike@273ventures.com>`
  alongside the existing `Author-email: 273 Ventures LLC`.

### Fixed

- **Audit-01 KLC-01 (Medium): MCP tool layer now honours
  `KaosContext._config`.** Previously each of the seven tool classes in
  `kaos_llm_client/tools.py` instantiated `KaosLLMSettings()` directly
  and passed `settings=` to `create_client()`, which silently dropped
  the documented per-request override path described in
  `KaosLLMSettings.from_context`. The fix routes inference tools through
  `create_client(model, context=context)` (so `from_context` is invoked
  inside `BaseProviderClient.__init__`); `KaosLLMProviderCheckTool`
  builds `settings = KaosLLMSettings.from_context(context)` directly
  since it introspects settings without constructing a client. Added
  `tests/unit/test_tools.py::TestToolContextConfigHonoured` regression
  test asserting that a `_config={"openai_api_key": "…"}` override on
  the context wins over a cleared environment.

- **Audit-01 KLC-02 (Medium): `[mcp]` extra promotion contract
  documented.** The monorepo source declares `[mcp]` (resolves locally
  via `[tool.uv.sources]`); this per-module repo strips the extra at
  release time per F009 lesson #4 (uv lock refuses to resolve
  unpublished siblings). The strip is the natural CI gate. Re-add the
  extra at the next patch release (`0.1.0a3`+) once `kaos-mcp` ships
  to PyPI. Closing the audit finding by making the contract explicit.

### Refactored

- **Audit-01 KLC-03 (Low): split `kaos_llm_client/tools.py` into a
  `tools/` subpackage.** The single 1554-line module mixed seven MCP
  tool classes with shared helpers, artifact storage, error formatting,
  and pricing lookup. New layout (largest file 265 lines, well under
  the 800-line review threshold):
  - `tools/__init__.py` — public re-exports + `register_llm_tools`
  - `tools/_common.py` — shared helpers (logger, `_tool_log_extra`,
    `_store_artifact`, `_format_llm_error`, annotations, constants)
  - `tools/chat.py`, `structured.py`, `tool_call.py`, `pydantic.py` —
    generative tools (one per file)
  - `tools/embed.py` — `KaosLLMEmbedTool` + `_estimate_tokens`
  - `tools/provider_check.py`, `cost_estimate.py` — local/info tools
  No public-API change: every name reachable as
  `from kaos_llm_client.tools import …` before remains reachable now,
  including private helpers (`_estimate_tokens`, `_lookup_pricing`)
  preserved for downstream test compatibility.

### Audit closure note

- **Audit-01 KLC-04 (Low): top-level `__all__` already alphabetically
  sorted.** Re-verified with `ruff check --fix` (no edits proposed).
  The audit applied a stricter pure-alphabetical view than the
  project's enforced linter; the existing RUF022 canonical
  (SCREAMING_CASE / PascalCase / dunders / snake_case, alphabetical
  within each group) is what we ship.

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

[Unreleased]: https://github.com/273v/kaos-llm-client/compare/v0.1.0a2...HEAD
[0.1.0a2]: https://github.com/273v/kaos-llm-client/compare/v0.1.0a1...v0.1.0a2
[0.1.0a1]: https://github.com/273v/kaos-llm-client/releases/tag/v0.1.0a1
