# Changelog

All notable changes to `kaos-llm-client` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.1.5] — 2026-05-22

Launch-blocker plan §Issue 3 — provider-served model snapshot capture
(see `kaos-modules/docs/plans/2026-05-22-launch-blocker-top-10.md`).

### Added

- **`ProviderResponse.model_snapshot: str | None`** — every provider
  parser now captures the resolved versioned snapshot from the
  response body alongside the requested model. ``model`` carries
  what we ASKED for; ``model_snapshot`` carries what was SERVED.
  Required for EU AI Act Article 12 (record-keeping) / Annex III §6
  reproducibility: an auditor 18 months later must identify which
  exact snapshot generated a given response.
- **Anthropic** (`providers/anthropic.py`) — captures
  ``raw["model"]`` (e.g. ``claude-sonnet-4-6-20260415``).
- **OpenAI Chat Completions + OpenAI-compatible providers**
  (`providers/openai_compat.py`) — captures ``raw["model"]``. Same
  field is consumed by Groq / xAI / Mistral / DeepSeek / OpenRouter
  via the shared OpenAI-compatible base.
- **OpenAI Responses API** (`providers/openai_responses.py`) —
  captures ``raw["model"]``.
- **Google Gemini** (`providers/google.py`) — captures
  ``raw["modelVersion"]`` (Google uses a different field name from
  OpenAI / Anthropic; pinned in test fixture).

### Tests

- **`tests/unit/test_provider_model_snapshot.py`** (7 tests) —
  pinned field defaults to ``None``, JSON round-trip carries the
  value, per-provider parser captures the response model (Anthropic
  / OpenAI Chat / OpenAI Responses / Google), defensive missing-
  field fallback, Google's modelVersion alias.
- **`tests/integration/test_provider_model_snapshot_live.py`**
  (5 live cases) — proves model_snapshot capture against the real
  OpenAI Chat, OpenAI Responses (o4-mini), Anthropic, and Google
  APIs + stability-across-calls invariant.
- **`tests/unit/test_provider_response_audit_fields.py`** (9 tests) —
  field-shape contract for the audit-trail triple
  (request_id / response_id / model_snapshot).


## [0.1.2] — 2026-05-22

Broad-reliability roadmap B1.3
(see `kaos-modules/docs/plans/2026-05-22-broad-reliability-adaptability-roadmap.md`).

### Added

- **`KaosLLMStreamInterruptedError`** — new typed exception raised by
  `parse_sse_stream` / `parse_sse_stream_sync` when the HTTP/network
  connection drops after the first byte has been received. Carries
  `partial_text` and `bytes_received` so SPA consumers can choose
  between (a) ship-partial-with-footer ("_streaming interrupted at
  N tokens; partial response follows_") and (b) retry-as-fresh-call.
  Inherits from `KaosLLMTransportError` for backward compatibility
  with existing `except KaosLLMTransportError` handlers; exposed
  from the top-level package.

### Fixed

- **#570 B1.3 — mid-stream provider disconnect surfaced as opaque
  transport error.** Pre-0.1.2, an httpx `ReadError`,
  `RemoteProtocolError`, or `ProtocolError` mid-stream surfaced as a
  generic `KaosLLMTransportError` with no partial-text payload. SPA
  consumers shipped a half-streamed message with no recovery signal
  — the user saw the assistant cut off mid-word. Now the typed error
  carries the byte counter + underlying cause so the consumer can
  render an honest interruption footer instead of silently dropping
  the partial response. Pre-first-byte failures still surface as the
  legacy transport error so existing retry policies continue to fire.

### Added

- `tests/unit/test_stream_interrupt.py` — 12 regression tests
  covering: clean-stream baseline; pre-first-byte vs mid-stream
  error type distinction; mid-stream `ReadError` + `RemoteProtocolError`
  paths; byte counter accuracy; sync sibling parity; error shape
  contract (inherits from `KaosLLMTransportError`, carries
  `partial_text` / `bytes_received` / `cause`, public export).

### Verified

- `ruff format --check kaos_llm_client tests`
- `ruff check kaos_llm_client tests`
- `ty check kaos_llm_client tests`
- `pytest tests/unit/ -q --no-cov` — **955 passed**


## [0.1.1] — 2026-05-21

Reliability roadmap R0.3 — Gemini tool dispatch fix
(see `kaos-modules/docs/plans/2026-05-21-reliability-roadmap.md`).

### Fixed

- **#560: Google Gemini tool dispatch returned HTTP 400 on every tool turn
  when the tool's parameters JSONSchema contained `$ref`/`$defs`,
  `const`, `default`, or `title` keywords.** Both Gemini Pro and Flash were
  unusable for tool-using legal research in the kaos-ui SPA. Root cause:
  `_tool_def_to_google` forwarded `ToolDefinition.parameters` verbatim
  instead of running it through `GoogleJsonSchemaTransformer`, which
  already inlines `$ref`/`$defs`, rewrites `const → enum: [value]`, and
  strips `title`, `default`, `format`. The transformer was correctly
  applied to the structured-output `responseSchema` path but not to the
  `functionDeclarations` tool path. Fix: thread the profile's
  `json_schema_transformer` (Gemini-family profiles all set it to
  `GoogleJsonSchemaTransformer`) into `_tool_def_to_google` and apply
  it to each tool's parameter block before sending. Confirmed by the
  reliability-roadmap worker-honesty audit
  (`kaos-modules/docs/audits/2026-05-21-worker-honesty.md`).

### Added

- `tests/unit/test_google.py::TestGoogleToolDispatch` — six regression
  tests covering: (a) the legacy (no-transformer) path still passes
  schemas verbatim; (b) `$ref` / `$defs` inlined; (c) `const`/`title`/
  `default` stripped/rewritten on root and nested nodes; (d) end-to-end
  `_build_request` plumbs the profile's transformer through;
  (e) tools-omitted path unchanged; (f) profile without a transformer
  still works.

### Verified

- `ruff format --check kaos_llm_client tests`
- `ruff check kaos_llm_client tests`
- `ty check kaos_llm_client tests`
- `pytest tests/unit/ -q --no-cov` — **943 passed**


## [0.1.0] — 2026-05-20

### Changed — WU-L of 0.1.0 GA plan

- 0.1.0 GA — WU-L of the 0.1.0 GA plan. First stable release of
  `kaos-llm-client`. The public API is frozen for the 0.1.x line: no
  breaking changes will land until 0.2.0. Only runtime kaos-* dep is
  `kaos-core`; pin floor raised from `>=0.1.0rc1,<0.2` to
  `>=0.1.0,<0.2`. No code changes relative to 0.1.0rc1; this release
  is the pin-floor + version bump that signals GA to downstream
  consumers (kaos-content, kaos-llm-core, kaos-agents, kaos-pdf, etc).

### Verified
- `ruff format --check kaos_llm_client tests`
- `ruff check kaos_llm_client tests`
- `ty check kaos_llm_client tests`
- `pytest -m "not live and not network and not slow and not integration" --no-cov -q`
  (937 passed, 100 deselected)


## [0.1.0rc1] — 2026-05-20

### Changed — WU-J of 0.1.0 GA plan

- Release candidate cut per WU-J of the 0.1.0 GA plan. Freezes the
  public Python API surface ahead of GA. No source changes relative
  to 0.1.0a5; this release exists to raise the kaos-core runtime pin
  floor to the rc track and signal API freeze to downstream consumers.
- Pin floor raised to `kaos-core>=0.1.0rc1,<0.2`. The `<0.2` ceiling
  protects against legacy `0.2.0a*` lines (e.g. kaos-nlp-transformers)
  leaking into resolution.
- `kaos_llm_client/_version.py` bumped to `0.1.0rc1`.

### Verified
- `ruff format --check kaos_llm_client tests`
- `ruff check kaos_llm_client tests`
- `ty check kaos_llm_client tests`
- `pytest -m "not live and not network and not slow and not integration" --no-cov -q`
  (937 passed, 100 deselected)


## [0.1.0a5] — 2026-05-20

Work unit WU-E of the [0.1.0 GA plan](https://github.com/273v/kaos-modules/blob/main/docs/plans/2026-05-20-0.1.0-ga-plan.md).

### Fixed

- **#466: `cost_usd=0` for `openai:gpt-5.4-mini` (and every other
  `provider:model` form).** `lookup_pricing()` now strips the
  `provider:` prefix internally before matching, so callers passing
  the canonical `openai:gpt-5.4-mini` / `anthropic:claude-opus-4-7`
  / `google:gemini-2.5-flash` form resolve to the same entry as the
  bare model name. Previously the caller was "responsible" for
  stripping the prefix, which silently produced `None` pricing and
  `cost_usd=0` on the default SPA model.
- **Test assertions updated for `kaos-core 0.1.0a12` error contract.**
  `KaosCoreError.__str__()` now returns the message only; structured
  fields (`provider`, `fix`, `model`, ...) live in `err.details`.
  Three unit tests (`test_azure.py`, `test_bedrock.py`,
  `test_errors.py`) were migrated to assert against
  `err.details["fix"]` rather than `str(err)`. No library code
  change — the actionable env-var names are still produced; only
  the rendering surface moved.

### Added

- **Hot-reloadable pricing overlay.** `kaos_llm_client.cost` now
  exports `load_pricing_overlay()` + `apply_pricing_overlay()`.
  Set `KAOS_LLM_PRICING_OVERLAY_PATH` to a JSON file mapping
  `model -> {"input": float, "output": float, ...}` and the table
  is merged at module import. Malformed / missing overlays log a
  WARN and no-op rather than crash. Lets the pricing table catch
  up with new model launches between releases without a code
  change.
- **Regression test `TestRequiredModelPricingForGA`.** Asserts every
  SPA + bench-harness default model resolves to non-zero input +
  output cost via the canonical `provider:model` form:
  `openai:gpt-5.4-mini`, `openai:gpt-5.4-nano`, `openai:gpt-5.4`,
  `anthropic:claude-opus-4-7`, `anthropic:claude-sonnet-4-6`,
  `anthropic:claude-haiku-4-5`, `google:gemini-2.5-flash`.

### Changed

- **`kaos-core` floor bumped: `>=0.1.0a1` → `>=0.1.0a12,<0.2`.**
  Aligns with the rest of the kaos-* DAG ahead of 0.1.0 GA
  (post-URI-redesign + Capability type).
- **Anthropic `ANTHROPIC_TOOL_FALLBACK` `default_max_tokens`
  raised from 8192 to 64000.** Attorney-grade long-form deliverables
  were truncating mid-output on the prior 2023-era default. Models
  whose own API caps are lower (Claude 3 = 4K) will be clamped by
  the provider upstream.


## [0.1.0a4] — 2026-05-16

### Fixed

- **Reasoning models no longer receive `temperature` / `top_p`.**
  When `ProfileMetadata.is_reasoning_model` is True, the client now
  strips both fields from the outbound request envelope before
  invoking the provider. Reasoning models reject these parameters
  with an error from many providers (PA16), so silently dropping
  them lets the same `RequestProfile` flow through reasoning and
  non-reasoning endpoints without caller-side branching.

### Added

- **`MODEL_PRICING` extended with six previously-mirrored models.**
  Pricing entries added for models that were already in the
  cross-repo `MODEL_REGISTRY` but missing from `cost.py`'s
  lookup table — so `LlmClient.estimate_cost(...)` no longer
  raises `KeyError` for them. Provenance and exact pricing
  values match the upstream provider docs at the date of the
  commit; see `kaos_llm_client/cost.py` for the per-model
  entries.

### Documentation

- Fixture provenance README backfilled per audit-03 D9; each
  fixture under `tests/fixtures/` now has a clear source / date /
  redaction note.

### Infrastructure

- Dependabot migrated to the uv ecosystem with a 72-hour cooldown
  matching the rest of the kaos-* org.
- Public-PR CI workflow hardening, CycloneDX SBOM release asset,
  CODEOWNERS expansion, OpenSSF Scorecard rollout.


## [0.1.0a3] — 2026-05-11

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
### Security

- **bandit + vulture now run in both pre-commit and CI.** The
  ``.pre-commit-config.yaml`` gains two new hooks (bandit static
  security scan + vulture dead-code scan), mirrored by jobs in
  ``security.yml`` so the scan is publicly visible on every PR.
  Bandit skip list is justified inline per audit
  (``B101,B404,B603,B607``); vulture runs at ``--min-confidence
  100`` with a shared ``--ignore-names`` list for framework
  callbacks / signal handlers / OAuth field names that vulture
  can't infer from the import graph alone. Both hooks currently
  pass clean. Mirrors the rollout pattern from kaos-core.
### Changed

- **uv.lock is now tracked in git.** Previously gitignored at v0.1.0a1
  because the ``[mcp]`` optional extra (and the ``kaos-mcp`` dev
  dependency) referenced a sibling not yet on PyPI; ``uv lock``
  couldn't resolve them. ``kaos-mcp`` shipped (0.1.0a2), so the
  original gating reason no longer applies. Tracking the lockfile
  gives reproducible local dev environments, lets Dependabot surface
  sibling-version bumps as PRs, and makes the supply-chain pin set
  publicly auditable. Mirrors the org-wide convention being adopted
  across all 16 kaos-* repos.

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

[Unreleased]: https://github.com/273v/kaos-llm-client/compare/v0.1.0a3...HEAD
[0.1.0a3]: https://github.com/273v/kaos-llm-client/compare/v0.1.0a2...v0.1.0a3
[0.1.0a2]: https://github.com/273v/kaos-llm-client/compare/v0.1.0a1...v0.1.0a2
[0.1.0a1]: https://github.com/273v/kaos-llm-client/releases/tag/v0.1.0a1
