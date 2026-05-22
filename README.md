# kaos-llm-client

[![PyPI - Version](https://img.shields.io/pypi/v/kaos-llm-client)](https://pypi.org/project/kaos-llm-client/)
[![Python](https://img.shields.io/pypi/pyversions/kaos-llm-client)](https://pypi.org/project/kaos-llm-client/)
[![License](https://img.shields.io/pypi/l/kaos-llm-client)](https://github.com/273v/kaos-llm-client/blob/main/LICENSE)

Thin, provider-native LLM client for the Kelvin Agentic OS — direct model calls across OpenAI, Anthropic, Google, xAI, Groq, Mistral, OpenRouter, **Azure OpenAI** (api-key + AAD/Entra), and **AWS Bedrock** (OpenAI-compatible Responses API), with one interface.

## Install

```bash
uv add "kaos-llm-client>=0.1.0"
# or
pip install "kaos-llm-client>=0.1.0"

# Azure OpenAI with Microsoft Entra ID / DefaultAzureCredential
# (api-key auth works without this extra — only needed for AAD).
uv add 'kaos-llm-client[azure]>=0.1.0'

# MCP server runtime (pulls in kaos-mcp)
uv add 'kaos-llm-client[mcp]>=0.1.0'
```

Set at least one provider API key (`KAOS_LLM_OPENAI_API_KEY`, `KAOS_LLM_ANTHROPIC_API_KEY`, `KAOS_LLM_GOOGLE_API_KEY`, …). Standard names (`OPENAI_API_KEY`, etc.) are accepted as fallbacks. For Azure with AAD, see the [Quick start](#azure-openai-with-microsoft-entra-id-aad) below.

## Features

- **Direct providers** — OpenAI, Anthropic, Google, xAI, Groq, Mistral, OpenRouter, plus a generic OpenAI-compatible client (VLLM, Ollama, LiteLLM, custom endpoints)
- **Cloud-hosted gateways** — **Azure OpenAI** (chat completions + Responses API; api-key OR Microsoft Entra ID via `DefaultAzureCredential`) and **AWS Bedrock** (OpenAI-compatible Responses API on `bedrock-mantle.<region>.api.aws`)
- **Multimodal** — images (URL, path, bytes), audio input, document input (PDF, text)
- **Streaming, tools, structured output** — SSE `StreamAccumulator`; `ToolDefinition` / `ToolChoice`; `json()` and `pydantic()` with native/tool/prompted modes and validation retries
- **Embeddings** — `embed()` / `embed_async()` for embedding-capable providers
- **Composition wrappers** — `FallbackClient`, `ConcurrencyLimitedClient`, `InstrumentedClient`
- **Response caching** — pluggable `CacheBackend` with BLAKE2b-keyed `FileCache`
- **Profile-driven behavior** — `ModelProfile` encodes provider/model differences (no `if provider ==` branches)
- **Lifecycle hooks** — `RequestHooks(on_request, on_response, on_error, on_retry)` for observability
- **Per-call observability** — every successful call emits one `LLM call complete` structured info-log with provider, model, request_id, token counts, and `estimated_usd` cost
- **CLI + MCP** — `kaos-llm-client` CLI with `--json` output and `kaos-llm-serve` MCP server

## Quick start

```python
from kaos_llm_client import create_client

# Direct OpenAI (or Anthropic, Google, xAI, Groq, Mistral, OpenRouter)
client = create_client("openai:gpt-5.4-mini")
response = client.chat([{"role": "user", "content": "Hello!"}])
print(response.text)
# logs: INFO LLM call complete provider=openai model=gpt-5.4-mini request_id=... estimated_usd=...
```

### Azure OpenAI with Microsoft Entra ID (AAD)

> **Install the `[azure]` extra first**: `uv add 'kaos-llm-client[azure]'`. This
> pulls in Microsoft's `azure-identity` SDK (~16 MB transitive, mostly
> `cryptography`). Without the extra, api-key auth still works on every
> Azure endpoint — only AAD needs `azure-identity`.

`DefaultAzureCredential` gives you managed-identity / `az login` /
service-principal auth without storing static keys. The Responses-API
client (`azure-responses:`) is the recommended path for `gpt-5.4+`
deployments where tool calling is required.

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from kaos_llm_client import create_client

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",
)
client = create_client(
    "azure-responses:gpt-5.4-mini",
    azure_ad_token_provider=token_provider,
)
response = client.chat([{"role": "user", "content": "Hello!"}])
```

`azure-identity` ships 22 credential classes — `ManagedIdentityCredential`, `WorkloadIdentityCredential`, `ClientSecretCredential`, `CertificateCredential`, etc. Any of them works as the first argument to `get_bearer_token_provider`. The async variants live in `azure.identity.aio` and are awaited automatically by the kaos-llm-client provider.

AAD requires a custom-subdomain endpoint
(`https://<resource>.openai.azure.com/`); regional endpoints accept
api-key only. Both forms work for `azure:` (chat completions) and
`azure-responses:` (Responses API).

### AWS Bedrock (OpenAI-compatible Responses API)

```python
import os
from kaos_llm_client import create_client

# Bearer token from `aws bedrock create-bearer-token` or your AWS auth flow
os.environ["KAOS_LLM_BEDROCK_API_KEY"] = "..."
client = create_client("bedrock:openai.gpt-oss-120b")
response = client.chat([{"role": "user", "content": "Hello!"}])
```

## Providers

Direct-API clients:

| Prefix | Client | Models | Auth |
|---|---|---|---|
| `openai:` | `OpenAIClient` | GPT-5.5/5.4/5/4.1, o1/o3/o4 reasoning | `KAOS_LLM_OPENAI_API_KEY` |
| `anthropic:` | `AnthropicClient` | Claude 4.7 Opus, 4.6 Sonnet, 4.5 Haiku, 3.5/3.7 | `KAOS_LLM_ANTHROPIC_API_KEY` |
| `google:` | `GoogleClient` | Gemini 2.5/3.x Pro/Flash | `KAOS_LLM_GOOGLE_API_KEY` |
| `xai:` | `XAIClient` | Grok-3, Grok-4 | `KAOS_LLM_XAI_API_KEY` |
| `groq:` | `GroqClient` | LLaMA, Mixtral (OpenAI-compat) | `KAOS_LLM_GROQ_API_KEY` |
| `mistral:` | `MistralClient` | Mistral, Mixtral | `KAOS_LLM_MISTRAL_API_KEY` |
| `openrouter:` | `OpenRouterClient` | Any model via OpenRouter | `KAOS_LLM_OPENROUTER_API_KEY` |
| `openai-compatible:` | `OpenAICompatibleClient` | VLLM, Ollama, LiteLLM, custom | varies (`base_url=...`) |

Cloud-hosted gateways:

| Prefix | Client | Notes |
|---|---|---|
| `azure:` / `azure-openai:` | `AzureOpenAIClient` (chat completions) | Legacy path; works for any deployment |
| `azure-responses:` / `azure-foundry:` | `AzureOpenAIResponsesClient` | **Recommended for `gpt-5.4+`** — chat-completions tool calling with `reasoning: none` is unsupported by Azure on those models |
| `bedrock:` | `BedrockClient` | OpenAI-compatible Responses API on `bedrock-mantle.<region>.api.aws` |

Azure auth is `api-key` (works on regional + custom-subdomain endpoints) or **AAD/Entra** (`Authorization: Bearer <token>` — custom-subdomain endpoint required). Use `azure_ad_token=...` for a static bearer or `azure_ad_token_provider=...` for `DefaultAzureCredential` / managed identity / `az login` flows. See the [Quick start](#quick-start) for the canonical Entra ID example.

Model strings use `provider:model` format. If no prefix is given, the provider is inferred from the model name:

```python
create_client("openai:gpt-5.4-mini")          # explicit provider
create_client("claude-sonnet-4-6")            # inferred: anthropic
create_client("gemini-2.5-pro")               # inferred: google
create_client("grok-3")                       # inferred: xai
create_client("azure-responses:gpt-5.4-mini") # Azure Responses API
create_client("bedrock:openai.gpt-oss-120b")  # AWS Bedrock
```

## Compatibility & status

| Item | Value |
|---|---|
| Python | 3.13, 3.14 |
| OS | Linux, macOS, Windows |
| Maturity | 0.1.0 GA; SemVer, pre-1.0 minor bumps may break public API |
| Tests | 924 unit + 5 live integration |
| Type checker | `ty` (clean) |

## Configuration

All settings use the `KAOS_LLM_` prefix via `KaosLLMSettings` (`ModuleSettings` subclass). Each provider key has a legacy fallback (e.g. `OPENAI_API_KEY`) for backward compatibility.

| Variable | Default | Description |
|----------|---------|-------------|
| `KAOS_LLM_{OPENAI,ANTHROPIC,GOOGLE,XAI,GROQ,MISTRAL,OPENROUTER}_API_KEY` | — | Direct-provider API key (`SecretStr`) |
| `KAOS_LLM_OPENAI_BASE_URL` | `https://api.openai.com` | Override for proxies / local models (per-provider variants exist) |
| `KAOS_LLM_AZURE_OPENAI_ENDPOINT` | — | Azure resource URL (e.g. `https://my-resource.openai.azure.com/`) |
| `KAOS_LLM_AZURE_OPENAI_API_KEY` | — | Azure resource subscription key (alternative to AAD) |
| `KAOS_LLM_AZURE_OPENAI_AD_TOKEN` | — | Static AAD bearer (use `azure_ad_token_provider=` for refresh) |
| `KAOS_LLM_AZURE_OPENAI_API_VERSION` | `2024-12-01-preview` | Azure API version (bump to `2025-04-01-preview` for newer Responses-API features) |
| `KAOS_LLM_BEDROCK_API_KEY` | — | AWS Bedrock bearer token; legacy fallback `AWS_BEARER_TOKEN_BEDROCK` |
| `KAOS_LLM_BEDROCK_BASE_URL` | `https://bedrock-mantle.us-east-2.api.aws` | Bedrock endpoint (override for other regions) |
| `KAOS_LLM_DEFAULT_TIMEOUT` | `120.0` | Request timeout (seconds) |
| `KAOS_LLM_DEFAULT_MAX_RETRIES` | `3` | Max retry attempts |
| `KAOS_LLM_MAX_RESPONSE_BYTES` | `33554432` | 32 MiB cap on non-streaming responses |
| `KAOS_LLM_STREAM_MAX_DURATION` | `600.0` | Wall-clock cap on a streaming response (seconds) |
| `KAOS_LLM_CACHE_ENABLED` | `false` | Enable response caching |
| `KAOS_LLM_CACHE_PATH` | `~/.cache/kaos/llm` | Cache directory |

Per-request overrides flow through `KaosContext._config` for MCP callers.

## CLI

```bash
kaos-llm-client check [--provider openai,anthropic] [--json]   # verify credentials
kaos-llm-client chat --model openai:gpt-5 --message "Hello!" [--system "..."] [--json]
kaos-llm-client profiles [--json]                              # list known model profiles
kaos-llm-client config [--json]                                # resolved settings (redacted)
```

All commands support `--json` with a consistent envelope: `{"command": "...", ...}`.

## MCP Server

```bash
kaos-llm-serve                                                # stdio (Claude Code / Desktop)
kaos-llm-serve --http --port 8000                             # streamable HTTP
kaos-llm-serve --model openai:gpt-5 --http --debug            # default model + debug logging
```

Exposes `kaos-llm-chat`, `kaos-llm-json`, and `kaos-llm-embed` MCP tools.

> **Security**: the HTTP transport has no built-in authentication or rate
> limiting. The default `--host 127.0.0.1` binds to loopback, which is
> the safe default. **Do not bind to a non-loopback interface unless you
> put an authenticated reverse proxy (mTLS, OAuth, IP allowlist, etc.)
> in front of it** — anyone who can reach the port can spend your
> configured LLM credits. The server emits a startup warning when
> `--host` is not loopback. See `kaos_llm_client/serve.py` module
> docstring for the full guidance.

## Documentation

Per-package reference: see the in-tree docstrings and the
[CHANGELOG.md](CHANGELOG.md).

Cross-cutting KAOS guides (agentic patterns, persona presets, settings
policy, citations, MCP data flow, migration to 0.1.0 GA) live in
[`kaos-modules/docs/guides/`](https://github.com/273v/kaos-modules/tree/main/docs/guides).

## Companion packages

Direct dependencies in the KAOS stack:

- **kaos-core** — runtime, `ModuleSettings`, `KaosContext`, structured logging
- **kaos-mcp** (optional via `[mcp]`) — FastMCP bridge for `kaos-llm-serve`

Higher layers consume `kaos-llm-client` for inference: `kaos-llm-core` (typed programs), `kaos-agents` (runtime), `kaos-citations` (verification). Full module roster at [docs.kelvin.legal/kaos-llm-client](https://docs.kelvin.legal/kaos-llm-client/).

## Development

```bash
uv sync --group dev
uv run ruff format kaos_llm_client/ tests/
uv run ruff check kaos_llm_client/ tests/
uv run ty check kaos_llm_client/ tests/
uv run pytest tests/unit/ -q
# live tier requires provider keys; see tests/integration/
uv run pytest tests/integration/ -q
```

## Build from source

```bash
uv build
uv pip install dist/kaos_llm_client-*.whl
```

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md)
for setup, quality gates, pull request expectations, and engineering
standards. By contributing you agree to follow the
[project conduct expectations](CODE_OF_CONDUCT.md) and certify the
[Developer Certificate of Origin v1.1](https://developercertificate.org/) —
sign every commit with `git commit -s`. Please open an issue before starting
on a non-trivial change so we can align on scope.

## Security

For security issues, **please do not file a public issue**. Report privately
via [GitHub Private Vulnerability Reporting](https://github.com/273v/kaos-llm-client/security/advisories/new)
or email **security@273ventures.com**. See [SECURITY.md](SECURITY.md) for the
full disclosure policy.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Copyright 2026 [273 Ventures LLC](https://273ventures.com).
Built for [kelvin.legal](https://kelvin.legal).
