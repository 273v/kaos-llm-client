# Agent Guidance

## Scope

This file is the canonical instruction file for coding agents working in
this repository. Follow it together with [CONTRIBUTING.md](CONTRIBUTING.md)
and the detailed standards under [docs/standards/](docs/standards/).

Keep changes focused and preserve user work already present in the
worktree. Do not edit generated files or release artifacts unless the
task explicitly requires that class of change.

## Project Identity

- Distribution: `kaos-llm-client`
- Import package: `kaos_llm_client`
- Python: 3.13+
- Package manager and environment runner: `uv`
- CLI entry points: `kaos-llm-client`, `kaos-llm-serve`
- Public optional extra currently declared: `azure`

The package is a thin, provider-native LLM client. It exposes typed
model, message, schema, tool, streaming, embedding, caching, provider
wrapper, CLI, and MCP surfaces.

## Setup

Use the repository tooling:

```bash
uv sync --group dev
```

Install pre-commit when preparing normal contribution work:

```bash
uvx pre-commit install
```

Do not add undeclared runtime dependencies. Keep optional provider or
cloud integrations behind extras and lazy imports.

## Local Checks

Use the quality gate from [CONTRIBUTING.md](CONTRIBUTING.md):

```bash
uv run ruff format --check kaos_llm_client tests
uv run ruff check kaos_llm_client tests
uv run ty check kaos_llm_client tests
uv run pytest tests/unit/ -q --no-cov
```

Type checking uses `ty`, not mypy. Inline suppressions use
`# ty: ignore[...]` with the narrowest practical rule and a reason when
the reason is not obvious.

When packaging, metadata, README rendering, or release behavior changes,
also run:

```bash
uv build
uvx --from twine twine check --strict dist/*
```

For docs-only changes, also run `git diff --check` and a markdown/link
sanity check over the changed Markdown files.

## Architecture Rules

Follow the architecture standards in
[docs/standards/python-design-and-architecture.md](docs/standards/python-design-and-architecture.md).
Keep the public API stable and intentional:

- Treat exports from `kaos_llm_client.__all__`, documented classes and
  functions, Pydantic models, CLI flags and JSON output, MCP tools and
  schemas, environment variables, and documented configuration as public
  contracts.
- Keep import-time work cheap. Do not perform network calls, provider
  initialization, filesystem scans, logging setup, or expensive model
  work at import time.
- Use Pydantic at external boundaries and typed dataclasses or explicit
  result types internally where they fit.
- Preserve the package's typed public model, message, schema, tool, and
  response contracts.
- Keep dependency-specific code centralized in provider or integration
  adapters so provider details do not leak through the package.

## LLM Transport Principles

Provider clients must stay behind typed adapters. New provider behavior
should fit the same sync, async, streaming, tools, structured-output,
embedding, error, retry, timeout, cost, token-accounting, logging, and
hook contracts as existing providers.

When changing transport behavior:

- Preserve consistent sync, async, and streaming semantics.
- Keep retries, timeouts, response-size limits, stream-duration limits,
  token accounting, and cost estimation explicit and testable.
- Do not leak raw provider payloads, credentials, auth headers, bearer
  tokens, internal filesystem paths, or provider stack traces in public
  errors, logs, CLI JSON, or MCP responses.
- Keep provider request and response translation inside typed adapters.
- Keep optional provider dependencies behind declared extras and lazy
  imports.
- Preserve existing model/message/schema contracts unless the change is
  intentionally public, tested, documented, and reflected in release
  notes.
- Prefer `ModelProfile` or equivalent typed capability metadata over
  provider-name conditionals scattered through call sites.

## Testing

Follow [docs/standards/tests-fixtures-ci.md](docs/standards/tests-fixtures-ci.md).
Unit tests must be deterministic, offline, and free of credentials.

Live and network tests are opt-in. They may require provider credentials
and must never run as part of the default unit gate. Mark live-provider
tests clearly, skip them when credentials are absent, and never commit,
print, log, snapshot, or record real secrets.

New public API, CLI behavior, schema output, provider behavior, security
behavior, and bug fixes need tests through the real entry point whenever
practical. Mocked-only tests are not enough for security-sensitive
behavior.

## Security

Follow [SECURITY.md](SECURITY.md) for vulnerability reporting and
[docs/standards/code-quality-standards.md](docs/standards/code-quality-standards.md)
for security expectations.

- Never commit secrets, tokens, credential material, `.env` files,
  provider transcripts containing secrets, or credential-bearing cassette
  data.
- Use secret-aware types and redaction for settings, logs, CLI output,
  JSON output, errors, hooks, and test fixtures.
- Keep HTTP, file, path, cache, streaming, and subprocess behavior
  bounded by explicit size, time, path, and concurrency limits.
- Do not add GPL, AGPL, unknown-license, non-commercial, or
  no-derivatives dependencies.
- The MCP HTTP transport must remain safe by default. Do not weaken
  loopback defaults or warnings without explicit security review.

## Commits, PRs, And Releases

Follow [docs/standards/engineering-process.md](docs/standards/engineering-process.md)
and [CONTRIBUTING.md](CONTRIBUTING.md).

- Use conventional commit messages.
- Sign commits with `git commit -s`.
- Keep PRs to one logical change and explain what changed, why, and how
  it was verified.
- Consider public API, CLI, MCP schema, package metadata, fixture,
  security, and release impact before requesting review.
- Update `CHANGELOG.md` for user-visible changes, including public API,
  CLI behavior, schema output, package metadata, security behavior, and
  deprecations.
- Do not move public tags. Release tags use `v<version>` and point to
  the exact commit used for published artifacts.
