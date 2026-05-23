"""Run the KAOS MCP server with LLM client tools.

Usage:
    # stdio (for Claude Code / Claude Desktop)
    kaos-llm-serve

    # streamable HTTP
    kaos-llm-serve --http --port 8000

    # with a default model
    kaos-llm-serve --model openai:gpt-5

    # all options + debug logging
    kaos-llm-serve --model anthropic:claude-sonnet-4-6 --http --debug

Security
--------

The HTTP transport (``--http``) exposes tools that consume the
configured LLM-provider credentials. There is **no built-in
authentication or rate limiting** in this server — the default
``--host 127.0.0.1`` binds to loopback, which is the only safe default.

If you bind to a non-loopback interface (``--host 0.0.0.0``,
``--host 192.168.x.x``, ``--host ::``):

- **Put authenticated reverse proxy in front** (mTLS, OAuth, signed
  JWT, IP allowlist, or a service mesh sidecar). Anyone who can reach
  the port can spend your provider credits.
- **Apply rate limits** at the proxy. A single misbehaving caller can
  exhaust your monthly quota in minutes.
- **Treat the port as a credential boundary.** The server holds the
  api-keys; do not expose it to networks you do not control.

This warning is mirrored in the README and in the runtime startup log
when ``--host`` differs from ``127.0.0.1``.

NOTE: Add to pyproject.toml [project.scripts]:
    kaos-llm-serve = "kaos_llm_client.serve:main"
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """Entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="KAOS MCP Server with LLM client tools")
    parser.add_argument("--http", action="store_true", help="Use streamable HTTP transport")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--model",
        default=None,
        help="Default model for tools (e.g., openai:gpt-5, anthropic:claude-sonnet-4-6)",
    )
    args = parser.parse_args(argv)

    try:
        from kaos_core import KaosRuntime

        # kaos-mcp is the [mcp] optional dep; absent from the base
        # install and from `uv sync --group dev`, so ty cannot resolve
        # it at check time. The try/except handles the import failure.
        from kaos_mcp import KaosMCPServer, KaosMCPSettings  # ty: ignore[unresolved-import]
    except ImportError:
        print(
            "Error: MCP server requires kaos-mcp.\nInstall with: pip install kaos-llm-client[mcp]",
            file=sys.stderr,
        )
        sys.exit(1)

    from kaos_llm_client.tools import register_llm_tools

    # Create runtime and register LLM tools
    runtime = KaosRuntime()
    n_tools = register_llm_tools(runtime, default_model=args.model)
    print(f"Registered {n_tools} LLM client tools", file=sys.stderr)
    if args.model:
        print(f"Default model: {args.model}", file=sys.stderr)

    # Configure server
    instructions = (
        "kaos-llm-client provides LLM inference tools for calling language models. "
        "Use kaos-llm-chat for one-shot chat completions with any supported provider. "
        "Use kaos-llm-json for structured JSON output matching a schema. "
        "Use kaos-llm-embed for generating text embeddings. "
        "Model strings use 'provider:model' format (e.g., 'openai:gpt-5', "
        "'anthropic:claude-sonnet-4-6'). If no provider prefix is given, "
        "the provider is inferred from the model name."
    )
    settings = KaosMCPSettings(
        name="kaos-llm-client-server",
        instructions=instructions,
        transport="streamable-http" if args.http else "stdio",
        host=args.host,
        port=args.port,
        debug=args.debug,
    )

    server = KaosMCPServer(runtime=runtime, settings=settings)

    if args.http:
        print(f"Starting HTTP server on {args.host}:{args.port}/mcp", file=sys.stderr)
        if args.host not in ("127.0.0.1", "localhost", "::1"):
            print(
                "WARNING: HTTP server is bound to a non-loopback interface "
                f"({args.host}). This server has no built-in auth or rate "
                "limiting; anyone who can reach the port can spend your "
                "configured LLM credits. Put an authenticated reverse proxy "
                "in front, or restrict access at the network layer. See the "
                "module docstring for details.",
                file=sys.stderr,
            )
        server.run_streamable_http()
    else:
        print("Starting stdio server", file=sys.stderr)
        server.run_stdio()


if __name__ == "__main__":
    main()
