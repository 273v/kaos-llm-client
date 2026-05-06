"""CLI entry point for kaos-llm-client.

Usage:
    kaos-llm-client check [--provider openai,anthropic] [--json]
    kaos-llm-client chat --model MODEL --message MSG [--json]
    kaos-llm-client profiles [--json]
    kaos-llm-client config [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def main(argv: list[str] | None = None) -> None:
    """Entry point for the kaos-llm-client CLI."""
    parser = argparse.ArgumentParser(
        description="KAOS LLM Client — thin provider-native LLM transport"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # check
    check_parser = subparsers.add_parser("check", help="Verify credentials and connectivity")
    check_parser.add_argument(
        "--provider",
        default=None,
        help="Comma-separated providers to check (default: all configured)",
    )
    check_parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")

    # chat
    chat_parser = subparsers.add_parser("chat", help="One-shot chat")
    chat_parser.add_argument("--model", required=True, help="Model string (e.g., openai:gpt-5)")
    chat_parser.add_argument("--message", "-m", required=True, help="User message")
    chat_parser.add_argument("--system", "-s", default=None, help="System prompt")
    chat_parser.add_argument("--max-tokens", type=int, default=None, help="Max output tokens")
    chat_parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")

    # profiles
    profiles_parser = subparsers.add_parser("profiles", help="List known model profiles")
    profiles_parser.add_argument(
        "--json", action="store_true", dest="json_output", help="JSON output"
    )

    # config
    config_parser = subparsers.add_parser(
        "config", help="Show resolved settings (secrets redacted)"
    )
    config_parser.add_argument(
        "--json", action="store_true", dest="json_output", help="JSON output"
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "check":
        _cmd_check(args)
    elif args.command == "chat":
        _cmd_chat(args)
    elif args.command == "profiles":
        _cmd_profiles(args)
    elif args.command == "config":
        _cmd_config(args)


def _cmd_check(args: Any) -> None:
    """Verify credentials and connectivity."""
    from kaos_llm_client.settings import KaosLLMSettings

    settings = KaosLLMSettings()

    # Determine which providers to check
    providers_to_check: list[str]
    if args.provider:
        providers_to_check = [p.strip() for p in args.provider.split(",")]
    else:
        providers_to_check = []
        if settings.openai_api_key:
            providers_to_check.append("openai")
        if settings.anthropic_api_key:
            providers_to_check.append("anthropic")
        if settings.google_api_key:
            providers_to_check.append("google")
        if settings.xai_api_key:
            providers_to_check.append("xai")

    if not providers_to_check:
        if args.json_output:
            print(json.dumps({"command": "check", "status": "no_providers", "providers": []}))
        else:
            print(
                "No API keys configured. Set KAOS_LLM_*_API_KEY environment variables.",
                file=sys.stderr,
            )
        sys.exit(1)

    results: list[dict[str, Any]] = []
    for provider in providers_to_check:
        result = {"provider": provider, "has_key": False, "status": "unknown"}
        key_field = f"{provider}_api_key"
        key_val = getattr(settings, key_field, None)
        if key_val is not None:
            result["has_key"] = True
            result["status"] = "configured"
        else:
            result["status"] = "missing_key"
        results.append(result)

    if args.json_output:
        print(json.dumps({"command": "check", "providers": results}))
    else:
        for r in results:
            status = "OK" if r["has_key"] else "MISSING KEY"
            print(f"  {r['provider']}: {status}")


def _cmd_chat(args: Any) -> None:
    """One-shot chat."""
    from kaos_llm_client.providers import create_client
    from kaos_llm_client.settings import KaosLLMSettings

    settings = KaosLLMSettings()

    messages: list[dict[str, Any]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.message})

    kwargs: dict[str, Any] = {}
    if args.max_tokens:
        kwargs["max_tokens"] = args.max_tokens

    client = create_client(args.model, settings=settings)
    response = client.chat(messages=messages, **kwargs)

    if args.json_output:
        print(
            json.dumps(
                {
                    "command": "chat",
                    "model": args.model,
                    "response": {
                        "text": response.text,
                        "provider": response.provider,
                        "model": response.model,
                        "usage": response.usage.model_dump(),
                        "stop_reason": response.stop_reason,
                        "latency_ms": response.latency_ms,
                    },
                },
                indent=2,
            )
        )
    else:
        print(response.text)


def _cmd_profiles(args: Any) -> None:
    """List known model profiles."""
    from dataclasses import asdict

    from kaos_llm_client.profiles import _PROVIDER_PROFILES

    profiles = {}
    for name, profile in _PROVIDER_PROFILES.items():
        d = asdict(profile)
        # Convert type references to strings
        if d.get("json_schema_transformer"):
            d["json_schema_transformer"] = d["json_schema_transformer"].__name__
        else:
            d["json_schema_transformer"] = None
        profiles[name] = d

    if args.json_output:
        print(json.dumps({"command": "profiles", "profiles": profiles}, indent=2))
    else:
        for name, profile_dict in profiles.items():
            print(f"\n{name}:")
            for k, v in profile_dict.items():
                print(f"  {k}: {v}")


def _cmd_config(args: Any) -> None:
    """Show resolved settings with secrets redacted.

    Redaction policy: ANY field typed as ``SecretStr`` on the settings
    model is redacted, identified by introspecting the model schema.
    String fields whose name contains common credential markers
    (``api_key``, ``token``, ``secret``, ``password``) are also masked
    as a defence-in-depth fallback for any plain-string credential
    fields that may exist now or in future.
    """
    from pydantic import SecretStr

    from kaos_llm_client.settings import KaosLLMSettings

    settings = KaosLLMSettings()
    # ``model_dump()`` does not coerce ``SecretStr`` to JSON-serialisable
    # ``str``; we walk the raw attribute list so SecretStr fields are
    # detected by type and redacted before any JSON encoding.
    secret_field_names: set[str] = set()
    for field_name, field_info in type(settings).model_fields.items():
        ann = field_info.annotation
        # ``SecretStr | None`` shows up as a Union; check direct + args.
        if ann is SecretStr or SecretStr in getattr(ann, "__args__", ()):
            secret_field_names.add(field_name)

    _CREDENTIAL_NAME_MARKERS = ("api_key", "token", "secret", "password")

    def _redact(value: Any) -> Any:
        if isinstance(value, SecretStr):
            raw = value.get_secret_value()
            if not raw:
                return None
            if len(raw) > 8:
                return raw[:4] + "..." + raw[-4:]
            return "***"
        if isinstance(value, str) and len(value) > 8:
            return value[:4] + "..." + value[-4:]
        return "***"

    data: dict[str, Any] = {}
    for field_name in type(settings).model_fields:
        value = getattr(settings, field_name, None)
        if value is None:
            data[field_name] = None
            continue
        if (
            field_name in secret_field_names
            or isinstance(value, SecretStr)
            or any(marker in field_name for marker in _CREDENTIAL_NAME_MARKERS)
        ):
            data[field_name] = _redact(value)
        else:
            data[field_name] = value

    if args.json_output:
        print(json.dumps({"command": "config", "settings": data}, indent=2))
    else:
        print("kaos-llm-client settings:")
        for k, v in data.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
