"""Pins the install contract for `kaos-llm-serve`.

audit-04 F-001: `README.md:20-21` documents
`uv add 'kaos-llm-client[mcp]>=0.1.0'` but `pyproject.toml` did not
declare the `[mcp]` extra. The `kaos-llm-serve` console script imports
`kaos_mcp` at runtime; without the extra users hit ImportError with
no remediation path.

This test runs in an environment that intentionally does not install
`kaos-mcp` (it's an optional extra), so the failure path is exercised
without any mocking — the `from kaos_mcp import ...` inside `main`
raises ImportError naturally.
"""

from __future__ import annotations

import pytest

from kaos_llm_client import serve


def test_serve_main_exits_with_mcp_extra_hint_when_kaos_mcp_missing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without kaos-mcp the entry point must exit 1 and name `[mcp]`.

    The import is inside ``main`` (not at module top level) so importing
    the module succeeds without the extra; only invoking ``main``
    triggers the failure. The error message must cite the canonical
    install extra (``pip install kaos-llm-client[mcp]``) rather than
    naming ``kaos-mcp`` directly, so the message tracks the declared
    ``[project.optional-dependencies]`` table.
    """
    with pytest.raises(SystemExit) as excinfo:
        serve.main([])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "[mcp]" in err, f"expected '[mcp]' in stderr, got: {err!r}"
    assert "kaos-llm-client[mcp]" in err, (
        f"expected the canonical install hint to name the package + extra, got: {err!r}"
    )
