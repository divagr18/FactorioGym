"""`factoriorl demo`: its flags, its output path, and its spend cap.

The demonstration is still the five-phase driver -- commission, measure a
production window, inject a fuel outage, recover, measure again -- because that
structure is what DESIGN 5.7's published evidence *is*. These cover the three
defects around it, not the phases themselves.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys as _sys
from pathlib import Path

from factoriorl.agent import provider as provider_module
from factoriorl.agent.guarded import BudgetedAdapter

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "tools" / "demonstration.py"


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_sys.executable, "-m", "factoriorl.cli", *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


# --- the flags are visible ---------------------------------------------------


def test_demo_help_shows_its_flags() -> None:
    """It was declared `nargs=REMAINDER`, so `--help` printed nothing at all --
    including the two flags that decide whether a run bills a provider and
    whether it overwrites tracked evidence."""
    result = _cli("demo", "--help")
    assert result.returncode == 0
    for flag in ("--model", "--out", "--publish-evidence", "--max-cost-usd"):
        assert flag in result.stdout, flag


def test_every_flag_the_cli_declares_is_one_the_tool_accepts() -> None:
    """Two argparse layers that disagree would accept a flag and drop it.

    The CLI forwards only what was supplied, so the tool keeps owning the
    defaults -- but the *names* must match or the forward fails at the far end.
    """
    cli_source = (ROOT / "src" / "factoriorl" / "cli.py").read_text(encoding="utf-8")
    block = cli_source[cli_source.index('sub.add_parser("demo"') :]
    block = block[: block.index("sub.add_parser(", 20)]
    declared = set(_long_options(block))

    tool = set(_long_options(DEMO.read_text(encoding="utf-8")))
    missing = sorted(declared - tool - {"--help"})
    assert not missing, f"the CLI forwards flags tools/demonstration.py does not accept: {missing}"


#: `add_argument("--flag"` -- matched on source text rather than parsed,
#: because the CLI side is a slice from the middle of a function and does not
#: stand alone as a module.
_LONG_OPTION = re.compile(r'add_argument\(\s*"(--[a-z0-9-]+)"')


def _long_options(source: str) -> list[str]:
    return _LONG_OPTION.findall(source)


# --- it no longer overwrites tracked evidence unasked ------------------------


def _tests_publish_flag(test: ast.expr) -> bool:
    return any(
        isinstance(node, ast.Attribute) and node.attr == "publish_evidence"
        for node in ast.walk(test)
    )


def _mentions_tracked_evidence(node: ast.AST) -> int:
    return sum(
        1
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and child.value == "phase5-demonstration.json"
    )


def test_publishing_over_the_tracked_evidence_file_is_opt_in() -> None:
    """It wrote `docs/evidence/phase5-demonstration.json` unconditionally with no
    `--out`, so every run dirtied the working tree and destroyed committed
    evidence. `docs/AGENT.md` had to tell people to stash afterwards.

    Checked structurally rather than by looking for the guard near the filename:
    a text search would keep passing after the guard was moved. Every mention of
    the filename must sit inside the body of an `if args.publish_evidence:`.
    """
    tree = ast.parse(DEMO.read_text(encoding="utf-8"))
    total = _mentions_tracked_evidence(tree)
    guarded = sum(
        _mentions_tracked_evidence(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and _tests_publish_flag(node.test)
    )
    assert total, "the tracked evidence filename is gone entirely; this check is now vacuous"
    assert guarded == total, (
        f"{total - guarded} of {total} mention(s) of the tracked evidence file sit "
        f"outside an `if args.publish_evidence:` branch, so a plain run can "
        f"overwrite committed evidence again"
    )


def test_the_default_destination_is_the_run_directory() -> None:
    source = DEMO.read_text(encoding="utf-8")
    assert 'loop.run_dir / "demonstration.json"' in source


# --- it cannot bill a provider without a ceiling -----------------------------


def test_the_demonstration_builds_its_provider_through_the_shared_guard() -> None:
    """It constructed a bare `OpenAICompatibleAdapter` and so had no spend cap at
    all -- harmless against a local endpoint, a liability the moment any
    entrypoint points at a paid provider."""
    source = DEMO.read_text(encoding="utf-8")
    assert "provider_module.build(" in source
    assert "OpenAICompatibleAdapter(" not in source, "it is building a raw adapter again"


def test_the_builder_always_returns_a_guarded_adapter() -> None:
    """The property that makes the check above worth anything: there is no
    argument to `build()` that yields an unguarded provider."""
    built = provider_module.build(model="local-model", max_cost_usd=1.0)
    assert isinstance(built.adapter, BudgetedAdapter)
    assert built.budget.cap_usd == 1.0
    assert built.adapter.inner is built.inner


def test_the_builder_refuses_an_unpriced_model_before_launching_anything() -> None:
    from factoriorl.pricing import UnknownModelPrice

    try:
        provider_module.build(model="a-model-nobody-priced")
    except UnknownModelPrice as refused:
        assert "snapshotted price" in str(refused)
    else:  # pragma: no cover - the assertion is the point
        raise AssertionError("an unpriced model must fail the preflight")


def test_the_guarded_adapter_reports_the_provider_not_the_wrapper() -> None:
    built = provider_module.build(model="local-model")
    assert built.adapter.describe()["adapter"] == built.inner.describe()["adapter"]
    assert "budget" in built.to_dict()
