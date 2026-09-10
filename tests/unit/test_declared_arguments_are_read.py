"""Every declared command-line argument must be read by the module declaring it.

This is a countermeasure, not a style rule. The single most expensive defect
class in this repository is **something declared that nothing reads**, and it
has landed six times:

- `tools/clone_expert.py` accepted `--holdout` and never referenced it, while
  its own docstring promised "same frozen holdout if one is given". Every
  behaviour-cloning number was scored on the run's own scenes and published as
  though it were the frozen set -- the same unpaired comparison that got 4.4
  withdrawn. A day of results had to be re-run.
- `factoriorl evaluate` had no `--holdout` at all, so the one supported way to
  score a provided checkpoint could not reproduce a single published number.
- `RegisteredTask.solve` was declared and never read, so authoring a task
  required editing framework internals -- an R6 gate clause.
- A docstring promised a disruption check that did not exist.
- `clone_expert` published success rates with no random floor, against an
  explicit project rule that a rate is quoted only with its floor.
- `episode_indices` was hardcoded to `range(eval_episodes)`, recording a false
  provenance claim for exactly the row whose provenance matters.

Only the first two are mechanically detectable, and this test detects them. It
passes today, which is the point: it is here to catch the seventh occurrence at
the moment it is written rather than after it has invalidated a day of results.

The check is deliberately shallow -- a name appearing anywhere in the module
counts as read. A shallow check that runs on every commit beats a precise one
nobody maintains, and the failure it guards against is an argument referenced
*nowhere*, not one referenced badly.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# `argparse` fills these in itself; a module never names them.
BUILT_IN = {"help", "version"}


def _dest(call: ast.Call) -> str | None:
    """The attribute name `argparse` will set on the namespace."""
    for keyword in call.keywords:
        if keyword.arg == "dest" and isinstance(keyword.value, ast.Constant):
            return str(keyword.value.value)
    chosen = None
    for argument in call.args:
        if not (isinstance(argument, ast.Constant) and isinstance(argument.value, str)):
            continue
        value = argument.value
        if value.startswith("--"):
            chosen = value  # a long option wins, as argparse does
        elif not value.startswith("-") and chosen is None:
            chosen = value  # positional
    return chosen.lstrip("-").replace("-", "_") if chosen else None


def _modules() -> list[Path]:
    files = sorted(p for p in (ROOT / "tools").glob("*.py") if not p.name.startswith("_"))
    files.append(ROOT / "src" / "factoriorl" / "cli.py")
    return [p for p in files if p.is_file()]


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_every_declared_argument_is_referenced(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if "add_argument" not in source:
        pytest.skip("declares no command-line arguments")

    tree = ast.parse(source)

    # A module that forwards the namespace wholesale reads its arguments by a
    # route this analysis cannot see, so name-matching would report false
    # failures. None do today; the branch exists so that adding one is a
    # deliberate choice rather than a silently broken test.
    forwards_namespace = any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "vars"
        for node in ast.walk(tree)
    )
    if forwards_namespace:
        pytest.skip(f"{path.name} forwards the namespace with vars(); names are not traceable")

    referenced = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    referenced |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

    unread = sorted(
        {
            dest
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and (dest := _dest(node))
            and dest not in BUILT_IN
            and dest not in referenced
        }
    )

    assert not unread, (
        f"{path.relative_to(ROOT)} declares command-line arguments that nothing in the "
        f"module reads: {unread}. An accepted-and-ignored argument is worse than a "
        f"missing one -- it reports success while doing nothing, which is how "
        f"`clone_expert --holdout` published a day of unpaired results as frozen. "
        f"Either read it or remove it."
    )


def test_the_check_would_catch_the_defect_it_exists_for() -> None:
    """Guard the guard: a shallow check that silently stops working is worthless."""
    source = (
        "import argparse\n"
        "def main():\n"
        "    p = argparse.ArgumentParser()\n"
        "    p.add_argument('--holdout')\n"
        "    p.add_argument('--task')\n"
        "    args = p.parse_args()\n"
        "    return args.task\n"
    )
    tree = ast.parse(source)
    referenced = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    referenced |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    unread = sorted(
        {
            dest
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and (dest := _dest(node))
            and dest not in BUILT_IN
            and dest not in referenced
        }
    )
    # Exactly the shape of the real defect: `--task` is read, `--holdout` is not.
    assert unread == ["holdout"]
