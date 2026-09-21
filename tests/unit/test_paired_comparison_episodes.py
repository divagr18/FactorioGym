"""Both comparison arms must score the *whole* frozen set (DESIGN 4.4, 4b.3).

`tools/shaping_comparison.py` and `tools/skill_ablation.py` carry a
byte-identical `pairing()` guard that compares each arm's recorded
`scored_episodes` and refuses a verdict when they differ. The guard is right,
and it is what withdrew 4.4's headline: shaped 35/50 against sparse 48/50, run
without a holdout, so the arms never saw the same worlds.

But both tools then asked for **fewer episodes than the frozen range** — 25 and
50 against 100. `evaluate_parallel` scores the first N of the range *to finish*
(`train.py:416`, `vecenv.py:181-207`), and which N that is depends on episode
length and worker scheduling. So two policies can legitimately score different
subsets, the guard fires, and the re-run is refused for a reason that has
nothing to do with shaping or skills. Only the full range makes `only_indices`
cover it and the episode identities identical by construction.

That is a defect that would have wasted the 4.4 re-run rather than one that
would have produced a wrong answer, which is why it is worth a test: a refusal
looks like caution, not like a bug.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
HOLDOUT = ROOT / "docs" / "evidence" / "holdout_v3.json"


def _tool(name: str):
    """Load a tool by path; `tools/` is not an importable package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def shaping():
    return _tool("shaping_comparison")


def test_the_default_is_the_whole_frozen_range(shaping):
    frozen = json.loads(HOLDOUT.read_text(encoding="utf-8"))["holdout"]["episodes_per_task"]
    assert shaping.frozen_episodes("docs/evidence/holdout_v3.json", None) == frozen
    assert frozen == 100, "if the freeze changes size, this test should say so"


def test_an_explicit_count_still_wins(shaping):
    """Deliberately overridable: a smaller run is legitimate for a smoke test,
    it just cannot produce a paired verdict."""
    assert shaping.frozen_episodes("docs/evidence/holdout_v3.json", 30) == 30


def test_without_a_holdout_it_falls_back_rather_than_crashing(shaping):
    """`--holdout ''` already forfeits pairing, so any count is as good as
    another; it must not raise on the missing file."""
    assert shaping.frozen_episodes("", None) == shaping.DEFAULT_UNPAIRED_EPISODES


def test_both_tools_resolve_it_the_same_way():
    """The two tools had the same latent defect because they are near-copies.
    Sharing the resolver is what stops them drifting apart again."""
    ablation = _tool("skill_ablation")
    source = (ROOT / "tools" / "skill_ablation.py").read_text(encoding="utf-8")
    assert "from shaping_comparison import frozen_episodes" in source
    # The resolver reached through the ablation tool, not merely imported by it.
    # `assert ablation is not None` was the first version of this line and could
    # not fail: `module_from_spec` either raises or returns a module.
    assert ablation.frozen_episodes("docs/evidence/holdout_v3.json", None) == 100


def test_neither_tool_hardcodes_an_episode_count_any_more():
    """The specific regression: a literal count in the subprocess command."""
    for name in ("shaping_comparison", "skill_ablation"):
        source = (ROOT / "tools" / f"{name}.py").read_text(encoding="utf-8")
        index = source.index('"--eval-episodes",')
        following = source[index : index + 120]
        assert "str(episodes)" in following, f"{name} passes a literal: {following[:80]!r}"
