"""Two nominally identical comparisons must be shown to score identical scenes.

R0.2: a numeric seed does not name a scene set, so a run sharing `master_seed`
with another is not a replication of it.
"""

from __future__ import annotations

import json

import pytest

from factoriorl.baselines import cache_key
from factoriorl.manifest import _git, amend, runs_dir
from factoriorl.tasks import get


def _key(**overrides):
    args = {
        "task": get("deliver"),
        "split": "test",
        "master_seed": 20260908,
        "episodes": 100,
        "action_space": "primitive+skills",
        "seed_run_id": "holdout-v3",
        "start_index": 3000,
    }
    args.update(overrides)
    return cache_key(**args)


class TestBaselineCacheIdentity:
    """The floor is a property of the scenes, not just of the seed number."""

    def test_the_holdouts_that_actually_collided_now_differ(self):
        # All three holdouts share master=20260908; deliver is v1.3.0 in each.
        v1 = _key(seed_run_id="holdout-v1", start_index=1000)
        v2 = _key(seed_run_id="holdout-v2", start_index=2000)
        v3 = _key(seed_run_id="holdout-v3", start_index=3000)
        assert len({v1, v2, v3}) == 3

    def test_run_id_alone_separates_the_key(self):
        assert _key(seed_run_id="holdout-v1") != _key(seed_run_id="holdout-v3")

    def test_start_index_alone_separates_the_key(self):
        assert _key(start_index=1000) != _key(start_index=3000)

    def test_the_action_space_still_separates_the_key(self):
        assert _key(action_space="primitive-v1") != _key(action_space="primitive+skills")

    def test_the_task_version_still_separates_the_key(self):
        assert _key(task=get("deliver")) != _key(task=get("repair_belt"))

    def test_an_identical_stream_is_a_cache_hit(self):
        assert _key() == _key()


class TestManifestAmend:
    def test_merges_one_level_deep_without_dropping_siblings(self, tmp_path, monkeypatch):
        monkeypatch.setattr("factoriorl.manifest.runtime_dir", lambda: tmp_path)
        directory = runs_dir() / "run-x"
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps({"seeds": {"master": 1, "run_id": "a"}, "model": None}),
            encoding="utf-8",
        )

        amend("run-x", {"seeds": {"eval_streams": {"structures": {"start_index": 3000}}}})
        document = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

        assert document["seeds"]["master"] == 1, "amend must not replace the seeds block"
        assert document["seeds"]["run_id"] == "a"
        assert document["seeds"]["eval_streams"]["structures"]["start_index"] == 3000

    def test_a_missing_manifest_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr("factoriorl.manifest.runtime_dir", lambda: tmp_path)
        assert amend("no-such-run", {"model": {"checkpoint_sha256": "x"}}) is None


class TestDirtyTreeIdentity:
    def test_a_dirty_tree_carries_a_digest_and_its_paths(self):
        state = _git()
        assert "commit" in state
        if state["dirty"]:
            assert len(state["diff_digest"]) == 16
            assert state["dirty_paths"], "a dirty tree must name what is dirty"
            # The old fixed slice ate the first character of every path.
            assert all(not p.startswith("rc/") for p in state["dirty_paths"])
        else:
            assert "diff_digest" not in state


class TestHoldoutCitationIsFindable:
    """`manifests_citing` looked for a dict; runs wrote a path string."""

    def test_the_freeze_tool_finds_a_top_level_holdout_dict(self, tmp_path):
        import sys

        sys.path.insert(0, "tools")
        import freeze_holdout as fh

        run = tmp_path / "release-1"
        run.mkdir()
        (run / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": "release-1",
                    "holdout": {
                        "id": "holdout_v3",
                        "content_hash": "4227eb56",
                        "task_entry_hash": "32f8a13a2b4772b1",
                        "task": "repair_belt",
                    },
                }
            ),
            encoding="utf-8",
        )

        found = fh.manifests_citing("holdout_v3", runs_root=tmp_path)
        assert len(found) == 1
        assert found[0]["citation"]["task_entry_hash"] == "32f8a13a2b4772b1"

    def test_a_path_string_is_not_mistaken_for_a_citation(self, tmp_path):
        import sys

        sys.path.insert(0, "tools")
        import freeze_holdout as fh

        run = tmp_path / "old-run"
        run.mkdir()
        (run / "manifest.json").write_text(
            json.dumps({"config": {"holdout": "docs/evidence/holdout_v3.json"}}),
            encoding="utf-8",
        )
        assert fh.manifests_citing("holdout_v3", runs_root=tmp_path) == []


@pytest.mark.parametrize("field", ["content_hash", "task_entry_hash", "start_index"])
def test_the_citation_carries_what_identifies_the_scenes(field):
    """Guards the shape train.py writes, so the two cannot drift apart."""
    import inspect

    from factoriorl.learn import train

    source = inspect.getsource(train.train)
    assert f'"{field}"' in source
