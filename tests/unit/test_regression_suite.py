"""Synthesis section 10: a benchmark number and a diagnostic one, kept apart.

Two halves. `evaluate_parallel` recorded counts only, so "how many failed" was
answerable and "which scenes failed" was not -- which also means no failure
could ever be turned into a regression case. And there was no regression suite:
adversarially found scenes had nowhere to live except a split, where they would
be pooled into a representative rate they do not represent.
"""

from __future__ import annotations

import json

import pytest

from factoriorl import regression
from factoriorl.learn.train import evaluate_parallel
from factoriorl.tasks import coverage, get


@pytest.fixture(scope="module")
def cases() -> list[regression.Case]:
    return regression.load()


class TestTheCommittedSuite:
    def test_it_loads_and_is_not_empty(self, cases):
        assert cases

    def test_every_case_names_a_registered_task_and_family(self, cases):
        for case in cases:
            task = get(case.task)
            assert case.layout_family in {f.name for f in task.spec.layout_families}

    def test_every_case_still_names_the_scene_it_was_recorded_against(self, cases):
        """The guarantee that makes a case an object rather than an index."""
        assert regression.verify(cases) == []

    def test_every_case_states_a_finding(self, cases):
        for case in cases:
            assert len(case.finding) > 40, case.case_id

    def test_the_seed_plan_is_recorded_with_the_cases(self):
        document = json.loads(regression.suite_path().read_text(encoding="utf-8"))
        assert document["seed_plan"]["master"] == regression.REGRESSION_MASTER
        assert document["seed_plan"]["run_id"] == regression.REGRESSION_RUN_ID

    def test_the_plate_line_finding_is_present_and_marked_open(self, cases):
        walled = [c for c in cases if c.task == "plate_line"]
        assert len(walled) == 10, "10 of 200 commissioning_walled scenes were affected"
        assert all(c.expected_present for c in walled), (
            "the finding was deliberately not fixed, so it must read as open"
        )

    def test_the_recorded_scenes_really_do_start_the_character_on_a_wall(self, cases):
        """The case is only worth keeping if it still reproduces."""
        import math
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
        import generator_diagnostics as gd

        for case in (c for c in cases if c.task == "plate_line"):
            task = get(case.task)
            blueprint = regression.blueprint_for(task, case)
            blocked = gd.blocked_tiles(blueprint)
            start = (
                math.floor(blueprint.character_position[0]),
                math.floor(blueprint.character_position[1]),
            )
            assert start in blocked, f"{case.case_id} no longer starts on a blocked tile"


class TestItIsNotADistribution:
    def test_the_report_carries_no_pooled_rate(self, cases):
        """Averaging hand-picked hard scenes makes a number describing nothing."""
        report = regression.report(cases)
        flat = json.dumps(report)
        assert "success_rate" not in flat
        assert not any("rate" in key for key in report)
        assert not any(key.startswith("mean_") for key in report)

    def test_it_reports_per_case(self, cases):
        report = regression.report(cases)
        assert len(report["cases"]) == len(cases)
        assert all("scene_still_matches" in row for row in report["cases"])

    def test_open_findings_are_listed_explicitly(self, cases):
        report = regression.report(cases)
        assert report["open_findings"], "a deliberately unfixed finding must be visible"

    def test_the_suite_is_not_one_of_the_declared_splits(self, cases):
        """It must not become `val`, which `freeze_holdout` refuses to freeze."""
        from factoriorl.tasks import all_tasks

        splits = {f.split for name in all_tasks() for f in get(name).spec.layout_families}
        assert splits <= {"train", "val", "test"}
        # Cases carry no split of their own; they are individual scenes.
        assert not hasattr(regression.Case, "split")


class TestStalenessIsDetected:
    def test_a_changed_digest_is_reported(self, cases, tmp_path):
        original = cases[0]
        tampered = regression.Case(
            **{**original.to_dict(), "blueprint_digest": "0" * 16},
        )
        problems = regression.verify([tampered])
        assert problems and "no longer names the scene" in problems[0]

    def test_a_missing_family_is_reported(self):
        case = regression.Case(
            case_id="x",
            task="plate_line",
            layout_family="no_such_family",
            episode_index=0,
            blueprint_digest="0" * 16,
            finding="a finding long enough to be a real sentence about something",
        )
        problems = regression.verify([case])
        assert problems and "no layout family" in problems[0]

    def test_a_round_trip_through_disk_preserves_every_case(self, cases, tmp_path):
        path = tmp_path / "suite.json"
        regression.save(cases, path)
        assert regression.load(path) == sorted(cases, key=lambda c: c.case_id)

    def test_an_unknown_schema_is_refused(self, tmp_path):
        path = tmp_path / "suite.json"
        path.write_text(json.dumps({"schema": "other/9", "cases": []}), encoding="utf-8")
        with pytest.raises(regression.RegressionError):
            regression.load(path)

    def test_the_digest_is_the_one_the_environment_installs_by(self, cases):
        from factoriorl.env import blueprint_digest

        case = cases[0]
        task = get(case.task)
        payload = regression.blueprint_for(task, case).to_dict(
            public_markers=task.spec.public_markers
        )
        assert blueprint_digest(payload) == coverage._digest(payload) == case.blueprint_digest


class TestPerSceneOutcomes:
    """The other half: evaluation has to say *which* scenes failed.

    Driven against the same engine-free fake vec env
    `test_frozen_eval_coverage.py` uses, so this exercises the real
    `evaluate_parallel` rather than reading its source.
    """

    def _run(self, fail: tuple[int, ...] = ()):
        from test_frozen_eval_coverage import COUNT, START, FrozenVecEnv, _Model

        env = FrozenVecEnv(num_envs=2)
        # `FrozenVecEnv` marks every episode a success; make some fail as a
        # *task* outcome (not an infrastructure exclusion, which is a different
        # thing and must not appear as a scene failure).
        real_step = env.step

        def step(actions):
            observations, rewards, dones, infos = real_step(actions)
            for info in infos:
                if info.get("episode_index") in fail:
                    info["success"] = False
                    info["TimeLimit.truncated"] = True
            return observations, rewards, dones, infos

        env.step = step
        return evaluate_parallel(
            env, _Model(), COUNT, only_indices=set(range(START, START + COUNT))
        )

    def test_a_row_is_recorded_for_every_scored_episode(self):
        from test_frozen_eval_coverage import COUNT, START

        result = self._run()
        assert len(result["per_scene"]) == COUNT
        assert [row["episode_index"] for row in result["per_scene"]] == list(
            range(START, START + COUNT)
        )

    def test_which_scenes_failed_is_answerable_not_only_how_many(self):
        from test_frozen_eval_coverage import START

        failed = (START + 1, START + 4)
        result = self._run(fail=failed)
        assert result["successes"] == 4
        assert result["failed_scenes"] == sorted(failed)
        assert {row["episode_index"] for row in result["per_scene"] if not row["success"]} == set(
            failed
        )

    def test_a_row_says_why_the_episode_ended(self):
        from test_frozen_eval_coverage import START

        result = self._run(fail=(START,))
        rows = {row["episode_index"]: row for row in result["per_scene"]}
        assert rows[START]["truncated"] is True
        assert rows[START + 1]["truncated"] is False

    def test_rows_are_ordered_by_identity_not_by_finish_order(self):
        """Worker scheduling decides finish order, so two runs over the same
        frozen set would diff as if the outcomes had changed."""
        indices = [row["episode_index"] for row in self._run()["per_scene"]]
        assert indices == sorted(indices)

    def test_an_infrastructure_exclusion_is_not_recorded_as_a_failed_scene(self):
        """It is not a task outcome, so it must not become a regression case."""
        from test_frozen_eval_coverage import COUNT, START, FrozenVecEnv, _Model

        env = FrozenVecEnv(num_envs=2, fail_scenes=(START + 2,), fail_times=1)
        result = evaluate_parallel(
            env, _Model(), COUNT, only_indices=set(range(START, START + COUNT))
        )
        assert result["failed_scenes"] == []
        assert len(result["per_scene"]) == COUNT

    def test_a_failed_scene_can_be_turned_into_a_regression_case(self):
        """The point of recording identities: a failure becomes an object."""
        from test_frozen_eval_coverage import START

        result = self._run(fail=(START + 3,))
        assert result["failed_scenes"] == [START + 3]
        case = regression.Case(
            case_id="synthetic-failure",
            task="plate_line",
            layout_family="commissioning_walled",
            episode_index=result["failed_scenes"][0],
            blueprint_digest="0" * 16,
            finding="a synthetic failure, long enough to read as a real finding",
        )
        assert case.episode_index == START + 3
