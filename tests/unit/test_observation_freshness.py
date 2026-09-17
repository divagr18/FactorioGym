"""R4.1: an intervention must not leave the policy reading a stale world.

The defect, measured against the committed Phase 5 demonstration
(`docs/evidence/phase5-demonstration.json`): the driver advanced 3,600 ticks
with `session.step(..., ticks=chunk)` and injected a fuel outage with raw Lua,
neither of which the env was told about, then rendered the next prompt from
`env._observation`. **That prompt said "game tick 450" while the world was at
4050**, and reported the drill holding 48 coal and `working` at the moment both
machines were empty. `memory.observe` then stamped those stale facts as
confirmed at the current step, so they were not even marked aged.

Only the first prompt per hand-back was stale, because `_execute` goes through
`step_payload`, which refreshes. That is why it survived: the symptom is a
wrong prompt, not a crash.

R4.1's gate: *"Every post-intervention prompt/observation has the correct tick
and current permitted state. A regression check reproduces and prevents the
previous stale-observation case."* Both halves are below, driven against the
real `FactorioEnv` with a stub session, so no engine is involved.
"""

from __future__ import annotations

import pytest

from factoriorl.seeding import Branch, SeedPlan
from factoriorl.tasks import get

pytest.importorskip("gymnasium")

DECISION_TICKS = 30


class _Response:
    def __init__(self, result):
        self.result = result
        self.ok = True


class _Timed:
    def __init__(self, result):
        self.response = _Response(result)


class StubSession:
    """A worker that advances a tick counter and produces a plate per step.

    Deliberately *not* `test_agent_loop.py`'s `StubEnv`, whose `step` never
    mutates the observation, so its tick is frozen at 120 and no freshness
    regression could be written against it.
    """

    def __init__(self, tick: int = 0):
        self.tick = tick
        self.plates = 0
        self.fuel = 48
        self.observes = 0
        self.truths = 0
        self.steps: list[int] = []

    # ---- what the env calls -------------------------------------------
    def define_scenario(self, payload, digest):
        return _Timed({})

    def reset(self, **kwargs):
        return _Timed({"episode_id": "e1"})

    def _observation(self) -> dict:
        return {
            "tick": self.tick,
            "character": {"position": [0.0, 0.0], "present": True},
            "inventory": {},
            "entities": [
                {
                    "h": "h1",
                    "p": [1.0, 1.0],
                    "type": "mining-drill",
                    "st": "working" if self.fuel else "no_fuel",
                    "working": bool(self.fuel),
                    "fuel": {"coal": self.fuel} if self.fuel else {},
                }
            ],
            "resources": {"tiles": []},
            "terrain": {"blocked": []},
            "inflight": [],
            "events": [],
            "recipes": [],
        }

    def _truth(self) -> dict:
        return {"markers": {}, "containers": {}, "produced": {"iron-plate": self.plates}}

    def observe(self):
        self.observes += 1
        return _Timed(self._observation())

    def truth(self):
        self.truths += 1
        return _Timed(self._truth())

    def step(self, payload, ticks: int = DECISION_TICKS):
        self.steps.append(ticks)
        self.tick += ticks
        # One plate per 240 ticks, the measured furnace rate. Derived from the
        # absolute tick, not the chunk: `ticks // 240` is 0 for every 30-tick
        # chunk, so an incremental version never produced a single plate.
        self.plates = self.tick // 240
        return _Timed(
            {
                "action": {"status": "completed"},
                "observation": self._observation(),
                "truth": self._truth(),
            }
        )

    # ---- what an evaluator-side intervention does ---------------------
    def empty_the_fuel(self):
        """The disruption, as raw as the demonstration's: a world write the
        env issued no request for and therefore cannot see."""
        self.fuel = 0


def _env(session=None):
    from factoriorl.env import FactorioEnv

    session = session or StubSession()
    env = FactorioEnv(
        get("plate_line"),
        session,
        SeedPlan(master=1, run_id="freshness"),
        branch=Branch.TRAIN,
        split="train",
    )
    env._episode_index = -1
    env.reset()
    return env, session


class TestTheDefectReproduces:
    """The pre-fix pattern, so the regression has something to prevent."""

    def test_advancing_around_the_env_leaves_the_cache_behind(self):
        env, session = _env()
        before = int(env._observation["tick"])
        # Exactly what `tools/demonstration.py:205` did: session.step in a
        # loop, response discarded.
        for _ in range(120):
            session.step({"action": "wait"}, ticks=DECISION_TICKS)
        assert session.tick - before == 3600
        assert int(env._observation["tick"]) == before, (
            "the env should be unchanged -- this is the defect, not the fix"
        )

    def test_an_evaluator_write_is_invisible_until_something_refreshes(self):
        env, session = _env()
        session.empty_the_fuel()
        drill = env._observation["entities"][0]
        assert drill["fuel"] == {"coal": 48}
        assert drill["working"] is True, (
            "the cached view still shows a fuelled, working drill -- the exact "
            "state the recorded prompt reported at tick 4050"
        )


class TestAdvanceKeepsTheEnvInSync:
    def test_the_observation_tick_tracks_the_world(self):
        env, session = _env()
        advanced = env.advance(3600)
        assert advanced == 3600
        assert int(env._observation["tick"]) == session.tick

    def test_the_rendered_prompt_carries_the_current_tick(self):
        """The gate's wording is about the *prompt*, so that is what is checked."""
        from factoriorl.agent.summary import TaskBrief, summarise

        env, session = _env()
        env.advance(3600)
        summary = summarise(
            env._observation, brief=TaskBrief.from_spec(env.spec_), actions=(), step=0
        )
        assert summary.tick == session.tick
        assert f"game tick {session.tick}" in summary.render()

    def test_it_advances_in_decision_sized_chunks(self):
        """A single jump would leave a two-sample window, which
        `window_sampled` refuses -- so the window stays measurable."""
        env, session = _env()
        session.steps.clear()
        env.advance(3600)
        assert set(session.steps) == {DECISION_TICKS}
        assert len(session.steps) == 3600 // DECISION_TICKS

    def test_the_production_window_stays_sampled_across_an_advance(self):
        from factoriorl.tasks.spec import Predicate, PredicateKind

        env, _ = _env()
        env.advance(7200)
        predicate = Predicate(
            PredicateKind.SUSTAINED_OUTPUT, item="iron-plate", at_least=10, over_ticks=3600
        )
        assert predicate.window_elapsed(env._truth)
        assert predicate.window_sampled(env._truth), (
            "an advance that jumped would leave a window nothing sampled"
        )
        assert predicate.window_output(env._truth) > 0

    def test_a_partial_final_chunk_is_not_overshot(self):
        env, session = _env()
        start = session.tick
        assert env.advance(70) == 70
        assert session.tick - start == 70

    def test_zero_or_negative_advances_do_nothing(self):
        env, session = _env()
        session.steps.clear()
        assert env.advance(0) == 0
        assert env.advance(-30) == 0
        assert session.steps == []

    def test_advancing_does_not_spend_the_decision_budget(self):
        """Evaluator time is not the agent's; charging it would let an
        intervention shorten the episode."""
        env, _ = _env()
        before = env._steps
        env.advance(3600)
        assert env._steps == before


class TestResyncAfterAnEvaluatorWrite:
    def test_the_change_becomes_visible(self):
        env, session = _env()
        session.empty_the_fuel()
        env.resync("fuel outage injected")
        drill = env._observation["entities"][0]
        assert drill["fuel"] == {}
        assert drill["working"] is False

    def test_the_reason_and_tick_are_recorded(self):
        env, session = _env()
        env.advance(600)
        session.empty_the_fuel()
        env.resync("fuel outage injected")
        assert env._resyncs == [{"tick": session.tick, "reason": "fuel outage injected"}]

    def test_an_episode_does_not_inherit_the_previous_one_s_interventions(self):
        env, session = _env()
        env.resync("first episode")
        assert env._resyncs
        env.reset()
        assert env._resyncs == []


class TestTheOneRefreshPath:
    def test_step_payload_and_advance_share_it(self):
        """`_adopt` exists so there is exactly one place the cache is
        refreshed; two would drift the way `play()` drifted from the loop."""
        import inspect

        from factoriorl.env import FactorioEnv

        for method in (FactorioEnv.step_payload, FactorioEnv.advance):
            assert "_adopt" in inspect.getsource(method)

    def test_the_metrics_see_every_advance(self):
        env, session = _env()
        env.advance(3600)
        report = env.metrics.report()
        assert report["episode_ticks"] == session.tick
        assert report["samples"] > 1


class TestTheDemonstrationDoesNotReimplementTheLoop:
    """B6: `play()` was a hand-copy of `AgentLoop.run_episode` and drifted.

    It omitted `arguments=`/`requires=` from `summarise` -- so a
    `parameterized-v1` catalog would have been prompted with no argument
    values at all -- and omitted `target=` from the memory record. Its own
    docstring recorded that the same re-implementation had already cost one
    measured defect (`unknown_target` x 50).

    A comment saying "do not re-implement the stepping" is what was there
    before. These are the checks.
    """

    def _source(self) -> str:
        import pathlib

        return pathlib.Path("tools/demonstration.py").read_text(encoding="utf-8")

    def test_the_hand_copied_loop_is_gone(self):
        import ast

        tree = ast.parse(self._source())
        functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "play" not in functions

    def test_it_does_not_step_or_advance_the_session_itself(self):
        import ast

        tree = ast.parse(self._source())
        calls = {
            f"{getattr(node.func.value, 'id', '')}.{node.func.attr}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        for forbidden in ("session.step", "session.advance", "session.act"):
            assert forbidden not in calls, forbidden

    def test_it_hands_back_to_the_maintained_loop(self):
        assert "run_segment" in self._source()

    def test_it_resyncs_after_its_own_world_write(self):
        """The disruption is raw Lua the env issued no request for."""
        source = self._source()
        assert "env.resync(" in source

    def test_it_does_not_import_the_prompt_building_pieces(self):
        """Nothing outside the loop should need them. If this fails, someone
        has started re-implementing the stepping again."""
        source = self._source()
        for name in ("action_vocabulary", "legal_actions", "summarise"):
            assert f"import {name}" not in source
            assert f"    {name},\n" not in source


class TestRunSegmentIsTheOneSteppingPath:
    def test_run_episode_delegates_to_it(self):
        import inspect

        from factoriorl.agent.loop import AgentLoop

        source = inspect.getsource(AgentLoop.run_episode)
        assert "run_segment" in source
        assert "self.env.reset()" in source

    def test_a_segment_continues_the_decision_numbering(self):
        """A hand-back must read as one episode, not several starting at 0."""
        import inspect

        from factoriorl.agent.loop import AgentLoop

        source = inspect.getsource(AgentLoop.run_segment)
        assert "first_step + steps" in source

    def test_a_segment_records_where_it_started(self):
        import inspect

        from factoriorl.agent.loop import AgentLoop

        assert '"first_step": first_step' in inspect.getsource(AgentLoop.run_segment)
