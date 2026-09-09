"""A disruption is declared on the task, applied by the env, and recorded (R4.3).

Every disruption in this repo before R4 was a driver-side `bridge.run` string
fired at a hard-coded point in a five-phase script. Three things followed, and
all three are what these tests pin against:

* **Nothing declared it.** The tick lived in the driver and the targets lived
  in the Lua string, so two runs of "the same disruption" were not comparable
  and no trace could say what had been done to the world.
* **The env could not see it.** The write went around the session, so the next
  prompt described the world as it was before -- measured at 3,600 ticks of
  staleness, reporting 48 coal in a drill whose fuel had just been cleared.
* **Nothing recorded whether it worked.** A disruption that matched no entity
  looked exactly like one that landed.

Driven against the real `FactorioEnv` with a stub session, so no engine is
involved.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from factoriorl.seeding import Branch, SeedPlan
from factoriorl.tasks import get
from factoriorl.tasks.spec import DISRUPTION_KINDS, Disruption

pytest.importorskip("gymnasium")

# Bare module name, not `tests.unit....`: there is no `__init__.py` under
# `tests/`, so pytest puts this directory on `sys.path` and the package path
# does not resolve. Imported rather than copied -- the stub already carries the
# measured Phase 5 numbers, and a second hand-written copy is how
# `tools/demonstration.py` drifted away from the agent loop twice.
from test_observation_freshness import DECISION_TICKS, StubSession, _Timed  # noqa: E402

OUTAGE_TICK = 600

#: A module-level singleton, because a frozen dataclass built in an argument
#: default is constructed once at import either way and ruff would rather it
#: said so.
OUTAGE = (Disruption("empty_fuel", OUTAGE_TICK, ("drill", "furnace")),)


class DisruptibleSession(StubSession):
    """The freshness stub, plus the typed `disrupt` request the env issues."""

    def __init__(self, tick: int = 0, matches: bool = True):
        super().__init__(tick)
        self.disruptions: list[dict] = []
        self.matches = matches

    def disrupt(self, kind: str, targets):
        self.disruptions.append({"kind": kind, "targets": list(targets), "tick": self.tick})
        if not self.matches:
            timed = _Timed(None)
            timed.response.ok = False
            timed.response.error = "no entity matched drill,furnace"
            return timed
        if kind == "empty_fuel":
            # What the engine does, and only that: the fuel *inventory*.
            # `entity.energy` is untouched, which is why a drill keeps running
            # afterwards and why the window cannot start here.
            self.fuel = 0
        return _Timed(
            {
                "kind": kind,
                "tick": self.tick,
                "targets": list(targets),
                "touched": ["burner-mining-drill=48"],
            }
        )


def _env(session, disruptions=OUTAGE):
    from factoriorl.env import FactorioEnv

    task = get("plate_line")
    task = replace(task, spec=replace(task.spec, disruptions=tuple(disruptions)))
    env = FactorioEnv(
        task,
        session,
        SeedPlan(master=1, run_id="disrupt"),
        branch=Branch.TRAIN,
        split="train",
    )
    env._episode_index = -1
    env.reset()
    return env


class TestItIsDeclared:
    def test_the_kinds_are_a_closed_set(self):
        assert "empty_fuel" in DISRUPTION_KINDS
        assert "no_such_kind" not in DISRUPTION_KINDS

    def test_a_declared_disruption_is_part_of_what_a_run_means(self):
        """In `to_dict()`, so it reaches the config digest: a run with an outage
        and a run without one are runs of different tasks."""
        from factoriorl.manifest import config_digest

        spec = get("plate_line").spec
        without = config_digest(spec.to_dict())
        with_outage = config_digest(
            replace(spec, disruptions=(Disruption("empty_fuel", 3600, ("drill",)),)).to_dict()
        )
        assert without != with_outage

    def test_the_persistent_operation_family_declares_one(self):
        spec = get("keep_line_running").spec
        assert len(spec.disruptions) == 1
        assert spec.disruptions[0].kind in DISRUPTION_KINDS

    def test_its_window_cannot_open_inside_the_measured_run_on(self):
        """`empty_fuel` leaves `entity.energy` alone and a burner drill keeps
        producing about 1,170 ticks on the item in its burner, so a window
        opening at the outage would score production the outage had not
        stopped. That is precisely the claim withdrawn from the Phase 5
        evidence, which called a 90-tick refuel a recovery."""
        from factoriorl.tasks.families import keep_line_running as family

        predicate = get("keep_line_running").spec.success[0]
        earliest_window_start = predicate.not_before_tick - predicate.over_ticks
        assert earliest_window_start >= family.OUTAGE_TICK + family.RUN_ON_TICKS


class TestTheEnvAppliesIt:
    def test_it_fires_when_the_world_reaches_the_declared_tick(self):
        session = DisruptibleSession()
        env = _env(session)
        assert session.disruptions == []
        env.advance(OUTAGE_TICK)
        assert len(session.disruptions) == 1
        assert session.disruptions[0]["kind"] == "empty_fuel"

    def test_it_cannot_be_skipped_by_a_step_that_advances_past_it(self):
        """`at_tick <= now`, not equality: an advance chunked at 30 ticks steps
        straight over a disruption declared at 601."""
        session = DisruptibleSession()
        env = _env(session, (Disruption("empty_fuel", OUTAGE_TICK + 1, ("drill",)),))
        env.advance(OUTAGE_TICK + DECISION_TICKS)
        assert len(session.disruptions) == 1

    def test_it_fires_once_and_not_on_every_step_after(self):
        session = DisruptibleSession()
        env = _env(session)
        env.advance(OUTAGE_TICK * 3)
        assert len(session.disruptions) == 1

    def test_a_later_episode_gets_its_own_disruption(self):
        """Cleared on reset, or a task declaring one would fire it in the first
        episode only and every later episode would be a different task."""
        session = DisruptibleSession()
        env = _env(session)
        env.advance(OUTAGE_TICK)
        assert len(session.disruptions) == 1
        env.reset()
        assert env._disruptions_applied == []
        env.advance(OUTAGE_TICK)
        assert len(session.disruptions) == 2


class TestTheObservationIsFreshAfterwards:
    def test_the_first_post_outage_observation_shows_the_emptied_fuel(self):
        """The whole point. The driver that fired raw Lua reported 48 coal in a
        drill whose fuel inventory it had just cleared."""
        session = DisruptibleSession()
        env = _env(session)
        env.advance(OUTAGE_TICK)
        drill = (env._observation.get("entities") or [])[0]
        assert not drill.get("fuel")
        assert drill.get("st") == "no_fuel"

    def test_the_refresh_is_recorded_as_a_resync_with_a_reason(self):
        session = DisruptibleSession()
        env = _env(session)
        env.advance(OUTAGE_TICK)
        reasons = [row["reason"] for row in env._resyncs]
        assert any("empty_fuel" in reason for reason in reasons), reasons

    def test_the_tick_still_matches_the_world(self):
        session = DisruptibleSession()
        env = _env(session)
        env.advance(OUTAGE_TICK)
        assert int(env._observation["tick"]) == session.tick


class TestItRecordsWhatHappened:
    def test_the_record_names_the_declared_tick_and_the_applied_one(self):
        session = DisruptibleSession()
        env = _env(session)
        env.advance(OUTAGE_TICK)
        row = env._disruptions_applied[0]
        assert row["declared_at_tick"] == OUTAGE_TICK
        assert row["applied_at_tick"] >= OUTAGE_TICK
        assert row["ok"] is True
        assert row["touched"]

    def test_a_disruption_that_matched_nothing_is_recorded_as_a_failure(self):
        """A disruption matching no entity is a defect in the task, not a
        quieter disruption -- and it was indistinguishable from one that
        landed."""
        session = DisruptibleSession(matches=False)
        env = _env(session)
        env.advance(OUTAGE_TICK)
        row = env._disruptions_applied[0]
        assert row["ok"] is False
        assert "no entity matched" in row["error"]


class TestItAnnouncesNothing:
    def test_no_marker_is_published_for_it(self):
        session = DisruptibleSession()
        env = _env(session)
        env.advance(OUTAGE_TICK)
        assert not (env._observation.get("goal") or {})

    def test_and_it_appends_no_event(self):
        """`events` carries the outcome of the agent's *own* requests. An
        unannounced disruption is detectable only by monitoring status, fuel and
        output, which is what R4.3 asks for."""
        session = DisruptibleSession()
        env = _env(session)
        before = len(env._observation.get("events") or [])
        env.advance(OUTAGE_TICK)
        assert len(env._observation.get("events") or []) == before
