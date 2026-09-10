"""A stopped machine's cause has to reach the prompt (R4.2).

`local-v2` was bumped to v4 to publish four per-entity fields, and its own note
says why: *"per-entity status name, working flag, fuel and output contents --
what a stopped machine's cause actually is."* `sensor.entity_record` builds all
four. `_entity_row` copied none of them, so every language-model prompt ever
rendered by this repo showed `status 18` -- the exact string `sensor.lua`'s
comment says *"neither client could act on"*.

It was found by reading R4.2's agent arm: 25 decisions on a line whose fuel had
just been emptied, and the drill's record in the recorded prompt carried
`status` and nothing else. The agent refuelled twice, but blind.

These tests pin the fields into the rendered prompt, because the wire had them
all along and the render is where they were lost.
"""

from __future__ import annotations

from factoriorl.agent.summary import TaskBrief, summarise

BRIEF = TaskBrief(
    id="plate_line", version="1.2.0", description="commission it", max_decision_steps=400
)


def _summary(record: dict):
    observation = {
        "tick": 3600,
        "character": {"present": True, "position": [0.0, 0.0]},
        "inventory": {"coal": 120},
        "entities": [record],
    }
    return summarise(observation, brief=BRIEF, actions=(), step=0)


def _drill(**extra) -> dict:
    record = {
        "h": "h2",
        "name": "burner-mining-drill",
        "type": "mining-drill",
        "p": [0.0, -7.0],
        "status": 12,
    }
    record.update(extra)
    return record


class TestTheCauseIsRendered:
    def test_the_status_name_is_shown_rather_than_its_integer(self):
        """`status 12` is not a diagnosis; `no_fuel` is the engine's own word."""
        text = _summary(_drill(st="no_fuel", working=False)).render()
        assert "no_fuel" in text
        assert "status 12" not in text

    def test_fuel_contents_reach_the_prompt(self):
        text = _summary(_drill(st="working", working=True, fuel={"coal": 47})).render()
        assert "fuel coal x47" in text

    def test_output_contents_reach_the_prompt(self):
        """The furnace's held plates are how "blocked" is told from "idle"."""
        record = {
            "h": "h1",
            "name": "stone-furnace",
            "type": "furnace",
            "p": [0.0, -5.0],
            "status": 1,
            "st": "working",
            "working": True,
            "output": {"iron-plate": 3},
        }
        assert "output iron-plate x3" in _summary(record).render()

    def test_the_fields_survive_into_the_row_for_programmatic_readers(self):
        row = _summary(_drill(st="no_fuel", working=False, fuel={"coal": 1}))
        assert row.entities[0]["st"] == "no_fuel"
        assert row.entities[0]["working"] is False
        assert row.entities[0]["fuel"] == {"coal": 1}


class TestWhatItRefusesToClaim:
    def test_an_absent_fuel_field_is_not_reported_as_an_empty_one(self):
        """`sensor.inventory_contents` returns nil both for an empty fuel
        inventory and for an entity that has none, so an absent field cannot
        tell a drained drill from a transport belt. Saying "fuel: empty" here
        would be inventing the distinction the wire does not carry."""
        text = _summary(
            {
                "h": "h3",
                "name": "transport-belt",
                "type": "transport-belt",
                "p": [1.0, 0.0],
                "status": 1,
                "st": "working",
                "working": True,
            }
        ).render()
        line = next(row for row in text.splitlines() if "transport-belt" in row)
        assert "fuel" not in line.lower()

    def test_an_unnamed_status_still_shows_its_integer(self):
        """A status the engine has no name for must not vanish from the prompt."""
        text = _summary(_drill()).render()
        assert "status 12" in text


class TestADiagnosisNamesARemedy:
    """`no power` is true, and on its own it is a dead end.

    A run crafted a lab, placed it, was told `no power` every turn for the rest
    of the run, and never did anything about it. It had never seen electricity:
    nothing in the prompt said what power is, where it comes from, or that a lab
    has no fuel slot to put coal in. A burner already got a remedy with its
    diagnosis -- `FUEL SLOT EMPTY -- give it some` -- and an electric machine got
    only the symptom.

    The engine's own word stays at the front of every line: these tests pin that
    the remedy is *added to* the status, never substituted for it, because a
    previous version of this file's neighbouring table invented status names and
    told a run its furnace was out of output space when it was out of ore.
    """

    def _factory(self, record: dict) -> str:
        observation = {
            "tick": 3600,
            "character": {"present": True, "position": [0.0, 0.0]},
            "inventory": {},
            "entities": [],
            "built": [record],
        }
        return summarise(observation, brief=BRIEF, actions=(), step=0).render()

    def test_an_unpowered_machine_is_told_it_needs_a_generator(self):
        text = self._factory(
            {"h": "h9", "name": "lab", "type": "lab", "p": [4.0, 4.0], "st": "no_power"}
        )
        assert "no power" in text
        assert "generator" in text
        # The specific wrong move it would otherwise reach for, having only ever
        # fuelled burners: a lab has no fuel slot at all.
        assert "no fuel slot" in text

    def test_an_exhausted_drill_is_told_to_pick_it_up_and_move_it(self):
        text = self._factory(
            {
                "h": "h1",
                "name": "burner-mining-drill",
                "type": "mining-drill",
                "p": [0.0, 0.0],
                "st": "no_minable_resources",
            }
        )
        assert "no minable resources" in text
        assert "place it where ore is under every tile it covers" in text

    def test_an_unknown_status_falls_back_to_the_engines_own_word(self):
        """The remedies are keyed on engine names. A key that is ever wrong must
        degrade to the engine's word rather than to silence or to a guess."""
        text = self._factory(
            {"h": "h1", "name": "boiler", "type": "boiler", "p": [0.0, 0.0], "st": "some_new_code"}
        )
        assert "some new code" in text
