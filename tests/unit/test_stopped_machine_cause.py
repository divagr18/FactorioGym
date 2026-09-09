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
