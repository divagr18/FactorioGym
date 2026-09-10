"""The prompt has to say what is missing, and where to go and get it.

A live run built one drill and one furnace, reached iron and copper plates, and
then stopped expanding. Its own plan recorded the reason: *"Practical maximum
without stone."* It spent its remaining two hundred decisions walking a
sixty-tile round trip to coal, mining one tile per reply, and banking gears and
belts it had no use for.

Every fact it needed was already in the prompt, and each one had been stripped
of the part that made it actionable:

* the recipe list said `stone-furnace x0`, which says *something* is missing and
  never which thing;
* `THE CHARTED MAP` said `stone: 182 tiles centred on (-9, -112)`, sixty
  thousand characters up in a static prefix the turn does not repeat;
* the `destination` domain offered `[-9.5, -111.5]` -- the stone -- as one bare
  coordinate among sixteen other bare coordinates.

So the run was not short of information. It was short of a sentence naming the
missing ingredient next to the place that has it. These tests pin both halves.
"""

from __future__ import annotations

from factoriorl.agent.summary import TaskBrief, argument_domains, summarise

BRIEF = TaskBrief(
    id="open_factory", version="0.3.0", description="build it out", max_decision_steps=400
)


class TestTheMissingIngredientIsNamed:
    def _render(self, recipes: list[dict]) -> str:
        observation = {
            "tick": 100,
            "character": {"present": True, "position": [0.0, 0.0]},
            "inventory": {"iron-plate": 35},
            "entities": [],
            "recipes": recipes,
        }
        return summarise(
            observation,
            brief=BRIEF,
            actions=(),
            step=0,
            arguments={"recipes": [entry["name"] for entry in recipes]},
        ).render()

    def test_an_unaffordable_recipe_says_which_ingredient_is_short(self):
        text = self._render(
            [
                {
                    "name": "stone-furnace",
                    "craftable": 0,
                    "missing": [{"name": "stone", "need": 5, "have": 0}],
                }
            ]
        )
        assert "stone-furnace x0" in text
        assert "need 5 stone, have 0" in text

    def test_several_short_ingredients_are_all_named(self):
        text = self._render(
            [
                {
                    "name": "lab",
                    "craftable": 0,
                    "missing": [
                        {"name": "electronic-circuit", "need": 10, "have": 2},
                        {"name": "iron-gear-wheel", "need": 10, "have": 4},
                    ],
                }
            ]
        )
        assert "need 10 electronic-circuit, have 2" in text
        assert "need 10 iron-gear-wheel, have 4" in text

    def test_an_affordable_recipe_carries_no_shortfall_noise(self):
        text = self._render([{"name": "iron-gear-wheel", "craftable": 17}])
        assert "iron-gear-wheel x17" in text
        assert "need" not in text.split("recipe:")[1].split("\n")[0]


class _Env:
    """The two methods `argument_domains` reads, and the label map beside them."""

    def __init__(self, domains: dict, labels: dict):
        self._domains = domains
        self._destination_labels = labels

        class _Template:
            parameterized = True

        class _Catalog:
            templates = (_Template(),)

        self.catalog = _Catalog()

    def argument_domains(self) -> dict:
        return self._domains


class TestADestinationSaysWhatIsThere:
    def test_a_charted_patch_is_named_beside_its_coordinate(self):
        """`[-9.5, -111.5]` is not a reason to walk a hundred tiles. `stone
        patch (182 tiles)` is."""
        rendered = argument_domains(
            _Env(
                {"destinations": [[-9.5, -111.5], [-30.5, 1.5]]},
                {(-9.5, -111.5): "stone patch (182 tiles)", (-30.5, 1.5): "your stone-furnace"},
            )
        )
        values = rendered["destinations"]["values"]
        assert "[-9.5, -111.5] stone patch (182 tiles)" in values
        assert "[-30.5, 1.5] your stone-furnace" in values

    def test_an_unlabelled_destination_still_renders(self):
        """The label map is best-effort; a coordinate without one must not
        vanish from a domain, or somewhere the agent could go stops existing."""
        rendered = argument_domains(_Env({"destinations": [[4.5, 4.5]]}, {}))
        assert rendered["destinations"]["values"] == ["[4.5, 4.5]"]

    def test_every_destination_survives_labelling(self):
        points = [[float(x) + 0.5, 0.5] for x in range(12)]
        rendered = argument_domains(_Env({"destinations": points}, {(0.5, 0.5): "coal"}))
        assert rendered["destinations"]["count"] == 12
        assert len(rendered["destinations"]["values"]) == 12


def test_the_rendered_turn_prints_labelled_destinations():
    """The render path, not just the domain builder.

    `argument_domains` and `render` both decided what a destination looks like,
    and only one of them was updated when the labels went in. Every offline test
    above passed; the first turn of a live world raised `Unknown format code 'f'
    for object of type 'str'`, because `render` was still formatting the pair
    itself. One place decides now, and this covers it.
    """
    observation = {
        "tick": 10,
        "character": {"present": True, "position": [0.0, 0.0]},
        "inventory": {},
        "entities": [],
    }
    text = summarise(
        observation,
        brief=BRIEF,
        actions=(),
        step=0,
        arguments={
            "destinations": {
                "rule": "walk here",
                "count": 2,
                "values": ["[-9.5, -111.5] stone patch (182 tiles)", "[1.5, 2.5]"],
            }
        },
    ).render()
    assert "[-9.5, -111.5] stone patch (182 tiles)" in text
    assert "[1.5, 2.5]" in text
