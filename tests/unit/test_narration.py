"""Saying what the agent is doing must not change what the agent can see.

A thirty-minute run printed nothing. Everything was recoverable afterwards from
`decisions.jsonl`, which is the wrong time to learn it has been hauling coal in
a circle for twenty minutes -- the two runs before this were both diagnosed by
reading a transcript after the fact.

The narrator fixes that, and the risk it introduces is the one worth testing: it
writes to the *world* to draw an overlay, and anything written to the world could
in principle be read back by the sensor. It cannot be here -- a `rendering`
object is not an entity and not a resource -- and these tests pin the two
properties that keep it that way: it never touches game state, and it never
raises into the run.
"""

from __future__ import annotations

from factoriorl.agent.narrate import Narrator


class _Decision:
    def __init__(self, step=1, reason="do a thing", plan="", outcomes=None, resolution="model"):
        self.step = step
        self.reason = reason
        self.plan = plan
        self.outcomes = outcomes or []
        self.resolution = resolution


class _Client:
    def __init__(self, fail=False):
        self.sent: list[str] = []
        self.fail = fail

    def lua(self, code: str):
        self.sent.append(code)
        if self.fail:
            raise OSError("socket gone")
        return "updated"


class _Session:
    def __init__(self, fail=False):
        self._client = _Client(fail=fail)


class TestTheOverlayCannotReachTheAgent:
    def test_it_draws_and_never_creates_an_entity(self):
        """`rendering.draw_text` only. `create_entity`, `insert` or a tile write
        would all be observable by the sensor and would make a watched run a
        different run from an unwatched one."""
        session = _Session()
        Narrator(session, console=False).decision(_Decision(reason="place a chest"))
        code = session._client.sent[0]
        assert "rendering.draw_text" in code
        for forbidden in ("create_entity", "destroy", "insert", "set_tiles", "teleport"):
            assert forbidden not in code, forbidden

    def test_the_text_is_data_and_not_code(self):
        """A reason is model output going into a Lua program. Pasting it in
        unescaped is a quoting bug at best and arbitrary execution at worst.

        Asserted by round-tripping rather than by string matching. The first
        version of this test stripped backslashes out of the emitted code before
        looking for the payload -- which defeats the very escaping it meant to
        check, and failed against an implementation that was in fact correct.
        """
        import json

        hostile = '" .. game.print("pwned") .. "'
        session = _Session()
        Narrator(session, console=False).decision(_Decision(reason=hostile))
        literal = session._client.sent[0].split("local text = ", 1)[1].split("\n", 1)[0]
        assert json.loads(literal) == hostile, "the reason survives as exactly one string"

    def test_a_dead_socket_never_ends_the_run(self):
        session = _Session(fail=True)
        narrator = Narrator(session, console=False)
        for step in range(3):
            narrator.decision(_Decision(step=step))
        assert narrator.overlay_errors == 3
        assert narrator.overlay is True, "still trying, because one blip is not a verdict"

    def test_it_gives_up_after_repeated_failures_and_says_so(self):
        session = _Session(fail=True)
        narrator = Narrator(session, console=False)
        for step in range(6):
            narrator.decision(_Decision(step=step))
        assert narrator.overlay is False
        assert narrator.to_dict()["overlay_errors"] >= 5


class TestTheConsoleRecord:
    def test_a_new_plan_is_announced_and_an_unchanged_one_is_not(self, capsys):
        narrator = Narrator(None, console=True, overlay=False)
        narrator.decision(_Decision(step=1, plan="automate coal"))
        narrator.decision(_Decision(step=2, plan="automate coal"))
        narrator.decision(_Decision(step=3, plan="build a lab"))
        printed = capsys.readouterr().out
        assert printed.count("PLAN") == 2, "a plan is news when it changes, not every turn"
        assert "automate coal" in printed
        assert "build a lab" in printed

    def test_a_failed_action_is_marked_differently_from_a_completed_one(self, capsys):
        narrator = Narrator(None, console=True, overlay=False)
        narrator.decision(
            _Decision(
                outcomes=[
                    {"key": "place_at", "arguments": {"item": "iron-chest"}, "status": "completed"},
                    {
                        "key": "give_to",
                        "arguments": {"item": "coal"},
                        "status": "failed",
                        "action_error": "no_items",
                    },
                ]
            )
        )
        printed = capsys.readouterr().out
        assert "place_at" in printed and "give_to" in printed
        assert "no_items" in printed
        assert "  x" in printed, "a failure has to be findable by eye in a scrolling log"

    def test_narration_can_be_switched_off_entirely(self, capsys):
        narrator = Narrator(None, console=False, overlay=False)
        narrator.decision(_Decision(reason="silent"))
        assert capsys.readouterr().out == ""


class TestTheTwoWaitsAreNotDescribedAsTheSameThing:
    """`wait` and `wait_for` both dispatch to the mod's `wait`, and both were
    described as "do nothing this decision interval".

    They differ by a factor of sixty. `wait` buys one decision interval -- 30
    ticks, half a second -- and `wait_for` blocks on a stated condition for up
    to thirty seconds. With identical descriptions the only visible difference
    was that one needed arguments and the other did not, and a live run spent
    **71 of its 135 decisions** on the cheap one: about 35 seconds of game time
    bought with 71 model calls, while `wait_for` sat legal in the same list.
    """

    def _described(self) -> dict[str, str]:
        from factoriorl import catalog as catalog_module
        from factoriorl.agent.summary import describe_template

        return {
            t.key: describe_template(t)
            for t in catalog_module.resolve("open-v1").templates
            if t.key in ("wait", "wait_for")
        }

    def test_they_do_not_share_a_description(self):
        described = self._described()
        assert described["wait"] != described["wait_for"]

    def test_each_names_what_it_actually_buys(self):
        described = self._described()
        assert "half a second" in described["wait"]
        assert "30 seconds" in described["wait_for"]

    def test_the_cheap_one_points_at_the_useful_one(self):
        """The bare wait cannot be removed -- it is the guaranteed legal no-op
        every mask leaves available -- so the description has to do the work."""
        assert "wait_for" in self._described()["wait"]
