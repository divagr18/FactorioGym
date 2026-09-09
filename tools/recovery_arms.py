"""Three arms from a validated common pre-intervention state (R4.2).

R4.2's gate: *"From a validated common pre-intervention state, compare
intervention-plus-agent, intervention-plus-no-action, and preferably
undisturbed operation. Use validated save/replay reconstruction if state
branching is unavailable; do not assume arbitrary snapshots already work. Show
production loss and subsequent sustained restoration attributable to agent
actions."*

**There is no state branching.** `RequestType` has no snapshot or restore, no
mid-run save exists, autosaves are disabled in both config writers, and the
only save on disk is the `--create` output. So the arms are reached by
*replaying an identical prefix from reset*, and R4.2's "do not assume" clause
is taken literally: the prefix is replayed, the extended world digest is
compared across arms at the pre-intervention tick, and **if the digests
disagree the run stops and reports it** rather than comparing arms whose common
state was never established.

The digest was widened for this. It previously omitted
`burner.remaining_burning_fuel` and `entity.energy` -- so two worlds holding
different amounts of energy in a burner had equal digests, which is exactly the
variable these arms differ on. Widening it found no new leak: six repeated
resets still agree, and a reset after a fuelled 600-tick run equals a clean one.

**Each arm gets its own worker, and that is not tidiness.** The first run of
this tool put all three on one worker and the digest check refused the
comparison, reporting `tech|steam-power` present in the third arm and absent in
the first two. Measured cause: `steam-power`'s research trigger is
`craft-item iron-plate count=50`, and a plate line crafts iron plates. The
trigger's counter is cumulative on the force, it is *consumed* rather than
cleared when it fires -- 67 plates fired it and left 17, then 33 more fired it
again -- and no Lua accessor resets it. `clear_statistics` does clear the
production statistics, and the counter is not those: statistics read 0 after
the reset and the trigger still re-fired. `disable_research()` does not stop it
and `research_enabled` is read-only.

So sequential episodes on one worker **cannot** reach the same technology
state, whatever reset does. A fresh worker starts the counter at zero, and the
prefix produces about 14 plates -- far below the threshold -- so no trigger
fires during it in any arm.

The three arms, all from that state:

* **no_action** -- disruption, then nothing. The control. Whatever production
  appears here is the line's own stored energy, not a repair.
* **agent** -- disruption, then the model gets the same decision budget.
* **undisturbed** -- no disruption at all. The ceiling.

Production attributable to the agent is `agent - no_action`, and it is only
meaningful once `no_action` has a *confirmed outage*: without one, the
disruption is not shown to be capable of stopping the line, and the difference
between arms measures nothing.

**A confirmed outage in the control arm does not make the agent's arm a
recovery.** The three outcomes this tool distinguishes are different claims:

* `recovery_demonstrated` -- the *agent's own* arm stopped for the declared
  duration and then returned to rate and held it.
* `loss_averted` -- the control stopped, the agent's arm never did. Real, and
  measurable as the difference, but there was no outage in the agent's arm to
  have recovered from.
* `nothing_demonstrated` -- the control never stopped.

Reporting the second as the first is exactly the claim being withdrawn from
`phase5-demonstration.json`, so the distinction is in the report and not left
to a reader.

Run: uv run python tools/recovery_arms.py --model gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl import assistance as assistance_module  # noqa: E402
from factoriorl import recovery as recovery_module  # noqa: E402
from factoriorl.agent.adapters import OpenAICompatibleAdapter, ScriptedAdapter  # noqa: E402
from factoriorl.agent.loop import AgentConfig, AgentLoop, _prompt_digest  # noqa: E402
from factoriorl.engine_config import resolve_game_speed  # noqa: E402
from factoriorl.env import FactorioEnv  # noqa: E402
from factoriorl.paths import evidence_dir  # noqa: E402
from factoriorl.rcon import RCONClient  # noqa: E402
from factoriorl.seeding import Branch, SeedPlan  # noqa: E402
from factoriorl.session import WorkerSession  # noqa: E402
from factoriorl.tasks import get  # noqa: E402
from factoriorl.worker import WorkerManager  # noqa: E402

TASK = "plate_line"
ARMS = ("no_action", "agent", "undisturbed")

#: Evaluator setup: fuel both machines so the line reaches steady state without
#: an agent. Identical in all three arms, and part of the replayed prefix.
FUEL_BOTH = """
local s = game.surfaces[1]
for _, name in pairs({'burner-mining-drill', 'stone-furnace'}) do
  for _, e in pairs(s.find_entities_filtered({name = name})) do
    local inv = e.get_fuel_inventory()
    if inv then inv.insert({name = 'coal', count = %d}) end
  end
end
return 'fuelled'
"""

#: The disruption. Fuel inventories only; `entity.energy` untouched, which is
#: why the run-on exists and why it had to be measured first.
EMPTY_FUEL = """
local s = game.surfaces[1]
local removed = {}
for _, name in pairs({'burner-mining-drill', 'stone-furnace'}) do
  for _, e in pairs(s.find_entities_filtered({name = name})) do
    local inv = e.get_fuel_inventory()
    if inv then
      local taken = 0
      for _, stack in pairs(inv.get_contents()) do taken = taken + stack.count end
      inv.clear()
      removed[#removed + 1] = name .. '=' .. taken
    end
  end
end
return 'tick=' .. game.tick .. '|' .. table.concat(removed, ',')
"""


def _digest_lines(session: WorkerSession) -> list[str]:
    result = session.world_digest().response.result
    if isinstance(result, str):
        return result.split("\n")
    return list((result or {}).get("lines") or [])


def _history(env: FactorioEnv) -> list[tuple[int, dict]]:
    """The production history the env has been sampling, as the judge wants it."""
    return [(tick, dict(counts)) for tick, counts in (env._truth.get("window") or [])]


def _label(row: dict) -> str:
    """How one run is named in the report. Repeats of the agent arm are
    numbered; the single-run control arms keep their bare names."""
    if row["arm"] == "agent":
        return f"agent#{row['repeat'] + 1}"
    return row["arm"]


def _prefix(env: FactorioEnv, endpoint: str, settle_ticks: int, coal: int) -> None:
    """The identical run-up every arm replays: fuel, then settle."""
    with RCONClient(endpoint, timeout=30.0) as client:
        client.lua(FUEL_BOTH % coal)
    env.resync("evaluator fuelled both machines")
    advanced = 0
    while advanced < settle_ticks:
        advanced += env.advance(env.spec_.decision_ticks * 10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.environ.get("FRRL_AGENT_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--settle-ticks", type=int, default=3600)
    parser.add_argument(
        "--observe-ticks",
        type=int,
        default=9600,
        help="ticks to watch after the disruption. Must clear the criteria's "
        "deadline, or every verdict is UNKNOWN for lack of observation",
    )
    parser.add_argument("--recovery-budget", type=int, default=25)
    parser.add_argument(
        "--agent-repeats",
        type=int,
        default=3,
        help="how many times to run the agent arm. More than one because the "
        "outcome is not a measurement at n=1: `give_coal_N` targets the *nearest* "
        "entity and the drill and furnace are two tiles apart, so which machine "
        "gets refuelled turns on where the character happens to be standing. Two "
        "runs of the same model on the same scene differed by 39 plates on "
        "exactly that. The control arms are deterministic -- their equal digests "
        "are the evidence the prefix replays -- so they run once.",
    )
    parser.add_argument("--coal", type=int, default=50)
    parser.add_argument("--out", default="r4-recovery-arms.json")
    args = parser.parse_args()

    if not os.environ.get(args.api_key_env):
        print(f"{args.api_key_env} is not set in the environment")
        return 2

    criteria = recovery_module.Criteria()
    task = get(TASK)
    report: dict = {
        "gate": "R4.2 disruption and recovery, separately",
        "task": {"id": TASK, "version": task.spec.version},
        "state_reconstruction": {
            "method": "replay an identical prefix from reset",
            "why": (
                "RequestType has no snapshot or restore, no mid-run save exists, "
                "autosaves are disabled, and the only save on disk is the --create "
                "output. State branching is unavailable, so R4.2's replay clause "
                "applies -- and its 'do not assume' clause means the replay is "
                "validated rather than trusted."
            ),
            "validator": "world.digest(), widened to carry entity.energy and "
            "burner.remaining_burning_fuel",
        },
        "criteria": criteria.to_dict(),
        "supplied_knowledge": {
            "system_prompt_digest": _prompt_digest(),
            "task_description": task.spec.description,
        },
        "assistance": assistance_module.describe_assistance(task.spec),
        "arms": {},
        "runs": [],
    }

    report["state_reconstruction"]["worker_per_arm"] = (
        "one fresh worker per arm. The research trigger counter is cumulative on "
        "the force and unresettable, so sequential episodes on one worker cannot "
        "reach the same technology state -- see the module docstring."
    )
    manager = WorkerManager()

    # One worker per *run*, not per arm: the trigger-counter leak is per force,
    # so repeating the agent arm on one worker would reintroduce it.
    schedule = [("no_action", 0)]
    schedule += [("agent", index) for index in range(max(1, args.agent_repeats))]
    schedule.append(("undisturbed", 0))

    for arm, repeat in schedule:
        label = arm if arm != "agent" else f"agent#{repeat + 1}"
        print(f"\n--- arm: {label}", flush=True)
        handle = manager.launch(f"r4-{arm}-{repeat}")
        session = None
        try:
            with RCONClient(handle.spec.rcon_endpoint, timeout=30.0) as client:
                client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
            session = WorkerSession(handle, timeout=60.0)
            session.status()
            endpoint = handle.spec.rcon_endpoint

            env = FactorioEnv(
                task,
                session,
                SeedPlan(master=20260909, run_id="recovery-arms"),
                branch=Branch.TRAIN,
                split="train",
            )
            # The same scene every arm: one episode index, replayed.
            env._episode_index = 0
            env.reset()
            _prefix(env, endpoint, args.settle_ticks, args.coal)

            row: dict = {
                "arm": arm,
                "repeat": repeat,
                "pre_intervention_tick": int(env._observation.get("tick") or 0),
                "pre_intervention_digest": _digest_lines(session),
                "pre_intervention_plates": float(
                    (env._truth.get("produced") or {}).get(criteria.item, 0)
                ),
            }

            if arm != "undisturbed":
                with RCONClient(endpoint, timeout=30.0) as client:
                    row["disruption"] = str(client.lua(EMPTY_FUEL))
                env.resync("fuel inventories emptied")
                row["disrupted_at"] = int(env._observation.get("tick") or 0)
            else:
                row["disruption"] = None
                # The ceiling arm is judged from the same tick, so the three
                # histories are directly comparable.
                row["disrupted_at"] = row["pre_intervention_tick"]

            if arm == "agent":
                adapter = OpenAICompatibleAdapter(
                    base_url=args.base_url,
                    model=args.model,
                    api_key_env=args.api_key_env,
                    timeout=args.timeout,
                )
                config = AgentConfig(
                    task_id=TASK,
                    split="train",
                    max_steps=args.recovery_budget,
                    run_prefix="r4arm",
                )
                loop = AgentLoop(
                    env,
                    adapter,
                    config,
                    provenance={
                        "engine": handle.engine.to_dict(),
                        "workers": [handle.spec.manifest()],
                        "assistance": report["assistance"],
                    },
                )
                loop.run_dir.mkdir(parents=True, exist_ok=True)
                loop._write_manifest()
                row["segment"] = loop.run_segment(
                    0, first_step=0, budget=args.recovery_budget, fresh_memory=True
                )
                row["run_dir"] = str(loop.run_dir)
                row["model"] = args.model
            else:
                # The control arms take the identical *shape* of run with no
                # model: the same number of decisions, all `wait`. Not "skip
                # the decisions" -- that would give the arms different decision
                # counts and so different tick alignments.
                scripted = ScriptedAdapter(
                    [json.dumps({"action": env.catalog.wait_index, "reason": "no action"})]
                    * (args.recovery_budget + 4)
                )
                config = AgentConfig(
                    task_id=TASK,
                    split="train",
                    max_steps=args.recovery_budget,
                    run_prefix=f"r4{arm}",
                    record_summaries=False,
                )
                loop = AgentLoop(env, scripted, config)
                loop.run_dir.mkdir(parents=True, exist_ok=True)
                row["segment"] = loop.run_segment(
                    0, first_step=0, budget=args.recovery_budget, fresh_memory=True
                )

            # Watch the aftermath, tick-aligned, through the same env.
            advanced = 0
            while advanced < args.observe_ticks:
                advanced += env.advance(env.spec_.decision_ticks * 5)
            row["final_tick"] = int(env._observation.get("tick") or 0)
            row["final_plates"] = float((env._truth.get("produced") or {}).get(criteria.item, 0))
            row["verdicts"] = recovery_module.judge(
                _history(env), criteria, disrupted_at=row["disrupted_at"]
            )
            report["runs"].append(row)
            # `arms` keeps the first run of each arm, so the digest block
            # and every existing reader still resolve one row per arm.
            report["arms"].setdefault(arm, row)
            print(
                f"    plates {row['pre_intervention_plates']:.0f} -> "
                f"{row['final_plates']:.0f}; outage "
                f"{row['verdicts']['outage']['state']}, recovery "
                f"{row['verdicts']['recovery']['state']}",
                flush=True,
            )
        finally:
            # Only here: the `finally` already closes it, and closing twice
            # made the success path do it once and the failure path twice.
            if session is not None:
                try:
                    session.close()
                except OSError:
                    pass
            manager.cleanup(handle)

    # ---- was the common state actually common? --------------------------
    # Every run, not one per arm: a repeat that started from a different
    # world is as disqualifying as an arm that did.
    digests = {_label(row): row["pre_intervention_digest"] for row in report["runs"]}
    reference = digests["no_action"]
    mismatched = {name: lines for name, lines in digests.items() if lines != reference}
    report["common_state"] = {
        "validated": not mismatched,
        "compared_at_tick": {arm: report["arms"][arm]["pre_intervention_tick"] for arm in ARMS},
        "digest_lines": len(reference),
        "mismatched_arms": sorted(mismatched),
        "diff": {
            arm: {
                "only_here": sorted(set(lines) - set(reference))[:8],
                "only_in_reference": sorted(set(reference) - set(lines))[:8],
            }
            for arm, lines in mismatched.items()
        },
    }

    if not report["common_state"]["validated"]:
        report["comparison"] = None
        report["verdict"] = (
            "REFUSED: the arms did not start from the same state, so nothing here "
            "compares them. R4.2 requires a *validated* common pre-intervention "
            "state; replaying a prefix is not the same as reaching one."
        )
    else:

        def produced(row: dict) -> float:
            return row["final_plates"] - row["pre_intervention_plates"]

        runs_by_arm: dict[str, list[dict]] = {}
        for row in report["runs"]:
            runs_by_arm.setdefault(row["arm"], []).append(row)

        control = produced(runs_by_arm["no_action"][0])
        ceiling = produced(runs_by_arm["undisturbed"][0])
        agent_runs = runs_by_arm["agent"]
        agent_plates = sorted(produced(row) for row in agent_runs)
        # Median, not mean: with three runs and a bimodal outcome -- the coal
        # reaches the drill or it does not -- a mean reports a middle that no
        # run occupies.
        median = agent_plates[len(agent_plates) // 2]
        report["comparison"] = {
            "plates_after_the_intervention_point": {
                "no_action": control,
                "agent_each": agent_plates,
                "agent_median": median,
                "undisturbed": ceiling,
            },
            "outage": {
                "no_action": runs_by_arm["no_action"][0]["verdicts"]["outage"]["state"],
                "agent_each": [row["verdicts"]["outage"]["state"] for row in agent_runs],
                "undisturbed": runs_by_arm["undisturbed"][0]["verdicts"]["outage"]["state"],
            },
            "recovery": {
                "no_action": runs_by_arm["no_action"][0]["verdicts"]["recovery"]["state"],
                "agent_each": [row["verdicts"]["recovery"]["state"] for row in agent_runs],
                "undisturbed": runs_by_arm["undisturbed"][0]["verdicts"]["recovery"]["state"],
            },
            "attributable_to_the_agent": round(median - control, 2),
            "attributable_range": [
                round(agent_plates[0] - control, 2),
                round(agent_plates[-1] - control, 2),
            ],
            "ceiling": round(ceiling, 2),
            "agent_runs": len(agent_runs),
            "note": (
                "attributable = median(agent) - no_action, with undisturbed as "
                "the ceiling and the full per-run range beside it. The control "
                "arm is what says the disruption was capable of stopping the "
                "line at all; without a confirmed outage there, a difference "
                "between arms measures nothing. The range is reported because "
                "the arm is not deterministic: `give_coal_N` fuels the nearest "
                "entity and the two machines are two tiles apart."
            ),
        }

        # Four outcomes per agent run, and they are not the same claim.
        # Calling the third one "recovery" is precisely the error being
        # withdrawn from `phase5-demonstration.json`: an outage has to be
        # confirmed in the run that is said to have recovered from it, not
        # merely in a sibling arm.
        def classify(row: dict) -> str:
            outage_state = row["verdicts"]["outage"]["state"]
            recovery_state = row["verdicts"]["recovery"]["state"]
            if outage_state == "confirmed" and recovery_state == "confirmed":
                return "recovery_demonstrated"
            if outage_state == "confirmed":
                # Its own line stopped for the declared duration and never came
                # back. The first version of this code called that "loss
                # averted" because it only tested the recovery clause, and
                # printed "the agent's arm never stopped" over an arm whose
                # outage read `confirmed`.
                return "agent_failed"
            if outage_state == "refused":
                return "loss_averted"
            return "indeterminate"

        per_run = {_label(row): classify(row) for row in agent_runs}
        report["per_agent_run_outcome"] = per_run
        control_outage = runs_by_arm["no_action"][0]["verdicts"]["outage"]
        if control_outage["state"] != "confirmed":
            report["outcome"] = "nothing_demonstrated"
            report["verdict"] = (
                "the control arm never suffered a confirmed outage, so the "
                "disruption is not shown to be capable of stopping the line and "
                "no difference between arms can be attributed to the agent"
            )
        else:
            tally = {
                name: sum(1 for v in per_run.values() if v == name)
                for name in set(per_run.values())
            }
            report["outcome"] = max(tally, key=lambda name: (tally[name], name))
            report["verdict"] = (
                f"the control arm's outage was confirmed at tick "
                f"{control_outage['at_tick']}, so the disruption can stop this "
                f"line. Across {len(agent_runs)} agent runs the outcomes were "
                f"{tally}: "
                + ", ".join(f"{name} {outcome}" for name, outcome in sorted(per_run.items()))
                + f". Median production after the intervention point was "
                f"{median:.0f} plates against the control's {control:.0f}, "
                f"ceiling {ceiling:.0f}; per-run {agent_plates}. "
                "`loss_averted` is not `recovery_demonstrated`: a run whose own "
                "line never stopped had no outage to recover from, and reporting "
                "one would repeat the claim R4 exists to withdraw."
            )

    path = evidence_dir() / args.out
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\ncommon state validated: {report['common_state']['validated']}", flush=True)
    if report["comparison"]:
        print(json.dumps(report["comparison"], indent=1), flush=True)
    print(report["verdict"], flush=True)
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
