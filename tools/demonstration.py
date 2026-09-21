"""The Phase 5 agent demonstration (DESIGN.md 5.7).

    Construct an automated plate line, inject a documented disruption, and
    restore sustained production.

Five phases, in order, with the agent in control of two of them and the driver
in control of nothing that could be mistaken for repair:

1. **Commission.** The agent fuels a drill and a furnace that the scene placed
   but left empty, until the line produces its first plate.
2. **Window one.** The agent is stopped. The world advances and production is
   counted. Nothing replenishes anything, which is what makes "operates without
   manual replenishment" a measurement rather than a claim.
3. **Disruption.** The driver empties both fuel inventories, records exactly
   what it removed and at which tick, and touches nothing else.
4. **Recovery.** The agent is given control again.
5. **Window two.** The agent is stopped again and production is counted again.

The acceptance clause that shapes the design is "no undeclared human repair or
factory-building script completes the run". The driver's only write to the world
is the disruption in phase 3, and it is recorded in the report with the item
counts it took. Everything that restores production is a model decision, and
every one of those is in `decisions.jsonl` with the prompt that produced it.

What this demonstration does not do
-----------------------------------
The agent commissions the line; it does not build it from parts. That is a
limit of the primitive catalog rather than a shortcut, and
`factoriorl.tasks.families.plate_line` records the measurement behind it: a
placement's position is the character's position plus a one-tile offset, 2x2
machines snap to integer centres, and the only furnace centre that catches a
drill's drop tile has the builder standing inside its footprint. Reporting that
honestly is worth more than a demonstration that pretended otherwise.

Run: uv run python tools/demonstration.py --model gpt-5.6-luna
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl import assistance as assistance_module  # noqa: E402

# `action_vocabulary`, `legal_actions` and `summarise` are deliberately *not*
# imported any more. They were needed only by `play()`, this file's hand-copy of
# `AgentLoop.run_episode`; the loop owns the stepping now, and the fact that
# nothing here needs the prompt-building pieces is the check that the copy is
# really gone.
from factoriorl.agent.loop import AgentConfig, AgentLoop  # noqa: E402
from factoriorl.engine_config import resolve_game_speed  # noqa: E402
from factoriorl.rcon import RCONClient  # noqa: E402

#: Ticks each measurement window runs with the agent stopped. Long enough for a
#: line that works to make several plates -- the furnace produces roughly one
#: every 240 ticks -- and short enough that the output slot cannot fill and
#: stall a line that is otherwise healthy.
WINDOW_TICKS = 3600

#: Plates a healthy line makes in one window. The furnace produces roughly one
#: every 240 ticks, so a 3,600-tick window should yield about fifteen. Requiring
#: a real fraction of that is what separates a restored line from a furnace
#: burning off residual heat: the first demonstration to "pass" made two plates
#: in its second window while the drill sat at status 53, out of fuel, and a
#: `produced > 0` test called that sustained production.
SUSTAINED_MIN_PLATES = 6

#: `defines.entity_status.working`. The drill is the upstream end of the line --
#: if it is not running, nothing downstream will be for long, whatever the
#: furnace is doing with what it already has.
STATUS_WORKING = 1

#: What the disruption removes, and how it is put back. Emptying the fuel is a
#: real Factorio failure mode with a recovery the catalog can express; the
#: alternative -- destroying a machine -- would need the agent to rebuild it,
#: which `plate_line`'s docstring measures as not reliably expressible here.
DISRUPTION = "fuel_outage"

EMPTY_FUEL = """
local s = game.surfaces[1]
local removed = {}
for _, name in pairs({'burner-mining-drill', 'stone-furnace'}) do
  for _, e in pairs(s.find_entities_filtered{name=name}) do
    local inv = e.get_fuel_inventory()
    if inv then
      local contents = inv.get_contents()
      for _, stack in pairs(contents) do
        removed[#removed + 1] = { entity = name, item = stack.name, count = stack.count }
      end
      inv.clear()
    end
  end
end
return helpers.table_to_json{ tick = game.tick, removed = removed }
"""

PRODUCTION = """
local s = game.surfaces[1]
local stats = game.forces['player'].get_item_production_statistics(s)
local drill = s.find_entities_filtered{name='burner-mining-drill'}[1]
local furnace = s.find_entities_filtered{name='stone-furnace'}[1]
local function fuel(e)
  if not e then return 0 end
  local inv = e.get_fuel_inventory()
  if not inv then return 0 end
  local total = 0
  for _, stack in pairs(inv.get_contents()) do total = total + stack.count end
  return total
end
return helpers.table_to_json{
  tick = game.tick,
  plates = stats.get_input_count('iron-plate'),
  drill_status = drill and drill.status or nil,
  furnace_status = furnace and furnace.status or nil,
  drill_fuel = fuel(drill),
  furnace_fuel = fuel(furnace),
}
"""


def probe(endpoint, code: str) -> dict:
    with RCONClient(endpoint, timeout=30.0) as client:
        return json.loads(client.lua(code))


def window(env, endpoint, ticks: int) -> dict:
    """Advance the world with the agent stopped, and count what it produced.

    Through `env.advance`, not `session.step`. The old version stepped the
    session directly in 300-tick chunks and discarded the observation and truth
    each response carried, so the env's cache -- and with it the next prompt --
    was left describing the world as it had been before the window. Measured on
    the committed run: the first post-outage prompt said "game tick 450" while
    the world was at 4050 (R4.1).

    `env.advance` also chunks by `decision_ticks` rather than 300, which keeps
    the production history *sampled*: a coarser jump leaves a window nothing
    observed, and `Predicate.window_sampled` refuses those.
    """
    before = probe(endpoint, PRODUCTION)
    started = time.perf_counter()
    advanced = env.advance(ticks)
    after = probe(endpoint, PRODUCTION)
    return {
        "ticks": advanced,
        "wall_seconds": round(time.perf_counter() - started, 2),
        "plates_before": before["plates"],
        "plates_after": after["plates"],
        "plates_produced": after["plates"] - before["plates"],
        "drill_status_after": after.get("drill_status"),
        "furnace_status_after": after.get("furnace_status"),
        "drill_fuel_after": after.get("drill_fuel"),
        "furnace_fuel_after": after.get("furnace_fuel"),
        # Both halves, because either alone is satisfiable without a line. A
        # furnace with residual heat makes a plate or two after its supply has
        # died, and a fuelled drill with a dead furnace makes none at all.
        "sustained": bool(
            after["plates"] - before["plates"] >= SUSTAINED_MIN_PLATES
            and after.get("drill_status") == STATUS_WORKING
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--split", default="train")
    parser.add_argument("--commission-budget", type=int, default=40)
    parser.add_argument("--recovery-budget", type=int, default=25)
    parser.add_argument("--window-ticks", type=int, default=WINDOW_TICKS)
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument(
        "--out",
        default=None,
        help="where to write the report. Defaults to the run's own directory; "
        "this file used to have no way to write anywhere else",
    )
    parser.add_argument(
        "--publish-evidence",
        action="store_true",
        help="also overwrite docs/evidence/phase5-demonstration.json, which is "
        "tracked. Opt-in because a demonstration run used to destroy committed "
        "evidence and dirty the working tree with no way to decline",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=5.0,
        help="hard cap on provider spend. A request whose worst case would "
        "exceed what is left is not sent at all",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=3600.0,
        help="ceiling on the run, measured from the first gameplay observation",
    )
    args = parser.parse_args()

    # Credentials come from the environment; the adapter redacts before writing.
    # Shared with `factoriorl doctor-agent` rather than inlined here, which is
    # what let the diagnostic report a key as unset about a run that then found
    # it. Returns names only, never values.
    from factoriorl.agent.credentials import load_env_file

    load_env_file(args.env_file)

    from factoriorl import manifest as manifest_module
    from factoriorl.agent import provider as provider_module
    from factoriorl.env import FactorioEnv
    from factoriorl.seeding import Branch, SeedPlan, seed_everything
    from factoriorl.session import WorkerSession
    from factoriorl.tasks import get
    from factoriorl.worker import WorkerManager

    # Through the shared builder, so this run has a spend cap. It used to
    # construct a bare adapter and therefore had none at all: harmless while it
    # pointed at a local endpoint, and a liability the moment an entrypoint
    # points at a paid provider. `factoriorl.agent.provider` exists so that
    # cannot happen by omission again.
    provider = provider_module.build(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        timeout=180.0,
        token_parameter="max_completion_tokens",
        temperature=None,
        max_cost_usd=args.max_cost_usd,
        max_wall_seconds=args.max_wall_seconds,
    )
    adapter = provider.adapter
    config = AgentConfig(
        task_id="plate_line",
        episodes=1,
        max_steps=args.commission_budget + args.recovery_budget,
        split=args.split,
        run_prefix="demo",
    )

    run_id = manifest_module.new_run_id(config.run_prefix)
    seeded = seed_everything(20260908)
    plan = SeedPlan(master=20260908, run_id=run_id)
    task = get("plate_line")

    manager = WorkerManager()
    handle = manager.launch(f"demo-{run_id[-8:]}")
    endpoint = handle.spec.rcon_endpoint
    started = time.perf_counter()
    report: dict = {"run_id": run_id, "plan_section": "5.7", "model": args.model}
    session = None
    try:
        with RCONClient(endpoint, timeout=30.0) as client:
            client.lua(f"game.speed = {resolve_game_speed()} return game.speed")
        session = WorkerSession(handle, timeout=30.0)
        session.status()
        branch = Branch.TRAIN if args.split == "train" else Branch.EVAL
        env = FactorioEnv(task, session, plan, branch=branch, split=args.split)
        loop = AgentLoop(
            env,
            adapter,
            config,
            run_id=run_id,
            provenance={
                "engine": handle.engine.to_dict(),
                "workers": [handle.spec.manifest()],
                "seeds": {**plan.to_dict(), "seeded": seeded},
            },
        )
        # `run` would do this, and the demonstration drives the loop itself, so
        # the artifacts a replay needs have to be created here instead.
        loop.run_dir.mkdir(parents=True, exist_ok=True)
        loop._write_json(loop.run_dir / "status.json", {"run_id": run_id, "state": "running"})
        loop._write_manifest()

        env.reset()
        opening = probe(endpoint, PRODUCTION)

        def producing() -> bool:
            return probe(endpoint, PRODUCTION)["plates"] > opening["plates"]

        report["phase_1_commission"] = loop.run_segment(
            0, first_step=0, budget=args.commission_budget, fresh_memory=True, until=producing
        )
        report["phase_2_window"] = window(env, endpoint, args.window_ticks)

        disruption = probe(endpoint, EMPTY_FUEL)
        report["phase_3_disruption"] = {
            "kind": DISRUPTION,
            "declared": "the driver emptied both fuel inventories and wrote nothing else",
            **disruption,
        }

        # The intervention is a raw world write the env issued no request for,
        # so it cannot see it. Without this the recovery prompt would report a
        # drill holding 48 coal and `working` at the moment it was empty --
        # which is exactly what the committed run recorded.
        env.resync(f"{DISRUPTION}: both fuel inventories emptied")

        # Recorded so the report shows the state the recovery began from.
        report["phase_3_disruption"]["state_after"] = probe(endpoint, PRODUCTION)
        report["phase_3_disruption"]["observation_after_resync"] = {
            "tick": int(env._observation.get("tick") or 0),
            "entities": [
                {
                    "type": record.get("type"),
                    "status": record.get("st"),
                    "working": record.get("working"),
                    "fuel": record.get("fuel") or {},
                }
                for record in (env._observation.get("entities") or [])
            ],
        }

        def refuelled() -> bool:
            """Both machines hold fuel again.

            Not "plates went up": the furnace keeps producing for a while on
            heat it already had, so a plate counter says the line recovered
            when nothing has been refuelled at all. That false positive ended
            the first passing run's recovery phase after eight decisions, one
            of which was a `no_space` refusal.
            """
            state = probe(endpoint, PRODUCTION)
            return bool(state.get("drill_fuel") and state.get("furnace_fuel"))

        commission = report["phase_1_commission"]
        report["phase_4_recovery"] = loop.run_segment(
            0,
            first_step=commission["first_step"] + commission["steps"],
            budget=args.recovery_budget,
            until=refuelled,
        )
        report["phase_4_recovery"]["refuelled"] = refuelled()
        report["phase_5_window"] = window(env, endpoint, args.window_ticks)

        final = probe(endpoint, PRODUCTION)
        report["totals"] = {
            "game_ticks": final["tick"],
            "game_seconds": round(final["tick"] / 60.0, 1),
            "wall_seconds": round(time.perf_counter() - started, 1),
            "plates_total": final["plates"],
            "decisions": len(loop.decisions),
            # `describe_assistance`, not the action profile. The old field put
            # "primitive-v1" under a name that reads as an assistance record,
            # so this run had no assistance record at all.
            "assistance": assistance_module.describe_assistance(task.spec),
            "action_profile": task.spec.action_profile,
            # Every refresh the env made outside a step, with its declared
            # reason. An evaluator write is not a decision and must not be
            # invisible in the record either.
            "out_of_band_refreshes": list(env._resyncs),
            "deliberation_profile": "language-model-v1",
        }
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        latencies = []
        for decision in loop.decisions:
            for attempt in decision.attempts:
                block = attempt.reply.usage or {}
                for key in usage:
                    usage[key] += int(block.get(key) or 0)
                latencies.append(attempt.reply.latency_ms)
        report["model_usage"] = usage
        report["latency_ms"] = {
            "calls": len(latencies),
            "mean": round(sum(latencies) / max(len(latencies), 1), 1),
        }
        report["restored"] = bool(report["phase_5_window"]["sustained"])
        report["operated_before_disruption"] = bool(report["phase_2_window"]["sustained"])
        loop._write_json(loop.run_dir / "status.json", {"run_id": run_id, "state": "completed"})
        session.close()
    finally:
        manager.cleanup(handle)

    report["limits"] = provider.to_dict()

    # Run-local by default. This wrote straight into `docs/evidence/` with no
    # way to redirect, so every demonstration run overwrote committed evidence
    # and dirtied the working tree -- and `docs/AGENT.md` had to warn people to
    # stash afterwards. Publishing is now something you ask for.
    destination = Path(args.out) if args.out else (loop.run_dir / "demonstration.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = provider.redact(json.dumps(report, indent=2, default=str))
    destination.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2, default=str))
    print(f"\nwrote {destination}")
    if args.publish_evidence:
        published = ROOT / "docs" / "evidence" / "phase5-demonstration.json"
        published.write_text(rendered + "\n", encoding="utf-8")
        print(f"published {published.relative_to(ROOT)} (tracked; commit or restore it)")
    print(
        f"spent ${provider.budget.committed_usd:.4f} of ${args.max_cost_usd:.2f} "
        f"over {provider.budget.calls} calls"
    )
    print(f"replay: uv run python tools/replay.py runtime/runs/{run_id}")
    return 0 if report.get("restored") else 1


if __name__ == "__main__":
    sys.exit(main())
