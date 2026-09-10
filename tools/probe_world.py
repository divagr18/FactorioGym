"""Gate A1, measured against a live engine rather than asserted.

The roadmap's A1 gate asks for a real-engine probe showing natural deposits and
terrain, the expected starting inventory, no prebuilt production assets or bonus
items, normal recipe locks, ticks advancing while the controller is idle -- and
then a save reloaded with its inventory, machines, technology and production
state intact.

Every one of those facts was measured by hand while A0 and A1 were written, and
nothing re-checked them afterwards. This does, and writes what it found to
`docs/evidence/a1-world-lifecycle.json`.

It needs an engine, so it is a tool rather than a test. Two workers are launched
in sequence: one to build and save, one to resume from that save.

Run:
  uv run python tools/probe_world.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl import worlds  # noqa: E402
from factoriorl.agent.checkpoint import Checkpointer  # noqa: E402
from factoriorl.freeplay import starting_inventory  # noqa: E402
from factoriorl.rcon import RCONClient  # noqa: E402
from factoriorl.session import WorkerSession  # noqa: E402
from factoriorl.worker import WorkerManager  # noqa: E402

#: A prototype the agent starts with, placed to prove a machine survives a
#: reload. Its own starting inventory supplies it, so nothing is conjured.
WITNESS = "stone-furnace"


def _lua(client: RCONClient, code: str):
    return client.lua(code)


def survey(client: RCONClient) -> dict:
    """What the generated map actually contains, by radius."""
    out: dict = {}
    for radius in (32, 128, 256):
        out[str(radius)] = {
            "resources": _lua(
                client,
                f'local s=game.surfaces["nauvis"] return s.count_entities_filtered('
                f'{{position={{0,0}}, radius={radius}, type="resource"}})',
            ),
            "trees": _lua(
                client,
                f'local s=game.surfaces["nauvis"] return s.count_entities_filtered('
                f'{{position={{0,0}}, radius={radius}, type="tree"}})',
            ),
            "cliffs": _lua(
                client,
                f'local s=game.surfaces["nauvis"] return s.count_entities_filtered('
                f'{{position={{0,0}}, radius={radius}, type="cliff"}})',
            ),
            "enemies": _lua(
                client,
                f'local s=game.surfaces["nauvis"] return s.count_entities_filtered('
                f'{{position={{0,0}}, radius={radius}, force="enemy"}})',
            ),
        }
    out["nearest_ore"] = _lua(
        client,
        'local s=game.surfaces["nauvis"] '
        'local e=s.find_entities_filtered({position={0,0}, radius=400, type="resource"}) '
        "local best,bd=nil,1e9 for _,x in pairs(e) do "
        "local d=math.sqrt(x.position.x^2+x.position.y^2) "
        "if d<bd then bd=d best=x end end "
        'if best then return best.name.."@"..math.floor(bd) end return "none"',
    )
    out["ore_kinds"] = _lua(
        client,
        'local s=game.surfaces["nauvis"] local t={} '
        'for _,x in pairs(s.find_entities_filtered({position={0,0}, radius=256, type="resource"})) '
        "do t[x.name]=(t[x.name] or 0)+1 end "
        'local o={} for k,v in pairs(t) do o[#o+1]=k..":"..v end return table.concat(o, ",")',
    )
    return out


def force_state(client: RCONClient) -> dict:
    """Recipe locks and research, which a resume must not reset."""
    return {
        "enabled_recipes": _lua(
            client,
            "local f=game.forces['player'] local n=0 "
            "for _,r in pairs(f.recipes) do if r.enabled then n=n+1 end end return n",
        ),
        "total_recipes": _lua(
            client, "local n=0 for _ in pairs(game.forces['player'].recipes) do n=n+1 end return n"
        ),
        "researched": _lua(
            client,
            "local f=game.forces['player'] local n=0 "
            "for _,t in pairs(f.technologies) do if t.researched then n=n+1 end end return n",
        ),
    }


def character_state(client: RCONClient) -> dict:
    return {
        "inventory": _lua(
            client,
            "local ch=storage.frrl_character if not ch or not ch.valid then return 'absent' end "
            "local inv=ch.get_inventory(defines.inventory.character_main) "
            "local t=inv.get_contents() local o={} "
            "for _,e in pairs(t) do o[#o+1]=e.name..':'..e.count end "
            "table.sort(o) return table.concat(o, ',')",
        ),
        "position": _lua(
            client,
            "local ch=storage.frrl_character if not ch or not ch.valid then return 'absent' end "
            "return math.floor(ch.position.x)..','..math.floor(ch.position.y)",
        ),
    }


def player_entities(client: RCONClient) -> dict:
    return {
        "player_force_entities": _lua(
            client,
            'local s=game.surfaces["nauvis"] local n=0 '
            'for _,e in pairs(s.find_entities_filtered({force="player"})) do '
            'if e.type ~= "character" then n=n+1 end end return n',
        ),
        "witness_present": _lua(
            client,
            f'local s=game.surfaces["nauvis"] return s.count_entities_filtered('
            f'{{name="{WITNESS}"}})',
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--out", default=str(ROOT / "docs" / "evidence" / "a1-world-lifecycle.json")
    )
    args = parser.parse_args()

    mode = worlds.get("open_factory")
    manager = WorkerManager()
    freeplay = starting_inventory(manager.engine.executable)
    report: dict = {
        "measures": "roadmap A1's gate, against a live engine",
        "world": mode.to_dict(),
        "freeplay_source": {k: freeplay[k] for k in ("source", "sha256", "items")},
        "engine": manager.engine.to_dict(),
    }
    started = time.perf_counter()

    # --- segment one: a fresh world, built on and saved ---------------------
    handle = manager.launch("probe-a1", map_seed=args.seed, terrain=mode.terrain)
    checkpoint = None
    try:
        with RCONClient(handle.spec.rcon_endpoint, timeout=60.0) as client:
            client.lua("game.speed = 1.0 return game.speed")
            client.lua(
                'local s=game.surfaces["nauvis"] s.request_to_generate_chunks({0,0}, 8) '
                "s.force_generate_chunk_requests() return true"
            )
            report["terrain"] = survey(client)
        session = WorkerSession(handle, timeout=60.0)
        session.status()
        session.configure(free_running=True)
        opened = session.open_world(
            inventory=freeplay["items"],
            observation_profile=mode.observation_profile,
            action_profile=mode.action_profile,
            chart_radius=mode.chart_radius,
        )
        report["open_world"] = opened.response.result

        with RCONClient(handle.spec.rcon_endpoint, timeout=60.0) as client:
            report["fresh"] = {
                **character_state(client),
                **player_entities(client),
                **force_state(client),
            }
            # Ticks advance while the controller is idle: the free-running clause.
            before = client.lua("return game.tick")
            time.sleep(2.0)
            after = client.lua("return game.tick")
            report["fresh"]["ticks_advanced_while_idle"] = after - before

            # Place the witness from the character's own inventory, so a reload
            # has something to have preserved.
            report["witness_placed"] = client.lua(
                "local ch=storage.frrl_character "
                "local p={ch.position.x+3, ch.position.y} "
                f'local e=game.surfaces["nauvis"].create_entity({{name="{WITNESS}", '
                'position=p, force="player"}) '
                "if e then "
                "ch.get_inventory(defines.inventory.character_main)"
                f'.remove({{name="{WITNESS}", count=1}}) '
                'return "placed" end return "failed"'
            )
            report["after_build"] = {**character_state(client), **player_entities(client)}

        keeper = Checkpointer(
            session=session,
            write_data=handle.spec.write_data,
            run_dir=ROOT / "runtime" / "runs" / "probe-a1",
        )
        record = keeper.save("probe")
        report["checkpoint"] = record
        checkpoint = Path(record["path"]) if record.get("verified") else None
        session.close()
    finally:
        manager.cleanup(handle)

    report["worker_directory_removed"] = not handle.spec.directory.exists()

    # --- segment two: resume from that save ---------------------------------
    if checkpoint is None:
        report["resume"] = {"skipped": "no verified checkpoint to resume from"}
    else:
        resumed = manager.launch(
            "probe-a1-resume",
            map_seed=args.seed,
            terrain=mode.terrain,
            save_source=checkpoint,
        )
        try:
            session = WorkerSession(resumed, timeout=60.0)
            session.status()
            session.configure(free_running=True)
            reopened = session.open_world(
                observation_profile=mode.observation_profile,
                action_profile=mode.action_profile,
                fresh=False,
            )
            with RCONClient(resumed.spec.rcon_endpoint, timeout=60.0) as client:
                report["resume"] = {
                    "open_world": reopened.response.result,
                    **character_state(client),
                    **player_entities(client),
                    **force_state(client),
                }
            session.close()
        finally:
            manager.cleanup(resumed)

    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    report["verdict"] = verdict(report)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2))
    print(f"wrote {args.out}")
    return 0 if all(v is True for v in report["verdict"].values()) else 1


def verdict(report: dict) -> dict:
    """Each gate clause, as a boolean nobody has to squint at."""
    terrain = report.get("terrain") or {}
    fresh = report.get("fresh") or {}
    after = report.get("after_build") or {}
    resume = report.get("resume") or {}
    expected = ",".join(
        sorted(f"{k}:{v}" for k, v in (report["freeplay_source"]["items"] or {}).items())
    )
    return {
        "natural_deposits_present": (terrain.get("32", {}).get("resources") or 0) > 0,
        "trees_present": (terrain.get("256", {}).get("trees") or 0) > 0,
        "no_enemies": (terrain.get("256", {}).get("enemies") or 0) == 0,
        "starting_inventory_matches_freeplay": fresh.get("inventory") == expected,
        "no_prebuilt_production_assets": fresh.get("player_force_entities") == 0,
        "recipes_not_all_unlocked": (
            (fresh.get("enabled_recipes") or 0) < (fresh.get("total_recipes") or 0)
        ),
        "ticks_advance_while_idle": (fresh.get("ticks_advanced_while_idle") or 0) > 0,
        "checkpoint_verified": bool((report.get("checkpoint") or {}).get("verified")),
        "checkpoint_survives_worker_teardown": report.get("worker_directory_removed") is True,
        "resume_kept_the_machine": resume.get("witness_present") == after.get("witness_present"),
        "resume_kept_the_inventory": resume.get("inventory") == after.get("inventory"),
        "resume_kept_research": resume.get("researched") == fresh.get("researched"),
        "resume_destroyed_nothing": ((resume.get("open_world") or {}).get("destroyed")) == 0,
    }


if __name__ == "__main__":
    sys.exit(main())
