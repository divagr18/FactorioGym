"""Turn a run directory into the report A5.3 asks for.

A5.3 wants: the replay and save links, actual factory progress, the first
consequential failure, representative trace evidence, and the next three ranked
fixes -- with defects separated into interface/runtime, inaccessible knowledge,
agent decisions, and time/cost limits.

Everything except the ranking is *measured*, and this computes it. The ranking
and the narrative are a person's job and are left as headings to fill in: a tool
that ranked its own findings would be presenting an opinion as a measurement,
which is the failure mode this repository keeps correcting.

The one piece of judgement it does make is naming a **first consequential
failure**, and it says exactly what it means by it: the earliest decision whose
action failed or was refused and whose failure signature then recurred. A single
refusal an agent recovered from is not consequential; the first one it went on
to repeat is where the run started going wrong.

Run:
  uv run python tools/run_report.py <run-id-or-dir> [--out docs/evidence/a5-first-run.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl import manifest as manifest_module  # noqa: E402

#: How many times a failure signature must appear before the first instance is
#: called consequential. Same threshold the agent's own stall detector uses, and
#: for the same reason: twice is a coincidence.
RECURRENCE = 3


def _resolve(target: str) -> Path:
    candidate = Path(target)
    if candidate.is_dir():
        return candidate
    found = manifest_module.runs_dir() / target
    if found.is_dir():
        return found
    raise SystemExit(f"no such run: {target}")


def _lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _relative(path: Path) -> str:
    """A repository-relative path, or the bare directory name if it is outside."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def first_consequential_failure(decisions: list[dict]) -> dict | None:
    """The earliest failure that went on to recur.

    Not simply the first failure: an agent that is refused once, understands
    why, and does something else has not hit a problem worth ranking. The first
    one it repeats is where the run started going wrong.
    """
    signatures: Counter[str] = Counter()
    first_seen: dict[str, dict] = {}
    for decision in decisions:
        for record in (decision.get("actions") or []) + (decision.get("refused") or []):
            error = record.get("action_error") or record.get("detail") or record.get("failure")
            if not error or record.get("status") == "completed":
                continue
            signature = json.dumps(
                [
                    record.get("key"),
                    record.get("target"),
                    sorted((record.get("arguments") or {}).items()),
                ],
                sort_keys=True,
                default=str,
            )
            signatures[signature] += 1
            first_seen.setdefault(
                signature,
                {
                    "decision": decision.get("step"),
                    "key": record.get("key"),
                    "arguments": record.get("arguments"),
                    "error": str(error)[:300],
                    "reason_given": decision.get("reason"),
                    "plan_at_the_time": decision.get("plan"),
                },
            )
    recurring = [(sig, count) for sig, count in signatures.items() if count >= RECURRENCE]
    if not recurring:
        return None
    best = min(recurring, key=lambda pair: first_seen[pair[0]]["decision"] or 0)
    entry = dict(first_seen[best[0]])
    entry["times_repeated"] = best[1]
    return entry


#: What a field says when the run never wrote the file that holds it.
#:
#: `None` was what it said before, and rendered as prose it produced lines like
#: "gameplay Nones of a Nones limit" and "spend $None of a $None cap" -- which
#: read as a measurement rather than as its absence. Marking absence matters
#: more here than anywhere else in the repo: this document is the one a person
#: reads to decide what the run showed.
#:
#: Short, because the banner at the top of an interrupted report already
#: explains why these are missing; spelling the reason out again in each of the
#: six fields that use it turned the section into a wall of one sentence.
UNRECORDED = "*unrecorded*"


def _or_unrecorded(value: object, suffix: str = "") -> str:
    return UNRECORDED if value is None else f"{value}{suffix}"


def stated_plans(decisions: list[dict]) -> list[dict]:
    """Every plan the agent actually stated, in order, deduplicated.

    Read from `decisions.jsonl` rather than from the memory snapshot in
    `summary.json`, because a run that is stopped never writes that snapshot --
    and the report then declared "the agent never used the `plan` field", which
    for the run this was first pointed at was the opposite of true: it stated a
    plan on 126 of 253 decisions, and one of them ("Practical maximum without
    stone") is the single most informative line in the whole run.

    A report that quietly turns a missing file into a claim about the agent is
    worse than one that admits it has nothing.
    """
    seen: list[dict] = []
    for decision in decisions:
        plan = (decision.get("plan") or "").strip()
        if not plan:
            continue
        if seen and seen[-1]["plan"] == plan:
            seen[-1]["held_until"] = decision.get("step")
            continue
        seen.append({"plan": plan, "stated_at": decision.get("step"), "held_until": None})
    return seen


def _finalization_seconds(result: dict) -> float | None:
    """How long finalization took, summed over its steps.

    `finalization` records a list of steps, each with its own `seconds`, and
    never a total -- so reading `finalization["seconds"]` was always `None`.
    """
    steps = (result.get("finalization") or {}).get("steps") or []
    seconds = [s.get("seconds") for s in steps if isinstance(s, dict)]
    measured = [float(v) for v in seconds if isinstance(v, int | float)]
    return round(sum(measured), 3) if measured else None


def first_machine_output(samples: list[dict]) -> int | None:
    """The tick of the first sample showing any machine-made item."""
    for sample in samples:
        if sample.get("machine_produced"):
            return sample.get("tick")
    return None


def render(run_dir: Path) -> str:
    summary = _load(run_dir / "summary.json")
    result = _load(run_dir / "result.json")
    config = _load(run_dir / "config.json")
    decisions = _lines(run_dir / "decisions.jsonl")
    events = _lines(run_dir / "tool_events.jsonl")
    samples = _lines(run_dir / "production.jsonl")
    saves = sorted(p.name for p in (run_dir / "saves").glob("*.zip"))

    limits = result.get("limits") or {}
    # `budget`, not `spend`. The key was wrong from the day this was written, so
    # `spend` was always `{}` and every report ever generated showed the cost of
    # the run as `None` -- eleventh instance in this repository of a field that
    # is declared and never read, and the first one that hid a number the run
    # printed to the console on its way out.
    spend = limits.get("budget") or {}
    clock = limits.get("clock") or {}
    episodes = result.get("episodes") or []
    production = (episodes[0].get("production") if episodes else {}) or {}
    machine = production.get("cumulative_produced") or {}
    # `result.json` is written when the run ends, and a run that was interrupted
    # has none -- but `production.jsonl` is appended as it goes. Falling back to
    # the last sample is the difference between a report that says "nothing" and
    # one that says what actually happened.
    if not machine and samples:
        machine = samples[-1].get("machine_produced") or {}
    # Same argument as `machine` above: `tool_events.jsonl` is appended as the
    # run goes, so a stopped run still has every action in it. Reading the verb
    # tally only from `summary.json` made a 381-action run report "actions by
    # verb: none" two lines under "381 tool actions".
    by_key = summary.get("tool_actions_by_key") or Counter(
        row.get("key") for row in events if row.get("key")
    )
    statuses = Counter(row.get("status") for row in events)
    fallbacks = result.get("fallback_decisions")
    if fallbacks is None:
        fallbacks = sum(1 for d in decisions if "fallback" in str(d.get("resolution") or ""))
    ticks_to_first = production.get("ticks_to_first_output") or first_machine_output(samples)
    interrupted = not summary and not result
    latency = result.get("latency_ms") or {}
    consequential = first_consequential_failure(decisions)

    last_sample = samples[-1] if samples else {}
    lines = [
        f"# A5 first run -- {run_dir.name}",
        "",
        "Exploratory result, not a benchmark success claim. An open world has no",
        "success predicate, no layout families and no reward components, so nothing",
        "measured here is comparable to a benchmark task.",
        "",
        *(
            [
                "**This run was stopped rather than finished.** It never reached",
                "finalization, so `summary.json` and `result.json` do not exist and every",
                "number they would have held is marked unrecorded below. What *is* here was",
                "appended as the run went -- decisions, tool events and production samples --",
                "and is complete up to the moment it was stopped.",
                "",
            ]
            if interrupted
            else []
        ),
        "## What was run",
        "",
        f"- model `{(config.get('adapter') or {}).get('model')}`, "
        f"adapter `{(config.get('adapter') or {}).get('adapter') or 'openai-compatible'}`",
        f"- world `{(config.get('task') or {}).get('id')}` "
        f"v{(config.get('task') or {}).get('version')}, "
        f"deliberation `{config.get('deliberation_profile')}`",
        f"- clock `{(limits.get('clock') or {}).get('measures', 'n/a')}`",
        f"- prompt digest `{config.get('prompt_digest')}`, "
        f"static prefix {config.get('static_prefix_chars')} chars",
        "",
        "## Time and cost",
        "",
        f"- gameplay {_or_unrecorded(clock.get('elapsed_seconds'), 's')} of a "
        f"{_or_unrecorded(clock.get('limit_seconds'), 's')} limit",
        f"- finalization {_or_unrecorded(_finalization_seconds(result), 's')}, outside gameplay",
        f"- simulated ticks {_or_unrecorded(summary.get('simulated_ticks'))}"
        + (f" (last production sample at tick {last_sample.get('tick')})" if last_sample else ""),
        f"- spend {_or_unrecorded(spend.get('committed_usd'))} of a "
        f"{_or_unrecorded(spend.get('cap_usd'))} cap over "
        f"{_or_unrecorded(spend.get('calls'))} calls",
        f"- cache hit rate {_or_unrecorded(spend.get('cache_hit_rate'))}",
        f"- model latency mean {_or_unrecorded(latency.get('mean'), ' ms')}, "
        f"p95 {_or_unrecorded(latency.get('p95'), ' ms')}",
        "",
        "## What the agent did",
        "",
        f"- {summary.get('decisions') or len(decisions)} decisions, "
        f"{summary.get('tool_actions') or len(events)} tool actions, "
        f"{summary.get('refused_actions') or sum(len(d.get('refused') or []) for d in decisions)} "
        "refused before execution",
        f"- fallbacks {fallbacks} (a fallback is a decision nobody chose)",
        "- actions by verb: "
        + (", ".join(f"{k} {v}" for k, v in sorted(by_key.items())) or "none"),
        "- outcomes: " + (", ".join(f"{k} {v}" for k, v in statuses.items() if k) or "none"),
        "",
        "## Factory progress",
        "",
        "Machine-produced output only. This is a derivation -- the engine counts",
        "every item entering the force's inventory space -- so handcrafting and",
        "mining are subtracted and published separately; see LIMITATIONS.",
        "",
        "- machine-produced: " + (json.dumps(machine) if machine else "nothing"),
        "- handcrafted: " + json.dumps(last_sample.get("handcrafted") or {}),
        "- mined by hand: " + json.dumps(last_sample.get("mined_by_hand") or {}),
        "- machines placed: " + json.dumps(last_sample.get("placed_counts") or {}),
        "- machines working at the end: " + json.dumps(last_sample.get("working_counts") or {}),
        f"- time to first machine output: tick {ticks_to_first}"
        if ticks_to_first is not None
        else "- time to first machine output: never produced any",
        f"- production samples {len(samples)} (wall-clock, independent of the decision rate)",
        "",
        "## First consequential failure",
        "",
        "The earliest failure that went on to recur. A single refusal the agent",
        f"recovered from is not ranked; this is the first one repeated {RECURRENCE}+ times.",
        "",
    ]
    if consequential:
        lines += [
            f"**Decision {consequential['decision']}: `{consequential['key']}`**, "
            f"repeated {consequential['times_repeated']} times.",
            "",
            f"- arguments: `{json.dumps(consequential['arguments'])}`",
            f"- error: {consequential['error']}",
            f"- the model's stated reason: {consequential['reason_given'] or '(none)'}",
            f"- the plan it was pursuing: {consequential['plan_at_the_time'] or '(none stated)'}",
        ]
    else:
        lines.append("None: no failure signature recurred. Fill in what limited the run instead.")

    lines += [
        "",
        "## Plans the agent stated",
        "",
    ]
    plans = summary.get("plans") or stated_plans(decisions)
    lines += (
        [
            f"- decision {p.get('stated_at')}"
            + (f"-{p['held_until']}" if p.get("held_until") else "")
            + (f" `{p['outcome']}`" if p.get("outcome") else "")
            + f" -- {p.get('plan')}"
            for p in plans
        ]
        if plans
        else ["- none: the agent never used the `plan` field."]
    )
    lines += [
        "",
        f"- stalls announced: {summary.get('stalls') or 'none'}",
        f"- interventions: {summary.get('interventions') or 'none'}",
        "",
        "## Artifacts",
        "",
        # Relative to the repository, never absolute. This document is committed
        # to a public repository, and `package_release.audit` refuses a bundle
        # carrying a drive letter for the same reason: an absolute path
        # publishes the author's machine layout and reproduces nothing. The run
        # id above is what actually identifies the run.
        f"- run directory: `{_relative(run_dir)}`",
        f"- replay: `uv run factoriorl replay {run_dir.name}`",
        "- saves: " + (", ".join(saves) or "none"),
        "",
        "## Next three fixes, ranked",
        "",
        "_To fill in from the evidence above. A5.3 asks for these separated by",
        "kind, because the four have different owners and different costs:_",
        "",
        "### Interface and runtime defects",
        "",
        "1. ",
        "",
        "### Knowledge the agent could not reach",
        "",
        "1. ",
        "",
        "### Agent decisions",
        "",
        "1. ",
        "",
        "### Time and cost limits",
        "",
        "1. ",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="run id or run directory")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = _resolve(args.run)
    text = render(run_dir)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
