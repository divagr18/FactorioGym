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


def render(run_dir: Path) -> str:
    summary = _load(run_dir / "summary.json")
    result = _load(run_dir / "result.json")
    config = _load(run_dir / "config.json")
    decisions = _lines(run_dir / "decisions.jsonl")
    events = _lines(run_dir / "tool_events.jsonl")
    samples = _lines(run_dir / "production.jsonl")
    saves = sorted(p.name for p in (run_dir / "saves").glob("*.zip"))

    limits = result.get("limits") or {}
    spend = limits.get("spend") or {}
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
    by_key = summary.get("tool_actions_by_key") or {}
    statuses = Counter(row.get("status") for row in events)
    consequential = first_consequential_failure(decisions)

    last_sample = samples[-1] if samples else {}
    lines = [
        f"# A5 first run -- {run_dir.name}",
        "",
        "Exploratory result, not a benchmark success claim. An open world has no",
        "success predicate, no layout families and no reward components, so nothing",
        "measured here is comparable to a benchmark task.",
        "",
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
        f"- gameplay {clock.get('elapsed_seconds')}s of a {clock.get('limit_seconds')}s limit",
        f"- finalization {(result.get('finalization') or {}).get('seconds')}s, outside gameplay",
        f"- simulated ticks {summary.get('simulated_ticks')}",
        f"- spend ${spend.get('committed_usd')} of a ${spend.get('cap_usd')} cap "
        f"over {spend.get('calls')} calls",
        f"- cache hit rate {spend.get('cache_hit_rate')} "
        f"({spend.get('cache_hit_tokens')} hit / {spend.get('cache_miss_tokens')} miss)",
        f"- model latency mean {(result.get('latency_ms') or {}).get('mean')} ms, "
        f"p95 {(result.get('latency_ms') or {}).get('p95')} ms",
        "",
        "## What the agent did",
        "",
        f"- {summary.get('decisions') or len(decisions)} decisions, "
        f"{summary.get('tool_actions') or len(events)} tool actions, "
        f"{summary.get('refused_actions') or sum(len(d.get('refused') or []) for d in decisions)} "
        "refused before execution",
        f"- fallbacks {result.get('fallback_decisions')} (a fallback is a decision nobody chose)",
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
        f"- time to first machine output: {production.get('ticks_to_first_output')}",
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
    plans = summary.get("plans") or []
    lines += (
        [f"- `{p.get('outcome')}` -- {p.get('plan')}" for p in plans]
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
        f"- run directory: `{run_dir}`",
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
