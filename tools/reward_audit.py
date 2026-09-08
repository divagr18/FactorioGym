"""Can a policy get paid for stopping? (PLAN.md 3.3, 4.4)

`deliver` seed 3 converged on `take_iron-plate_20` and never delivered: it ran
95.9 steps of a 120 budget for +0.062 and scored 0.00 on every evaluation row.
That is not an optimiser failure. It is the reward's arithmetic working exactly
as written, and the arithmetic is checkable without an engine.

The invariant
-------------
Shaping is supposed to point at the goal. It stops doing that when a component
can be *maximised in a state where the success predicate is false*, because the
agent can then bank the payout and idle. The only thing left opposing that is
the step cost, so the test is a comparison of two numbers:

    plateau  = the shaping still earnable while success is false
    pressure = step-cost weight x the decision budget

When ``plateau >= pressure`` the episode has a positive-return absorbing
strategy that is not success, and the budget has stopped being pressure. On
`deliver` the plateau is 0.35 (0.15 for holding plates, 0.20 for putting four of
the required twenty in the destination) against a pressure of 0.12 -- so
reaching the plateau and standing still returns **+0.23**, and a seed that finds
it first has nothing pulling it out.

Two components are treated specially:

* **Potential-based** shaping is exempt. It telescopes to a policy-independent
  constant (Ng, Harada and Russell), so it cannot create a plateau -- provided
  Phi(terminal) is zero, which `rewards.py` now enforces and did not always.
* A component whose predicate is *implied* by success is not a plateau: earning
  its cap means the episode is already won. `repair_belt` pays for a plate in
  the sink and succeeds on one plate in the sink, so its cap is unreachable
  without success and it is correctly exempt.

The other half of the invariant
------------------------------
A plateau test alone is only one side, and `repair_belt` is the point that makes
that obvious: it satisfies the plateau rule *perfectly* -- its shaping predicate
**is** its success predicate, so the cap is unreachable while the task is
unfinished -- and it is therefore completely unlearnable. Its only shaping pays
on a plate reaching the sink, which is the win condition, and success needs
exactly one. So nothing pays before success and the reward is a constant
negative drip.

Measured: 190 training episodes, **zero successes**, mean episode reward -0.300
against a step cost of exactly 300 x 0.001. The policy had nothing to ascend,
and more steps cannot fix that -- 190 episodes of zero signal and 1,900 episodes
of zero signal are the same thing to a policy gradient.

So a reward needs both properties, and they pull against each other:

* **non-exploitable** -- the shaping reachable short of the goal must cost less
  than exhausting the budget, or idling on it beats continuing;
* **informative** -- something must be payable *before* success, or there is no
  gradient at all.

A component is informative when its quantity can be non-zero while success is
still false: either it measures something success does not (deliver's `carried`),
or it measures the success quantity with a threshold above one, so partial
progress pays (plate_line's thirty plates). A component measuring the success
quantity at a threshold of one can only ever pay for winning.

Run: uv run python tools/reward_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factoriorl.rewards import RewardKind  # noqa: E402
from factoriorl.tasks import all_tasks, get  # noqa: E402


def units_to_cap(component) -> float | None:
    """How many units of the predicate's quantity the cap pays for."""
    if component.cap is None or not component.weight:
        return None
    return component.cap / abs(component.weight)


def reachable_without_success(component, successes) -> tuple[bool, str]:
    """Can this component's cap be earned while the task is still unfinished?"""
    predicate = component.predicate
    if predicate is None:
        return False, "no predicate"
    for success in successes:
        same_target = (
            predicate.kind == success.kind
            and predicate.marker == getattr(success, "marker", None)
            and getattr(predicate, "item", None) == getattr(success, "item", None)
        )
        if not same_target:
            continue
        needed = getattr(success, "at_least", None)
        units = units_to_cap(component)
        if needed is None or units is None:
            return False, "measures the success quantity itself"
        if units < needed:
            return True, f"cap reached at {units:g} of the {needed:g} success requires"
        return False, f"cap needs {units:g}, at or past the {needed:g} for success"
    return True, "measures something success does not require"


def informative_components(spec) -> list[str]:
    """Shaping that can pay while the success predicate is still false."""
    thresholds = {}
    for predicate in spec.success:
        key = (predicate.kind, predicate.marker, getattr(predicate, "item", None))
        thresholds[key] = getattr(predicate, "at_least", None)

    found: list[str] = []
    for component in spec.rewards:
        if component.kind in (RewardKind.SPARSE_SUCCESS, RewardKind.STEP_COST):
            continue
        if not component.shaping:
            continue
        if component.kind is RewardKind.POTENTIAL:
            # A potential function is dense by construction: it is a distance,
            # and it changes on every step that changes the state.
            found.append(f"{component.name} (potential)")
            continue
        predicate = component.predicate
        if predicate is None:
            continue
        key = (predicate.kind, predicate.marker, getattr(predicate, "item", None))
        if key not in thresholds:
            found.append(f"{component.name} (measures a quantity success does not)")
        elif (thresholds[key] or 0) > 1:
            found.append(f"{component.name} (partial credit up to {thresholds[key]:g})")
    return found


def audit(task_id: str) -> dict:
    spec = get(task_id).spec
    step_weight = 0.0
    for component in spec.rewards:
        if component.kind is RewardKind.STEP_COST:
            step_weight = abs(component.weight)
    pressure = round(step_weight * spec.max_decision_steps, 4)

    plateau = 0.0
    contributors, exempt = [], []
    for component in spec.rewards:
        if component.kind in (RewardKind.SPARSE_SUCCESS, RewardKind.STEP_COST):
            continue
        if not component.shaping:
            continue
        if component.kind is RewardKind.POTENTIAL:
            exempt.append({"name": component.name, "why": "potential-based; telescopes"})
            continue
        if component.cap is None:
            # Uncapped high-water is unbounded payout for a non-goal state.
            contributors.append({"name": component.name, "cap": None, "why": "uncapped"})
            continue
        earnable, why = reachable_without_success(component, spec.success)
        if earnable:
            plateau += component.cap
            contributors.append({"name": component.name, "cap": component.cap, "why": why})
        else:
            exempt.append({"name": component.name, "why": why})

    plateau = round(plateau, 4)
    informative = informative_components(spec)
    return {
        "task": task_id,
        "informative_components": informative,
        # No component payable before success means no gradient at all, whatever
        # the training budget.
        "has_gradient": bool(informative),
        "version": spec.version,
        "budget": spec.max_decision_steps,
        "step_cost_weight": step_weight,
        "pressure": pressure,
        "plateau": plateau,
        "idle_return": round(plateau - pressure, 4),
        "violates": plateau >= pressure and plateau > 0,
        "margin": round(plateau / pressure, 2) if pressure else None,
        "contributors": contributors,
        "exempt": exempt,
    }


def main() -> int:
    reports = [audit(task_id) for task_id in sorted(all_tasks())]
    violations = [r for r in reports if r["violates"]]
    starved = [r for r in reports if not r["has_gradient"]]

    for report in reports:
        flag = "TRAP" if report["violates"] else "ok  "
        print(
            f"{flag} {report['task']:16s} plateau={report['plateau']:.2f} "
            f"pressure={report['pressure']:.2f} "
            f"idle_return={report['idle_return']:+.2f} "
            f"budget={report['budget']}"
        )
        for entry in report["contributors"]:
            print(f"        + {entry['name']:16s} cap={entry['cap']}  {entry['why']}")
        if not report["has_gradient"]:
            print("        ! nothing pays before success: this reward has no gradient")

    out = ROOT / "docs" / "evidence" / "reward-audit.json"
    out.write_text(
        json.dumps(
            {
                "what": (
                    "Shaping earnable while the success predicate is false, against the "
                    "step cost of exhausting the budget. A task where the first exceeds "
                    "the second has a positive-return strategy that is not success."
                ),
                "reports": reports,
                "violating": [r["task"] for r in violations],
                "no_gradient": [r["task"] for r in starved],
                "no_gradient_note": (
                    "Nothing in these rewards pays before the success predicate fires, so a "
                    "policy has nothing to ascend and the training budget is irrelevant. "
                    "repair_belt: 190 episodes, zero successes, mean reward -0.300 against a "
                    "step cost of exactly 300 x 0.001."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if violations:
        print(
            f"\n{len(violations)} of {len(reports)} families pay for stopping: "
            f"{[r['task'] for r in violations]}"
        )
    print(f"wrote {out.relative_to(ROOT)}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
