"""Reward accounting (PLAN.md 3.5).

Computed in **Python**, over monotone counters the Lua side maintains. Lua stays
authoritative for world state; Python owns scoring. That split buys three
things: shaping weights change without a mod rebuild and worker relaunch,
evaluator truth stays structurally separate from policy observation, and the
exploit tests below run without an engine -- which is the only way they will
actually run.

``total`` is ``sum(components.values())`` **by construction**, so PLAN's
"reward components sum to the returned reward" is not something a test has to
catch.

Two component kinds make the anti-exploit criteria structurally true rather
than merely tested:

* **potential-based** shaping is ``gamma*phi(s') - phi(s)``, so any closed loop
  in state space sums to zero -- dismantle/rebuild farming pays nothing;
* **high-water** shaping rewards the increase of a maximum, never the level, so
  moving items out of the goal and back cannot pay twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from factoriorl.tasks.spec import Predicate, PredicateKind, RewardComponent, RewardKind

GAMMA = 0.99


def _measure(predicate: Predicate | None, observation: dict, truth: dict) -> float:
    """The scalar a shaping component tracks."""
    if predicate is None:
        return 0.0
    if predicate.kind is PredicateKind.CONTAINER_HOLDS:
        return float(
            (truth.get("containers") or {}).get(predicate.marker, {}).get(predicate.item, 0)
        )
    if predicate.kind is PredicateKind.INVENTORY_HOLDS:
        return float((observation.get("inventory") or {}).get(predicate.item, 0))
    if predicate.kind is PredicateKind.PRODUCED:
        return float((truth.get("produced") or {}).get(predicate.item, 0))
    if predicate.kind is PredicateKind.CHARACTER_WITHIN:
        target = (truth.get("markers") or {}).get(predicate.marker)
        position = (observation.get("character") or {}).get("position")
        if not target or not position:
            return 0.0
        distance = ((position[0] - target[0]) ** 2 + (position[1] - target[1]) ** 2) ** 0.5
        # Closer is better, bounded, so the potential is well defined.
        return float(max(0.0, 1.0 - distance / 64.0))
    if predicate.kind is PredicateKind.ENTITY_WORKING:
        return 1.0 if (truth.get("working") or {}).get(predicate.marker) else 0.0
    return 0.0


@dataclass
class RewardAccountant:
    """Per-episode reward state. Reset with the episode, never across one."""

    components: tuple[RewardComponent, ...]
    shaping_enabled: bool = True
    _high_water: dict[str, float] = field(default_factory=dict)
    _potential: dict[str, float] = field(default_factory=dict)
    #: Cumulative payout per capped component, reset with the episode.
    _paid: dict[str, float] = field(default_factory=dict)

    def reset(self, observation: dict, truth: dict) -> None:
        self._high_water = {}
        self._potential = {}
        self._paid = {}
        for component in self.components:
            value = _measure(component.predicate, observation, truth)
            if component.kind is RewardKind.HIGH_WATER:
                self._high_water[component.name] = value
            elif component.kind is RewardKind.POTENTIAL:
                self._potential[component.name] = value

    def step(
        self,
        observation: dict,
        truth: dict,
        succeeded: bool,
        terminated: bool = False,
    ) -> dict[str, float]:
        """Reward components for one transition. Keys are stable per task."""
        out: dict[str, float] = {}
        for component in self.components:
            if component.kind is RewardKind.SPARSE_SUCCESS:
                out[component.name] = component.weight if succeeded else 0.0
                continue
            if not self.shaping_enabled:
                out[component.name] = 0.0
                continue
            if component.kind is RewardKind.STEP_COST:
                out[component.name] = -abs(component.weight)
                continue

            value = _measure(component.predicate, observation, truth)
            if component.kind is RewardKind.HIGH_WATER:
                previous = self._high_water.get(component.name, value)
                gain = max(0.0, value - previous)
                self._high_water[component.name] = max(previous, value)
                payout = component.weight * gain * component.scale
                if component.cap is not None:
                    # Bound the cumulative payout, not the per-step one: the
                    # exploit is earning the cap many times over across an
                    # episode, not earning a lot once.
                    paid = self._paid.get(component.name, 0.0)
                    payout = max(0.0, min(payout, component.cap - paid))
                    self._paid[component.name] = paid + payout
                out[component.name] = payout
            elif component.kind is RewardKind.POTENTIAL:
                previous = self._potential.get(component.name, value)
                self._potential[component.name] = value
                # Phi(terminal) must be 0, or the shaping does not telescope to
                # a policy-independent -w*Phi(s0) and the invariance guarantee
                # is void. Termination is entry to an absorbing state whose
                # value is zero by definition; truncation is NOT termination --
                # the episode is cut, the state still has value -- so only
                # `terminated` zeroes it. Left unbranched, this paid a second
                # success bonus on `navigate` (Phi(s_T) ~ 0.97 against a sparse
                # weight of 1.0), and would pay for failing *near* the goal
                # once failure predicates exist.
                phi_next = 0.0 if terminated else value
                out[component.name] = component.weight * (GAMMA * phi_next - previous)
            else:
                out[component.name] = 0.0
        return out

    @staticmethod
    def total(components: dict[str, float]) -> float:
        # By construction, not by assertion.
        return float(sum(components.values()))
