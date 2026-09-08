"""Engine-free coverage for the task engine, catalogs, rewards and seeding.

These are the tests that must run on every commit. Anything that needs a live
Factorio worker runs only in the gates, so the properties most likely to rot --
reward exploits, catalog indices, seed disjointness, import direction -- are
deliberately checkable without one.
"""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path

import pytest

from factoriorl import catalog as catalog_module
from factoriorl.rewards import GAMMA, RewardAccountant
from factoriorl.seeding import Branch, SeedPlan
from factoriorl.tasks import all_tasks, get, validate_all
from factoriorl.tasks.spec import (
    Predicate,
    PredicateKind,
    RewardComponent,
    RewardKind,
)

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("navigate", "deliver", "mine_smelt", "supply_furnace", "repair_belt", "restore_power")


# ------------------------------------------------------------------ registry


def test_the_six_introductory_families_are_registered():
    registered = all_tasks()
    for name in FAMILIES:
        assert name in registered, f"missing introductory family: {name}"


def test_every_task_validates():
    report = validate_all()
    assert report["ok"], report


@pytest.mark.parametrize("task_id", FAMILIES)
def test_each_task_declares_a_held_out_layout_family(task_id):
    """PLAN.md section 3: structural holdouts, not merely unseen seeds."""
    spec = get(task_id).spec
    assert spec.families("train"), f"{task_id} has no training family"
    assert spec.families("test"), f"{task_id} has no held-out family"
    train = {f.name for f in spec.families("train")}
    test = {f.name for f in spec.families("test")}
    assert not (train & test), f"{task_id} reuses a training family as a holdout"


@pytest.mark.parametrize("task_id", FAMILIES)
def test_generators_are_deterministic_in_their_seed(task_id):
    task = get(task_id)
    family = task.spec.layout_families[0]
    first = task.generate(family, random.Random(1234)).to_dict()
    second = task.generate(family, random.Random(1234)).to_dict()
    assert first == second, f"{task_id} generator is not deterministic in its seed"


@pytest.mark.parametrize("task_id", FAMILIES)
def test_generated_scenes_are_structurally_sound(task_id):
    task = get(task_id)
    for family in task.spec.layout_families:
        for index in range(25):
            blueprint = task.generate(family, random.Random(index))
            assert not blueprint.out_of_box(), blueprint.out_of_box()
            assert not blueprint.footprint_conflicts(), blueprint.footprint_conflicts()


def test_tasks_never_import_worker_management():
    """ "A task can be added without editing worker management", as a test.

    Import direction is the mechanical form of that claim: if the task package
    reached into worker/pool/supervisor, adding a task could require changing
    them.
    """
    forbidden = {
        "factoriorl.worker",
        "factoriorl.worker_config",
        "factoriorl.pool",
        "factoriorl.supervisor",
        "factoriorl.rcon",
        "factoriorl.session",
    }
    root = Path(__file__).resolve().parents[2] / "src" / "factoriorl" / "tasks"
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in forbidden:
                offenders.append(f"{path.name} imports {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden:
                        offenders.append(f"{path.name} imports {alias.name}")
    assert not offenders, offenders


# ------------------------------------------------------------------- catalog


def test_catalog_always_offers_a_no_op():
    resolved = catalog_module.resolve("primitive-v1")
    assert resolved.templates[resolved.wait_index].action == "wait"


def test_task_subsets_are_indexed_by_catalog_order_not_declaration_order():
    """Reproducibility depends on this: a task listing actions in a different
    order must not shift the meaning of an action index."""
    full = catalog_module.resolve("primitive-v1")
    forward = catalog_module.resolve("primitive-v1", ("move_east", "move_north", "wait"))
    backward = catalog_module.resolve("primitive-v1", ("wait", "move_north", "move_east"))
    assert forward.keys() == backward.keys()
    order = [k for k in full.keys() if k in {"move_east", "move_north", "wait"}]
    assert list(forward.keys()) == order


def test_a_subset_without_wait_still_gets_one():
    resolved = catalog_module.resolve("primitive-v1", ("move_east",))
    assert any(t.action == "wait" for t in resolved.templates)


def test_catalog_digest_changes_only_with_the_catalog():
    a = catalog_module.resolve("primitive-v1", ("move_east", "wait"))
    b = catalog_module.resolve("primitive-v1", ("move_east", "wait"))
    c = catalog_module.resolve("primitive-v1", ("move_west", "wait"))
    assert a.digest() == b.digest()
    assert a.digest() != c.digest()


def test_unknown_template_is_rejected_loudly():
    with pytest.raises(ValueError, match="no templates"):
        catalog_module.resolve("primitive-v1", ("not_a_real_action",))


# ------------------------------------------------------------------- rewards


def _accountant(component: RewardComponent) -> RewardAccountant:
    sparse = RewardComponent("success", RewardKind.SPARSE_SUCCESS, 1.0, shaping=False)
    accountant = RewardAccountant((sparse, component))
    accountant.reset({}, {})
    return accountant


def test_components_sum_to_the_total_by_construction():
    components = {"a": 0.5, "b": -0.25, "c": 1.0}
    assert RewardAccountant.total(components) == pytest.approx(1.25)


def test_transfer_loop_cannot_be_farmed():
    """Items out of the goal and back in must not pay twice.

    High-water shaping rewards the increase of a maximum, never the level, so
    the loop nets exactly what one delivery was worth.
    """
    component = RewardComponent(
        "delivered",
        RewardKind.HIGH_WATER,
        weight=1.0,
        predicate=Predicate(PredicateKind.CONTAINER_HOLDS, marker="dst", item="iron-plate"),
    )
    accountant = _accountant(component)
    total = 0.0
    for count in (0, 10, 0, 10, 0, 10):
        truth = {"containers": {"dst": {"iron-plate": count}}}
        total += RewardAccountant.total(accountant.step({}, truth, False))
    assert total == pytest.approx(10.0), "the loop paid more than one delivery"


def test_dismantle_rebuild_loop_is_never_profitable():
    """Potential-based shaping cannot be farmed by cycling through states.

    Around a closed loop the shaping sums to ``(gamma - 1) * sum(phi)``, which
    is *negative*, not zero -- the discount makes a round trip cost a little
    rather than pay. Asserting "close to zero" would be the wrong property; the
    one that matters is that the loop is never profitable, and that the leak is
    bounded by the discount rather than growing with the number of laps.
    """
    component = RewardComponent(
        "progress",
        RewardKind.POTENTIAL,
        weight=1.0,
        predicate=Predicate(PredicateKind.INVENTORY_HOLDS, item="iron-plate"),
    )
    accountant = _accountant(component)
    accountant.reset({"inventory": {"iron-plate": 5}}, {})
    counts = (5, 9, 5, 9, 5)
    total = 0.0
    for count in counts:
        total += RewardAccountant.total(
            accountant.step({"inventory": {"iron-plate": count}}, {}, False)
        )
    assert total <= 1e-9, f"a closed loop paid {total}, so it can be farmed"
    assert abs(total) <= (1 - GAMMA) * sum(counts) + 1e-9, (
        "the loss around a loop should be exactly the discount on the potentials visited"
    )

    # And laps do not compound into a payout: more laps only cost more.
    longer = _accountant(component)
    longer.reset({"inventory": {"iron-plate": 5}}, {})
    long_total = 0.0
    for count in counts * 4:
        long_total += RewardAccountant.total(
            longer.step({"inventory": {"iron-plate": count}}, {}, False)
        )
    assert long_total <= total + 1e-9


def test_repeated_observation_earns_nothing():
    component = RewardComponent(
        "delivered",
        RewardKind.HIGH_WATER,
        weight=1.0,
        predicate=Predicate(PredicateKind.CONTAINER_HOLDS, marker="dst", item="iron-plate"),
    )
    accountant = _accountant(component)
    truth = {"containers": {"dst": {"iron-plate": 7}}}
    first = RewardAccountant.total(accountant.step({}, truth, False))
    rest = sum(RewardAccountant.total(accountant.step({}, truth, False)) for _ in range(10))
    assert first == pytest.approx(7.0)
    assert rest == pytest.approx(0.0), "observing the same state again paid out"


def test_disabling_shaping_leaves_success_untouched():
    component = RewardComponent(
        "delivered",
        RewardKind.HIGH_WATER,
        weight=1.0,
        predicate=Predicate(PredicateKind.CONTAINER_HOLDS, marker="dst", item="iron-plate"),
    )
    sparse = RewardComponent("success", RewardKind.SPARSE_SUCCESS, 1.0, shaping=False)
    off = RewardAccountant((sparse, component), shaping_enabled=False)
    off.reset({}, {})
    truth = {"containers": {"dst": {"iron-plate": 50}}}
    components = off.step({}, truth, succeeded=True)
    assert components["success"] == pytest.approx(1.0)
    assert components["delivered"] == pytest.approx(0.0)
    assert RewardAccountant.total(components) == pytest.approx(1.0)


# ------------------------------------------------------------------- seeding


def test_train_and_eval_seed_streams_are_disjoint():
    """An evaluation episode must never be one the policy trained on."""
    plan = SeedPlan(master=99, run_id="r1")
    train = {plan.episode_seed(Branch.TRAIN, i) for i in range(2000)}
    evaluation = {plan.episode_seed(Branch.EVAL, i) for i in range(2000)}
    assert not (train & evaluation)


def test_episode_seeds_depend_on_the_index_not_the_order():
    plan = SeedPlan(master=7, run_id="r2")
    assert plan.episode_seed(Branch.TRAIN, 731) == plan.episode_seed(Branch.TRAIN, 731)
    assert plan.episode_seed(Branch.TRAIN, 731) != plan.episode_seed(Branch.TRAIN, 732)


def test_different_runs_do_not_share_episode_seeds():
    a = SeedPlan(master=7, run_id="run-a")
    b = SeedPlan(master=7, run_id="run-b")
    assert a.episode_seed(Branch.TRAIN, 0) != b.episode_seed(Branch.TRAIN, 0)


# ---------------------------------------------------- potential-based shaping


def _potential_task_components():
    return (
        RewardComponent(
            name="approach",
            kind=RewardKind.POTENTIAL,
            weight=1.0,
            predicate=Predicate(PredicateKind.NEAR_MARKER, marker="goal", threshold=30.0),
        ),
    )


def _phi_run(accountant, values, terminate_last):
    """Score a trajectory whose potential takes `values`, returning the
    discounted sum of the shaping term."""
    total, discount = 0.0, 1.0
    for index, value in enumerate(values):
        last = index == len(values) - 1
        observation = {"character": {"position": [value, 0.0]}, "entities": []}
        truth = {"markers": {"goal": [0.0, 0.0]}}
        components = accountant.step(
            observation, truth, succeeded=False, terminated=terminate_last and last
        )
        total += discount * components["approach"]
        discount *= GAMMA
    return total


def test_potential_shaping_telescopes_to_a_policy_independent_constant():
    """Ng-Harada-Russell invariance: the discounted shaping over an episode must
    equal -w*Phi(s0) regardless of the path taken. That only holds if Phi is
    zeroed at a terminal state -- without the branch, the sum keeps a
    path-dependent gamma^T*Phi(s_T) term and the guarantee is void."""
    task = get("navigate")
    components = [c for c in task.spec.rewards if c.kind is RewardKind.POTENTIAL]
    if not components:
        pytest.skip("navigate declares no potential component")

    start = {"character": {"position": [10.0, 0.0]}, "entities": []}
    truth = {"markers": {"goal": [0.0, 0.0]}}

    sums = []
    for path in ([8.0, 5.0, 1.0], [9.0, 9.0, 1.0]):
        accountant = RewardAccountant(task.spec.rewards, shaping_enabled=True)
        accountant.reset(start, truth)
        phi0 = accountant._potential[components[0].name]
        sums.append(round(_phi_run(accountant, path, terminate_last=True), 9))
        expected = round(-components[0].weight * phi0, 9)
    assert sums[0] == sums[1], "shaping depends on the path taken"
    assert abs(sums[0] - expected) < 1e-9, f"expected {expected}, got {sums[0]}"


@pytest.mark.parametrize("task_id", FAMILIES)
def test_every_high_water_component_is_capped_below_success(task_id):
    """PLAN section 3: no shaping component's achievable total may reach the
    sparse success weight. `mine_smelt` paid 0.2 per mined ore against a
    success weight of 1.0, so a policy that mined and never smelted could
    out-earn finishing the task several times over."""
    spec = get(task_id).spec
    sparse = sum(c.weight for c in spec.rewards if c.kind is RewardKind.SPARSE_SUCCESS)
    assert sparse > 0, f"{task_id} declares no sparse success reward"
    for component in spec.rewards:
        if component.kind is not RewardKind.HIGH_WATER:
            continue
        assert component.cap is not None, f"{task_id}.{component.name} is uncapped"
        assert component.cap < sparse, (
            f"{task_id}.{component.name} can pay {component.cap} against success {sparse}"
        )


def test_the_cap_actually_bounds_the_cumulative_payout():
    """A cap on the per-step payout would not help: the exploit is earning it
    many times over across an episode."""
    component = RewardComponent(
        name="carried",
        kind=RewardKind.HIGH_WATER,
        weight=1.0,
        cap=0.25,
        predicate=Predicate(PredicateKind.INVENTORY_HOLDS, item="iron-ore"),
    )
    accountant = RewardAccountant((component,), shaping_enabled=True)
    empty = {"inventory": {}, "character": {}, "entities": []}
    accountant.reset(empty, {})
    total = 0.0
    for held in range(1, 21):
        observation = {"inventory": {"iron-ore": held}, "character": {}, "entities": []}
        total += accountant.step(observation, {}, succeeded=False)["carried"]
    assert total == pytest.approx(0.25), f"cumulative payout {total} exceeded the cap"


# ------------------------------------------------------------------ landmarks


@pytest.mark.parametrize("task_id", FAMILIES)
def test_landmarks_do_not_read_truth(task_id):
    """Landmarks enter the goal vector, which is a policy input, so they must be
    computable from the declared observation alone. A landmark that reads
    evaluator truth would be privileged information -- the same violation the
    action-mask purity test exists to prevent."""
    spec = get(task_id).spec
    observation = {
        "character": {"position": [3.0, 4.0]},
        "inventory": {"iron-ore": 4, "iron-plate": 1},
        "entities": [{"name": "stone-furnace", "p": [5.0, 5.0], "h": "e1"}],
        "resources": {"tiles": []},
        "tick": 120,
    }
    truth = {
        "markers": {"goal": [10.0, 10.0], "dst": [8.0, 8.0], "furnace": [5.0, 5.0]},
        "containers": {"dst": {"iron-plate": 99}, "sink": {"iron-plate": 99}},
        "produced": {"iron-plate": 99},
        "working": {"drill": True, "generator": True},
    }
    for predicate in spec.landmarks:
        with_truth = predicate.evaluate(observation, truth)
        without_truth = predicate.evaluate(observation, {})
        assert with_truth == without_truth, (
            f"{task_id} landmark {predicate.kind.value} changes with evaluator truth"
        )


@pytest.mark.parametrize("task_id", FAMILIES)
def test_goal_vector_slots_fit(task_id):
    """Success predicates plus landmarks must fit the goal vector beside the
    budget fraction, or declaring a landmark would silently drop another."""
    from factoriorl import encoders

    spec = get(task_id).spec
    assert 1 + len(spec.success) + len(spec.landmarks) <= encoders.GOAL_FEATURES


# --------------------------------------------------- decision interval invariant


@pytest.mark.parametrize("task_id", FAMILIES)
def test_the_decision_interval_outlasts_the_longest_move(task_id):
    """A move is timed by the mod, not by the decision loop.

    ``actions.lua`` sets ``deadline_tick = game.tick + payload.ticks`` and
    ``inflight.lua`` stops the walk there, so a stride longer than the decision
    interval is still running when the next action arrives. Only another move
    supersedes a running move, so the next ``place_*``/``transfer``/``craft_*``
    executes mid-walk -- and placement binds to ``floor(position)``, which puts
    the entity on whatever tile the walk happened to reach. Nothing raises; the
    episode is simply wrong. The margin here is currently zero (30 == 30), so
    any change to either constant has to trip this test.
    """
    from factoriorl.tasks import MAX_ADVANCE_TICKS

    spec = get(task_id).spec
    assert spec.decision_ticks >= catalog_module.LONG_MOVE_TICKS, (
        f"{task_id} decides every {spec.decision_ticks} ticks but the catalog's longest "
        f"stride runs for {catalog_module.LONG_MOVE_TICKS}"
    )
    assert 1 <= spec.decision_ticks <= MAX_ADVANCE_TICKS, (
        f"{task_id} decision_ticks {spec.decision_ticks} is outside what the mod will advance"
    )


@pytest.fixture
def temporary_task():
    """Register a variant of an existing task, then take it back out.

    Validation reads the registry, so the only way to test that it rejects a
    misconfigured task is to put one there. Removal happens even on failure --
    a leaked entry would fail every later test that validates the registry.
    """
    from dataclasses import replace

    from factoriorl import tasks as tasks_module

    added: list[str] = []

    def _add(task_id: str, **changes) -> str:
        base = get("navigate")
        spec = replace(base.spec, id=task_id, **changes)
        tasks_module._REGISTRY[task_id] = tasks_module.RegisteredTask(
            spec=spec, generate=base.generate
        )
        added.append(task_id)
        return task_id

    yield _add

    from factoriorl import tasks as tasks_module_cleanup

    for task_id in added:
        tasks_module_cleanup._REGISTRY.pop(task_id, None)


@pytest.mark.parametrize(
    ("decision_ticks", "expected"),
    [
        (catalog_module.LONG_MOVE_TICKS - 1, "still running"),
        (1, "still running"),
        (0, "must be in"),
        (-30, "must be in"),
        (36001, "must be in"),
    ],
)
def test_a_bad_decision_interval_fails_validation(temporary_task, decision_ticks, expected):
    """`tasks validate` runs before any worker launches, which is the whole
    point: an interval shorter than a stride corrupts episodes silently at
    runtime, and an out-of-range one is only rejected by the mod mid-episode."""
    bad = temporary_task("synthetic_bad_interval", decision_ticks=decision_ticks)
    control = temporary_task("synthetic_good_interval")

    report = validate_all(sample_seeds=1)

    problems = report["tasks"][bad]["problems"]
    assert not report["tasks"][bad]["ok"], "a violating task validated clean"
    assert any("decision_ticks" in p for p in problems), problems
    assert any(expected in p for p in problems), problems
    assert not report["ok"], "validate_all reported ok with a violating task registered"
    # The check indicts the interval, not tasks in general.
    assert report["tasks"][control]["ok"], report["tasks"][control]["problems"]


# ------------------------------------------- what the shaping flag may change


def _unshaped_accountant(*components):
    """An accountant with shaping switched off, for the 4.4 ablation checks."""
    accountant = RewardAccountant(components, shaping_enabled=False)
    accountant.reset({"inventory": {}, "character": {}, "entities": []}, {})
    return accountant


def test_step_cost_survives_disabling_shaping():
    """PLAN 4.4 separates 'shaping made learning faster' from 'shaping changed
    the problem'. It cannot do that if one flag moves several variables, and
    step cost is not shaping: it is the task's requirement that the goal be
    reached promptly, present in the sparse formulation too."""
    component = RewardComponent(name="step_cost", kind=RewardKind.STEP_COST, weight=0.001)
    out = _unshaped_accountant(component).step({}, {}, succeeded=False)
    assert out["step_cost"] == pytest.approx(-0.001)


def test_disabling_shaping_zeroes_shaping_components():
    """The other half: everything that *is* shaping must go."""
    component = RewardComponent(
        name="carried",
        kind=RewardKind.HIGH_WATER,
        weight=1.0,
        cap=1.0,
        predicate=Predicate(PredicateKind.INVENTORY_HOLDS, item="iron-ore"),
    )
    observation = {"inventory": {"iron-ore": 5}, "character": {}, "entities": []}
    assert _unshaped_accountant(component).step(observation, {}, succeeded=False)["carried"] == 0.0


def test_truncation_does_not_zero_the_potential():
    """Only termination zeroes Phi. Truncation cuts the episode without making
    the state worthless, and the learner bootstraps the shaped value there --
    zeroing it would inject a spurious terminal penalty into every episode that
    merely ran out of budget."""
    task = get("navigate")
    components = [c for c in task.spec.rewards if c.kind is RewardKind.POTENTIAL]
    if not components:
        pytest.skip("navigate declares no potential component")
    name = components[0].name
    truth = {"markers": {"goal": [0.0, 0.0]}}
    start = {"character": {"position": [10.0, 0.0]}, "entities": []}
    near = {"character": {"position": [1.0, 0.0]}, "entities": []}

    cut = RewardAccountant(task.spec.rewards, shaping_enabled=True)
    cut.reset(start, truth)
    truncated_value = cut.step(near, truth, succeeded=False, terminated=False)[name]

    ended = RewardAccountant(task.spec.rewards, shaping_enabled=True)
    ended.reset(start, truth)
    terminated_value = ended.step(near, truth, succeeded=False, terminated=True)[name]

    assert truncated_value != terminated_value, "terminated and truncated scored identically"
    assert terminated_value < truncated_value, "terminating should forfeit the remaining potential"


def test_the_baseline_cache_key_separates_action_spaces():
    """A random policy over skills is a different agent from a random policy
    over primitives, so they must not share a cached floor.

    They did. In the Phase 4b ablation the flat arm ran first and cached a
    floor of 0.12 measured over primitive actions; the skill arms then reported
    that number as their own, and `deliver` was published as 1.00 against 0.12
    when its true floor with skills is nearer 0.80. Skills are added by a
    wrapper rather than by the catalog, so the catalog digest -- which the key
    already contained -- could not tell the two apart.
    """
    from factoriorl.baselines import cache_key

    task = get("deliver")
    flat = cache_key(task, "test", 11, 25, "primitive-v1")
    skills = cache_key(task, "test", 11, 25, "skills-v1")
    assert flat != skills, "a skills floor would be served from the primitive cache entry"


# ------------------------------------------------- paying for stopping


def test_shaping_plateaus_are_pinned_to_the_families_that_have_them():
    """A task must not pay more for stopping than the budget charges for it.

    Shaping earnable while the success predicate is still false is a plateau: an
    agent banks it and idles, and the only thing opposing that is the step cost.
    When the plateau exceeds the step cost of exhausting the budget, the episode
    has a positive-return absorbing strategy that is not success.

    `deliver` seed 3 is the worked example. It converged on `take_iron-plate_20`,
    ran 95.9 steps of a 120 budget for +0.062 and scored 0.00 on all three
    evaluation rows -- the reward's arithmetic working as written.

    This pins the set rather than asserting it is empty, because emptying it
    means changing reward weights, which bumps every task version and invalidates
    a frozen holdout validated against the current ones. A new violation fails
    here; fixing one also fails here, deliberately, so the pin is updated by
    someone who meant to.
    """
    import subprocess
    import sys as _sys

    result = subprocess.run(
        [_sys.executable, str(ROOT / "tools" / "reward_audit.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    report = json.loads((ROOT / "docs" / "evidence" / "reward-audit.json").read_text("utf-8"))
    assert result.returncode == 1, "the audit must exit non-zero while violations stand"

    assert report["violating"] == ["deliver", "mine_smelt", "supply_furnace"], (
        "the set of families that pay for stopping changed; if this is a fix, update "
        "the pin, and if it is not, a new reward plateau was introduced"
    )

    by_task = {r["task"]: r for r in report["reports"]}
    # The three clean families are clean because nothing is earnable short of
    # the goal -- repair_belt succeeds on the very quantity it pays for.
    for task_id in ("navigate", "repair_belt", "restore_power"):
        assert by_task[task_id]["plateau"] == 0.0
        assert by_task[task_id]["idle_return"] < 0

    # deliver's margin is the one that was measured collapsing.
    assert by_task["deliver"]["idle_return"] > 0.2
