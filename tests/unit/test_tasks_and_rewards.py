"""Engine-free coverage for the task engine, catalogs, rewards and seeding.

These are the tests that must run on every commit. Anything that needs a live
Factorio worker runs only in the gates, so the properties most likely to rot --
reward exploits, catalog indices, seed disjointness, import direction -- are
deliberately checkable without one.
"""

from __future__ import annotations

import ast
import inspect
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


def test_no_family_pays_a_policy_for_stopping():
    """Shaping earnable while success is false is a plateau: bank it and idle.

    The only thing opposing that is the step cost, so the invariant is a
    comparison of two numbers -- shaping still earnable short of the goal
    against the step cost of exhausting the budget -- and neither needs an
    engine.

    `deliver` seed 3 is the worked example of getting this wrong. It converged
    on `take_iron-plate_20`, ran 95.9 steps of a 120 budget for +0.062 and
    scored 0.00 on all three evaluation rows. At the time three of six families
    paid for stopping: deliver +0.23, mine_smelt +0.45, supply_furnace +0.50.

    The fix was not to shrink the shaping but to grade it. Each progress
    component's weight is now cap divided by the quantity success requires, so
    it pays smoothly all the way to the goal and tops out exactly when the task
    is done -- the shape `repair_belt` always had, and the reason its cap was
    never reachable short of success. Only the acquisition terms remain
    earnable early, and they sit well under pressure.
    """
    import subprocess
    import sys as _sys

    # Regenerates the report; the exit code covers both halves of the invariant
    # and is asserted by the gradient test below, not here.
    subprocess.run(
        [_sys.executable, str(ROOT / "tools" / "reward_audit.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    report = json.loads((ROOT / "docs" / "evidence" / "reward-audit.json").read_text("utf-8"))
    assert report["violating"] == [], (
        f"these families pay for stopping: {report['violating']}. A shaping cap "
        "reachable while success is false must stay below the step cost of the budget."
    )
    by_task = {r["task"]: r for r in report["reports"]}
    for task_id, entry in by_task.items():
        assert entry["idle_return"] < 0, f"{task_id} returns {entry['idle_return']:+} for idling"

    # Progress components must top out at the goal, not before it: that is what
    # makes them exempt rather than merely small.
    for task_id in ("deliver", "mine_smelt", "supply_furnace"):
        names = {c["name"] for c in by_task[task_id]["contributors"]}
        assert "at_destination" not in names
        assert "plates_produced" not in names


def test_only_the_objective_and_its_work_site_are_published_never_the_decoys():
    """Publishing the goal must not publish the scene.

    Two sources are allowed and no others: a marker a success or failure
    predicate names, and a marker the task declares in `extra_public_markers`.
    The second exists because a goal that *is* a state has its work somewhere
    else -- `repair_belt` succeeds on a plate in the sink and the plate only
    moves if a belt goes in the gap, and the measured consequence of leaving the
    gap unpublished was a potential that paid for approaching a point the
    observation never contained.

    `deliver`'s decoys are in neither source. They are what took its skill-space
    random floor from 0.80 to 0.04, and publishing them would undo the
    hardening.
    """
    for task_id in all_tasks():
        spec = get(task_id).spec
        named = {p.marker for p in (*spec.success, *spec.failure) if getattr(p, "marker", None)}
        allowed = named | set(spec.extra_public_markers)
        public = set(spec.public_markers)
        assert public == allowed, f"{task_id}: public {public} is not {allowed}"
        assert not {m for m in public if "decoy" in m}, f"{task_id} publishes a decoy"
        # A declared extra must be a real marker the generator emits, or the
        # observation would advertise a position that does not exist.
        #
        # Sampled across every family rather than one blueprint, because a
        # second fault is legitimately optional: `repair_belt` gives a third of
        # its training scenes two holes and its holdout always two, so `gap2`
        # exists in some scenes and not others. What must hold is that the
        # marker is real *somewhere* -- and, more strictly than before, that the
        # first declared extra is present in *every* scene, since it is what the
        # goal vector falls back to and a scene missing it would point the
        # policy at whatever name happened to sort first.
        if spec.extra_public_markers:
            emitted: set[str] = set()
            for family in spec.layout_families:
                for seed in range(12):
                    markers = set(get(task_id).generate(family, random.Random(seed)).markers)
                    emitted |= markers
                    focus = spec.extra_public_markers[0]
                    assert focus in markers, (
                        f"{task_id}/{family.name} seed {seed} omits {focus}, "
                        "which the goal vector falls back to"
                    )
            for name in spec.extra_public_markers:
                assert name in emitted, f"{task_id} publishes absent marker {name}"

    task = get("deliver")
    assert task.spec.public_markers == ("dst",)
    every_marker: set[str] = set()
    for family in task.spec.layout_families:
        blueprint = task.generate(family, random.Random(11))
        every_marker |= set(blueprint.markers)
        every_marker |= {e.marker for e in blueprint.entities if getattr(e, "marker", None)}
    assert {"decoy_0", "decoy_1", "src", "dst"} <= every_marker
    assert every_marker & set(task.spec.public_markers) == {"dst"}


def test_the_goal_vector_points_at_where_the_agent_must_act():
    """Not at where success is measured, and not at whatever sorts first.

    `repair_belt` is scored on the sink and has to place a belt in the gap 3.5
    tiles short of it; `restore_power` is scored on the drill and has to place a
    pole four tiles short. The goal vector used to take the first published
    marker in sort order, which handed `repair_belt` the gap by luck and
    `restore_power` the drill.
    """
    for task_id, expected in (
        ("repair_belt", "gap"),
        ("restore_power", "gap"),
        ("deliver", "dst"),
        ("navigate", "goal"),
    ):
        spec = get(task_id).spec
        assert spec.focus_marker == expected, task_id
        assert spec.focus_marker in spec.public_markers, (
            f"{task_id} focuses a marker it does not publish, so the goal vector "
            "would carry a position the agent cannot otherwise see"
        )

    # And the reward's potential must point at the same place the observation
    # does, or the agent is paid for approaching something it cannot locate.
    for task_id in ("repair_belt", "restore_power"):
        spec = get(task_id).spec
        potentials = [r for r in spec.rewards if r.kind is RewardKind.POTENTIAL and r.predicate]
        assert potentials, task_id
        assert all(r.predicate.marker == spec.focus_marker for r in potentials), task_id


def test_begin_episode_sends_both_declared_profiles():
    """Both profiles are episode state on the worker, so both must be re-sent.

    Omitting one is silent: the worker's default still yields a well-formed
    observation and a working catalog, the manifest still records what the task
    declared, and nothing downstream can tell the two apart. The observation
    profile was decorative for exactly that reason until it was passed; the
    action profile had the same bug and no task had yet declared a non-default
    value to expose it.
    """
    from factoriorl.env import FactorioEnv

    source = inspect.getsource(FactorioEnv.begin_episode)
    assert "observation_profile=self.spec_.observation_profile" in source
    assert "action_profile=self.spec_.action_profile" in source


def test_every_family_has_something_that_pays_before_success():
    """The other half of the reward invariant, and the half that was missing.

    `repair_belt` used to satisfy the plateau rule *perfectly* -- its shaping
    predicate was its success predicate, so the cap was unreachable while the
    task was unfinished -- and it was completely unlearnable for exactly that
    reason. Measured: 190 training episodes over 50,000 steps, zero successes,
    mean episode reward -0.300 against a step cost of exactly 300 x 0.001. The
    policy had nothing to ascend, and no training budget fixes that.
    `restore_power` was worse: no shaping component at all.

    Both now carry a potential on the distance to the gap they have to close.
    Sutton & Barto §17.4 (p.386) recommends initialising the value function
    rather than inventing subgoal rewards; Wiewiora (2003) proves that
    equivalent to potential-based shaping, which is what this is -- so it cannot
    change which policy is optimal.
    """
    report = json.loads((ROOT / "docs" / "evidence" / "reward-audit.json").read_text("utf-8"))
    assert report["no_gradient"] == [], (
        f"nothing pays before success in {report['no_gradient']}; a policy has nothing "
        "to ascend and no training budget helps"
    )

    by_task = {r["task"]: r for r in report["reports"]}
    for task_id, entry in by_task.items():
        assert entry["informative_components"], task_id

    # The three routes are distinguishable: a different quantity, partial credit
    # toward the same one, or a potential.
    assert any("does not" in c for c in by_task["deliver"]["informative_components"])
    assert any("partial credit" in c for c in by_task["plate_line"]["informative_components"])
    assert any("potential" in c for c in by_task["repair_belt"]["informative_components"])


def test_shaping_may_read_truth_but_only_declared_markers_are_published():
    """The observation/evaluator boundary, stated at the level that is true.

    An earlier version of this test asserted that `gap` must *never* reach an
    observation, on the principle that shaping reads evaluator truth and the
    policy's input may not. The principle is right and the conclusion was wrong:
    the boundary is not "markers are secret", it is "only what a task declares
    is published". `repair_belt` was paid for approaching a gap that appeared
    nowhere in its observation, and `deliver` scored 0.35 in exactly that
    condition and 0.92 once its target was published.

    What must stay true is the rest of truth. `containers`, `produced` and
    `working` drive success predicates and shaping, and none of them is a
    marker, so none can be published by this mechanism at all.
    """
    import random

    for task_id in ("repair_belt", "restore_power"):
        task = get(task_id)
        spec = task.spec
        assert "gap" in spec.public_markers, "the gap is meant to be visible now"
        blueprint = task.generate(spec.layout_families[0], random.Random(3))
        assert "gap" in blueprint.markers, f"{task_id} stopped emitting the gap marker"
        shaping = [r for r in spec.rewards if r.predicate and r.predicate.marker == "gap"]
        assert shaping, f"{task_id} has a gap marker nothing reads"
        assert all(r.kind is RewardKind.POTENTIAL for r in shaping)

    # Publication is a marker mechanism, so no other truth channel can ride it.
    for task_id in all_tasks():
        spec = get(task_id).spec
        assert not ({"containers", "produced", "working"} & set(spec.public_markers)), task_id
        # Pooled over every family and several seeds: an optional second fault
        # marker is real but absent from most scenes, so one blueprint cannot
        # decide whether a published name exists.
        declared: set[str] = set()
        for family in spec.layout_families:
            for seed in range(8):
                blueprint = get(task_id).generate(family, random.Random(seed))
                declared |= set(blueprint.markers)
                declared |= {
                    e.marker for e in blueprint.entities if getattr(e, "marker", None)
                }
        assert set(spec.public_markers) <= declared, (
            f"{task_id} publishes a marker no generator emits"
        )


def test_a_familys_discount_horizon_covers_its_episode():
    """The weight was never the thing to fix.

    Potential shaping telescopes to `-w*Phi(s0) + (gamma-1)*w*sum Phi(s_t)`, so a
    policy sitting at high potential pays `(1-gamma)*w*Phi` every step. For a
    policy that never succeeds, approaching gains a one-time `w*dPhi` and costs
    an ongoing `(1-gamma)*w*dPhi*T_remaining` -- and **`w` cancels**. What is
    left is `(1-gamma)*T` against 1, so the only question is whether the episode
    fits inside the discount horizon `1/(1-gamma)`.

    At the 0.99 default that horizon is 100 steps. `repair_belt` runs 300, so
    the drag beat the gain threefold and the first probe measured a reward of
    -2.5 -- worse than the -0.30 flat line it replaced.

    None of this contradicts Ng, Harada and Russell: the *optimal* policy is
    unchanged either way. What shaping cannot do at too small a gamma is
    bootstrap, because among the failing policies that are all a learner sees
    early on, the shaped ordering prefers the one that does not linger near a
    goal it cannot reach.
    """
    for task_id in all_tasks():
        spec = get(task_id).spec
        potentials = [r for r in spec.rewards if r.kind is RewardKind.POTENTIAL and r.shaping]
        if not potentials:
            continue
        gamma = spec.gamma or GAMMA
        drag_ratio = (1.0 - gamma) * spec.max_decision_steps
        assert drag_ratio < 1.0, (
            f"{task_id}: (1-gamma)*T = {drag_ratio:.2f}; its potential shaping costs more "
            f"drag than the approach is worth, at any weight. Raise gamma above "
            f"{1 - 1 / spec.max_decision_steps:.4f}."
        )
        # Deliberately nothing about the *weight*. A first version of this test
        # also required the potential to be smaller than the sparse reward, and
        # that is not a real constraint: potential shaping is policy-invariant,
        # so its magnitude cannot make the guidance outrank the goal. Magnitude
        # matters for high-water shaping, which can, and `tools/reward_audit.py`
        # is where that is checked.
