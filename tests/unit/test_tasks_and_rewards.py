"""Engine-free coverage for the task engine, catalogs, rewards and seeding.

These are the tests that must run on every commit. Anything that needs a live
Factorio worker runs only in the gates, so the properties most likely to rot --
reward exploits, catalog indices, seed disjointness, import direction -- are
deliberately checkable without one.
"""

from __future__ import annotations

import ast
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
