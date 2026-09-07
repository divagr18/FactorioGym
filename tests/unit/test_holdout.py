"""Engine-free coverage for the frozen release holdout (PLAN.md 4.5).

PLAN 4.5 asks for "frozen held-out evaluation" and for the candidate family set
to be declared before any held-out result is read. A JSON file full of digests
does not deliver either of those on its own -- it delivers them only if something
fails loudly when the file stops describing the generators. These tests are that
something.

``test_committed_holdout_still_verifies`` is the load-bearing one, and it is
*designed* to fail whenever anyone edits a test-split generator or bumps a task
version. That is not a fragile test; it is the mechanism. Three test-split
families were edited on 2026-09-07 and the only evidence of it was a set of
success rates that moved, which is indistinguishable from the policy having
changed. The correct response to this test going red is to decide whether the
generator change was intended and then to re-freeze deliberately, never to
regenerate the file until the test passes again.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import freeze_holdout as fh  # noqa: E402

from factoriorl.seeding import SeedPlan  # noqa: E402
from factoriorl.tasks import get  # noqa: E402
from factoriorl.tasks.spec import Blueprint, EntitySpec  # noqa: E402


@pytest.fixture(scope="module")
def document() -> dict:
    if not fh.OUTPUT_PATH.is_file():
        pytest.fail(
            f"{fh.OUTPUT_PATH} is missing. PLAN 4.5 requires the frozen holdout to be "
            "committed; regenerate it with `uv run python tools/freeze_holdout.py --write`."
        )
    return json.loads(fh.OUTPUT_PATH.read_text(encoding="utf-8"))


# ------------------------------------------------------- the load-bearing test


def test_committed_holdout_still_verifies(document):
    """The committed holdout must still be what the current generators produce.

    This is the whole mechanism. If it fails, either a test-split generator
    changed, a task version was bumped, or the seed derivation moved -- and in
    every one of those cases a held-out rate published against the committed
    ``content_hash`` now refers to scenes that no longer exist.

    Do not "fix" this by re-running the freeze tool. Decide first whether the
    change to the task was intended; a re-freeze resets the candidate declaration
    and invalidates every prior citation, which is a cost that should be paid
    knowingly.
    """
    problems = fh.verify(document)
    assert problems == [], "the frozen holdout no longer matches the generators:\n" + "\n".join(
        f"  - {p}" for p in problems
    )


def test_no_manifest_cites_a_stale_holdout_hash(document):
    """PLAN 4.5: the hash must match every manifest citing it.

    A run that names this holdout under a different hash reports numbers measured
    on other scenes. Passes vacuously while no run cites the holdout, which is
    the honest state before the release run and is *not* evidence that the
    citation path works -- see the report accompanying this file.
    """
    problems = fh.stale_citations(document)
    assert problems == [], "\n".join(f"  - {p}" for p in problems)


# ----------------------------------------------------------------- the digest


def test_digest_matches_the_digest_the_worker_installs_scenes_by():
    """A frozen digest must name the scene the environment would actually install.

    The freeze tool takes its digest from ``tools/generator_diagnostics.py``
    rather than from ``factoriorl.env`` (which pulls in gymnasium and numpy, and
    freezing a list of pure-Python blueprints has no reason to require the `rl`
    extra). That indirection can drift, and a drifted digest would produce a
    holdout of hashes no worker ever computes -- a file that verifies against
    itself forever while describing nothing. This pins the chain from this side.
    """
    pytest.importorskip("gymnasium")
    from factoriorl.env import blueprint_digest

    blueprint = Blueprint(
        entities=(EntitySpec("wooden-chest", (3.0, 4.0), marker="goal"),),
        markers={"goal": (3.0, 4.0)},
    )
    assert fh.scene_digest(blueprint) == blueprint_digest(blueprint.to_dict())


def test_episode_enumeration_matches_the_environment():
    """The frozen episodes must be the ones ``FactorioEnv.prepare_scene`` produces.

    The digest being right is not enough: the tool also has to reach the same
    blueprint. ``prepare_scene`` draws ``rng.randrange(len(families))`` from the
    episode RNG *before* handing it to the generator, and ``randrange(1)``
    consumes a variable number of ``getrandbits`` calls even though its result is
    always 0. Omitting that draw -- the obvious simplification for a split with
    one family -- would give every frozen episode a scene the environment never
    installs, and nothing else in this file would notice: the digests would be
    self-consistent, stable across processes, and wrong.

    Driven against the real environment with a stub session, so no worker and no
    engine are involved; ``prepare_scene`` touches the session only to install
    the scenario it already computed.
    """
    pytest.importorskip("gymnasium")
    from factoriorl.env import FactorioEnv

    class _StubSession:
        def define_scenario(self, payload, digest):  # noqa: D401 - stub
            return None

    plan = SeedPlan(master=fh.HOLDOUT_MASTER_SEED, run_id=fh.HOLDOUT_RUN_ID)
    for task_id in ("navigate", "restore_power"):
        task = get(task_id)
        env = FactorioEnv(
            task,
            _StubSession(),
            plan,
            branch=fh.HOLDOUT_BRANCH,
            split=fh.HOLDOUT_SPLIT,
        )
        families = task.spec.families(fh.HOLDOUT_SPLIT)
        for index in (fh.HOLDOUT_START_INDEX, fh.HOLDOUT_START_INDEX + 37):
            expected = fh.episode_spec(task, families, plan, index)
            assert env.prepare_scene(index) == expected["blueprint_digest"]
            assert env._family.name == expected["layout_family"]


# ------------------------------------------------------------------- hashing


def test_content_hash_is_stable_across_processes():
    """``hash()`` is randomised per process and this repo has been bitten by it.

    Seeding blueprint sampling with the built-in ``hash()`` once made the task
    validation suite pass or fail at random. A holdout hash with that property
    would report drift on roughly every run, and a check that cries wolf is a
    check everyone learns to override -- which is strictly worse than not having
    frozen the holdout at all. Two fresh interpreters with different hash seeds
    must agree, on the real committed body rather than on a toy dict.
    """
    program = (
        "import json, sys\n"
        "sys.path.insert(0, r'%s')\n"
        "import freeze_holdout as fh\n"
        "doc = json.loads(fh.OUTPUT_PATH.read_text(encoding='utf-8'))\n"
        "print(fh.content_hash(doc['holdout']))\n" % (ROOT / "tools")
    )
    hashes = set()
    for seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT / "src"), "PATH": ""},
        )
        hashes.add(result.stdout.strip())
    assert len(hashes) == 1, f"content hash differs across processes: {hashes}"


def test_content_hash_covers_the_episodes(document):
    """A changed digest must change the hash, or the hash certifies nothing."""
    tampered = copy.deepcopy(document["holdout"])
    tampered["tasks"]["navigate"]["episodes"][0]["blueprint_digest"] = "0" * 16
    assert fh.content_hash(tampered) != document["content_hash"]


def test_content_hash_ignores_the_prose_around_it(document):
    """Editing a note must not invalidate a published citation.

    The declaration and the limitation notes describe the holdout rather than
    being it. If sharpening a sentence changed the hash, every manifest citing
    the file would go stale for no content change, and the natural response --
    updating the cited hashes -- would erase the record of what each result was
    actually measured on.
    """
    edited = copy.deepcopy(document)
    edited["notes"]["known_limitations"].append("a clarification added later")
    edited["declaration"]["candidate_families"] = ["navigate", "deliver", "repair_belt"]
    assert fh.content_hash(edited["holdout"]) == document["content_hash"]


def test_hand_editing_the_file_is_detected(document):
    """The recorded hash is checked against the body, not merely regenerated.

    Regeneration alone would not notice a JSON edited by hand to match whatever
    the generators now produce, because the edit and the regeneration would agree
    with each other and disagree only with the hash nobody rechecked.
    """
    tampered = copy.deepcopy(document)
    tampered["holdout"]["tasks"]["navigate"]["episodes"][0]["blueprint_digest"] = "0" * 16
    problems = fh.verify(tampered)
    assert any("content_hash does not match" in p for p in problems)
    assert any("navigate" in p and "episodes changed" in p for p in problems)


# --------------------------------------------------------------- invalidation


def test_a_task_version_change_invalidates_its_frozen_holdout(document):
    """A version bump invalidates the holdout even if every blueprint survives.

    ``version`` lives on the ``TaskSpec`` and the generator is a separate
    function, so budgets, predicates or reward weights can move while every
    scene stays byte-identical. A success rate on the new task is not comparable
    to one on the old, so the version has to be part of the frozen identity
    rather than a label beside it.
    """
    stale = copy.deepcopy(document)
    frozen_version = stale["holdout"]["tasks"]["mine_smelt"]["task_version"]
    stale["holdout"]["tasks"]["mine_smelt"]["task_version"] = "0.0.1-previous"
    stale["content_hash"] = fh.content_hash(stale["holdout"])

    problems = fh.verify(stale)
    assert any(
        "mine_smelt" in p and "task_version" in p and frozen_version in p for p in problems
    ), problems
    # And only that: the scenes themselves are untouched, so nothing should be
    # reported as a changed episode. A check that flagged everything on a version
    # bump would leave a real generator edit indistinguishable from a relabel.
    assert not any("episodes changed" in p for p in problems), problems


def test_a_changed_generator_is_reported_per_episode():
    """Drift must name the episodes, not merely announce itself.

    "The holdout no longer verifies" is not actionable. Whoever caused the break
    needs to see whether one episode moved or ninety, because a generator edit
    typically moves most of a family and leaves a few coincidentally identical.
    """
    plan = SeedPlan(master=fh.HOLDOUT_MASTER_SEED, run_id=fh.HOLDOUT_RUN_ID)
    frozen = fh.build_body(["navigate"], plan, start_index=1000, episodes=10)
    current = copy.deepcopy(frozen)
    for episode in current["tasks"]["navigate"]["episodes"][:4]:
        episode["blueprint_digest"] = "f" * 16

    problems = fh.diff_bodies(frozen, current)
    assert any("4/10 episodes changed" in p for p in problems), problems
    assert any("[1000]" in p for p in problems), problems


def test_a_newly_registered_task_is_not_silently_treated_as_held_out():
    """A family absent from the freeze has no held-out result.

    Adding a task family after the freeze and reporting a rate for it against
    this holdout would present a number nobody committed to in advance, which is
    exactly what the frozen file exists to prevent.
    """
    plan = SeedPlan(master=fh.HOLDOUT_MASTER_SEED, run_id=fh.HOLDOUT_RUN_ID)
    frozen = fh.build_body(["navigate"], plan, start_index=1000, episodes=5)
    current = fh.build_body(["navigate", "deliver"], plan, start_index=1000, episodes=5)
    problems = fh.diff_bodies(frozen, current)
    assert any("deliver" in p and "absent from the holdout" in p for p in problems), problems


# ---------------------------------------------------------------- seed stream


def test_offset_indices_are_not_degenerate():
    """Freezing at index 1000 rather than 0 must not cost scene diversity.

    The holdout deliberately skips the index prefix burned by exploratory runs
    that scored the test split. That is only safe because an episode seed is
    ``blake2b(master | run_id | branch | index)`` -- a hash of the index, not a
    position in a stream -- so any index is as good as any other. Asserted rather
    than assumed: if the seeding ever became stream-based, a large offset could
    quietly land in a degenerate region and the holdout would be 100 evaluations
    of a handful of scenes while still looking like 100 draws.
    """
    plan = SeedPlan(master=fh.HOLDOUT_MASTER_SEED, run_id=fh.HOLDOUT_RUN_ID)
    task = get("navigate")
    families = task.spec.families(fh.HOLDOUT_SPLIT)

    def digests(start: int) -> set[str]:
        return {
            fh.episode_spec(task, families, plan, i)["blueprint_digest"]
            for i in range(start, start + 50)
        }

    at_zero, at_offset = digests(0), digests(fh.HOLDOUT_START_INDEX)
    assert len(at_offset) >= len(at_zero) - 5, (
        f"index {fh.HOLDOUT_START_INDEX} yields {len(at_offset)} distinct scenes against "
        f"{len(at_zero)} at index 0; the offset is costing diversity"
    )
    # And the offset actually buys separation: no scene in the frozen range is
    # one the burned prefix could have shown. Not a proof of decontamination --
    # the generators themselves were tuned against observed test outcomes for two
    # families -- but it is the part that can be checked.
    assert not (at_zero & at_offset)


def test_holdout_is_the_test_split_only(document):
    """The threshold is carried by unfamiliar structures, so val must not be frozen.

    Freezing the validation split would turn a set intended for selection into a
    fixed target to tune against, which is how a validation split stops working.
    """
    assert document["holdout"]["split"] == "test"
    for task_id, entry in document["holdout"]["tasks"].items():
        declared = [f.name for f in get(task_id).spec.families("test")]
        assert entry["families_in_split"] == declared


def test_every_registered_family_is_frozen(document):
    """All six, not only today's release candidates.

    Freezing the whole set costs one file and removes the temptation to re-freeze
    when a fourth family later looks promising -- a re-freeze after results are
    known is precisely the selection PLAN 4.5 forbids.
    """
    from factoriorl.tasks import all_tasks

    assert sorted(document["holdout"]["tasks"]) == sorted(all_tasks())


def test_frozen_families_admit_enough_distinct_scenes(document):
    """A holdout that admits few scenes is evaluated few times, whatever N says.

    PLAN section 3: "a holdout whose generator admits one scene is evaluated
    once, no matter how many episodes are run against it". Publishing a Wilson
    interval over 100 correlated draws would overstate confidence, so the frozen
    file is checked against the same floor the split audit uses.
    """
    thin = {
        task_id: entry["distinct_scenes"]
        for task_id, entry in document["summary"].items()
        if entry["below_min_distinct_scenes"]
    }
    assert not thin, (
        f"frozen families below {fh.MIN_DISTINCT_SCENES} distinct scenes: {thin}; "
        "their held-out rates are not 100 independent draws"
    )


# ----------------------------------------------------------------- citations


def test_a_stale_manifest_citation_is_detected(tmp_path, document):
    """A run citing the holdout under the wrong hash reports foreign scenes."""
    for name, cited in (("good", document["content_hash"]), ("stale", "deadbeef" * 8)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "manifest.json").write_text(
            json.dumps({"run_id": name, "holdout": {"id": fh.HOLDOUT_ID, "content_hash": cited}}),
            encoding="utf-8",
        )
    # A manifest that names no holdout at all is not a failure: most runs are not
    # release runs, and demanding a citation from every run would push people to
    # paste one in without evaluating against it.
    (tmp_path / "unrelated").mkdir()
    (tmp_path / "unrelated" / "manifest.json").write_text(
        json.dumps({"run_id": "unrelated"}), encoding="utf-8"
    )

    problems = fh.stale_citations(document, runs_root=tmp_path)
    assert len(problems) == 1
    assert "stale" in problems[0]


# --------------------------------------------------------------------- CLI


def test_verify_exits_non_zero_on_drift(tmp_path, document):
    """``--verify`` is meant to be run by a gate, which reads the exit code first.

    A tool that printed a failure and exited 0 would teach the gate to ignore it,
    which is how ``tools/solvability.py``'s exit contract was decided and the
    same reasoning applies here.
    """
    tampered = copy.deepcopy(document)
    tampered["holdout"]["tasks"]["navigate"]["episodes"][0]["blueprint_digest"] = "0" * 16
    tampered["content_hash"] = fh.content_hash(tampered["holdout"])
    path = tmp_path / "holdout_v1.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "freeze_holdout.py"), "--verify", "--out", str(path)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 1, result.stdout
    assert "VERIFY FAILED" in result.stdout


# ------------------------------------------------------- the consumer side


def test_train_can_load_every_task_the_committed_holdout_covers():
    """The freeze is worthless if the training entrypoint cannot read it.

    ``train`` looked the task table up at the top level of the document while it
    lives under ``holdout``, so ``--holdout`` raised "does not cover task" for
    every task the file plainly covered and the frozen path had never once run.
    Nothing caught it: ``freeze_holdout.py --verify`` checks the file against the
    generators, and the tests above check the tool, but neither one ever asked
    the consumer to open it.

    That is precisely the gap ``RELEASE_RUNNER_CONTRACT`` warns about -- no
    downstream check can tell whether the episodes behind a published number
    were the frozen ones -- so the consumer is exercised here directly.
    """
    from factoriorl.learn.train import TrainConfig, _load_frozen_holdout
    from factoriorl.tasks import get

    path = ROOT / "docs" / "evidence" / "holdout_v1.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    covered = sorted(document["holdout"]["tasks"])
    assert covered, "the committed holdout covers no tasks"

    for task_id in covered:
        loaded = _load_frozen_holdout(TrainConfig(task_id=task_id, holdout=str(path)), get(task_id))
        assert loaded is not None
        assert loaded["content_hash"] == document["content_hash"]


def test_train_rejects_a_task_the_holdout_does_not_cover():
    """The error must name what the file *does* cover.

    The message it replaces said only that the task was missing, which was true
    of every task under the old lookup and so read as a problem with the task
    rather than with the lookup. Listing the covered set makes the two
    distinguishable at a glance.
    """
    from factoriorl.learn.train import TrainConfig, _load_frozen_holdout
    from factoriorl.tasks import get

    path = ROOT / "docs" / "evidence" / "holdout_v1.json"
    with pytest.raises(ValueError, match="does not cover task") as raised:
        _load_frozen_holdout(
            TrainConfig(task_id="no_such_family", holdout=str(path)), get("navigate")
        )
    assert "navigate" in str(raised.value), "the error does not say what is covered"
