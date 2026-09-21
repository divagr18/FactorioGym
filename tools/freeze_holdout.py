"""Freeze and verify the release held-out evaluation set (DESIGN.md 4.5).

DESIGN.md 4.5 asks for "frozen held-out evaluation": 100 episodes per family per
training seed, on the **test (structural) split**, with the candidate family set
declared *before* anyone looks at a held-out result. Nothing in the repo made
"frozen" mean anything. The evaluation set was whatever ``train.py`` happened to
generate at the moment it ran, so three training seeds scored three different
sets of scenes, and a re-run after a generator edit scored a fourth -- while
every one of them was published under the same words, "held-out success rate".

This tool turns that phrase into a checkable object. It enumerates the exact
scenes the environment would install for a fixed evaluation seed stream, writes
them to ``docs/evidence/holdout_v3.json`` with a content hash over the whole
structure, and -- the load-bearing half -- regenerates from current source and
reports which episodes differ.

Run:

    uv run python tools/freeze_holdout.py            # write the frozen file
    uv run python tools/freeze_holdout.py --verify   # check it, non-zero on drift

Why the holdout has its own seed plan
-------------------------------------
An episode seed is ``blake2b(master | run_id | branch | index)`` and ``train.py``
builds its ``SeedPlan`` with ``run_id = manifest.new_run_id(...)``, a timestamp
plus a random salt. So the evaluation scenes are a function of the *run*, not of
``--seed``: two runs launched with the same ``--seed`` one second apart evaluate
completely different content, and no evaluation this repo has ever performed can
be reproduced from its manifest's ``master`` alone.

That makes "freeze the holdout at master seed N" impossible on its own, so this
file freezes a whole seed plan -- ``master`` **and** ``run_id`` -- that belongs to
the holdout rather than to any run. Training seeds keep varying the policy; the
evaluated scenes stop varying with them. That is also the only arrangement in
which DESIGN 4.5's "per training seed" comparison, and 4b.3's two arms, are paired
measurements rather than three independent samples of the scene distribution,
which section 3 explicitly asks for.

The consequence for the release runner is a change this tool cannot make: the
evaluation must be driven from *this* seed plan, not from the training run's.
See ``RELEASE_RUNNER_CONTRACT`` below for the exact obligation.

Why the episodes do not start at index 0
----------------------------------------
Until 2026-09-08 ``train.py`` evaluated ``split="test"`` by default, and the
exploratory runs in ``runtime/runs`` -- a flat-vs-skills ablation and several
per-family runs -- scored that split from a shared cursor starting at 0. Task
targets were raised and a skill addressing scheme changed after those numbers
were read. Freezing over indices 0..99 would therefore freeze the index range
that selection decisions were made against.

Two things separate this holdout from those runs, and they are worth keeping
distinct because only one of them is airtight:

* the ``run_id`` differs, so the seed stream is disjoint by construction -- no
  run has ever drawn ``blake2b(master | "holdout-v1" | "eval" | i)`` for any i;
* the index range is offset past the burned prefix anyway.

The offset is belt-and-braces: it costs nothing and it survives someone later
pointing a run at this exact seed plan (which the release runner is *required*
to do, so the case is not hypothetical). A large index is not a worse index --
the seed is a hash of the index rather than a position in a stream, so index
1000 is as well distributed as index 0, and ``test_offset_indices_are_not_degenerate``
asserts that rather than trusting it.

What this does **not** decontaminate
------------------------------------
A disjoint seed stream makes the *scenes* unseen. It does not make the
*generators* unseen. ``mine_smelt`` and ``supply_furnace`` had their targets and
layouts adjusted after their test-split numbers were read, so the test families
themselves were partly tuned against observed held-out outcomes. No index offset
can undo that; only a generator nobody has scored against can, and this repo
does not have one. That limitation is written into the frozen file under
``notes.known_limitations`` so a reader of the release result meets it there
rather than inferring it, and it is the reason ``--candidates`` records a
declaration timestamp: what remains honest here is the *forward* commitment, not
a claim of retrospective independence.

What breaks, and why that is the point
--------------------------------------
Editing a test-split generator changes its blueprints, so ``--verify`` fails and
so does ``tests/unit/test_holdout.py``. That is the mechanism working. There is
deliberately no silent regeneration and no ``--force``: re-freezing is
``--write`` plus a commit plus a new declaration, which leaves a diff someone has
to justify. A tool that quietly re-froze on drift would reduce the holdout to a
record of the last time anyone ran it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

# The digest and the scene-count floor come from the split audit rather than
# being restated here. `scene_digest` is the algorithm `factoriorl.env` installs
# scenes by (canonical JSON, sha256, 16 hex chars) and an existing test pins the
# two together; `tests/unit/test_holdout.py` pins it again from this side, so
# neither tool can drift from the environment without a red test. Importing
# `factoriorl.env` directly would pull in gymnasium and numpy and make freezing a
# holdout require the `rl` extra, which a pure-Python blueprint enumeration has
# no reason to need.
from generator_diagnostics import MIN_DISTINCT_SCENES, scene_digest  # noqa: E402

from factoriorl.seeding import Branch, SeedPlan  # noqa: E402
from factoriorl.tasks import all_tasks, get  # noqa: E402

# ------------------------------------------------------------------- identity

#: Bumped, never edited in place. A holdout whose contents changed under a name
#: someone already published a number against is worse than no holdout: every
#: earlier result silently starts referring to scenes it never saw.
# The live holdout. Bumped rather than re-frozen in place: a holdout is
# supposed to be an immutable artifact, and `holdout_v2` was re-frozen five
# times, each edit silently invalidating the whole-file hash cited by every
# earlier run. `holdout_v1` and `holdout_v2` stay in the tree as the record of
# what earlier results were measured against; `--out` still targets any of them
# for --verify.
#
# v3 exists because `repair_belt` and `restore_power` reached v1.6.0: their
# evaluated splits used to test regimes training never showed (a two-hole
# holdout against one-hole training, and a terminal-gap holdout against
# interior-gap training), which is a task change and not a re-labelling.
HOLDOUT_ID = "holdout_v3"

#: Schema of the hashed body. Inside the hash on purpose -- a format change is a
#: content change from the point of view of anything comparing hashes.
SCHEMA = "factoriorl.holdout/1"

OUTPUT_PATH = ROOT / "docs" / "evidence" / f"{HOLDOUT_ID}.json"

# ------------------------------------------------------------------ seed plan

#: The holdout's master seed. Its exact value is arbitrary and deliberately so:
#: disjointness from every training run comes from `HOLDOUT_RUN_ID`, not from
#: this number, because `run_id` is part of the seed derivation and no training
#: run will ever carry the holdout's. It is written as the freeze date so the
#: file says when the stream was chosen. It is *not* `train.py`'s default
#: `master_seed`, and it must never be changed to track one: doing so would make
#: the frozen scenes a function of a training default.
HOLDOUT_MASTER_SEED = 20260908

#: The other half of the seed plan, and the half that does the work. Fixed, and
#: unlike any `manifest.new_run_id()` output (those carry a timestamp and a
#: random salt), so this stream cannot collide with a run's by accident.
HOLDOUT_RUN_ID = "holdout-v3"

#: Evaluation draws from the EVAL branch, disjoint from TRAIN by construction
#: (see `factoriorl.seeding`), so no frozen episode can be one the policy trained
#: on regardless of what index range training reached.
HOLDOUT_BRANCH = Branch.EVAL

#: DESIGN section 3: the acceptance threshold is carried by unfamiliar
#: *structures*, so the holdout is the test split and only the test split. The
#: `val` split stays available for selection and must not be frozen here -- a
#: frozen validation set invites tuning against a fixed target, which is the
#: opposite of what validation is for.
HOLDOUT_SPLIT = "test"

#: First episode index. Offset past the prefix burned by the exploratory runs
#: that scored `split="test"` from a cursor at 0 -- see the module docstring.
#: 1000 rather than 100 because `evaluate_parallel` over-runs its cursor: with N
#: workers it starts more episodes than it counts, so the burned prefix extends
#: an unknown handful past `eval_episodes` and a tight offset would leave that
#: uncertainty inside the holdout.
HOLDOUT_START_INDEX = 3000

#: DESIGN 4.5: "Evaluate 100 held-out episodes per family per training seed."
HOLDOUT_EPISODES = 100

#: What the release runner has to do for the frozen file to mean anything. Kept
#: as text in the artefact rather than only in this docstring, because the person
#: who reads the holdout months from now will have the JSON and not this module.
RELEASE_RUNNER_CONTRACT = (
    "The release evaluation must build its evaluation SeedPlan from this file's "
    "seed_plan (master and run_id) rather than from the training run's plan, and "
    "must start its episode cursor at start_index. The default path does neither: "
    "a run evaluates with SeedPlan(master=--seed, run_id=<fresh per-run id>) from "
    "index 0, which is a different seed stream and therefore different scenes. "
    "A run that does not opt in has NOT evaluated this holdout however its "
    "manifest is labelled, and no check downstream can tell the difference: "
    "tests/unit/test_holdout.py can only confirm that a cited hash matches the "
    "committed file, never that the episodes behind the number were these ones."
)

# ------------------------------------------------------------- known caveats

#: Recorded in the artefact so the release result carries its own caveats.
#: Outside the content hash (see `content_hash`): these are prose about the
#: content, and a reader who sharpens the wording must not thereby invalidate
#: every published citation of the hash.
KNOWN_LIMITATIONS = [
    "The seed stream is disjoint from every run in runtime/runs, so these scenes "
    "are unseen. The generators that produce them are not: mine_smelt and "
    "supply_furnace had targets and layouts adjusted after their test-split "
    "numbers were read on 2026-09-07, so those two families were partly tuned "
    "against observed held-out outcomes. Their frozen rates measure less "
    "independence than the other four and must be reported with this caveat.",
    "Until 2026-09-08 train.py evaluated split='test' by default, so the "
    "exploratory runs in runtime/runs scored the structural split repeatedly and "
    "task-design decisions followed. This holdout's honesty rests on the "
    "forward declaration in `declaration`, not on those runs having been blind.",
    "A frozen episode list fixes the scenes, not the policy's exposure to the "
    "task family during training. Structural transfer is still the claim being "
    "measured; freezing only removes scene-sampling variance from it.",
]

# --------------------------------------------------------------- enumeration


def episode_spec(task, families, plan: SeedPlan, index: int) -> dict:
    """One episode's frozen identity, generated exactly as ``FactorioEnv`` would.

    ``FactorioEnv.prepare_scene`` builds ``seed_plan.generator_rng(branch, index)``,
    draws ``rng.randrange(len(families))`` to pick the layout family, and hands
    that *same, already-advanced* RNG to ``task.generate``. The draw is replayed
    here even where the split holds a single family and its result is a foregone
    zero, because ``randrange(1)`` still consumes a variable number of
    ``getrandbits`` calls: skipping it would hand the generator an RNG state the
    environment never produces, and every digest in the frozen file would name a
    scene no run will ever install.
    """
    rng = plan.generator_rng(HOLDOUT_BRANCH, index)
    family = families[rng.randrange(len(families))]
    blueprint = task.generate(family, rng)
    return {
        "episode_index": index,
        "layout_family": family.name,
        # Digested exactly as `FactorioEnv._install` sends it, published
        # markers included -- otherwise the frozen digest names a payload no
        # run installs, which is the one thing this file exists to prevent.
        "blueprint_digest": scene_digest(
            blueprint, task.spec.public_markers, task.spec.extra_tracked_items
        ),
    }


def freeze_task(task_id: str, plan: SeedPlan, start_index: int, episodes: int) -> dict:
    """The frozen specs for one task family's held-out split."""
    task = get(task_id)
    families = task.spec.families(HOLDOUT_SPLIT)
    if not families:
        raise SystemExit(f"{task_id} declares no '{HOLDOUT_SPLIT}' layout family")
    specs = [
        episode_spec(task, families, plan, index)
        for index in range(start_index, start_index + episodes)
    ]
    return {
        "task_id": task_id,
        # A holdout is only meaningful for the task it was frozen against. A
        # version bump means the budgets, predicates or rewards moved, and a
        # success rate on the new task is not comparable to one on the old even
        # when every blueprint digest happens to survive -- which it can, since
        # `version` lives on the spec and the generator is a separate function.
        # Recorded inside the hashed body so a bump alone invalidates the file.
        "task_version": task.spec.version,
        "families_in_split": [f.name for f in families],
        "episodes": specs,
    }


def build_body(
    task_ids: list[str],
    plan: SeedPlan,
    start_index: int = HOLDOUT_START_INDEX,
    episodes: int = HOLDOUT_EPISODES,
    holdout_id: str = HOLDOUT_ID,
) -> dict:
    """The part of the artefact the content hash covers."""
    return {
        "schema": SCHEMA,
        "holdout_id": holdout_id,
        "seed_plan": {
            "master": plan.master,
            "run_id": plan.run_id,
            "branch": HOLDOUT_BRANCH.value,
            "episode_seed_algorithm": "blake2b(master|run_id|branch|index)[:8]",
        },
        "split": HOLDOUT_SPLIT,
        "start_index": start_index,
        "episodes_per_task": episodes,
        "digest_algorithm": "sha256(canonical_json(blueprint.to_dict()))[:16]",
        "tasks": {
            task_id: freeze_task(task_id, plan, start_index, episodes)
            for task_id in sorted(task_ids)
        },
    }


# -------------------------------------------------------------------- hashing


def content_hash(body: dict) -> str:
    """sha256 over the canonical JSON of the hashed body.

    ``hashlib``, never the built-in ``hash()``: ``hash()`` is randomised per
    process by PYTHONHASHSEED, and this repo has already been bitten by that --
    seeding blueprint sampling with it made the task validation suite pass or
    fail at random. A holdout hash that differed between processes would make the
    verification below report drift on every second run and would train everyone
    to ignore it. ``test_content_hash_is_stable_across_processes`` pins this.

    ``sort_keys`` and tight separators so the hash depends on the content and not
    on the indentation the file happens to be written with.

    Covers ``body`` only. ``declaration``, ``notes`` and ``generated_at`` sit
    outside it deliberately: they describe the holdout rather than being it, and
    a hash that moved when someone corrected a sentence would invalidate every
    manifest citing it for no content change.
    """
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ------------------------------------------------------------- verification


def diff_bodies(frozen: dict, current: dict) -> list[str]:
    """Every way the regenerated holdout differs from the committed one.

    Returns messages, not a bool, and enumerates rather than short-circuiting:
    "the holdout no longer verifies" is not actionable, and the first difference
    is rarely the whole story -- a generator edit typically moves most of a
    family's episodes and leaves a few coincidentally identical. Whoever caused
    the break needs to see the shape of it to decide whether it was intended.
    """
    problems: list[str] = []
    for key in ("schema", "split", "start_index", "episodes_per_task", "digest_algorithm"):
        if frozen.get(key) != current.get(key):
            problems.append(f"{key}: frozen {frozen.get(key)!r}, current {current.get(key)!r}")
    if frozen.get("seed_plan") != current.get("seed_plan"):
        problems.append(
            f"seed_plan: frozen {frozen.get('seed_plan')}, current {current.get('seed_plan')}; "
            "the regenerated episodes come from a different seed stream, so no per-episode "
            "comparison below means anything"
        )

    frozen_tasks = frozen.get("tasks", {})
    current_tasks = current.get("tasks", {})
    for task_id in sorted(set(frozen_tasks) - set(current_tasks)):
        problems.append(f"{task_id}: frozen in the holdout but no longer registered")
    for task_id in sorted(set(current_tasks) - set(frozen_tasks)):
        problems.append(
            f"{task_id}: registered now but absent from the holdout; it was never frozen, "
            "so it has no held-out result and must not be reported as having one"
        )

    for task_id in sorted(set(frozen_tasks) & set(current_tasks)):
        was, now = frozen_tasks[task_id], current_tasks[task_id]
        if was["task_version"] != now["task_version"]:
            # Reported separately from the digest comparison and never folded
            # into it: a version bump invalidates the frozen holdout on its own,
            # even when every blueprint is byte-identical, because the task the
            # rate is a rate *of* has changed.
            problems.append(
                f"{task_id}: task_version {was['task_version']} -> {now['task_version']}; "
                "the frozen holdout belongs to the earlier task and its rates are not "
                "comparable to results on this one"
            )
        if was["families_in_split"] != now["families_in_split"]:
            problems.append(
                f"{task_id}: test split families {was['families_in_split']} -> "
                f"{now['families_in_split']}"
            )
        was_episodes = {e["episode_index"]: e for e in was["episodes"]}
        now_episodes = {e["episode_index"]: e for e in now["episodes"]}
        differing = sorted(
            index
            for index in set(was_episodes) | set(now_episodes)
            if was_episodes.get(index) != now_episodes.get(index)
        )
        if differing:
            shown = ", ".join(str(i) for i in differing[:8])
            more = f" (+{len(differing) - 8} more)" if len(differing) > 8 else ""
            problems.append(
                f"{task_id}: {len(differing)}/{len(was_episodes)} episodes changed at "
                f"indices {shown}{more}"
            )
            for index in differing[:3]:
                problems.append(
                    f"    [{index}] frozen {was_episodes.get(index)} "
                    f"current {now_episodes.get(index)}"
                )
    return problems


def verify(document: dict, pending: set[str] | None = None) -> list[str]:
    """Check a committed artefact against the current generators.

    Two independent checks, because they fail for different reasons and a caller
    that saw only one would misdiagnose the other:

    * the committed hash must match the committed body -- catches the file being
      edited by hand, which no amount of regeneration would notice;
    * the regenerated body must match the committed body -- catches a generator,
      a task version or the seeding changing under it.

    ``pending`` names tasks that are registered but deliberately not yet in this
    file, and exists only for ``--add-task``: without it that mode could never
    run, because the condition it fixes -- a registered task absent from the
    holdout -- is the very thing verification refuses to proceed past. Nothing
    else may pass it, or "registered but never frozen" would stop being an
    error.
    """
    problems: list[str] = []
    body = document.get("holdout")
    if not isinstance(body, dict):
        return ["the file has no 'holdout' body; it is not a holdout artefact"]

    recorded = document.get("content_hash")
    recomputed = content_hash(body)
    if recorded != recomputed:
        problems.append(
            f"content_hash does not match the file's own body: recorded {recorded}, "
            f"recomputed {recomputed}; the JSON was edited after it was frozen"
        )

    seed_plan = body.get("seed_plan", {})
    plan = SeedPlan(master=seed_plan.get("master"), run_id=seed_plan.get("run_id"))
    registered = sorted(set(all_tasks()) - (pending or set()))
    # Regenerated with the *frozen* parameters, never with this module's current
    # defaults: verification must ask "is the committed content still what the
    # source produces", and re-parameterising from the defaults would answer a
    # different question and pass after someone changed a constant here.
    current = build_body(
        task_ids=sorted(set(body.get("tasks", {})) | set(registered)),
        plan=plan,
        start_index=body.get("start_index"),
        episodes=body.get("episodes_per_task"),
    )
    problems.extend(diff_bodies(body, current))
    return problems


# ------------------------------------------------------------ manifest audit

#: Where a run states which holdout its numbers came from. `RunManifest.extra`
#: puts unknown keys at the manifest's top level, so a run cites the holdout by
#: writing `extra={"holdout": {"id": ..., "content_hash": ...}}`.
MANIFEST_HOLDOUT_KEY = "holdout"


def manifests_citing(holdout_id: str = HOLDOUT_ID, runs_root: Path | None = None) -> list[dict]:
    """Every run manifest that claims to have evaluated against this holdout.

    DESIGN 4.5's requirement is that the hash "matches every manifest citing it",
    so the citation has to be checkable from the outside. A manifest that names
    no holdout is not an error here -- most runs are not release runs -- but one
    that names *this* holdout with a stale hash is: it reports a number against
    scenes it did not evaluate.
    """
    root = runs_root if runs_root is not None else ROOT / "runtime" / "runs"
    if not root.is_dir():
        return []
    citations = []
    for path in sorted(root.glob("*/manifest.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A malformed manifest is a manifest problem, not a holdout problem,
            # and `runs verify` is where it belongs. Silently skipping it here
            # would be wrong only if it cited the holdout, which it cannot be
            # read to determine either way.
            continue
        citation = data.get(MANIFEST_HOLDOUT_KEY)
        if isinstance(citation, dict) and citation.get("id") == holdout_id:
            citations.append({"path": str(path), "citation": citation})
    return citations


def stale_citations(document: dict, runs_root: Path | None = None) -> list[str]:
    """Manifests citing this holdout under a hash it does not have."""
    expected = document.get("content_hash")
    summary = document.get("summary") or {}
    problems = []
    for entry in manifests_citing(
        document.get("holdout", {}).get("holdout_id", HOLDOUT_ID), runs_root
    ):
        citation = entry["citation"]
        cited_entry = citation.get("task_entry_hash")
        task_id = citation.get("task") or entry.get("task")
        expected_entry = (summary.get(task_id) or {}).get("entry_hash")
        # Judge on the per-task digest when the run recorded one: a re-freeze of
        # another family leaves this run's scenes untouched and must not be
        # reported as stale. Runs from before the digest existed can only be
        # checked against the whole-file hash, and for those the coarse verdict
        # is the honest one -- they cannot prove which scenes they used.
        if cited_entry is not None and expected_entry is not None:
            if cited_entry != expected_entry:
                problems.append(
                    f"{entry['path']} cites {task_id} entry_hash {cited_entry}, but the "
                    f"committed holdout has {expected_entry}; that run's held-out numbers "
                    "were measured against different scenes"
                )
            continue
        cited = citation.get("content_hash")
        if cited != expected:
            problems.append(
                f"{entry['path']} cites holdout content_hash {cited}, but the committed "
                f"holdout is {expected}; that run predates per-task digests, so which "
                "scenes produced its numbers cannot be established"
            )
    return problems


# ------------------------------------------------------------------ assembly


def git_commit() -> str | None:
    """The commit the freeze was taken at, so the artefact pins its own source."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            cwd=ROOT,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def entry_hash(entry: dict) -> str:
    """Digest of one task's frozen episode set.

    Must agree with `factoriorl.learn.train.holdout_entry_hash`; a run cites
    what this produces, so the two are pinned together by
    ``tests/unit/test_holdout.py``.
    """
    canonical = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


def summarise(body: dict) -> dict:
    """Per-task facts a reader checks before trusting a rate computed on this.

    ``distinct_scenes`` is the one that matters: DESIGN section 3 is explicit that
    "a holdout whose generator admits one scene is evaluated once, no matter how
    many episodes are run against it". A frozen list of 100 episodes covering
    four scenes would still present a Wilson interval computed as if there were
    100 independent draws, so the count is published beside the rate rather than
    left for the split audit to mention elsewhere.
    """
    summary = {}
    for task_id, entry in body["tasks"].items():
        digests = [e["blueprint_digest"] for e in entry["episodes"]]
        distinct = len(set(digests))
        summary[task_id] = {
            # Scoped to one task, so it moves exactly when *these* scenes move.
            # The whole-file `content_hash` changes whenever any family is
            # re-frozen, which made `stale_citations` flag every task whenever
            # one was bumped -- and a warning that fires on untouched families
            # is one a reader learns to click through. `holdout_v2` was
            # re-frozen five times; `deliver`'s episodes were identical across
            # the last four, and only this digest can say so.
            "entry_hash": entry_hash(entry),
            "task_version": entry["task_version"],
            "families_in_split": entry["families_in_split"],
            "episodes": len(digests),
            "distinct_scenes": distinct,
            "distinct_fraction": round(distinct / len(digests), 4) if digests else 0.0,
            "below_min_distinct_scenes": distinct < MIN_DISTINCT_SCENES,
        }
    return summary


def build_document(
    task_ids: list[str],
    plan: SeedPlan,
    start_index: int,
    episodes: int,
    candidates: list[str] | None,
    holdout_id: str = HOLDOUT_ID,
) -> dict:
    body = build_body(task_ids, plan, start_index, episodes, holdout_id)
    return {
        "content_hash": content_hash(body),
        "holdout": body,
        # Everything below is outside the hash. See `content_hash`.
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "declaration": {
            # DESIGN 4.5: "Declare the candidate family set *before* looking at any
            # held-out result, with a timestamped record -- that declaration is
            # what keeps the holdout honest." Left null rather than defaulted to
            # all six, because a declaration nobody made must read as missing. A
            # file whose candidate list was filled in by a tool would document
            # the tool's opinion and would satisfy the requirement on paper only,
            # which is the exact failure the requirement exists to prevent.
            "candidate_families": candidates,
            "declared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if candidates
            else None,
            "declared_at_commit": git_commit() if candidates else None,
            "note": (
                "The candidate set must be recorded here before any result on this "
                "holdout is read. Adding it afterwards produces the same JSON and a "
                "different claim."
            ),
        },
        "notes": {
            "release_runner_contract": RELEASE_RUNNER_CONTRACT,
            "known_limitations": KNOWN_LIMITATIONS,
            "start_index_rationale": (
                f"Episodes start at {start_index}, not 0. Indices near 0 of the test "
                "split were evaluated repeatedly by the exploratory runs in "
                "runtime/runs while train.py still defaulted to split='test', and "
                "task-design decisions followed those numbers. The seed stream is "
                "already disjoint because run_id differs; the offset is redundant "
                "protection that survives a future run being pointed at this plan."
            ),
            "on_change": (
                "If a test-split generator or a task version changes, --verify fails "
                "and so does tests/unit/test_holdout.py. That is intended. Re-freezing "
                "is a deliberate act: run with --write, commit the diff, and record a "
                "new candidate declaration. Any result published against the previous "
                "content_hash refers to scenes that no longer exist and must be "
                "labelled with the hash it was measured on."
            ),
        },
        "summary": summarise(body),
    }


# ------------------------------------------------------------------ reporting


def print_summary(document: dict) -> None:
    body = document["holdout"]
    plan = body["seed_plan"]
    print(f"\n{body['holdout_id']}  ({body['schema']})")
    print(f"  content_hash : {document['content_hash']}")
    print(
        f"  seed plan    : master={plan['master']} run_id={plan['run_id']} branch={plan['branch']}"
    )
    print(
        f"  episodes     : {body['episodes_per_task']} per task, "
        f"indices {body['start_index']}..{body['start_index'] + body['episodes_per_task'] - 1}, "
        f"split '{body['split']}'"
    )
    print(f"\n  {'task':<16}{'version':<10}{'test family':<18}{'distinct':>9}{'':>3}")
    for task_id, entry in sorted(document["summary"].items()):
        flag = "  <-- below min" if entry["below_min_distinct_scenes"] else ""
        print(
            f"  {task_id:<16}{entry['task_version']:<10}"
            f"{','.join(entry['families_in_split']):<18}"
            f"{entry['distinct_scenes']:>9}{flag}"
        )
    declared = document["declaration"]["candidate_families"]
    print(f"\n  candidate declaration: {declared if declared else 'NOT DECLARED'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verify",
        action="store_true",
        help="regenerate from current source and compare against the committed file",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the frozen file; refuses to overwrite unless --refreeze is given",
    )
    parser.add_argument(
        "--refreeze",
        action="store_true",
        help=(
            "permit overwriting an existing holdout. A deliberate act: every result "
            "published against the old content_hash stops describing these scenes."
        ),
    )
    parser.add_argument(
        "--add-task",
        default="",
        help=(
            "freeze one newly registered task into an existing holdout without "
            "touching any existing entry or the candidate declaration. The added "
            "task is deliberately NOT declared a candidate."
        ),
    )
    parser.add_argument(
        "--replace-task",
        default="",
        help=(
            "re-freeze one task's entry in place. Refused if any manifest cites "
            "that task's entry_hash, because replacing it would silently change "
            "what a published number refers to."
        ),
    )
    parser.add_argument(
        "--declare",
        action="store_true",
        help=(
            "record the candidate family declaration on an already-frozen holdout "
            "without touching its content. Requires --candidates."
        ),
    )
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--tasks", default=",".join(sorted(all_tasks())))
    parser.add_argument("--episodes", type=int, default=HOLDOUT_EPISODES)
    parser.add_argument("--start-index", type=int, default=HOLDOUT_START_INDEX)
    parser.add_argument("--seed", type=int, default=HOLDOUT_MASTER_SEED)
    parser.add_argument("--run-id", default=HOLDOUT_RUN_ID)
    parser.add_argument(
        "--candidates",
        default="",
        help=(
            "comma-separated task ids declared as release candidates, recorded with a "
            "timestamp. DESIGN 4.5 requires this before any held-out result is read."
        ),
    )
    args = parser.parse_args()

    if args.verify:
        if not args.out.is_file():
            print(f"no frozen holdout at {args.out}; run with --write first")
            return 2
        document = json.loads(args.out.read_text(encoding="utf-8"))
        problems = verify(document) + stale_citations(document)
        print_summary(document)
        if problems:
            print(f"\nVERIFY FAILED ({len(problems)} problems):")
            for problem in problems:
                print(f"  - {problem}")
            print(
                "\nIf a generator or task version changed on purpose, re-freeze with "
                "--write --refreeze and record a new candidate declaration; do not "
                "reuse the previous content_hash."
            )
            return 1
        print(f"\nVERIFY OK: {args.out} still matches the current generators")
        return 0

    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()] or None
    plan = SeedPlan(master=args.seed, run_id=args.run_id)

    if args.replace_task:
        # Distinct from both --add-task and --refreeze. A task's spec can change
        # legitimately before anyone has evaluated against it -- `build_line`
        # gained a settling period the day it was frozen, after the gate showed
        # the window alone accepted a line that had stopped. Re-freezing the
        # whole file would drop the declaration timestamp that protects the
        # other seven; leaving the entry stale would fail every verification.
        #
        # The condition that makes this safe is not "recently frozen", it is
        # "no result refers to it", so that is what is checked.
        if not args.out.is_file():
            print(f"no frozen holdout at {args.out}; run with --write first")
            return 2
        document = json.loads(args.out.read_text(encoding="utf-8"))
        body = document["holdout"]
        summary = document.get("summary") or {}
        targets = [t.strip() for t in args.replace_task.split(",") if t.strip()]
        missing = sorted(set(targets) - set(body["tasks"]))
        if missing:
            print(f"not in this holdout: {missing}. Use --add-task to add a new task.")
            return 2
        cited = []
        for entry in manifests_citing(body.get("holdout_id", HOLDOUT_ID)):
            citation = entry["citation"]
            task_id = citation.get("task")
            if task_id not in targets:
                continue
            digest = (summary.get(task_id) or {}).get("entry_hash")
            if citation.get("task_entry_hash") == digest or citation.get(
                "content_hash"
            ) == document.get("content_hash"):
                cited.append(f"{entry['path']} cites {task_id}")
        if cited:
            print(
                "refusing to replace an entry a published run refers to; its numbers "
                "would silently start describing different scenes:"
            )
            for line in cited:
                print(f"  - {line}")
            print("Bump the holdout id and freeze a new one instead.")
            return 1
        frozen_plan = SeedPlan(
            master=body["seed_plan"]["master"], run_id=body["seed_plan"]["run_id"]
        )
        before_hashes = {t: (summary.get(t) or {}).get("entry_hash") for t in body["tasks"]}
        for task_id in targets:
            body["tasks"][task_id] = freeze_task(
                task_id, frozen_plan, body["start_index"], body["episodes_per_task"]
            )
        before = document["content_hash"]
        document["content_hash"] = content_hash(body)
        document["summary"] = summarise(body)
        document["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        document["git_commit"] = git_commit()
        untouched = [
            t
            for t in body["tasks"]
            if t not in targets and before_hashes[t] != document["summary"][t]["entry_hash"]
        ]
        if untouched:
            print(f"aborting: replacing {targets} moved other entries: {untouched}")
            return 1
        if not args.write:
            print_summary(document)
            print(f"\ndry run; nothing written. Pass --write to replace {targets}")
            return 0
        args.out.write_text(json.dumps(document, indent=2), encoding="utf-8")
        print_summary(document)
        print(f"\nreplaced {targets} in {args.out}")
        print(f"content_hash {before} -> {document['content_hash']}")
        for task_id in targets:
            print(
                f"  {task_id} entry_hash {before_hashes[task_id]} -> "
                f"{document['summary'][task_id]['entry_hash']}"
            )
        print("Every other entry_hash is unchanged, and no run cited the replaced ones.")
        return 0

    if args.add_task:
        # A separate mode from --refreeze, and the reason is the declaration.
        # --refreeze rebuilds the document from scratch, which drops
        # `declaration.declared_at` -- and that timestamp is the entire evidence
        # that the candidate set was chosen before any held-out result was read.
        # Adding an eighth task must not destroy the evidence about the other
        # seven. Existing entries are copied verbatim, so every existing
        # `entry_hash` is unchanged and `stale_citations` correctly reports no
        # published run as stale, which is the case it was already written for.
        if not args.out.is_file():
            print(f"no frozen holdout at {args.out}; run with --write first")
            return 2
        document = json.loads(args.out.read_text(encoding="utf-8"))
        pending = {t.strip() for t in args.add_task.split(",") if t.strip()}
        problems = verify(document, pending=pending)
        if problems:
            print("refusing to add a task to a holdout that does not verify:")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        added = [t.strip() for t in args.add_task.split(",") if t.strip()]
        present = sorted(set(added) & set(document["holdout"]["tasks"]))
        if present:
            print(
                f"already frozen: {present}. Changing an existing entry is a "
                "re-freeze, not an addition; its published results would stop "
                "describing these scenes."
            )
            return 1
        body = document["holdout"]
        # The seed plan is the frozen file's, not the CLI's: a task added under a
        # different plan would sit in the same file under the same holdout_id
        # while naming scenes from a different seed stream.
        frozen_plan = SeedPlan(
            master=body["seed_plan"]["master"], run_id=body["seed_plan"]["run_id"]
        )
        for task_id in added:
            body["tasks"][task_id] = freeze_task(
                task_id, frozen_plan, body["start_index"], body["episodes_per_task"]
            )
        body["tasks"] = {k: body["tasks"][k] for k in sorted(body["tasks"])}
        before = document["content_hash"]
        document["content_hash"] = content_hash(body)
        document["summary"] = summarise(body)
        document["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        document["git_commit"] = git_commit()
        if not args.write:
            print_summary(document)
            print(f"\ndry run; nothing written. Pass --write to add {added}")
            return 0
        args.out.write_text(json.dumps(document, indent=2), encoding="utf-8")
        print_summary(document)
        print(f"\nadded {added} to {args.out}")
        print(f"content_hash {before} -> {document['content_hash']}")
        print(
            "Per-task entry_hash values for the existing tasks are unchanged, so "
            "results published against them still describe their scenes. The added "
            f"task is not a declared candidate: declaration is still "
            f"{document['declaration']['candidate_families']}."
        )
        return 0

    if args.declare:
        # A separate mode from --write because the declaration is not content:
        # it sits outside the content hash, so recording it leaves every
        # published citation valid. Routing it through --refreeze would have
        # warned about invalidating results that are in fact untouched, and a
        # warning that is wrong is a warning people learn to click through.
        if candidates is None:
            print("--declare requires --candidates")
            return 2
        if not args.out.is_file():
            print(f"no frozen holdout at {args.out}; run with --write first")
            return 2
        document = json.loads(args.out.read_text(encoding="utf-8"))
        problems = verify(document)
        if problems:
            print("refusing to declare against a holdout that does not verify:")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        if document["declaration"]["candidate_families"] is not None:
            # The timestamp on the first declaration is the entire evidence that
            # the candidate set was chosen before a held-out result was read.
            # Overwriting it would destroy that evidence while leaving the file
            # looking equally well-formed, which is the one failure this whole
            # mechanism exists to make impossible.
            print(
                "refusing to overwrite an existing declaration "
                f"({document['declaration']['candidate_families']}, declared "
                f"{document['declaration']['declared_at']}). Re-declaring after a result "
                "has been read is exactly what DESIGN 4.5 forbids; bump the holdout id and "
                "freeze a new one instead."
            )
            return 1
        unknown = sorted(set(candidates) - set(document["holdout"]["tasks"]))
        if unknown:
            print(f"candidates not present in the frozen holdout: {unknown}")
            return 2
        document["declaration"]["candidate_families"] = candidates
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        document["declaration"]["declared_at"] = stamp
        document["declaration"]["declared_at_commit"] = git_commit()
        args.out.write_text(json.dumps(document, indent=2), encoding="utf-8")
        print_summary(document)
        print(f"\ndeclared {candidates} at {stamp}; content_hash unchanged")
        return 0

    # Derived from the output path: a second holdout written to another file
    # but still announcing `holdout_v1` would leave two content hashes sharing
    # one citation, which is precisely the ambiguity this file exists to remove.
    document = build_document(
        task_ids,
        plan,
        args.start_index,
        args.episodes,
        candidates,
        holdout_id=Path(args.out).stem if args.out else HOLDOUT_ID,
    )
    print_summary(document)

    if not args.write:
        print(f"\ndry run; nothing written. Pass --write to freeze into {args.out}")
        return 0
    if args.out.is_file() and not args.refreeze:
        # Overwriting is possible but never accidental. The value of the file is
        # that it does not change; a tool that silently rewrote it on every run
        # would make the content hash a record of the last invocation.
        print(
            f"\nrefusing to overwrite {args.out}: it is already frozen. Every published "
            "result citing its content_hash refers to the scenes it holds now. Pass "
            "--refreeze if the holdout genuinely must be replaced."
        )
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    if candidates is None:
        print(
            "WARNING: no candidate family declaration recorded. DESIGN 4.5 requires the "
            "candidate set to be declared before any held-out result is read; rerun with "
            "--candidates before evaluating."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
