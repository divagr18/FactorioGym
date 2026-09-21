"""R5.3's third arm: PPO started from another policy's parameters.

The comparison R5.3 asks for is scratch PPO against behaviour cloning against
BC-initialised PPO, and the third did not exist -- `train.py` had no
`--init-from`. The books make it the deciding arm rather than merely a nice
one: BC's training data is drawn from the expert's own state distribution, so
no number of demonstrations covers the states a cloned policy then visits.
Our own measurement says the same thing empirically -- 60 to 400 demonstrations
moved expert-action accuracy 0.79 to 0.90 and left structural transfer at
exactly 0.00 both times. RL on top is one of the two remedies, and the other
(DAgger, SMILe) is blocked on a queryable expert we do not have.

Four properties are worth pinning, and each corresponds to a way this could be
wrong while appearing to work.

**Only the policy transfers.** `MaskablePPO.load` would bring the optimizer
too, and BC's Adam moments were accumulated under a supervised cross-entropy
loss. Carrying those into a policy-gradient objective conditions the first PPO
updates on gradients of a different loss, which is not what "start from these
parameters" means.

**A mismatch is refused.** `load_state_dict(strict=False)` would leave every
mismatched layer at its random initialisation and produce a policy that is
neither scratch nor cloned, with a result attributable to nothing.

**The load has to actually do something.** Equal weight digests before and
after mean the flag was a no-op and the run is a scratch run wearing a label,
which is worse than an error because it publishes.

**The run says what it is.** DESIGN 4.2 forbids scripted-solution labels during
ordinary PPO training. This arm inherits parameters fitted on exactly those
labels, so it is not comparable to a scratch run, and that is only enforceable
if the manifest records it.

These use the three declared `deliver` checkpoints already on disk rather than
training anything, so no engine and no run is involved. They skip when those
checkpoints are absent -- the declaration is committed but `runtime/runs/` is
gitignored, so a fresh clone has the list and not the files.
"""

from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "docs" / "evidence" / "phase4-checkpoints.json"

pytest.importorskip("sb3_contrib")


def declared() -> list[pathlib.Path]:
    from factoriorl import manifest as manifest_module

    if not DECLARATION.is_file():
        return []
    entries = json.loads(DECLARATION.read_text(encoding="utf-8")).get("checkpoints") or []
    paths = [
        manifest_module.runs_dir() / entry["run_id"] / "model.zip"
        for entry in entries
        if entry.get("run_id")
    ]
    return [path for path in paths if path.is_file()]


@pytest.fixture(scope="module")
def checkpoints():
    found = declared()
    if len(found) < 2:
        pytest.skip(
            "needs at least two declared checkpoints on disk; runtime/runs/ is "
            "gitignored, so a fresh clone has the declaration and not the files"
        )
    return found


@pytest.fixture
def model(checkpoints):
    """A live model with the declared architecture, built without an engine.

    Loading a declared checkpoint gives one for free and avoids constructing a
    vec env: the architecture is what matters here, not the environment.

    Function-scoped deliberately. These tests mutate the policy's weights, so a
    module-scoped model would make `test_the_weights_actually_transfer` pass
    only because it happens to run before the others -- it asserts a change
    against weights an earlier test may already have overwritten.
    """
    from sb3_contrib import MaskablePPO

    return MaskablePPO.load(checkpoints[0], device="cpu")


def test_the_weights_actually_transfer(model, checkpoints):
    """The property that makes the flag meaningful. A no-op load would publish
    a scratch run under an initialised label."""
    from factoriorl.learn.policy import initialise_from

    record = initialise_from(model, checkpoints[1])
    assert record["weights_changed"] is True
    assert record["policy_weights_digest_before"] != record["policy_weights_digest_after"]


def test_loading_the_same_checkpoint_twice_is_detected_as_a_no_op(model, checkpoints):
    """Initialising from the weights already in the model changes nothing, and
    the record must say so rather than reporting success. `train` turns this
    into a refusal."""
    from factoriorl.learn.policy import initialise_from

    initialise_from(model, checkpoints[1])
    again = initialise_from(model, checkpoints[1])
    assert again["weights_changed"] is False
    assert again["policy_weights_digest_before"] == again["policy_weights_digest_after"]


def test_the_digest_after_equals_the_source(model, checkpoints):
    """A load that transferred *something* is not enough -- it has to transfer
    the source's parameters. Initialising two different models from one
    checkpoint must land them both on the same weights."""
    from sb3_contrib import MaskablePPO

    from factoriorl.learn.policy import _digest_values, initialise_from

    initialise_from(model, checkpoints[1])
    other = MaskablePPO.load(checkpoints[0], device="cpu")
    initialise_from(other, checkpoints[1])
    assert _digest_values(model.policy.state_dict()) == _digest_values(other.policy.state_dict())


def test_a_missing_checkpoint_is_a_clear_error(model, tmp_path):
    from factoriorl.learn.policy import initialise_from

    with pytest.raises(FileNotFoundError, match="no checkpoint to initialise from"):
        initialise_from(model, tmp_path / "absent.zip")


def test_an_architecture_mismatch_is_refused_with_both_signatures(model, tmp_path):
    """`strict=False` would leave mismatched layers randomly initialised and the
    result attributable to neither arm, so the refusal names both signatures
    instead."""
    import io
    import zipfile

    import torch

    from factoriorl.learn.policy import initialise_from

    # A checkpoint whose policy.pth has one parameter of the wrong shape. Built
    # by rewriting a real archive, so everything else about it is genuine.
    source = declared()[0]
    forged = tmp_path / "forged.zip"
    with zipfile.ZipFile(source) as original:
        state = torch.load(
            io.BytesIO(original.read("policy.pth")), map_location="cpu", weights_only=True
        )
        key = next(iter(state))
        state[key] = torch.zeros(state[key].shape[0] + 1, *state[key].shape[1:])
        blob = io.BytesIO()
        torch.save(state, blob)
        with zipfile.ZipFile(forged, "w") as out:
            for item in original.infolist():
                if item.filename == "policy.pth":
                    out.writestr("policy.pth", blob.getvalue())
                else:
                    out.writestr(item, original.read(item.filename))

    with pytest.raises(ValueError, match="different\\s+architecture"):
        initialise_from(model, forged)


def test_the_record_names_what_it_is_not_comparable_to(model, checkpoints):
    """DESIGN 4.2's prohibition is on the training signal, and this arm inherits
    parameters fitted on exactly the labels it forbids. The arm is legitimate
    and must not share a column with scratch PPO, which only the label
    enforces."""
    from factoriorl.learn.policy import INITIALISED_MODE, initialise_from

    record = initialise_from(model, checkpoints[1])
    assert record["training_mode"] == INITIALISED_MODE
    assert "scratch PPO" in record["not_comparable_to"]
    assert "4.2" in record["not_comparable_to"]


def test_the_record_says_the_optimizer_is_fresh(model, checkpoints):
    """The distinction from `MaskablePPO.load`, recorded where a reader of the
    manifest will see it."""
    from factoriorl.learn.policy import initialise_from

    record = initialise_from(model, checkpoints[1])
    assert record["optimizer_state"].startswith("fresh")
    assert "supervised loss" in record["optimizer_state"]


def test_a_zipless_path_is_accepted(model, checkpoints):
    """`model.save(dir / "model")` writes `model.zip`, so callers naturally
    hold the extensionless path SB3 gave them."""
    from factoriorl.learn.policy import initialise_from

    record = initialise_from(model, checkpoints[1].with_suffix(""))
    assert record["initialised_from"] == checkpoints[1].name


def test_the_config_records_the_flag_so_a_scratch_run_is_distinguishable():
    """An unrecorded `init_from` makes an initialised run indistinguishable
    from a scratch one in the manifest, which is the provenance failure the
    whole label exists to prevent."""
    from factoriorl.learn.train import TrainConfig

    assert TrainConfig(task_id="deliver").to_dict()["init_from"] is None
    assert (
        TrainConfig(task_id="deliver", init_from="runs/x/model.zip").to_dict()["init_from"]
        == "runs/x/model.zip"
    )
