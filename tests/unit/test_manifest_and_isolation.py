"""Manifests and dependency isolation (PLAN.md sections 2 and 4.1).

Two properties that are easy to claim and easy to break silently:

* a run manifest records everything PLAN section 2 requires -- otherwise it is
  decoration, which is exactly what the old four-key report was;
* the environment never imports the training stack, so evaluation cannot be
  changed by having torch installed.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from factoriorl import manifest as manifest_module
from factoriorl.protocol import PROTOCOL_VERSION


def _manifest(tmp_path, monkeypatch) -> dict:
    from factoriorl import paths

    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    (tmp_path / "mod" / "factoriorl").mkdir(parents=True)
    (tmp_path / "mod" / "factoriorl" / "info.json").write_text(
        '{"version": "0.1.0"}', encoding="utf-8"
    )
    (tmp_path / "mod" / "factoriorl" / "control.lua").write_text("-- x", encoding="utf-8")
    return manifest_module.RunManifest(
        run_id="test-run",
        engine={"version": "2.0.60", "build": 83512},
        task={"id": "navigate", "version": "1.0.0"},
        profiles={"observation": "local-v1", "action": "primitive-v1"},
        reward={"shaping_enabled": True, "components": []},
        seeds={"master": 1},
        model={"algorithm": "MaskablePPO"},
    ).to_dict()


def test_manifest_carries_every_required_field(tmp_path, monkeypatch):
    data = _manifest(tmp_path, monkeypatch)
    missing = [key for key in manifest_module.REQUIRED_FIELDS if not data.get(key)]
    assert not missing, f"manifest is missing PLAN section 2 fields: {missing}"


def test_manifest_records_the_protocol_and_mod_it_was_produced_with(tmp_path, monkeypatch):
    data = _manifest(tmp_path, monkeypatch)
    assert data["protocol"]["version"] == PROTOCOL_VERSION
    assert data["mod"]["source_digest"], "the mod digest pins the code that ran"


def test_manifest_keeps_a_model_slot_even_when_unused(tmp_path, monkeypatch):
    """Present-but-null, so its absence is never read as 'no model'."""
    from factoriorl import paths

    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    (tmp_path / "mod" / "factoriorl").mkdir(parents=True)
    (tmp_path / "mod" / "factoriorl" / "info.json").write_text(
        '{"version": "0.1.0"}', encoding="utf-8"
    )
    data = manifest_module.RunManifest(
        run_id="r",
        engine={"version": "2.0.60"},
        task={"id": "navigate"},
        profiles={},
        reward={},
        seeds={},
    ).to_dict()
    assert "model" in data
    assert data["model"] is None


def test_manifest_records_the_measured_host_not_a_claimed_one():
    """Every document in this repo said RTX 4060; the machine has a 3050.

    The manifest must read the hardware rather than inherit a claim from prose,
    or a published figure carries a false provenance record.
    """
    host = manifest_module.host_info()
    assert host["processor"], "processor is measured"
    assert "gpu" in host, "the GPU block is always present, even without torch"
    gpu = host["gpu"]
    if gpu.get("name"):
        assert gpu["vram_gb"] > 0
        assert gpu["capability"].startswith("sm_")
    else:
        assert gpu.get("reason"), "an absent GPU explains itself"


def test_config_digest_is_stable_and_order_independent():
    a = manifest_module.config_digest({"b": 2, "a": 1})
    b = manifest_module.config_digest({"a": 1, "b": 2})
    c = manifest_module.config_digest({"a": 1, "b": 3})
    assert a == b
    assert a != c


ENV_IMPORT_PROBE = """
import sys
import factoriorl.env  # noqa: F401
import factoriorl.encoders  # noqa: F401
import factoriorl.tasks  # noqa: F401
import factoriorl.cli  # noqa: F401
factoriorl_tasks = __import__("factoriorl.tasks", fromlist=["discover"])
factoriorl_tasks.discover()
banned = [m for m in ("torch", "stable_baselines3", "sb3_contrib") if m in sys.modules]
print(",".join(banned))
"""


def test_the_environment_never_imports_the_training_stack():
    """PLAN 4.1: evaluation must run without training dependencies changing
    environment behaviour. Import isolation is the mechanical form of that."""
    result = subprocess.run(
        [sys.executable, "-c", ENV_IMPORT_PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        # Distinguish "the env imports torch" from "this machine could not run
        # the probe at all"; only the first is a defect in the code under test.
        if "paging file" in result.stderr or "Memory allocation" in result.stderr:
            pytest.skip(f"host under memory pressure: {result.stderr.strip()[:120]}")
        pytest.fail(result.stderr)
    leaked = result.stdout.strip()
    assert not leaked, f"the env path imported training-only modules: {leaked}"


REFERENCE_SOLUTION_PROBE = """
import sys
import factoriorl.learn.train  # noqa: F401
leaked = [m for m in sys.modules if m.endswith("tasks.reference")]
print(",".join(leaked))
"""


@pytest.mark.skipif(
    subprocess.run(
        [sys.executable, "-c", "import torch"], capture_output=True, check=False
    ).returncode
    != 0,
    reason="training stack not installed",
)
def test_training_entrypoint_does_not_import_reference_solutions():
    """PLAN 4.2: no scripted solution labels during ordinary PPO training."""
    result = subprocess.run(
        [sys.executable, "-c", REFERENCE_SOLUTION_PROBE],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not result.stdout.strip()
