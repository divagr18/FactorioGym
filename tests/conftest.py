"""Shared pytest fixtures.

Engine tests need a real Factorio binary. When it is absent they are skipped
with a clear reason -- but DESIGN.md section 4 is explicit that "tests skipped
because Factorio was unavailable" is an *incomplete* state for engine-dependent
work, so the phase gates run with ``--require-engine``, which turns that skip
into a failure. A gate can then never pass by quietly skipping its evidence.

Workers are expensive (several hundred MB commit each); fixture tests share
one worker per module. Tests that kill or restart workers get their own.
"""

from __future__ import annotations

import os

import pytest

from factoriorl.engine_config import resolve_engine_config
from factoriorl.errors import FactorioRLError


def pytest_addoption(parser):
    parser.addoption(
        "--require-engine",
        action="store_true",
        default=False,
        help="fail (instead of skip) engine tests when Factorio is unavailable",
    )


def _engine_required(config) -> bool:
    return bool(config.getoption("--require-engine") or os.environ.get("FACTORIORL_REQUIRE_ENGINE"))


@pytest.fixture(scope="session")
def engine(request):
    try:
        return resolve_engine_config()
    except FactorioRLError as exc:
        if _engine_required(request.config):
            pytest.fail(f"engine required but unavailable: {exc}")
        pytest.skip(f"Factorio unavailable: {exc}")


@pytest.fixture(scope="session")
def worker_manager(engine):
    from factoriorl.worker import WorkerManager

    return WorkerManager(engine=engine)


def _worker_id(request) -> str:
    """Per-test worker id, unique to this run.

    Worker directories survive the run that created them; an orphaned engine
    keeps holding its write-data lock. Reusing a fixed id makes a fresh worker
    fail on a stale lock and misreport why.
    """
    name = request.node.name[:24].replace("_", "-").replace("[", "-").replace("]", "")
    return f"t-{name}-{os.getpid()}"


@pytest.fixture(scope="module")
def module_worker(worker_manager, request):
    """One worker for a whole module of non-destructive tests."""
    worker_id = f"m-{request.module.__name__.split('.')[-1][:20]}-{os.getpid()}"
    handle = worker_manager.launch(worker_id)
    yield handle
    worker_manager.cleanup(handle)


@pytest.fixture(scope="module")
def module_session(module_worker):
    from factoriorl.session import WorkerSession

    session = WorkerSession(module_worker)
    session.status()
    yield session
    session.close()


@pytest.fixture()
def live_worker(worker_manager, request):
    """One worker per test; for tests that kill or restart their worker."""
    handle = worker_manager.launch(_worker_id(request))
    yield handle
    worker_manager.cleanup(handle)


@pytest.fixture()
def live_session(live_worker):
    from factoriorl.session import WorkerSession

    session = WorkerSession(live_worker)
    session.status()
    yield session
    session.close()
