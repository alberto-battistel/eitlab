from importlib.util import find_spec

import pytest

from tests.support import load_backend, make_disk, make_model, make_pattern


@pytest.fixture(scope="session")
def backend():
    # Missing optional NGSolve is a skip, not a fake numerical backend.
    if find_spec("ngsolve") is None:
        pytest.skip("NGSolve is not installed; only inverse-layer checks can run")
    # Missing project files or broken imports are errors, rather than silent skips.
    return load_backend()


@pytest.fixture(scope="session")
def disk(backend):
    return make_disk(backend)


@pytest.fixture
def model(backend, disk):
    mesh, electrodes = disk
    return make_model(backend, mesh, electrodes)


@pytest.fixture
def pattern(backend, disk):
    _, electrodes = disk
    return make_pattern(backend, electrodes)
