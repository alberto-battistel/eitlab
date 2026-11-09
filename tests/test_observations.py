import numpy as np
import pytest

from tests.support import make_plan

pytestmark = pytest.mark.fem


@pytest.mark.parametrize("reference_name", ["MEAN_ZERO", "FIRST_ELECTRODE", "CURRENT_ELECTRODES_MEAN"])
def test_voltage_reference_preserves_all_pairwise_differences(backend, reference_name):
    U = np.array([1.7, -0.2, 0.4, -1.1])
    currents = {"el_0": 1.0, "el_1": 0.0, "el_2": -1.0, "el_3": 0.0}
    reference = getattr(backend.fem.PotentialReference, reference_name)
    referenced = reference.apply(U, currents=currents)
    np.testing.assert_allclose(referenced[:, None] - referenced[None, :],
                               U[:, None] - U[None, :], atol=1e-14)
    if reference_name == "MEAN_ZERO":
        assert abs(referenced.mean()) < 1e-14
    elif reference_name == "FIRST_ELECTRODE":
        assert abs(referenced[0]) < 1e-14
    else:
        assert abs((referenced[0] + referenced[2]) / 2) < 1e-14


@pytest.mark.parametrize("exclude_current", [False, True])
def test_measurement_values_and_flattening_match_injection_order(backend, model, disk, exclude_current):
    _, electrodes = disk
    plan = make_plan(backend, electrodes, exclude_current)
    runner = backend.measurement.MeasurementRunner(model, plan)
    results = runner.run()
    expected = []
    for current in plan.current_protocol:
        U = model.solve(current).electrode_potentials
        if exclude_current:
            # Independent selection, without reusing the protocol matrix under test.
            U = U[[i for i, e in enumerate(electrodes) if current.current_on(e) == 0]]
        expected.append(U)
    expected = np.asarray(expected)
    assert results.voltages.shape == (8, 6 if exclude_current else 8)
    np.testing.assert_allclose(results.voltages.ravel(), expected.ravel(), atol=1e-11)


@pytest.mark.regression
def test_current_reference_ignores_currents_below_tolerance(backend):
    U = np.array([1.0, 9.0, -1.0, 3.0])
    currents = {"el_0": 1.0, "el_1": 1e-14, "el_2": -1.0, "el_3": 0.0}
    referenced = backend.fem.PotentialReference.CURRENT_ELECTRODES_MEAN.apply(
        U, currents=currents, current_tolerance=1e-12,
    )
    np.testing.assert_allclose(referenced, U, atol=1e-14)
