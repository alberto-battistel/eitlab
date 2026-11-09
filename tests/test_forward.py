import numpy as np
import pytest

from tests.support import make_model, make_pattern

pytestmark = pytest.mark.fem


def test_disk_has_expected_electrodes_and_positive_boundary_measures(backend, model, disk):
    mesh, electrodes = disk
    assert len(electrodes) == len(set(electrodes)) == 8
    assert set(electrodes).issubset(mesh.GetBoundaries())
    lengths = np.array(list(model.electrode_lengths.values()))
    expected = np.pi / len(electrodes)  # radius 1, half the circumference is electrodes
    np.testing.assert_allclose(lengths, expected, rtol=0.02)
    assert np.all(lengths > 0)


def test_voltage_is_linear_in_current(backend, model, pattern, disk):
    _, electrodes = disk
    baseline = model.solve(pattern).electrode_potentials
    assert np.all(np.isfinite(baseline))
    assert np.linalg.norm(baseline) > 1e-8
    for amplitude in (-1.0, 2.0):
        scaled = model.solve(make_pattern(backend, electrodes, amplitude=amplitude))
        np.testing.assert_allclose(scaled.electrode_potentials, amplitude * baseline,
                                   rtol=1e-10, atol=1e-11)


def test_reciprocity_between_two_current_patterns(backend, model, pattern, disk):
    _, electrodes = disk
    other = make_pattern(backend, electrodes, source=4, sink=6)
    U_a = model.solve(pattern).electrode_potentials
    U_b = model.solve(other).electrode_potentials
    I_a = np.array([pattern.current_on(e) for e in electrodes])
    I_b = np.array([other.current_on(e) for e in electrodes])
    np.testing.assert_allclose(I_b @ U_a, I_a @ U_b, rtol=1e-10, atol=1e-11)


def test_numerical_ground_does_not_change_referenced_voltages(backend, model, pattern, disk):
    mesh, electrodes = disk
    other = make_model(backend, mesh, electrodes, ground=electrodes[5])
    np.testing.assert_allclose(model.solve(pattern).electrode_potentials,
                               other.solve(pattern).electrode_potentials,
                               rtol=1e-9, atol=1e-10)


def test_update_sigma_matches_a_fresh_model(backend, model, pattern, disk):
    mesh, electrodes = disk
    original = model.solve(pattern).electrode_potentials.copy()
    model.set_sigma(1.7)
    updated = model.solve(pattern).electrode_potentials
    fresh = make_model(backend, mesh, electrodes, sigma=1.7).solve(pattern)
    np.testing.assert_allclose(updated, fresh.electrode_potentials, rtol=1e-10, atol=1e-11)
    assert np.linalg.norm(updated - original) > 1e-4 * np.linalg.norm(original)


@pytest.mark.regression
def test_set_sigma_then_solve_uses_the_new_conductivity(backend, model, pattern, disk):
    mesh, electrodes = disk
    model.solve(pattern)  # populate the cached factorization
    model.set_sigma(1.7)
    actual = model.solve(pattern).electrode_potentials
    expected = make_model(backend, mesh, electrodes, sigma=1.7).solve(pattern).electrode_potentials
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
