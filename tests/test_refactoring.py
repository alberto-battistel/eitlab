"""Lifecycle, observation snapshots and independent module-boundary checks."""
from dataclasses import replace
import subprocess
import sys

import numpy as np
import pytest

from tests.support import make_model, make_plan


@pytest.mark.fem
def test_discrete_power_equals_specified_electrode_current_work(backend, model, pattern):
    """Curved-boundary quadrature must not rescale the prescribed total currents."""
    solution = model.solve(pattern)
    energy = backend.ng.InnerProduct(solution.gfu.vec, model.a.mat * solution.gfu.vec)
    currents = np.array([pattern.current_on(name) for name in model.electrode_names])
    power = currents @ solution.raw_electrode_potentials
    np.testing.assert_allclose(energy, power, rtol=1e-10, atol=1e-12)


@pytest.mark.fem
def test_updates_keep_spaces_and_reject_old_states_and_linearizations(backend, model, pattern):
    solution = model.solve(pattern)
    linearization = backend.fem.ElectricalLinearization(model)
    rhs = model.make_electrode_rhs({model.electrode_names[1]: 1})
    assert linearization.invAT is model.inverse_operator
    old_space, old_revision = model.fes, model.revision
    model.set_sigma(1.3)
    assert model.fes is old_space
    assert model.revision > old_revision
    with pytest.raises(ValueError, match="stale"):
        linearization.solve_adjoint([rhs])
    with pytest.raises(ValueError, match="stale"):
        model.validate_solution(solution)
    fresh_linearization = backend.fem.ElectricalLinearization(model)
    with pytest.raises(ValueError, match="stale"):
        fresh_linearization.assemble_parameter_jacobian(solution, [])


@pytest.mark.fem
def test_sigma_input_is_owned_and_failed_update_is_atomic(backend, model, pattern, disk):
    mesh, _ = disk
    space = backend.ng.L2(mesh, order=0)
    field = backend.ng.GridFunction(space)
    field.Set(1.2)
    model.set_sigma(field)
    expected = model.solve(pattern).electrode_potentials
    field.vec[:] = 2.4
    np.testing.assert_allclose(model.solve(pattern).electrode_potentials, expected, atol=1e-12)
    revision = model.revision
    with pytest.raises(ValueError):
        model.set_sigma(-1)
    assert model.revision == revision
    np.testing.assert_allclose(model.solve(pattern).electrode_potentials, expected, atol=1e-12)


@pytest.mark.fem
def test_reference_override_does_not_change_later_calls(backend, model, pattern):
    expected = model.solve(pattern).electrode_potentials
    override = model.solve(pattern, backend.fem.PotentialReference.FIRST_ELECTRODE)
    assert abs(override.electrode_potentials[0]) < 1e-12
    assert model.potential_reference is backend.fem.PotentialReference.MEAN_ZERO
    np.testing.assert_allclose(model.solve(pattern).electrode_potentials, expected, atol=1e-12)


@pytest.mark.fem
def test_variable_observation_layout_and_optional_states(backend, model, disk):
    _, names = disk
    P = backend.stimulation.CurrentPattern
    patterns = [P({names[0]: 1, names[1]: -1}),
                P({names[0]: .5, names[1]: .5, names[2]: -.5, names[3]: -.5})]
    plan = backend.measurement.MeasurementPlan(patterns, backend.measurement.SingleEndedNoCurrent)
    results = backend.measurement.MeasurementRunner(model, plan).run(store_solutions=False)
    assert results.vector.shape == (10,)
    np.testing.assert_array_equal(results.offsets, [0, 6, 10])
    assert len(results.labels) == 10
    assert not results.solutions
    with pytest.raises(ValueError, match="Variable channel"):
        _ = results.voltages
    with pytest.raises(ValueError, match="store_solutions"):
        backend.fem.JacobianAssembler(backend.fem.ElectricalLinearization(model), results)


@pytest.mark.fem
def test_observation_snapshot_survives_protocol_edits(backend, model, disk):
    _, names = disk
    runner = backend.measurement.MeasurementRunner(model, make_plan(backend, names, True))
    results = runner.run()
    original = results.vector.copy()
    runner.measurement_protocol.current_tolerance = 2.0
    J = backend.fem.JacobianAssembler(backend.fem.ElectricalLinearization(model), results).assemble_total_jacobian()
    assert J.shape[0] == 48
    np.testing.assert_array_equal(results.vector, original)
    assert runner.run().vector.shape == (64,)
    with pytest.raises(ValueError):
        results.voltage_blocks[0][0] = 99


@pytest.mark.fem
def test_result_validation_rejects_other_model_and_missing_states(backend, model, disk):
    mesh, names = disk
    results = backend.measurement.MeasurementRunner(model, make_plan(backend, names)).run()
    other = make_model(backend, mesh, names)
    with pytest.raises(ValueError, match="another model"):
        backend.fem.JacobianAssembler(backend.fem.ElectricalLinearization(other), results).assemble_total_jacobian()
    with pytest.raises(ValueError, match="one stored state"):
        replace(results, solutions=results.solutions[:-1])


@pytest.mark.fem
def test_numeric_linearization_survives_later_evaluations(backend, disk):
    from eit_forward.adapters import NGSolveConductivityForward
    mesh, names = disk
    forward = NGSolveConductivityForward(backend, mesh, names, make_plan(backend, names, True))
    p = np.ones(forward.n_parameters)
    snapshot = forward.linearize(p)
    numerical = forward.linearize(p, method="central")
    np.testing.assert_allclose(snapshot.matrix, numerical.matrix, rtol=2e-5, atol=1e-8)
    forward.evaluate(p * 1.5)
    direction = np.linspace(-.2, .3, len(p))
    weights = np.linspace(0, 1, len(snapshot.prediction))
    np.testing.assert_allclose(weights @ snapshot.jvp(direction), snapshot.vjp(weights) @ direction)
    np.testing.assert_array_equal(snapshot.reference_parameters, p)
    assert not snapshot.matrix.flags.writeable


@pytest.mark.fem
def test_geometry_edits_invalidate_mesh_and_rebuild_labels(backend):
    geometry = backend.geometry.ElectrodeDisk(radius=1, n_electrodes=4)
    old_mesh = geometry.make_mesh(maxh=.5)
    geometry.add_disk_inclusion(name="target", p=(0, 0), r=.2)
    assert geometry.geo is None and geometry.mesh is None
    updated = geometry.make_mesh(maxh=.5)
    assert updated is not old_mesh
    assert "target" in updated.GetMaterials()
    assert len(geometry.electrode_names) == 4


@pytest.mark.fem
def test_finite_difference_on_duplicate_preserves_original_model(backend, model, disk):
    _, names = disk
    plan = make_plan(backend, names, True)
    results = backend.measurement.MeasurementRunner(model, plan).run()
    J = backend.fem.JacobianAssembler(backend.fem.ElectricalLinearization(model), results).assemble_total_jacobian()
    revision = model.revision
    from eitlab import ElementwiseConductivity, FiniteDifferenceJacobian
    mapping = ElementwiseConductivity(model.mesh)
    baseline = mapping.from_coefficient(model.sigma_cf)
    working = model.duplicate()
    def evaluate(parameters):
        working.set_sigma(mapping.create(parameters))
        return backend.measurement.MeasurementRunner(working, plan).run(store_solutions=False).vector
    numerical = FiniteDifferenceJacobian(evaluate, step=1e-4, scheme="central").jacobian(baseline)
    np.testing.assert_allclose(numerical, J, rtol=2e-5, atol=1e-8)
    assert model.revision == revision


def test_stimulation_snapshots_and_protocol_subtraction():
    from eit_forward.stimulation import CurrentPattern, CurrentProtocol, ProtocolFactory
    values = {"a": 1., "b": -1.}
    first = CurrentPattern(values)
    values["a"] = 2.
    assert first.current_on("a") == 1.
    second = CurrentPattern({"b": 1., "c": -1.})
    full = CurrentProtocol([first, second])
    reduced = full - CurrentProtocol([second])
    assert list(reduced) == [first] and list(full) == [first, second]
    assert len(ProtocolFactory.jump_and_skip(["a", "b", "c"], "a")) == 3
    with pytest.raises(ValueError, match="positive"):
        ProtocolFactory.jump_and_skip(["a", "b", "c"], "a", jump=0)


def test_inverse_and_observations_import_without_ngsolve():
    program = '''
import sys
import eit_inverse
import eitlab
from eitlab import TikhonovSolver, CurrentPattern, PotentialReference
import eitlab.differentiation
import eit_forward.stimulation
from eit_forward.observations import ElectrodeObservation
assert "ngsolve" not in sys.modules
assert "FEM" not in sys.modules
assert "eit_forward.adapters" not in sys.modules
'''
    subprocess.run([sys.executable, "-c", program], check=True)


def test_tikhonov_zero_penalty_rank_deficiency():
    from eit_inverse import TikhonovSolver
    J = np.array([[1., 0, 0], [2., 0, 0]])
    y = np.array([1., 2.])
    np.testing.assert_allclose(TikhonovSolver(0, J).solve(y), np.linalg.lstsq(J, y, rcond=None)[0], atol=1e-12)


@pytest.mark.fem
def test_rewritten_single_circle_pipeline():
    from examples._single_circle import ExampleConfig, run
    config = ExampleConfig(n_electrodes=8, order=1, maxh_jacobian=.45, maxh_forward=.3)
    result = run(config)
    J, x, y = result["J"], result["delta_sigma_estimate"], result["delta_y"]
    assert J.shape == (64, len(x))
    assert np.any(result["delta_sigma_true_at_centers"] < 0)
    assert np.linalg.norm(result["V_inho"] - result["V_homo"]) > 1e-4
    np.testing.assert_allclose(J.T @ (J @ x - y) + config.regularization_lambda * x,
                               np.zeros_like(x), atol=1e-10)
