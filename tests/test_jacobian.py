import numpy as np
import pytest

from tests.support import make_model, make_plan
from eitlab.differentiation import FiniteDifferenceJacobian
from eit_forward.adapters import NGSolveConductivityForward, NGSolveAdjointJacobian

pytestmark = [pytest.mark.fem, pytest.mark.jacobian]


@pytest.mark.parametrize("exclude_current", [False, True])
def test_adjoint_jacobian_matches_central_directional_difference(backend, disk, exclude_current):
    """Check both the parameter map and the complete measurement transform."""
    mesh, electrodes = disk
    parameter_space = backend.ng.L2(mesh, order=0)
    sigma = backend.ng.GridFunction(parameter_space)
    sigma.Set(1.0)
    model = make_model(backend, mesh, electrodes, sigma=sigma)
    plan = make_plan(backend, electrodes, exclude_current)
    runner = backend.measurement.MeasurementRunner(model, plan)
    results = runner.run()
    adjoint = backend.fem.ElectricalLinearization(model, parameter_space=parameter_space)
    J = backend.fem.JacobianAssembler(adjoint, results).assemble_total_jacobian()
    n_measurements = 8 * (6 if exclude_current else 8)
    assert J.shape == (n_measurements, parameter_space.ndof)
    assert np.all(np.isfinite(J))

    rng = np.random.default_rng(17)
    directions = [np.ones(parameter_space.ndof), rng.uniform(-1, 1, parameter_space.ndof)]
    for direction in directions:
        predicted = J @ direction
        assert np.linalg.norm(predicted) > 1e-10
        # Two scales help distinguish a derivative problem from finite-difference roundoff.
        errors = []
        for step in (1e-3, 1e-4):
            values = []
            for sign in (1, -1):
                perturbed = backend.ng.GridFunction(parameter_space)
                perturbed.vec.FV().NumPy()[:] = 1.0 + sign * step * direction
                perturbed_model = make_model(backend, mesh, electrodes, sigma=perturbed)
                values.append(backend.measurement.MeasurementRunner(perturbed_model, plan).run().voltages.ravel())
            finite_difference = (values[0] - values[1]) / (2 * step)
            errors.append(np.linalg.norm(predicted - finite_difference) / np.linalg.norm(predicted))
        assert max(errors) < 5e-4, f"Relative derivative errors at h=1e-3,1e-4: {errors}"


@pytest.mark.parametrize("exclude_current", [False, True])
@pytest.mark.parametrize("scheme", ["forward", "central"])
def test_full_finite_difference_jacobian_matches_adjoint(backend, disk, exclude_current, scheme, monkeypatch):
    mesh, electrodes = disk
    plan = make_plan(backend, electrodes, exclude_current)
    forward = NGSolveConductivityForward(backend, mesh, electrodes, plan)
    # Use a nonuniform reference to check evaluation at the actual parameter vector.
    p = np.linspace(0.8, 1.2, forward.n_parameters)
    before = forward.evaluate(p)
    J_adjoint = NGSolveAdjointJacobian(forward).jacobian(p)
    # Deliberately make symbolic linearization unavailable for the fallback.
    def unavailable_adjoint(*args, **kwargs):
        raise NotImplementedError("Symbolic linearization is unavailable")
    monkeypatch.setattr(backend.fem, "ElectricalLinearization", unavailable_adjoint)
    J_fd = FiniteDifferenceJacobian(forward.evaluate, step=1e-4, scheme=scheme).jacobian(p)
    assert J_fd.shape == J_adjoint.shape
    relative_error = np.linalg.norm(J_fd - J_adjoint) / np.linalg.norm(J_adjoint)
    assert relative_error < (5e-4 if scheme == "forward" else 2e-6), relative_error
    np.testing.assert_allclose(forward.evaluate(p), before, rtol=1e-10, atol=1e-11)
