import numpy as np
import pytest

from eitlab.differentiation import FiniteDifferenceJacobian


@pytest.mark.parametrize("scheme", ["forward", "backward", "central"])
@pytest.mark.parametrize("step", [1e-5, np.array([1e-5, 2e-5])])
def test_finite_difference_matches_analytic_nonlinear_derivative(scheme, step):
    def forward(p):
        return np.array([p[0]**2 + np.sin(p[1]), np.exp(p[0]) + p[1]**3])
    p = np.array([0.0, 0.4])  # Includes a zero-valued parameter.
    exact = np.array([[2 * p[0], np.cos(p[1])], [np.exp(p[0]), 3 * p[1]**2]])
    actual = FiniteDifferenceJacobian(forward, step=step, scheme=scheme).jacobian(p)
    tolerance = 5e-9 if scheme == "central" else 3e-5
    np.testing.assert_allclose(actual, exact, atol=tolerance, rtol=tolerance)


def test_finite_difference_handles_complex_measurements_of_real_parameters():
    def forward(p):
        return np.array([p[0] + 1j * p[1], np.exp(1j * p[0]) + p[1]**2])
    p = np.array([0.3, -0.4])
    exact = np.array([[1, 1j], [1j * np.exp(1j * p[0]), 2 * p[1]]])
    J = FiniteDifferenceJacobian(forward, step=1e-5).jacobian(p)
    np.testing.assert_allclose(J, exact, rtol=1e-9, atol=1e-9)
    assert np.iscomplexobj(J)


@pytest.mark.parametrize("scheme", ["forward", "central"])
def test_finite_difference_does_not_mutate_inputs_or_reuse_callback_buffers(scheme):
    buffer = np.zeros(2)
    calls = []
    def forward(p):
        calls.append(p.copy())
        buffer[:] = [2 * p[0] + p[1], -p[0] + 3 * p[1]]
        p[:] = 999  # Simulate a callback that modifies its own input.
        return buffer
    p = np.array([0.2, 0.4])
    original = p.copy()
    J = FiniteDifferenceJacobian(forward, step=1e-4, scheme=scheme).jacobian(p)
    np.testing.assert_allclose(J, [[2, 1], [-1, 3]], atol=1e-10)
    np.testing.assert_array_equal(p, original)
    assert len(calls) == (5 if scheme == "central" else 3)


def test_finite_difference_rejects_changing_measurement_layout():
    def forward(p):
        return np.ones(2 if p[0] <= 0 else 3)
    with pytest.raises(ValueError, match="shape changed"):
        FiniteDifferenceJacobian(forward).jacobian([0.0])
