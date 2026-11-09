import numpy as np
import pytest

from eit_inverse import TikhonovSolver


@pytest.mark.parametrize("shape", [(12, 4), (4, 12)])
def test_tikhonov_satisfies_regularized_normal_equation(shape):
    """Check the defining optimality condition, including underdetermined data."""
    rng = np.random.default_rng(11)
    J = rng.normal(size=shape)
    y = rng.normal(size=shape[0])
    lam = 0.03
    x = TikhonovSolver(lam, J).solve(y)
    residual = J.T @ (J @ x - y) + lam * x
    scale = max(np.linalg.norm(J.T @ y), 1.0)
    assert np.linalg.norm(residual) / scale < 1e-12


def test_tikhonov_multiple_lambdas_have_correct_order_and_shape():
    J = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    y = np.array([0.4, -0.2, 0.3])
    lambdas = [0.01, 0.1, 1.0]
    solutions = TikhonovSolver(lambdas, J).solve(y)
    assert solutions.shape == (2, 3)
    for column, lam in enumerate(lambdas):
        expected = np.linalg.solve(J.T @ J + lam * np.eye(2), J.T @ y)
        np.testing.assert_allclose(solutions[:, column], expected, atol=1e-12)
