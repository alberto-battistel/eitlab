import numpy as np
import pytest

from eit_inverse import triangle_regularization_operators


def test_physical_l2_and_tv_on_two_triangles():
    points = np.array([[0, 0], [2, 0], [0, 1], [2, 1]])
    cells = np.array([[0, 1, 2], [1, 3, 2]])
    mass, jumps = triangle_regularization_operators(points, cells)
    # Each triangle has area 1; the shared diagonal has length sqrt(5).
    values = np.array([2., 5.])
    assert np.linalg.norm(mass @ values)**2 == pytest.approx(29.)
    assert np.abs(jumps @ values).sum() == pytest.approx(3*np.sqrt(5))
    np.testing.assert_allclose(jumps @ np.ones(2), 0)
    # Reversing parameter order preserves both physical penalties.
    m2, d2 = triangle_regularization_operators(points, cells[::-1])
    assert np.abs(d2 @ values[::-1]).sum() == pytest.approx(3*np.sqrt(5))
    np.testing.assert_allclose(m2 @ values[::-1], (mass @ values)[::-1])


def test_single_triangle_has_no_boundary_tv():
    mass, jumps = triangle_regularization_operators(
        [[0, 0], [2, 0], [0, 3]], np.array([[0, 1, 2]]))
    assert mass[0, 0]**2 == pytest.approx(3)
    assert jumps.shape == (0, 1)


@pytest.mark.parametrize('points,cells', [
    ([[0, 0], [1, 0], [2, 0]], [[0, 1, 2]]),
    ([[0, 0], [1, 0], [0, 1]], [[0, 1, 3]]),
    ([[0, 0], [1, 0], [0, 1]], [[0., 1., 2.]]),
    ([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]]*3),
])
def test_invalid_mesh_rejected(points, cells):
    with pytest.raises(ValueError):
        triangle_regularization_operators(points, np.array(cells))
