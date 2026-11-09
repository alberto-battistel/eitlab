"""Independent optimality, backend agreement and geometric conservation."""
import numpy as np
import pytest
from scipy import sparse

from eit_inverse import (DataSpace, Regularizer, InverseProblem, SciPySolver,
                         CVXPYSolver, grid_operator, ParameterSpace,
                         triangle_pixel_space, MappedForward)


def cvx_solver():
    pytest.importorskip("cvxpy")
    return CVXPYSolver(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("backend_name", ["scipy", "cvxpy"])
def test_weighted_l2_optimality_with_prior(order, backend_name):
    rng = np.random.default_rng(41)
    A = rng.normal(size=(14, 8))
    b, prior = rng.normal(size=14), rng.normal(size=8)
    L = grid_operator((8,), order, spacing=.7)
    ds = DataSpace(14, standard_deviation=np.linspace(.5, 1.4, 14))
    problem = InverseProblem(A, b, [Regularizer(L, .3, "l2", prior)], data_space=ds)
    solver = SciPySolver() if backend_name == "scipy" else cvx_solver()
    x = solver.solve(problem).parameters
    gradient = 2*(problem.A.T @ (problem.A @ x-problem.b)) + .6*(L.T @ (L @ (x-prior)))
    assert np.linalg.norm(gradient) < 1e-6


@pytest.mark.parametrize("backend_name", ["scipy", "cvxpy"])
def test_l1_matches_exact_soft_threshold_with_prior_and_bounds(backend_name):
    b = np.array([-2., -.1, .5, 3.])
    prior = np.array([.2, .3, -.4, .1])
    weight = .6
    expected = prior + np.sign(b-prior)*np.maximum(np.abs(b-prior)-weight/2, 0)
    expected = np.clip(expected, -.8, 1.2)
    p = InverseProblem(np.eye(4), b, [Regularizer(np.eye(4), weight, "l1", prior)], bounds=(-.8, 1.2))
    solver = SciPySolver(abs_tol=1e-9, rel_tol=1e-8) if backend_name == "scipy" else cvx_solver()
    result = solver.solve(p)
    np.testing.assert_allclose(result.parameters, expected, atol=2e-6)
    assert result.status == "optimal"


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("norm", ["l1", "l2"])
def test_backends_agree_for_six_2d_regularizers(order, norm):
    rng = np.random.default_rng(63)
    A, b = rng.normal(size=(24, 12)), rng.normal(size=24)
    L = grid_operator((3, 4), order, spacing=(.8, 1.2))
    p = InverseProblem(A, b, [Regularizer(L, .4, norm)], bounds=(-.2, .8))
    a = SciPySolver(abs_tol=1e-9, rel_tol=1e-8).solve(p)
    c = cvx_solver().solve(p)
    np.testing.assert_allclose(a.parameters, c.parameters, atol=2e-5)
    assert abs(a.objective-c.objective) < 1e-5


def test_mixed_l1_l2_regularization_and_correlated_whitening():
    rng = np.random.default_rng(3)
    A, b = rng.normal(size=(12, 7)), rng.normal(size=12)
    W = np.eye(12)+np.diag(np.full(11, .15), 1)
    regs = [Regularizer(np.eye(7), .1),
            Regularizer(grid_operator((7,), 1), .3, "l1", prior=np.linspace(0, .4, 7))]
    p = InverseProblem(A, b, regs, data_space=DataSpace(12, whitening=W))
    a, c = SciPySolver().solve(p), cvx_solver().solve(p)
    np.testing.assert_allclose(a.parameters, c.parameters, atol=2e-5)


def test_admm_reports_nonconvergence():
    p = InverseProblem(np.eye(4), np.arange(4.), [Regularizer(np.eye(4), .1, "l1")])
    with pytest.raises(RuntimeError, match="did not converge"):
        SciPySolver(max_iter=1, abs_tol=1e-12, rel_tol=1e-12).solve(p)


def test_grid_derivatives_have_expected_nullspaces_and_scale():
    shape = (4, 5)
    hy, hx = .3, .7
    Y, X = np.meshgrid(np.arange(shape[0])*hy, np.arange(shape[1])*hx, indexing="ij")
    G = grid_operator(shape, 1, spacing=(hy, hx))
    H = grid_operator(shape, 2, spacing=(hy, hx))
    np.testing.assert_allclose(G @ np.ones(np.prod(shape)), 0, atol=1e-14)
    np.testing.assert_allclose(H @ (2*X+3*Y+1).ravel(), 0, atol=1e-13)
    second = H @ (X**2+Y**2+X*Y).ravel()
    # yy rows then xx rows, followed by mixed rows; no boundary constraints.
    pure_count = (shape[0]-2)*shape[1] + shape[0]*(shape[1]-2)
    np.testing.assert_allclose(second[:pure_count], 2, atol=1e-13)
    np.testing.assert_allclose(second[pure_count:], np.sqrt(2), atol=1e-13)


def test_mask_does_not_connect_across_holes_or_penalize_constant_boundaries():
    mask = np.array([[True, False, True], [True, False, True], [True, False, True]])
    G, H = [grid_operator(mask.shape, o, mask=mask) for o in [1, 2]]
    # Independent constant values in two disconnected columns remain unpenalized.
    x = np.array([1, 7, 1, 7, 1, 7.])
    np.testing.assert_allclose(G @ x, 0)
    np.testing.assert_allclose(H @ x, 0)


TRIANGLES = np.array([[[0., 0.], [1., 0.], [1., 1.]],
                      [[0., 0.], [1., 1.], [0., 1.]]])


def test_overlap_mapping_matches_analytic_areas_and_mass_adjoint():
    s = triangle_pixel_space(TRIANGLES, [0, .25, 1], [0, 1])
    np.testing.assert_allclose(s.P.toarray(), [[.0625, .9375], [.4375, .5625]])
    np.testing.assert_allclose(s.R.toarray(), [[.125, .875], [.625, .375]])
    np.testing.assert_allclose(s.P @ np.ones(s.ndof), 1)
    np.testing.assert_allclose(s.R @ np.ones(s.n_fem), 1)
    np.testing.assert_allclose((sparse.diags(s.pixel_areas) @ s.R).toarray(),
                               (s.P.T @ sparse.diags(s.fem_areas)).toarray())
    assert not np.allclose(s.R.toarray(), s.P.T.toarray())
    assert not np.allclose((s.R @ s.P).toarray(), np.eye(s.ndof))
    np.testing.assert_allclose(s.fem_areas @ (s.P @ [2., 5.]), s.pixel_areas @ [2., 5.])


def test_inactive_pixels_offsets_and_grid_coverage():
    s = triangle_pixel_space(TRIANGLES, [-1, 0, 1, 2], [-1, 0, 1, 2], offset=[1, 2])
    assert s.ndof == 1
    assert s.active_indices.tolist() == [4]
    np.testing.assert_allclose(s.to_fem([3]), [4, 5])
    np.testing.assert_allclose(s.from_fem([4, 5]), [3])
    assert s.to_image([3])[1, 1] == 3
    assert np.isnan(s.to_image([3])[0, 0])
    with pytest.raises(ValueError, match="does not cover"):
        triangle_pixel_space(TRIANGLES, [0, .5], [0, 1])


def test_mapped_forward_chain_rule_and_transpose_action():
    from eitlab.differentiation import FiniteDifferenceJacobian
    class Nonlinear:
        def evaluate(self, p):
            return np.array([p[0]**2+p[1], np.exp(.1*p[1])])
    raw = Nonlinear()
    s = ParameterSpace([[1, .2, 0], [0, .3, 1]], offset=[1, 1])
    fine_provider = FiniteDifferenceJacobian(raw.evaluate, scheme="central", step=1e-5)
    f = MappedForward(raw, fine_provider, s)
    x, v, w = np.array([.1, .2, .3]), np.array([1., 2., -1.]), np.array([3., -2.])
    lin = f.linearize(x)
    eps = 1e-5
    np.testing.assert_allclose(lin.jvp(v), (f.evaluate(x+eps*v)-f.evaluate(x-eps*v))/(2*eps), rtol=1e-8)
    np.testing.assert_allclose(w @ lin.jvp(v), lin.vjp(w) @ v)


def test_input_errors_reject_complex_and_mismatched_dimensions():
    with pytest.raises(TypeError):
        InverseProblem(np.eye(2)*1j, np.zeros(2))
    with pytest.raises(ValueError):
        InverseProblem(np.eye(2), np.zeros(2), [Regularizer(np.eye(3), 1)])
    with pytest.raises(ValueError):
        DataSpace(2, standard_deviation=0)
    with pytest.raises(ValueError):
        InverseProblem(np.eye(2), np.zeros(2), bounds=(1, -1))


@pytest.mark.fem
def test_real_ngsolve_pixel_chain_rule_and_inverse(backend):
    from eit_forward.adapters import (NGSolveConductivityForward, NGSolveAdjointJacobian,
                                 triangles_in_parameter_order)
    from tests.support import make_plan
    geometry = backend.geometry.ElectrodeDisk(radius=1., n_electrodes=8)
    mesh = geometry.make_mesh(maxh=.45, curve_order=1)
    els = tuple(geometry.electrode_names)
    raw = NGSolveConductivityForward(backend, mesh, els, make_plan(backend, els, exclude_current=True))
    edges = np.linspace(-1, 1, 5)
    s = triangle_pixel_space(triangles_in_parameter_order(raw), edges, edges, offset=np.ones(raw.n_parameters))
    f = MappedForward(raw, NGSolveAdjointJacobian(raw), s)
    p = np.zeros(s.ndof)
    lin = f.linearize(p)
    direction = np.random.default_rng(4).normal(size=s.ndof)
    h = 1e-4
    np.testing.assert_allclose(lin.jvp(direction), (f.evaluate(p+h*direction)-f.evaluate(p-h*direction))/(2*h), rtol=1e-5, atol=1e-8)
    problem = InverseProblem(lin.matrix, lin.jvp(.01*direction), [Regularizer(np.eye(s.ndof), 1e-4)])
    x = SciPySolver().solve(problem).parameters
    gradient = problem.A.T @ (problem.A @ x-problem.b) + 1e-4*x
    assert np.linalg.norm(gradient) < 1e-7


@pytest.mark.fem
def test_curved_mesh_is_rejected_by_polygon_mapping(backend, disk):
    from eit_forward.adapters import NGSolveConductivityForward, triangles_in_parameter_order
    from tests.support import make_plan
    mesh, els = disk
    f = NGSolveConductivityForward(backend, mesh, els, make_plan(backend, els))
    with pytest.raises(ValueError, match="straight-sided"):
        triangles_in_parameter_order(f)


def test_scipy_rejects_portable_unsupported_regularizer_clearly():
    p = InverseProblem(np.eye(4), np.ones(4),
                       [Regularizer(np.eye(4), .2, "huber", name="robust")])
    with pytest.raises(NotImplementedError, match="unsupported.*robust"):
        SciPySolver().solve(p)


def test_cvxpy_huber_regularizer_matches_numeric_objective():
    pytest.importorskip("cvxpy")
    A = np.eye(5)
    b = np.array([0., .2, 1., -1.5, .4])
    reg = Regularizer(np.eye(5), .3, "huber", huber_M=.25)
    p = InverseProblem(A, b, [reg])
    result = cvx_solver().solve(p)
    assert result.objective == pytest.approx(p.objective(result.parameters), rel=1e-10)
    assert np.isfinite(result.objective)


def test_cvxpy_group_l2_regularizer():
    pytest.importorskip("cvxpy")
    L = np.eye(6)
    reg = Regularizer(L, .2, "group_l2", group_size=2)
    p = InverseProblem(np.eye(6), np.arange(6.)/5, [reg])
    result = cvx_solver().solve(p)
    assert result.status == "optimal"
    assert np.isfinite(result.objective)


def test_cvxpy_expression_regularizer_escape_hatch():
    pytest.importorskip("cvxpy")
    from eit_inverse import CVXPYExpressionRegularizer
    reg = CVXPYExpressionRegularizer(
        4, expression=lambda x, cp: cp.norm_inf(x),
        value=lambda x: np.linalg.norm(x, np.inf), weight=.2, name="linf")
    p = InverseProblem(np.eye(4), np.arange(4.)/3, [reg])
    result = cvx_solver().solve(p)
    assert result.objective == pytest.approx(p.objective(result.parameters), rel=1e-9)
    np.testing.assert_allclose(result.parameters, [0., 1/3, 2/3, .9], atol=1e-5)
    with pytest.raises(NotImplementedError, match="linf"):
        SciPySolver().solve(p)


def test_cvxpy_expression_regularizer_with_portable_prior_and_bounds():
    pytest.importorskip("cvxpy")
    from eit_inverse import CVXPYExpressionRegularizer
    b = np.array([-2., -.05, .6, 2.])
    prior = np.array([.4, -.2, .1, -.3])
    weight, l2_weight = .4, .5
    portable = Regularizer(np.eye(4), l2_weight, "l2", prior=prior)
    custom = CVXPYExpressionRegularizer(
        4, expression=lambda x, cp: cp.norm1(x),
        value=lambda x: np.linalg.norm(x, 1), weight=weight)
    p = InverseProblem(np.eye(4), b, [portable, custom], bounds=(-.7, .8))
    result = cvx_solver().solve(p)
    # Complete the square, then apply soft thresholding and box projection.
    center = (b + l2_weight*prior) / (1+l2_weight)
    expected = np.clip(np.sign(center)*np.maximum(
        np.abs(center)-weight/(2*(1+l2_weight)), 0), -.7, .8)
    np.testing.assert_allclose(result.parameters, expected, atol=1e-5)
    expected_objective = (np.sum((expected-b)**2)
                          + l2_weight*np.sum((expected-prior)**2)
                          + weight*np.abs(expected).sum())
    assert result.objective == pytest.approx(expected_objective, abs=1e-7)


def test_cvxpy_rejects_other_backend_without_portable_attributes():
    pytest.importorskip("cvxpy")
    from eit_inverse import BackendRegularizer

    class OtherRegularizer(BackendRegularizer):
        backend = "other"

    p = InverseProblem(np.eye(2), np.ones(2), [OtherRegularizer(2)])
    with pytest.raises(NotImplementedError, match="targets backend 'other'"):
        cvx_solver().solve(p)

