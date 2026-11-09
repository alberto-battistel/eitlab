"""Independent checks of selection geometry, correlation, and held-out fits."""
import numpy as np
import pytest
from scipy import sparse

from eit_inverse import (DataSpace, Regularizer, InverseProblem, SciPySolver,
                         CVXPYSolver, RegularizationPath, regularization_path,
                         select_lcurve, normalized_cross_correlation,
                         select_correlation, measurement_folds, cross_validate)


def test_ridge_path_matches_analytic_norms():
    weights = np.logspace(-4, 4, 81)
    p = InverseProblem(np.eye(4), [1., -2., 3., 1.], [Regularizer(np.eye(4), 1.)])
    path = regularization_path(p, SciPySolver(), weights)
    magnitude = np.linalg.norm(p.data)
    np.testing.assert_allclose(path.residual_norms, magnitude*weights/(1+weights), rtol=1e-9)
    np.testing.assert_allclose(path.penalty_norms, magnitude/(1+weights), rtol=1e-9)
    # A well-conditioned identity problem need not have a convex L-corner.
    with pytest.raises(ValueError, match="No positive"):
        select_lcurve(path)


def test_lcurve_selects_geometric_corner_with_correct_orientation():
    # Clockwise curvature is the opposite bend, not the L-corner we want.
    points = np.array([[0, 3], [.1, 2], [.2, 1], [1, .2], [2, .1], [3, 0.]])
    weights = np.logspace(-3, 2, len(points))
    path = RegularizationPath(weights, np.ones((6, 3)), np.exp(points[:, 0]),
                              np.exp(points[:, 1]), np.zeros(6), 0, ())
    selected = select_lcurve(path)
    assert selected.index in (2, 3)
    assert selected.scores[selected.index] > 0


def test_path_varies_one_weight_and_retains_prior_bounds_and_other_terms():
    p = InverseProblem(np.eye(3), [3., -2., 1.],
        [Regularizer(np.eye(3), .2, "l2", prior=[.3, .1, -.2]),
         Regularizer(np.eye(3), .1, "l1")], bounds=(-.5, .8),
        data_space=DataSpace(3, standard_deviation=[1., 2., 3.]))
    weights = np.array([.02, .2, 2.])
    path = regularization_path(p, SciPySolver(), weights, regularizer_index=1)
    for j, w in enumerate(weights):
        x = path.solutions[j]
        assert np.all((x >= -.5) & (x <= .8))
        expected = np.linalg.norm(p.A @ x-p.b)**2+p.regularizers[0].value(x)+w*np.abs(x).sum()
        np.testing.assert_allclose(path.objectives[j], expected)
        np.testing.assert_allclose(path.penalty_norms[j], np.abs(x).sum())
    assert p.regularizers[1].weight == .1  # no mutation of caller's problem


def test_degenerate_or_zero_norm_lcurve_does_not_invent_a_weight():
    weights = np.logspace(-3, 3, 7)
    path = RegularizationPath(weights, np.ones((7, 3)), np.exp(np.arange(7.)),
                              np.exp(-np.arange(7.)), np.zeros(7), 0, ())
    with pytest.raises(ValueError, match="No positive"):
        select_lcurve(path)
    path.penalty_norms[:] = 0
    with pytest.raises(ValueError, match="positive finite"):
        select_lcurve(path)


def test_correlation_is_amplitude_invariant_and_reports_known_lag():
    a = np.array([1., 3., -2., .5, 4., -1., 2., 0.])
    np.testing.assert_allclose(normalized_cross_correlation(2*a+7, a)["coefficient"], 1)
    delayed = np.r_[0., 0., a[:-2]]
    r = normalized_cross_correlation(delayed, a, max_lag=2)
    assert r["lag"] == 2
    np.testing.assert_allclose(r["coefficient"], 1)
    assert np.isnan(normalized_cross_correlation(np.ones(8), a)["coefficient"])
    assert np.isnan(normalized_cross_correlation(np.full(8, .1), a)["coefficient"])
    with pytest.raises(ValueError, match="1D"):
        normalized_cross_correlation(a.reshape(2, 4), a.reshape(2, 4), max_lag=1)


def test_reference_correlation_matches_independent_pearson_scores():
    A = np.diag([1., .5, .1, .05])
    p = InverseProblem(A, [1., .5, .5, -.1], [Regularizer(np.eye(4), 1.)])
    path = regularization_path(p, SciPySolver(), np.logspace(-4, 2, 20))
    truth = np.array([1., 1., 0., 0.])
    result = select_correlation(path, truth)
    independent = np.array([np.corrcoef(x, truth)[0, 1] for x in path.solutions])
    np.testing.assert_allclose(result.scores, independent)
    assert result.index == np.argmax(independent)


def test_masked_comparison_mapping_and_tied_scores():
    weights = np.array([.1, 1., 10.])
    solutions = np.array([[1., 2., 3.], [2., 4., 6.], [3., 6., 9.]])
    path = RegularizationPath(weights, solutions, np.ones(3), np.ones(3), np.ones(3), 0, ())
    P = sparse.csr_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    reference = np.array([[1., 2.], [3., np.nan]])
    result = select_correlation(path, reference, comparison_operator=P, mask=np.isfinite(reference))
    assert result.weight == .1
    assert any("tie" in m for m in result.messages)


def test_group_folds_keep_repeated_measurements_together_and_are_reproducible():
    groups = np.repeat(np.arange(8), 3)
    folds = measurement_folds(24, n_splits=4, groups=groups, seed=17)
    repeated = measurement_folds(24, n_splits=4, groups=groups, seed=17)
    seen = []
    for (train, test), (t2, v2) in zip(folds, repeated):
        assert not set(groups[train]).intersection(groups[test])
        np.testing.assert_array_equal(train, t2); np.testing.assert_array_equal(test, v2)
        seen.extend(test.tolist())
    assert sorted(seen) == list(range(24))


@pytest.mark.parametrize("backend", ["scipy", "cvxpy"])
def test_cv_matches_hand_computed_training_fits_and_full_refit(backend):
    rng = np.random.default_rng(13)
    A, b = rng.normal(size=(12, 3)), rng.normal(size=12)
    sd = np.linspace(.5, 1.3, 12)
    p = InverseProblem(A, b, [Regularizer(np.eye(3), 1.)],
                       data_space=DataSpace(12, standard_deviation=sd))
    solver = SciPySolver()
    if backend == "cvxpy":
        pytest.importorskip("cvxpy")
        solver = CVXPYSolver()
    weights = np.array([.01, .1, 1., 10.])
    folds = measurement_folds(12, n_splits=3, seed=5)
    selected = cross_validate(p, solver, weights, splits=folds)
    independent = []
    for w in weights:
        residuals = []
        for train, test in folds:
            At, bt = A[train]/sd[train, None], b[train]/sd[train]
            x = np.linalg.solve(At.T @ At + (len(train)/12)*w*np.eye(3), At.T @ bt)
            predicted = A[test] @ x
            residuals.extend(((predicted-b[test])/sd[test])**2)
        independent.append(np.mean(residuals))
    np.testing.assert_allclose(selected.scores, independent, rtol=1e-6, atol=1e-7)
    expected_index = np.argmin(independent)
    assert selected.index == expected_index
    fullA, fullb = A/sd[:, None], b/sd
    expected = np.linalg.solve(fullA.T @ fullA + weights[expected_index]*np.eye(3), fullA.T @ fullb)
    np.testing.assert_allclose(selected.parameters, expected, atol=1e-6)
    assert np.isfinite(selected.diagnostics["out_of_fold_predictions"]).all()


def test_l1_paths_and_cv_are_supported_without_smoothing_the_penalty():
    A = np.tile(np.eye(3), (4, 1))
    b = A @ np.array([1., 0., -.5]) + np.random.default_rng(7).normal(scale=.1, size=12)
    p = InverseProblem(A, b, [Regularizer(np.eye(3), .1, "l1")])
    weights = np.array([.05, .2, 1.])
    path = regularization_path(p, SciPySolver(), weights)
    np.testing.assert_allclose(path.penalty_norms, np.abs(path.solutions).sum(axis=1))
    result = cross_validate(p, SciPySolver(), weights, n_splits=3)
    assert result.weight in weights
    assert np.isfinite(result.scores).all()


def test_correlation_cv_uses_held_out_predictions():
    rng = np.random.default_rng(14)
    A = rng.normal(size=(18, 4)); b = A @ [1., 0., -.5, .2] + rng.normal(scale=.2, size=18)
    p = InverseProblem(A, b, [Regularizer(np.eye(4), 1.)])
    r = cross_validate(p, SciPySolver(), [.01, .1, 1.], n_splits=3, criterion="correlation")
    for j in range(3):
        scores = [np.corrcoef(r.diagnostics["out_of_fold_predictions"][j, test], b[test])[0, 1]
                  for _, test in r.diagnostics["splits"]]
        np.testing.assert_allclose(r.scores[j], np.mean(scores))


def test_cv_rejects_leaky_whitening_and_overlapping_splits():
    W = np.eye(6); W[0, 1] = .2
    p = InverseProblem(np.ones((6, 2)), np.arange(6.), [Regularizer(np.eye(2), 1.)],
                       data_space=DataSpace(6, whitening=W))
    with pytest.raises(ValueError, match="diagonal whitening"):
        cross_validate(p, SciPySolver(), [.1, 1.])
    p = InverseProblem(np.ones((6, 2)), np.arange(6.), [Regularizer(np.eye(2), 1.)])
    with pytest.raises(ValueError, match="disjoint"):
        cross_validate(p, SciPySolver(), [.1, 1.], splits=[([0, 1, 2], [2, 3]), ([3, 4], [0, 1])])
    with pytest.raises(ValueError, match="repeat"):
        cross_validate(p, SciPySolver(), [.1, 1.], splits=[([0, 1, 2], [3, 4]), ([0, 1, 5], [3, 4])])


def test_invalid_weight_grids_and_constant_reference_fail_explicitly():
    p = InverseProblem(np.eye(3), np.ones(3), [Regularizer(np.eye(3), 1.)])
    for weights in [[1., .1], [1., 1.], [-1., 1.], [np.nan, 1.]]:
        with pytest.raises(ValueError):
            regularization_path(p, SciPySolver(), weights)
    path = regularization_path(p, SciPySolver(), [.1, 1.])
    with pytest.raises(ValueError, match="undefined"):
        select_correlation(path, np.ones(3))
