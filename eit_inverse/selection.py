"""Backend-independent regularization paths and weight-selection diagnostics.

L-curve is a discrete corner heuristic; reference correlation requires known
truth; cross-validation holds out measurements. None guarantees an optimal
physical reconstruction, and each returns the complete selection diagnostics.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .problem import DataSpace, InverseProblem, Regularizer, real_matrix


def _weights(values):
    w = np.asarray(values, dtype=float)
    if w.ndim != 1 or len(w) < 2 or not np.isfinite(w).all() or np.any(w < 0) or np.any(np.diff(w) <= 0):
        raise ValueError("weights must be at least two finite, nonnegative, strictly increasing values")
    return w.copy()


def _with_weight(problem, weight, regularizer_index=0, *, matrix=None, data=None,
                 data_space=None, penalty_scale=1.):
    if not 0 <= regularizer_index < len(problem.regularizers):
        raise ValueError("regularizer_index is out of range")
    penalties = [Regularizer(r.L, penalty_scale*(weight if i == regularizer_index else r.weight),
                             r.norm, r.prior, r.name) for i, r in enumerate(problem.regularizers)]
    return InverseProblem(problem.matrix if matrix is None else matrix,
                          problem.data if data is None else data, penalties,
                          data_space=problem.data_space if data_space is None else data_space,
                          bounds=(problem.lower, problem.upper))


@dataclass
class RegularizationPath:
    weights: np.ndarray
    solutions: np.ndarray                 # (n_weights, n_parameters)
    residual_norms: np.ndarray            # ||W(Ax-b)||_2
    penalty_norms: np.ndarray             # ||L(x-prior)||_2 or ||...||_1, unweighted
    objectives: np.ndarray
    regularizer_index: int
    results: tuple


@dataclass
class WeightSelection:
    weight: float
    index: int
    parameters: np.ndarray
    method: str
    scores: np.ndarray
    diagnostics: dict
    messages: tuple = ()


def regularization_path(problem, solver, weights, *, regularizer_index=0):
    """Vary one penalty weight; leave other weights, bounds and whitening fixed."""
    w = _weights(weights)
    results, residuals, penalties = [], [], []
    for weight in w:
        p = _with_weight(problem, weight, regularizer_index)
        result = solver.solve(p)
        r = p.regularizers[regularizer_index]
        v = r.L @ (result.parameters-r.prior)
        results.append(result)
        residuals.append(result.weighted_residual_norm)
        penalties.append(np.linalg.norm(v) if r.norm == "l2" else np.abs(v).sum())
    return RegularizationPath(w, np.stack([r.parameters for r in results]),
                              np.asarray(residuals), np.asarray(penalties),
                              np.array([r.objective for r in results]),
                              regularizer_index, tuple(results))


def select_lcurve(path):
    """Select maximal positive discrete curvature in (log residual, log penalty).

    Uses signed three-point circumcircle curvature, with counterclockwise L-corners
    positive. Endpoints cannot be selected. Zero/nonfinite norms and duplicate
    curve points are excluded; an absent corner raises instead of inventing one.
    This is a simple discrete heuristic, not Hansen's adaptive pruning algorithm.
    """
    rho, eta = path.residual_norms, path.penalty_norms
    valid = np.isfinite(rho) & np.isfinite(eta) & (rho > 0) & (eta > 0)
    indices = np.flatnonzero(valid)
    if len(indices) < 5:
        raise ValueError("L-curve needs at least five positive finite norm pairs")
    points = np.column_stack([np.log(rho[indices]), np.log(eta[indices])])
    scale = max(np.ptp(points, axis=0).max(), 1.)
    keep = [0]
    for j in range(1, len(points)):
        if np.linalg.norm(points[j]-points[keep[-1]]) > 1e-8*scale:
            keep.append(j)
    indices, points = indices[keep], points[keep]
    if len(indices) < 5:
        raise ValueError("L-curve has too few distinct positive norm pairs")
    curvature = np.full(len(path.weights), np.nan)
    for j in range(1, len(points)-1):
        a, b = points[j]-points[j-1], points[j+1]-points[j]
        denominator = np.linalg.norm(a)*np.linalg.norm(b)*np.linalg.norm(a+b)
        if denominator > 0:
            curvature[indices[j]] = 2*(a[0]*b[1]-a[1]*b[0])/denominator
    candidates = np.flatnonzero(np.isfinite(curvature) & (curvature > 1e-8/scale))
    if not len(candidates):
        raise ValueError("No positive L-curve corner; broaden the grid or use validation")
    index = int(candidates[np.argmax(curvature[candidates])])
    messages = []
    location = int(np.flatnonzero(indices == index)[0])
    if location in (1, len(indices)-2):
        messages.append("Corner is near the sampled range boundary; broaden/refine the weight grid")
    return WeightSelection(float(path.weights[index]), index, path.solutions[index].copy(),
                           "lcurve", curvature, dict(valid_norms=valid, retained_indices=indices,
                           residual_norms=rho.copy(), penalty_norms=eta.copy()), tuple(messages))


def normalized_cross_correlation(first, second, *, max_lag=0, absolute=False):
    """Mean-subtracted normalized correlation; no amplitude/offset error penalty.

    Zero lag accepts matching real arrays of any shape. Nonzero lags require
    one-dimensional arrays. Positive lag means first is delayed relative to
    second: compare first[lag:] with second[:-lag]. Every overlap is centered
    and normalized separately. Constant overlaps have undefined (NaN) scores.
    """
    if np.iscomplexobj(first) or np.iscomplexobj(second):
        raise TypeError("Correlation inputs must be real")
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.shape != b.shape or a.size < 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Correlation needs matching finite arrays with at least three values")
    if int(max_lag) != max_lag or max_lag < 0 or max_lag > a.size-3:
        raise ValueError("max_lag must leave at least three overlapping values")
    if max_lag and a.ndim != 1:
        raise ValueError("Nonzero-lag correlation is only defined here for 1D arrays")
    a, b = a.ravel(), b.ravel()
    lags = np.arange(-int(max_lag), int(max_lag)+1)
    coefficients = np.full(len(lags), np.nan)
    for j, lag in enumerate(lags):
        aa, bb = (a[lag:], b[:-lag]) if lag > 0 else (a[:lag], b[-lag:]) if lag < 0 else (a, b)
        if np.ptp(aa) == 0 or np.ptp(bb) == 0:
            continue
        raw_a, raw_b = np.linalg.norm(aa), np.linalg.norm(bb)
        aa, bb = aa-aa.mean(), bb-bb.mean()
        norm_a, norm_b = np.linalg.norm(aa), np.linalg.norm(bb)
        if norm_a <= 16*np.finfo(float).eps*raw_a or norm_b <= 16*np.finfo(float).eps*raw_b:
            continue
        denom = norm_a*norm_b
        if denom > 0:
            coefficients[j] = np.clip((aa @ bb)/denom, -1., 1.)
    scores = np.abs(coefficients) if absolute else coefficients
    finite = np.flatnonzero(np.isfinite(scores))
    if not len(finite):
        return dict(coefficient=np.nan, lag=None, coefficients=coefficients, lags=lags)
    best = finite[np.argmax(scores[finite])]
    return dict(coefficient=float(coefficients[best]), lag=int(lags[best]),
                coefficients=coefficients, lags=lags)


def _choose_score(weights, scores, maximize):
    finite = np.flatnonzero(np.isfinite(scores))
    if not len(finite):
        raise ValueError("All selection scores are undefined")
    index = int(finite[np.argmax(scores[finite]) if maximize else np.argmin(scores[finite])])
    messages = []
    ties = finite[np.isclose(scores[finite], scores[index], rtol=1e-10, atol=1e-12)]
    if len(ties) > 1:
        index = int(ties[0])
        messages.append("Selection scores tie; chose the smallest tied sampled weight")
    if index in (0, len(weights)-1):
        messages.append("Selected weight is at the search boundary; broaden the weight grid")
    return index, tuple(messages)


def select_correlation(path, reference, *, comparison_operator=None, mask=None,
                       absolute=False, max_lag=0):
    """Select reference-image correlation; reference must be known independently.

    comparison_operator optionally maps reconstruction parameters to the common
    evaluation grid. Zero lag is the default so position errors remain visible.
    Spatial image registration is not implemented; optional shifts are 1D only.
    """
    ref = np.asarray(reference)
    if np.iscomplexobj(ref):
        raise TypeError("Reference must be real")
    operator = None if comparison_operator is None else real_matrix(comparison_operator, "comparison_operator")
    expected = path.solutions.shape[1] if operator is None else operator.shape[0]
    if ref.size != expected or (operator is not None and operator.shape[1] != path.solutions.shape[1]):
        raise ValueError("Reference/comparison operator dimensions must match the reconstruction")
    valid = np.ones(ref.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if valid.shape != ref.shape or valid.sum() < 3:
        raise ValueError("mask must match reference and include at least three values")
    if max_lag and (ref.ndim != 1 or not valid.all()):
        raise ValueError("Nonzero lags require an unmasked 1D reference")
    scores, lags, coefficients = [], [], []
    for solution in path.solutions:
        prediction = solution if operator is None else operator @ solution
        result = normalized_cross_correlation(np.asarray(prediction).reshape(ref.shape)[valid],
                                              ref[valid], max_lag=max_lag, absolute=absolute)
        coefficient = result["coefficient"]
        coefficients.append(coefficient); lags.append(result["lag"])
        scores.append(abs(coefficient) if absolute else coefficient)
    scores = np.asarray(scores)
    index, messages = _choose_score(path.weights, scores, True)
    return WeightSelection(float(path.weights[index]), index, path.solutions[index].copy(),
                           "reference_correlation", scores,
                           dict(coefficients=np.asarray(coefficients), best_lags=np.asarray(lags),
                                absolute=absolute, max_lag=max_lag), messages)


def measurement_folds(n_data, *, n_splits=5, seed=0, groups=None):
    """Random row K-fold, or group K-fold keeping dependent rows together."""
    if int(n_splits) != n_splits or n_splits < 2:
        raise ValueError("n_splits must be an integer >=2")
    labels = np.arange(n_data) if groups is None else np.asarray(groups)
    if labels.shape != (n_data,):
        raise ValueError("groups must have one label per measurement")
    _, inverse = np.unique(labels, return_inverse=True)
    units = np.arange(inverse.max()+1) if len(inverse) else np.array([])
    if len(units) < n_splits:
        raise ValueError("There must be at least n_splits rows/groups")
    np.random.default_rng(seed).shuffle(units)
    folds = []
    for selected in np.array_split(units, n_splits):
        test = np.flatnonzero(np.isin(inverse, selected))
        train = np.flatnonzero(~np.isin(inverse, selected))
        folds.append((train, test))
    return tuple(folds)


def cross_validate(problem, solver, weights, *, regularizer_index=0, n_splits=5,
                   seed=0, groups=None, splits=None, criterion="mse",
                   scale_training_weights=True):
    """Choose weight using held-out measurement error or held-out correlation.

    Row CV requires diagonal whitening so held-out measurements cannot leak
    through a full whitening matrix. Group reciprocal/dependent EIT rows.
    Custom (train,test) splits support blocked data; held-out rows cannot repeat.
    Every training penalty is scaled by n_train/n_full by default, preserving
    the full problem's mean-loss regularization convention. Refit on all data.
    """
    w = _weights(weights)
    if criterion not in ("mse", "correlation"):
        raise ValueError("criterion must be 'mse' or 'correlation'")
    W = problem.data_space.W
    diagonal = W.diagonal()
    off_diagonal = W-sparse.diags(diagonal)
    off_diagonal.eliminate_zeros()
    if off_diagonal.nnz or np.any(diagonal == 0):
        raise ValueError("Row cross-validation requires nonsingular diagonal whitening; use independent blocks otherwise")
    if splits is not None and groups is not None:
        raise ValueError("Choose custom splits or groups, not both")
    folds = measurement_folds(problem.n_data, n_splits=n_splits, seed=seed, groups=groups) if splits is None else tuple(splits)
    if len(folds) < 2:
        raise ValueError("At least two validation splits are required")
    checked, seen = [], set()
    for train, test in folds:
        train, test = np.asarray(train), np.asarray(test)
        for idx in (train, test):
            if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) or not len(idx) or len(np.unique(idx)) != len(idx) or np.any(idx < 0) or np.any(idx >= problem.n_data):
                raise ValueError("Split indices must be unique, nonempty in-range integer vectors")
        if np.intersect1d(train, test).size or seen.intersection(test.tolist()):
            raise ValueError("Train/test must be disjoint and validation rows cannot repeat across folds")
        if criterion == "correlation" and len(test) < 3:
            raise ValueError("Correlation validation folds need at least three measurements")
        seen.update(test.tolist()); checked.append((train, test))
    folds = tuple(checked)
    fold_scores = np.full((len(w), len(folds)), np.nan)
    predictions = np.full((len(w), problem.n_data), np.nan)
    for j, weight in enumerate(w):
        for k, (train, test) in enumerate(folds):
            train_space = DataSpace(len(train), whitening=sparse.diags(diagonal[train]))
            scale = len(train)/problem.n_data if scale_training_weights else 1.
            p = _with_weight(problem, weight, regularizer_index,
                             matrix=problem.matrix[train], data=problem.data[train],
                             data_space=train_space, penalty_scale=scale)
            solution = solver.solve(p).parameters
            prediction = np.asarray(problem.matrix[test] @ solution)
            predictions[j, test] = prediction
            observed, fitted = diagonal[test]*problem.data[test], diagonal[test]*prediction
            if criterion == "mse":
                fold_scores[j, k] = np.mean((fitted-observed)**2)
            else:
                fold_scores[j, k] = normalized_cross_correlation(fitted, observed)["coefficient"]
    counts = np.array([len(test) for _, test in folds])
    scores = np.average(fold_scores, axis=1, weights=counts)
    index, messages = _choose_score(w, scores, criterion == "correlation")
    full_result = solver.solve(_with_weight(problem, w[index], regularizer_index))
    return WeightSelection(float(w[index]), index, full_result.parameters.copy(),
                           "cross_validation_"+criterion, scores,
                           dict(fold_scores=fold_scores, fold_score_std=np.std(fold_scores, axis=1, ddof=1),
                                out_of_fold_predictions=predictions, splits=folds,
                                full_data_result=full_result, training_weights_scaled=scale_training_weights),
                           messages)
