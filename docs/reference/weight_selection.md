# Selecting a regularization weight

The FEM-independent **eit_inverse.selection** module adds a reusable weight
path, L-curve corner selection, normalized cross-correlation against known
reference images, and held-out measurement cross-validation. Both SciPy and
CVXPY backends use the same API, for L2 or L1 penalties of any supported order.

The result is a **selected candidate**, not a guarantee of the best physical
reconstruction. Inspect the curves and test the choice on independent data.
No additional dependency is required.

## What each method needs

| Method | Selection rule | Needs ground truth? | Main limitation |
| --- | --- | --- | --- |
| L-curve | Positive corner curvature in log residual vs log penalty | No | Some problems have no clear corner; L1 curves can have active-set bends |
| Reference correlation | Highest normalized correlation with a known image | Yes | Ignores amplitude and constant offset; may tie |
| Measurement CV, MSE | Lowest held-out weighted prediction error | No | Dependent rows must be held out together |
| Measurement CV, correlation | Highest held-out prediction correlation | No | Also ignores amplitude; held-out folds can be constant/undefined |

Correlation with the data used to fit a reconstruction is not an independent
weight-selection criterion: fitting noise can improve it. Use known image truth
or held-out measurements. The default CV score is MSE, which also measures
amplitude errors. Here “cross-correlation” is implemented literally as normalized
mean-subtracted correlation, with an optional 1D lag search; cross-validation
is a separate facility.

## Build the shared path

~~~python
import numpy as np
from eit_inverse import (
    InverseProblem, Regularizer, SciPySolver, CVXPYSolver,
    grid_operator, regularization_path, select_lcurve,
)

L = grid_operator(pixel_shape, order=1, mask=active_pixel_mask)
problem = InverseProblem(J_pixel, delta_y, [Regularizer(L, 1., "l2")])
solver = SciPySolver()       # or CVXPYSolver()
weights = np.logspace(-5, 1, 25)
path = regularization_path(problem, solver, weights)
chosen = select_lcurve(path)
print(chosen.weight)
estimate = chosen.parameters
~~~

Weights must be a strictly increasing vector of at least two finite,
nonnegative candidates. No automatic range is assumed: data units, whitening,
grid spacing, norm, and derivative order affect suitable values.
L-curve selection additionally needs at least five distinct positive norm pairs.

For each candidate the path stores the solution, objective, weighted residual
norm, unweighted selected-penalty norm, and original backend result.
It changes only one regularizer weight; other terms, priors, bounds and data
weighting stay fixed. Choose another term with regularizer_index:

~~~python
path = regularization_path(problem, solver, weights, regularizer_index=1)
~~~

This is a **one-dimensional search**. With several penalties, independently
selected weights are not a joint multidimensional optimum.
The existing objective uses lambda times squared L2, or lambda times L1;
lambda is not squared and there is no hidden factor 1/2.

## L-curve details

Horizontal coordinate:

$$\rho(\lambda)=\|W(Ax_\lambda-b)\|_2.$$

Vertical coordinate:

$$
\eta(\lambda)=
\begin{cases}
\|L(x_\lambda-x_\mathrm{prior})\|_2,&\mathrm{L2},\\
\|L(x_\lambda-x_\mathrm{prior})\|_1,&\mathrm{L1}.
\end{cases}
$$

The penalty coordinate does **not** include lambda, and the L2 coordinate is
the norm rather than its square. The implementation uses signed three-point
circumcircle curvature of the log-log points. With increasing lambda, the
counterclockwise bend from a mostly vertical leg toward a mostly horizontal
leg has positive curvature. The strongest positive interior bend is selected.
This is a simple discrete method, not Hansen's adaptive pruning algorithm.

Zero/nonfinite norm pairs and near-duplicate curve points are excluded.
Endpoints cannot be chosen. No positive corner or too few usable points raises
a ValueError instead of inventing a weight. A well-conditioned problem may
legitimately have no L-shaped curve.

The result exposes curvature scores and the retained points through diagnostics.
A corner close to the candidate-range boundary produces a message asking for
a broader/refined grid. For nonsmooth L1 paths or several visible bends, inspect
the entire curve and compare with validation.

~~~python
print(chosen.messages)
curvature = chosen.scores
rho = path.residual_norms
eta = path.penalty_norms
~~~

## Reference-image cross-correlation

~~~python
from eit_inverse import select_correlation

chosen = select_correlation(path, known_delta_conductivity)
# Optional comparison on a shared evaluation grid:
chosen = select_correlation(
    path, known_reference_image,
    comparison_operator=parameters_to_evaluation_grid,
    mask=evaluation_body_mask,
)
~~~

The score is Pearson-style mean-subtracted normalized correlation. Multidimensional
images are compared at the same coordinates, flattening only the included
evaluation cells. A supplied comparison operator maps parameters before scoring.
Masked-out reference NaNs are allowed; included reference/prediction values
must be finite.

Zero lag is the default, so position errors remain visible. The default is
**signed** correlation: inverted conductivity changes are not treated as correct.
Use absolute=True explicitly if sign-insensitive similarity is the intended
measure. Amplitude errors need an additional error metric: correlation of
2 x + c with x is one even though the reconstruction may be quantitatively wrong.
Constant or numerically near-constant signals have undefined scores.

Optional 1D lag search:

~~~python
from eit_inverse import normalized_cross_correlation

diagnostic = normalized_cross_correlation(signal1, signal2, max_lag=3)
print(diagnostic["coefficient"], diagnostic["lag"])
chosen = select_correlation(path, known_1d_reference, max_lag=3)
~~~

Positive lag means the first signal is delayed relative to the second.
Each overlap is centered/normalized independently and must contain at least
three samples. Nonzero lags require an unmasked 1D reference; image registration
is not implemented. Searching lags can hide localization errors and is not the
default for EIT image-quality comparisons.

Tied scores are reported; the smallest tied sampled weight is selected.
All scores and best lags remain available in the result. A boundary selection
asks for a broader search. If all scores are undefined, the selector raises.

## Held-out measurement cross-validation

~~~python
from eit_inverse import cross_validate

chosen = cross_validate(
    problem, solver, weights,
    n_splits=5, seed=42,
    criterion="mse",          # default; alternatively "correlation"
)
estimate = chosen.parameters  # already refitted on ALL measurements
~~~

For each candidate/fold, solve with training rows only, then predict the unseen
rows. MSE uses the held-out whitened residuals. Fold scores are averaged with
weights proportional to held-out row counts. Correlation similarly averages
fold correlations; it does not refit amplitudes to the held-out data.

All regularization weights in a training fold are multiplied by
n_training/n_full by default. This preserves the same mean-data-loss
regularization convention despite using a sum-of-squares objective and fewer
rows. Fixed penalties are scaled as well as the varied penalty.
Set scale_training_weights=False only if unscaled training objectives are
intentional. The selected candidate is finally solved with all original rows,
original whitening and full-scale weights.

Returned diagnostics include fold scores, their sample standard deviation,
the splits, out-of-fold predictions for every candidate, and the full-data result.
Fold-score scatter is descriptive; it is not a confidence interval for lambda.

### Group dependent measurements

Random row folds assume independent validation observations. EIT can have
reciprocal/redundant rows, shared injections, reference noise, or repeated
measurements. Keep the relevant dependent rows together:

~~~python
chosen = cross_validate(
    problem, solver, weights,
    groups=measurement_group_ids, n_splits=4, seed=7,
)
~~~

Every measurement group belongs to one held-out fold. Group labels must have
one entry per row, and at least n_splits distinct groups are needed.
Group construction depends on the actual protocol/noise model; the selector
does not infer reciprocal EIT groups from a bare Jacobian.

For blocked experiments or manually defined acquisition groups, supply explicit
(train_indices, test_indices) pairs using splits. Train/test indices must be
disjoint, in range, and unique. Test rows cannot repeat across folds.
Custom splits may leave some rows unvalidated; their out-of-fold predictions
remain NaN. Correlation folds require at least three observations.

**Row CV requires nonsingular diagonal whitening.** A full whitening matrix
can mix held-out measurements into training rows, causing leakage. The helper
rejects it. With correlated noise, use genuinely independent acquisition blocks
and an appropriately defined training/validation covariance workflow, rather
than prewhitening the full dataset and then blindly splitting its rows.

Reference conductivity, parameter maps, priors, and any preprocessing learned
from measurements also need independent/training-only construction. The helper
cannot detect how caller-provided objects were obtained.

CV evaluates predictive fit of the fixed linearized model. Model mismatch,
choice of Jacobian reference, parameterization and experimental noise can still
limit imaging accuracy. It does not implement nonlinear EIT tuning or GCV.

## Examples and tests

~~~bash
# L-curve + reference correlation + held-out MSE on one reproducible dataset:
python -m examples.regularization_weight_selection

# Choose either backend, norm or derivative order:
python -m examples.regularization_weight_selection --backend cvxpy --norm l1 --order 1
python -m examples.regularization_weight_selection --norm l1 --order 2

# Save the complete numerical path and a comparison figure:
python -m examples.regularization_weight_selection \
    --output results/weights.npz --plot results/weights.png

python -m pytest -q tests/test_weight_selection.py
python -m pytest -q
~~~

The default example selects approximately 0.0562 by L-curve, 0.0178 by
known-truth correlation, and 0.178 by held-out MSE. Different methods optimize
different criteria; agreement is not guaranteed. Ground-truth errors are
printed only because the example uses synthetic data.
Use --plot and --output to generate the figure and numerical results.

Tests cover analytic ridge-path norms, correct corner orientation and degenerate
curves, unchanged fixed penalties/priors/bounds, known lags, amplitude invariance,
reference-grid masks, ties, grouped folds, independently calculated CV training
fits/refits with both backends, L1 selection, held-out correlation, and rejection
of overlapping splits or leaky whitening.

Primary references:

- [Hansen, The L-curve and its use in inverse problems](https://www2.imm.dtu.dk/pubdb/pubs/2231-full.html)
- [Hansen et al., adaptive pruning for a discrete L-curve](https://www2.imm.dtu.dk/pubdb/pubs/3436-full.html)
- [Cross-validation and grouped/dependent observations](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Pearson correlation and constant inputs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)

[Documentation index](../README.md)
