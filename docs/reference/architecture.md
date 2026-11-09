# Architecture and inverse objective

The FEM layer computes physical states and numerical derivatives. The inverse
layer consumes explicit dense/sparse arrays and does not import NGSolve.
Examples compose these layers with data processing and optional plotting.

| Layer | Modules | Responsibility |
| --- | --- | --- |
| Electrical physics | `eit_forward.model`, `.measurement`, `.observations`, `.stimulation` | CEM, current protocols, referencing and measurements |
| Geometry | `eit_forward.geometries`, `.points` | Material/boundary labels, named points and meshing |
| Conductivity discretization | `eit_forward.parameters` | Parameter ownership, DOF order and mesh regularization operators |
| Numerical adapters | `eit_forward.adapters` | Predictions, electrical Jacobians |
| Shared numerical utilities | `eitlab.discretization`, `.differentiation` | Parameter maps, pixel overlap and finite differences |
| Inverse objective | `eit_inverse.problem` | Whitening, priors, bounds and penalties |
| Solvers | `eit_inverse.backends`, `.tikhonov` | SciPy, CVXPY and reusable SVD Tikhonov |
| Parameter choice | `eit_inverse.selection`, `.noise_tuning` | Weight paths, validation and noise amplification |
| Plotting | `eitlab.plotting` | Matplotlib/PyVista views and boundary inspection |

`eitlab` exposes convenient lazy imports. Importing its inverse, stimulation,
observation or bibliography interfaces does not load NGSolve. CVXPY and plotting
backends load when their operations are invoked.

## Shared objective

The common `Regularizer` API defines the objective solved by both inverse backends:

$$
\min_{\ell\le x\le u}\|W(Ax-b)\|_2^2
+\sum_{r\in L2}\lambda_r\|L_r(x-x_r)\|_2^2
+\sum_{r\in L1}\lambda_r\|L_r(x-x_r)\|_1.
$$

`DataSpace` defines W from independent standard deviations or explicit
whitening. `Regularizer` defines L, a nonnegative objective coefficient lambda,
a norm and a prior. There is no factor 1/2 or implicit squaring of the weight.
Operators must use the same parameter ordering.

For difference EIT, A is J at a declared reference, b is measured minus
reference voltage, and x is a conductivity/basis-coefficient change. Priors
and bounds apply to x. To constrain absolute conductivity, account for the
reference offset. A box in reduced coefficients need not constrain every element.

```python
import numpy as np
from eit_inverse import DataSpace, InverseProblem, Regularizer, SciPySolver

J = np.array([[1., 0.], [1., 2.], [0., 1.]])
b = np.array([.1, -.2, .05])
penalty = Regularizer(np.eye(2), weight=.01, norm="l2")
problem = InverseProblem(J, b, [penalty], data_space=DataSpace(3), bounds=(-.5, .5))
result = SciPySolver().solve(problem)
```

The same problem can be passed to `CVXPYSolver`. SciPy is the default backend:
it uses augmented least squares for L2 and ADMM for L1/mixed penalties, including
LASSO and mesh/grid TV. TV therefore does not require CVXPY. CVXPY independently
translates the same common objective and is useful for backend validation.

Portable mathematical regularizers belong in `Regularizer`, independently of a
solver package. Besides L1/L2 it represents Huber and grouped-L2 penalties;
CVXPY currently implements the latter two while SciPy reports them explicitly as
unsupported. Grouped L2 can express group sparsity or isotropic TV when operator
rows are arranged as gradient-component groups.

`CVXPYExpressionRegularizer` is a deliberately backend-specific escape hatch for
research formulations that are not yet part of the portable API. It accepts an
arbitrary convex CVXPY expression builder plus a matching numerical value
function. `BackendRegularizer` defines the small extension pattern so another
solver package can provide an analogous backend-specific regularizer without
putting package-specific objects into the common problem API. CVXPY remains
optional and is imported only when its solver is invoked.

Results contain parameters, objective, weighted residual, status
and diagnostics. `TikhonovSolver` provides a reusable SVD for identity L2 and
multiple weights.

## Regularization operators

| Penalty | Operator and norm |
| --- | --- |
| Identity Tikhonov | `Regularizer(I, weight, "l2")` |
| Geometric L2(0) | `Regularizer(space.l2_operator(), weight, "l2")` |
| P0 mesh TV | `Regularizer(space.tv_operator(), weight, "l1")` |
| Pixel derivatives | `grid_operator(shape, order, spacing=..., mask=...)` with L1/L2 |

The FEM adapter constructs elementwise-constant operators in parameter DOF
order. Their L2 energy is the element-volume integral; TV uses internal face
measures and vanishes on constants. These differ from pixel finite differences.

`grid_operator` supports orders 0, 1 and 2 in one/two dimensions. In 2D, order
2 stacks Dyy, Dxx and sqrt(2) Dxy. L2 penalizes a discrete Hessian Frobenius
norm; L1 gives componentwise Hessian sparsity, not TGV. Stencils crossing mask
boundaries are omitted. Array order is C.

See [weight selection](weight_selection.md) and [noise tuning](noise_tuning.md) for penalty construction and parameter choice.

## Parameter and pixel maps

`ParameterSpace` in `eitlab.discretization` represents

$$\sigma=\sigma_0+P x,\qquad J_x=J_\sigma P.$$

Rows of P follow the FEM coefficient order. Pixel or radial bases restrict
reconstruction without changing the physical solver. `MappedForward` composes
a forward adapter, Jacobian provider and map; its linearization supplies
prediction, matrix, Jv and transpose actions.

`PixelSpace` also stores grid/mask metadata and overlap maps: P prolongs
pixel values to element averages; R restricts elements to covered-pixel
averages. They are not inverses and R is not generally P.T. For geometric
masses, M_pixel R = P.T M_element. Partial pixels use covered areas. Exact
polygon overlap requires straight 2D triangles; curved elements are rejected.

Prefer `conductivity_space.pixel_space(x_edges, y_edges)` so the FEM adapter
owns extraction and DOF ordering. Use standalone `triangle_pixel_space` when
the numerical triangle coordinates are already known.

## Linearization and scope

Use `ElectricalLinearization` and `JacobianAssembler(linearization, results)`
for stored FEM states. Numerical adapters return explicit arrays and provide
adjoint/finite-difference alternatives. Numerical snapshots survive later
evaluations; FEM states/linearizations become invalid after a model update.
See the [forward guide](../guides/forward_model.md).

Common inverse solvers handle one fixed linearized problem. Rebuild
predictions/derivatives in a nonlinear outer loop; the joint contact example
demonstrates Gauss–Newton. General matrix-free optimization
and arbitrary linear inequalities are outside the current inverse API.

[Documentation index](../README.md)
