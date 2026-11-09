"""Real EIT: FEM predictions/Jacobian -> pixels -> either inverse backend.

Uses affine triangles and a coarse pixel grid. This is a software integration
example, not a mesh-convergence or realistic imaging-performance study.
"""

# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples._common import load_backend, make_plan
from eit_forward.adapters import NGSolveConductivityForward, NGSolveAdjointJacobian, triangles_in_parameter_order
from eitlab.differentiation import FiniteDifferenceJacobian
from eit_inverse import (MappedForward, triangle_pixel_space, DataSpace,
                         InverseProblem, Regularizer, grid_operator, SciPySolver, CVXPYSolver)

# %% Settings — edit these before running the following cells
solver_backend = 'scipy'  # ['scipy', 'cvxpy', 'both']
jacobian_method = 'adjoint'  # ['adjoint', 'central']
regularization_norm = 'both'  # ['l2', 'l1', 'both']
regularization_order = 'all'  # ['0', '1', '2', 'all']
output_dir = Path("results/pixel_inverse")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "pixel_inverse.npz"

# %% Compute the demonstration
backend = load_backend()
geometry = backend.geometry.ElectrodeDisk(radius=1., n_electrodes=8)
mesh = geometry.make_mesh(maxh=.45, curve_order=1)
electrodes = tuple(geometry.electrode_names)
raw = NGSolveConductivityForward(backend, mesh, electrodes, make_plan(backend, electrodes, exclude_current=True))
triangles = triangles_in_parameter_order(raw)
edges = np.linspace(-1, 1, 5)
space = triangle_pixel_space(triangles, edges, edges, offset=np.ones(raw.n_parameters))
provider = NGSolveAdjointJacobian(raw) if jacobian_method == "adjoint" else FiniteDifferenceJacobian(raw.evaluate, step=1e-4, scheme="central")
forward = MappedForward(raw, provider, space)
reference = np.zeros(space.ndof)  # x describes conductivity changes
linearization = forward.linearize(reference)
centers = space.centers
truth = .03*((centers[:, 0] > 0) & (np.abs(centers[:, 1]) < .4))
delta_y = forward.evaluate(truth)-linearization.prediction
# %% Create synthetic data
rng = np.random.default_rng(23)
delta_y += .001*np.std(delta_y)*rng.standard_normal(delta_y.shape)
J = linearization.jacobian()
scale = np.linalg.norm(J.toarray(), 2)
# Common scaling for both backends. This is normalization, not a measured
# noise estimate. Change lambda when changing W or grid spacing.
data_space = DataSpace(len(delta_y), standard_deviation=scale)
solvers = {"scipy": SciPySolver(abs_tol=1e-8), "cvxpy": CVXPYSolver()}
chosen = solvers if solver_backend == "both" else {solver_backend: solvers[solver_backend]}
norms = ["l2", "l1"] if regularization_norm == "both" else [regularization_norm]
orders = range(3) if regularization_order == "all" else [int(regularization_order)]
saved = dict(triangles=triangles, x_edges=edges, y_edges=edges, mask=space.mask,
             active_indices=space.active_indices, pixel_areas=space.pixel_areas,
             fem_areas=space.fem_areas, reference_fem=space.offset,
             J_pixel=J.toarray(), delta_y=delta_y, truth_pixel=truth,
             truth_fem_change=space.P @ truth, data_scale=scale)
print(f"FEM parameters: {raw.n_parameters}; active pixels: {space.ndof}; J: {J.shape}")
print("backend norm order objective weighted_residual iterations")
for name, solver in chosen.items():
    for norm in norms:
        for order in orders:
            L = grid_operator(space.shape, order, mask=space.mask)
            weight = 1e-3 if norm == "l2" else 1e-5
            problem = InverseProblem(J, delta_y, [Regularizer(L, weight, norm)],
                                     data_space=data_space, bounds=(-.2, .2))
            result = solver.solve(problem)
            key = f"{name}_{norm}_order{order}"
            saved[key] = result.parameters
            saved[key+"_fem_change"] = space.P @ result.parameters
            saved[key+"_lambda"] = weight
            print(f"{name:7} {norm} {order} {result.objective:.6g} "
                  f"{result.weighted_residual_norm:.4g} {result.iterations}")
# %% Save numerical results
if output_file:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, **saved)
    print(f"Saved {output_file}; P and R can be rebuilt from saved triangles/edges")

# %% Show every selected backend, norm and derivative order
from examples._common import plot_pixel_maps
reconstruction_keys = [f"{name}_{norm}_order{order}"
                       for name in chosen for norm in norms for order in orders]
fig_fields, axes_fields = plot_pixel_maps(space,
    [truth, *[saved[key] for key in reconstruction_keys]],
    ["True change", *[key.replace("_", " ") for key in reconstruction_keys]])
fig_fields.savefig(output_dir / "pixel_inverse_fields.png", dpi=160)

# %% Data fit for every reconstruction
fig_data, ax_data = plt.subplots(figsize=(10, 4), layout="constrained")
ax_data.plot(delta_y, "k.", label="Noisy nonlinear data")
for key in reconstruction_keys:
    ax_data.plot(J @ saved[key], alpha=.7, label=key.replace("_", " "))
ax_data.set(xlabel="Measurement index", ylabel="Voltage change (V)")
ax_data.legend(fontsize=7, ncol=2)
fig_data.savefig(output_dir / "pixel_inverse_data.png", dpi=160)

plt.show()
