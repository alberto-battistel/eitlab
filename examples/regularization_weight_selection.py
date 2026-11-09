"""Compare L-curve, known-truth correlation and held-out CV weight selection."""

# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eit_inverse import (InverseProblem, Regularizer, grid_operator, SciPySolver,
                         CVXPYSolver, regularization_path, select_lcurve,
                         select_correlation, cross_validate)

# %% Settings — edit these before running the following cells
solver_backend = 'both'  # ['scipy', 'cvxpy']
norm = 'l2'  # ['l2', 'l1']
order = 1  # [0, 1, 2]
output_dir = Path("results/regularization_weight_selection")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "regularization_weight_selection.npz"
plot_file = output_dir / "regularization_weight_selection.png"

# %% Compute the demonstration
rng = np.random.default_rng(32)
t, obs = np.linspace(0, 1, 30), np.linspace(0, 1, 60)
A = np.exp(-.5*((obs[:, None]-t[None, :])/.04)**2)
A /= A.sum(axis=1, keepdims=True)
truth = .8*((t > .2) & (t < .45)) + np.maximum(0, 1-np.abs(t-.72)/.15)
b = A @ truth + .04*rng.standard_normal(len(obs))
problem = InverseProblem(A, b, [Regularizer(grid_operator((len(t),), order), 1., norm)])
solver = SciPySolver() if solver_backend == "scipy" else CVXPYSolver()
weights = np.logspace(-5, 1, 25)
path = regularization_path(problem, solver, weights)
selections = {
    "L-curve": select_lcurve(path),
    "known-truth correlation": select_correlation(path, truth),
    "held-out MSE": cross_validate(problem, solver, weights, n_splits=4, seed=6),
}
print(f"{solver_backend}; {norm}; order {order}; {len(weights)} weight candidates")
print("method weight relative_truth_error")
for name, selected in selections.items():
    error = np.linalg.norm(selected.parameters-truth)/np.linalg.norm(truth)
    print(f"{name:24} {selected.weight:.6g} {error:.4g}")
    for message in selected.messages:
        print("  "+message)
# %% Save numerical results
if output_file:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, weights=weights, solutions=path.solutions,
             residual_norms=path.residual_norms, penalty_norms=path.penalty_norms,
             lcurve_curvature=selections["L-curve"].scores,
             reference_correlations=selections["known-truth correlation"].scores,
             cv_scores=selections["held-out MSE"].scores,
             cv_fold_scores=selections["held-out MSE"].diagnostics["fold_scores"],
             selected_weights=np.array([r.weight for r in selections.values()]),
             selection_methods=np.array(list(selections)), coordinates=t, truth=truth,
             data=b, A=A, norm=norm, order=order, backend=solver_backend,
             noise_std=.04, rng_seed=32, cv_seed=6)
# %% Plot the results
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes[0, 0].loglog(path.residual_norms, path.penalty_norms, "o-")
corner = selections["L-curve"]
axes[0, 0].plot(path.residual_norms[corner.index], path.penalty_norms[corner.index], "r*", ms=14)
axes[0, 0].set(xlabel="weighted residual norm", ylabel="unweighted penalty norm", title="L-curve")
axes[0, 1].semilogx(weights, selections["known-truth correlation"].scores)
axes[0, 1].set(xlabel="weight", ylabel="Pearson correlation", title="Known-truth reference")
axes[1, 0].semilogx(weights, selections["held-out MSE"].scores)
axes[1, 0].set(xlabel="weight", ylabel="held-out MSE", title="Four-fold measurement CV")
axes[1, 1].plot(t, truth, "k--", label="truth")
for name, selected in selections.items():
    axes[1, 1].plot(t, selected.parameters, label=f"{name}: {selected.weight:.2g}")
axes[1, 1].set(xlabel="position", ylabel="reconstruction")
axes[1, 1].legend(fontsize=8)
fig.suptitle(f"Regularization weight selection: {norm.upper()}, order {order}")
fig.tight_layout()
plot_file.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(plot_file, dpi=160)

plt.show()
