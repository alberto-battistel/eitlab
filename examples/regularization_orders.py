"""Six regularizers on one reproducible 1D linear blur inverse problem.

This example is numerical only: it has no NGSolve dependency.
"""

# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eit_inverse import InverseProblem, Regularizer, SciPySolver, CVXPYSolver, grid_operator

# %% Settings — edit these before running the following cells
solver_backend = 'both'  # ['scipy', 'cvxpy', 'both']
output_dir = Path("results/regularization_orders")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "regularization_orders.npz"
plot_file = output_dir / "regularization_orders.png"

# %% Compute the demonstration
rng = np.random.default_rng(42)
t = np.linspace(0, 1, 60)
obs = np.linspace(0, 1, 45)
A = np.exp(-0.5*((obs[:, None]-t[None, :])/.035)**2)
A /= A.sum(axis=1, keepdims=True)
truth = .8*((t > .2) & (t < .42)) + np.maximum(0, 1-np.abs(t-.73)/.17)
b = A @ truth + .015*rng.standard_normal(len(obs))
solvers = {"scipy": SciPySolver(), "cvxpy": CVXPYSolver()}
chosen = solvers if solver_backend == "both" else {solver_backend: solvers[solver_backend]}
# spacing=1 means derivatives per sample index. These demonstration weights
# are not universal choices, and L1 and L2 weights have different units.
weights = {"l2": [.04, .25, 2.0], "l1": [.005, .03, .08]}
saved = dict(coordinates=t, truth=truth, A=A, data=b)
print("backend norm order objective residual relative_error iterations")
for name, solver in chosen.items():
    for norm in ("l2", "l1"):
        for order in range(3):
            L = grid_operator((len(t),), order)
            reg = Regularizer(L, weights[norm][order], norm, name=f"{norm}_order{order}")
            result = solver.solve(InverseProblem(A, b, [reg]))
            key = f"{name}_{norm}_order{order}"
            saved[key] = result.parameters
            saved[key+"_lambda"] = reg.weight
            error = np.linalg.norm(result.parameters-truth)/np.linalg.norm(truth)
            print(f"{name:7} {norm} {order} {result.objective:.6g} "
                  f"{result.weighted_residual_norm:.4g} {error:.4g} {result.iterations}")
# %% Save numerical results
if output_file:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, **saved)
    print(f"Saved {output_file}")
# %% Plot the results
fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True, layout="constrained")
for row, norm in enumerate(("l2", "l1")):
    for order in range(3):
        ax = axes[row, order]
        ax.plot(t, truth, "k--", label="truth")
        for name in chosen:
            ax.plot(t, saved[f"{name}_{norm}_order{order}"], label=name)
        ax.set_title(f"{norm.upper()}, order {order}")
        ax.set_xlabel("position")
axes[0, 0].legend()
for ax in axes[:, 0]:
    ax.set_ylabel("Parameter (a.u.)")
fig.suptitle("L2 and L1 regularization: orders 0, 1 and 2")
plot_file.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(plot_file, dpi=160)

print(f"Saved {plot_file}")

plt.show()
