"""Compare reconstruction features of portable regularizers and a CVXPY extension.

The example is numerical only and has no FEM dependency. SciPy and CVXPY solve
identical L2, LASSO, anisotropic-TV and Huber problems. A final custom
CVXPY expression demonstrates the backend-specific escape hatch. The purpose is to compare reconstruction character rather than
tune universally optimal weights.
"""

# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from eit_inverse import (CVXPYExpressionRegularizer, CVXPYSolver, InverseProblem,
                         Regularizer, SciPySolver, grid_operator)

# %% Settings
output_dir = Path("results/regularization_features")
output_dir.mkdir(parents=True, exist_ok=True)
rng_seed = 17
noise_standard_deviation = 0.02
weights = {"L2": 0.08, "LASSO": 0.025, "TV": 0.035, "Huber": 0.08, "Custom": 0.02}

# %% Synthetic piecewise/sparse 1D object and blurred measurements
rng = np.random.default_rng(rng_seed)
x_coordinate = np.linspace(0, 1, 90)
measurement_coordinate = np.linspace(0, 1, 65)
A = np.exp(-0.5*((measurement_coordinate[:, None]-x_coordinate[None, :])/0.035)**2)
A /= A.sum(axis=1, keepdims=True)
truth = (0.75*((x_coordinate > .16) & (x_coordinate < .36))
         + 0.5*((x_coordinate > .55) & (x_coordinate < .72)))
truth += 0.45*np.maximum(0, 1-np.abs(x_coordinate-.84)/.07)
data_clean = A @ truth
data = data_clean + noise_standard_deviation*rng.standard_normal(len(data_clean))

I = sparse.eye(len(x_coordinate), format="csr")
D = grid_operator((len(x_coordinate),), 1, spacing=x_coordinate[1]-x_coordinate[0])
problems = {
    "L2": InverseProblem(A, data, [Regularizer(I, weights["L2"], "l2", name="L2")]),
    "LASSO": InverseProblem(A, data, [Regularizer(I, weights["LASSO"], "l1", name="LASSO")]),
    "TV": InverseProblem(A, data, [Regularizer(D, weights["TV"], "l1", name="TV")]),
    "Huber": InverseProblem(A, data, [Regularizer(D, weights["Huber"], "huber",
                                                   huber_M=0.15, name="Huber gradient")]),
    # Backend-specific escape hatch: an arbitrary convex expression. This one
    # penalizes both the maximum absolute amplitude and its L1 norm.
    "Custom": InverseProblem(A, data, [CVXPYExpressionRegularizer(
        len(x_coordinate),
        expression=lambda x, cp: cp.norm_inf(x) + 0.1*cp.norm1(x),
        value=lambda x: np.linalg.norm(x, np.inf) + 0.1*np.linalg.norm(x, 1),
        weight=weights["Custom"], name="custom CVXPY penalty")]),
}

# %% Solve portable penalties; demonstrate the CVXPY-specific extension separately
scipy_solver = SciPySolver()
cvxpy_solver = CVXPYSolver(solver="CLARABEL")
results = {}
for name in ("L2", "LASSO", "TV"):
    results[(name, "SciPy")] = scipy_solver.solve(problems[name])
    results[(name, "CVXPY")] = cvxpy_solver.solve(problems[name])
results[("Huber", "CVXPY")] = cvxpy_solver.solve(problems["Huber"])
results[("Custom", "CVXPY")] = cvxpy_solver.solve(problems["Custom"])

print("regularizer backend objective residual relative_error")
for (name, backend), result in results.items():
    error = np.linalg.norm(result.parameters-truth)/np.linalg.norm(truth)
    print(f"{name:11} {backend:6} {result.objective:9.4g} "
          f"{result.weighted_residual_norm:8.4g} {error:8.4g}")
for name in ("L2", "LASSO", "TV"):
    difference = np.linalg.norm(results[(name, "SciPy")].parameters
                                - results[(name, "CVXPY")].parameters)
    print(f"{name} SciPy/CVXPY parameter difference: {difference:.3e}")

# %% Reconstruction comparison
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True,
                         layout="constrained")
for ax, name in zip(axes.ravel(), ("L2", "LASSO", "TV", "Huber")):
    ax.plot(x_coordinate, truth, "k--", label="Truth")
    if name != "Huber":
        ax.plot(x_coordinate, results[(name, "SciPy")].parameters, label="SciPy")
    ax.plot(x_coordinate, results[(name, "CVXPY")].parameters,
            ":" if name != "Huber" else "-", label="CVXPY")
    ax.set_title(name)
    ax.set_xlabel("Position")
    ax.grid(alpha=.2)
    ax.legend()
for ax in axes[:, 0]:
    ax.set_ylabel("Parameter (a.u.)")
fig.suptitle("Portable regularization features: smooth, sparse, edge-preserving and Huber")
fig.savefig(output_dir / "regularization_features.png", dpi=180)

# %% Data fits
fig_fit, ax_fit = plt.subplots(figsize=(8, 4), layout="constrained")
ax_fit.plot(measurement_coordinate, data, ".", label="Noisy data")
ax_fit.plot(measurement_coordinate, data_clean, "k--", label="Clean data")
for name in ("L2", "LASSO", "TV"):
    ax_fit.plot(measurement_coordinate, A @ results[(name, "SciPy")].parameters, label=name)
ax_fit.plot(measurement_coordinate, A @ results[("Huber", "CVXPY")].parameters, label="Huber")
ax_fit.plot(measurement_coordinate, A @ results[("Custom", "CVXPY")].parameters, label="Custom CVXPY")
ax_fit.set(xlabel="Measurement position", ylabel="Signal (a.u.)", title="Forward fits")
ax_fit.grid(alpha=.2)
ax_fit.legend(ncol=3)
fig_fit.savefig(output_dir / "regularization_features_fits.png", dpi=180)
plt.show()
