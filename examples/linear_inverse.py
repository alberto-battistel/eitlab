"""A NumPy-only example of the inverse layer, independent of FEM."""

# %% Imports
import matplotlib.pyplot as plt
import numpy as np

from eit_inverse import TikhonovSolver


from pathlib import Path

# %% Settings — edit these before running the following cells
output_dir = Path("results/linear_inverse")
output_dir.mkdir(parents=True, exist_ok=True)

# %% Compute the demonstration
rng = np.random.default_rng(23)
J = rng.normal(size=(12, 4))
truth = np.array([0.1, -0.2, 0.3, 0.05])
clean_data = J @ truth
data = clean_data + 0.01 * rng.standard_normal(clean_data.shape)
lambdas = [1e-4, 1e-2, 1.0]
estimates = TikhonovSolver(lambdas, J).solve(data)
print("True parameter change:", truth)
for column, lam in enumerate(lambdas):
    estimate = estimates[:, column]
    print(f"lambda={lam:g}: {estimate}, error={np.linalg.norm(estimate - truth):.3g}")
print("Objective: ||J delta_p - delta_y||^2 + lambda ||delta_p||^2")

# %% Parameter recovery and measured-data fits
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
indices = np.arange(len(truth))
axes[0].plot(indices, truth, "ko--", label="Truth")
axes[1].plot(data, "ko", label="Noisy observations")
for column, lam in enumerate(lambdas):
    axes[0].plot(indices, estimates[:, column], "x-", label=f"lambda={lam:g}")
    axes[1].plot(J @ estimates[:, column], label=f"lambda={lam:g}")
axes[0].set(xlabel="Parameter index", ylabel="Parameter change (a.u.)", xticks=indices)
axes[1].set(xlabel="Observation index", ylabel="Data (a.u.)")
for ax in axes:
    ax.legend(fontsize=8)
fig.savefig(output_dir / "linear_inverse.png", dpi=160)

plt.show()
