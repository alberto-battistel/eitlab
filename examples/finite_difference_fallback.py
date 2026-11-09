"""Compute an EIT Jacobian with either differentiation provider."""

# %% Imports

import matplotlib.pyplot as plt
import numpy as np

from examples._common import load_backend, make_disk, make_plan
from eitlab.differentiation import FiniteDifferenceJacobian
from eit_forward.adapters import NGSolveConductivityForward, NGSolveAdjointJacobian
from eit_inverse import TikhonovSolver


from pathlib import Path

# %% Settings — edit these before running the following cells
method = 'central'  # ['adjoint', 'forward', 'central']
step = 0.0001
compare = True
output_dir = Path("results/finite_difference_fallback")
output_dir.mkdir(parents=True, exist_ok=True)

# %% Compute the demonstration
backend = load_backend()
mesh, electrodes = make_disk(backend)
forward = NGSolveConductivityForward(backend, mesh, electrodes, make_plan(backend, electrodes, True))
parameters = np.ones(forward.n_parameters)
if method == "adjoint":
    provider = NGSolveAdjointJacobian(forward)
else:
    provider = FiniteDifferenceJacobian(forward.evaluate, step=step, scheme=method)
J = provider.jacobian(parameters)
print(f"Method: {method}, J shape: {J.shape}")
if compare:
    expected = NGSolveAdjointJacobian(forward).jacobian(parameters)
    error = np.linalg.norm(J - expected) / np.linalg.norm(expected)
    print(f"Relative error against adjoint: {error:.3e}")

# The inverse layer consumes the same arrays regardless of the provider.
true_change = np.zeros(forward.n_parameters)
true_change[:max(1, forward.n_parameters // 5)] = 0.01
delta_y = forward.evaluate(parameters + true_change) - forward.evaluate(parameters)
lam = 1e-3 * np.linalg.norm(J, ord=2)**2
estimate = TikhonovSolver(lam, J).solve(delta_y)
residual = np.linalg.norm(J @ estimate - delta_y) / np.linalg.norm(delta_y)
print(f"Regularized reconstruction data residual: {residual:.3e}")
print("Same-mesh implementation demonstration; measurement ordering and parameter basis are fixed.")

# %% Jacobian comparison and reconstruction
from examples._common import plot_parameter_maps
fig_jacobian, axes_jacobian = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
image = axes_jacobian[0].imshow(J, aspect="auto", cmap="RdBu_r",
                               vmin=-np.max(np.abs(J)), vmax=np.max(np.abs(J)))
axes_jacobian[0].set(title=f"{method.capitalize()} Jacobian", xlabel="Conductivity DOF", ylabel="Measurement index")
fig_jacobian.colorbar(image, ax=axes_jacobian[0], label="Voltage sensitivity (V / (S/m))")
if compare:
    difference = J-expected
    image_error = axes_jacobian[1].imshow(difference, aspect="auto", cmap="RdBu_r")
    axes_jacobian[1].set(title=f"Difference from adjoint: {error:.2e}",
                         xlabel="Conductivity DOF", ylabel="Measurement index")
    fig_jacobian.colorbar(image_error, ax=axes_jacobian[1], label="Sensitivity error (V / (S/m))")
else:
    axes_jacobian[1].plot(delta_y, label="Nonlinear voltage change")
    axes_jacobian[1].plot(J @ estimate, "--", label="Reconstructed fit")
    axes_jacobian[1].set(xlabel="Measurement index", ylabel="Voltage change (V)")
    axes_jacobian[1].legend()
fig_jacobian.savefig(output_dir / "finite_difference_jacobian.png", dpi=160)
fig_fields, axes_fields = plot_parameter_maps(mesh, forward.parameter_space,
    [true_change, estimate], ["True change", "Reconstructed change"])
fig_fields.savefig(output_dir / "finite_difference_fields.png", dpi=160)

plt.show()
