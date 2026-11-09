"""Optional small end-to-end FEM -> measurements/Jacobian -> NumPy inverse.

Uses one mesh for simulation and reconstruction as an implementation check.
It is not evidence of reconstruction accuracy on independent experimental data.
"""

# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples._common import load_backend, make_disk, make_model, make_plan
from eit_inverse import TikhonovSolver

# %% Settings — edit these before running the following cells
output_dir = Path("results/disk_reconstruction")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "disk_reconstruction.npz"

# %% Compute the demonstration
backend = load_backend()
mesh, electrodes = make_disk(backend)
space = backend.ng.L2(mesh, order=0)
sigma_ref = backend.ng.GridFunction(space)
sigma_ref.Set(1.0)
model = make_model(backend, mesh, electrodes, sigma_ref)
plan = make_plan(backend, electrodes, exclude_current=True)
runner = backend.measurement.MeasurementRunner(model, plan)
results = runner.run()
linearization = backend.fem.ElectricalLinearization(model, parameter_space=space)
J = backend.fem.JacobianAssembler(linearization, results).assemble_total_jacobian()
y_ref = results.voltages.ravel()

# Project an off-center disk into the same parameter space.
sigma_true = backend.ng.GridFunction(space)
region = backend.ng.IfPos(0.35**2 - (backend.ng.x - 0.25)**2 - backend.ng.y**2, 1, 0)
sigma_true.Set(1.0 + 0.03 * region)
true_change = sigma_true.vec.FV().NumPy().copy() - sigma_ref.vec.FV().NumPy()
truth_model = make_model(backend, mesh, electrodes, sigma_true)
y_true = backend.measurement.MeasurementRunner(truth_model, plan).run().voltages.ravel()
# %% Create synthetic data
rng = np.random.default_rng(23)
noise_sd = 1e-3 * np.std(y_true - y_ref)
delta_y = y_true - y_ref + noise_sd * rng.standard_normal(y_true.shape)

# Only these numerical arrays enter the inverse layer.
lam = 1e-3 * np.linalg.norm(J, ord=2)**2
estimate = TikhonovSolver(lam, J).solve(delta_y)
relative_residual = np.linalg.norm(J @ estimate - delta_y) / np.linalg.norm(delta_y)
linearization_error = np.linalg.norm(J @ true_change - (y_true - y_ref)) / np.linalg.norm(y_true - y_ref)
print(f"J shape: {J.shape}, lambda: {lam:.3g}")
print(f"Linearization relative error: {linearization_error:.3g}")
print(f"Reconstruction data residual: {relative_residual:.3g}")
print("Same-mesh synthetic demonstration; small residual does not prove accurate imaging.")
# %% Save numerical results
if output_file:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, J=J, delta_y=delta_y, y_ref=y_ref, y_true=y_true,
             delta_sigma_true=true_change, delta_sigma_estimate=estimate,
             regularization_lambda=lam, noise_sd=noise_sd,
             electrode_names=np.asarray(electrodes), parameter_ndof=space.ndof)
    print(f"Saved numerical arrays to {output_file}")
    print("Parameter arrays use this run's L2 DOF ordering; the mesh is not saved.")

# %% Compare conductivity maps on a common scale
from examples._common import plot_parameter_maps
fig_fields, axes_fields = plot_parameter_maps(
    mesh, space, [true_change, estimate, estimate-true_change],
    ["True change", "Tikhonov reconstruction", "Reconstruction error"])
fig_fields.savefig(output_dir / "disk_reconstruction_fields.png", dpi=160)

# %% Compare measured and fitted voltage changes
fig_data, axes_data = plt.subplots(2, 1, sharex=True, figsize=(9, 5), layout="constrained")
axes_data[0].plot(delta_y, label="Noisy nonlinear data")
axes_data[0].plot(J @ estimate, "--", label="Linearized fit")
axes_data[0].set(ylabel="Voltage change (V)", title="Same-mesh synthetic data fit")
axes_data[0].legend()
axes_data[1].plot(J @ estimate-delta_y)
axes_data[1].set(xlabel="Measurement index", ylabel="Fit residual (V)")
fig_data.savefig(output_dir / "disk_reconstruction_data.png", dpi=160)

plt.show()
