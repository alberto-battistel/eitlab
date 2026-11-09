"""Independent-mesh EIT simulation and reconstruction in editable # %% cells.

All computed arrays stay in the namespace. Figures are displayed and saved in
results/single_circle/. See examples._single_circle for an importable pipeline.
"""
# %% Imports
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from eit_forward import geometries, measurement, stimulation
import eit_forward as forward_api
from eit_forward.backend import load_backend
from eit_forward.adapters import NGSolveConductivityForward, NGSolveAdjointJacobian
from eitlab.differentiation import FiniteDifferenceJacobian
from eit_inverse import TikhonovSolver
from examples._single_circle import (ExampleConfig, build_geometry, generate_measurements,
    reconstruction_mesh_arrays, calc_abs_correlation)

# %% Settings
config = ExampleConfig(
    n_electrodes=16, order=2, maxh_jacobian=0.1, maxh_forward=0.05,
    noise_level=1e-3, regularization_lambda=1e-4,
)
jacobian_method = "adjoint"  # "forward", "central" or "backward" also work
output_dir = Path("results/single_circle")
output_dir.mkdir(parents=True, exist_ok=True)

# %% Coarse reference model and Jacobian
geometry = build_geometry(config)
mesh = geometry.make_mesh(maxh=config.maxh_jacobian)
electrodes = tuple(geometry.electrode_names)
protocol = stimulation.ProtocolFactory.adjacent(electrodes, amplitude=1.0)
plan = measurement.MeasurementPlan(protocol, measurement.SingleEnded)
backend = load_backend()
forward = NGSolveConductivityForward(
    backend, mesh, electrodes, plan, z_contact=config.z_contact,
    order=config.order, inverse=config.inverse,
    potential_reference=forward_api.PotentialReference.CURRENT_ELECTRODES_MEAN)
parameters = np.full(forward.n_parameters, config.sigma_background)
if jacobian_method == "adjoint":
    provider = NGSolveAdjointJacobian(forward)
elif jacobian_method in ("forward", "central", "backward"):
    provider = FiniteDifferenceJacobian(forward.evaluate, step=config.finite_difference_step,
                                         scheme=jacobian_method)
else:
    raise ValueError("Unknown Jacobian method")
J = provider.jacobian(parameters)
# %% Independent fine-mesh data and difference reconstruction
homogeneous, inclusion, noise, noise_sd = generate_measurements(config, electrodes, plan)
delta_y = inclusion + noise - homogeneous
# The inverse solver consumes arrays only. No mesh or FEM state is passed.
estimate = TikhonovSolver(config.regularization_lambda, J).solve(delta_y)
coordinates, triangles, dofs = reconstruction_mesh_arrays(mesh, forward.parameter_space)
centers = coordinates[triangles].mean(axis=1)
inside = np.linalg.norm(centers - config.inclusion_center, axis=1) <= config.inclusion_radius
true_change = np.zeros(forward.n_parameters)
true_change[dofs] = np.where(inside, config.sigma_inclusion - config.sigma_background, 0.0)
result = dict(J=J, reference_parameters=parameters, delta_sigma_estimate=estimate,
            delta_sigma_true_at_centers=true_change, V_homo=homogeneous,
            V_inho=inclusion, noise=noise, noise_sd=noise_sd, delta_y=delta_y,
            regularization_lambda=config.regularization_lambda,
            correlation=calc_abs_correlation(true_change, estimate),
            coordinates=coordinates, triangles=triangles, parameter_dofs=dofs,
            electrode_names=np.asarray(electrodes), jacobian_method=jacobian_method)

# %% Save numerical results
np.savez_compressed(output_dir / "single_circle.npz", **result)
print(f"Jacobian: {J.shape}; absolute correlation: {result['correlation']:.3f}")

# %% Voltages and conductivity maps
from matplotlib.patches import Circle
fig_voltage, axes_voltage = plt.subplots(3, 1, sharex=True, figsize=(10, 8), layout="constrained")
axes_voltage[0].plot(result["V_homo"], label="Homogeneous")
axes_voltage[0].plot(result["V_inho"], label="Inclusion")
axes_voltage[0].set_ylabel("Voltage / V")
axes_voltage[0].legend()
axes_voltage[1].plot(delta_y, label="Noisy nonlinear difference")
axes_voltage[1].plot(J @ estimate, "--", label="Linearized reconstruction fit")
axes_voltage[1].set(ylabel="Voltage change (V)")
axes_voltage[1].legend()
axes_voltage[2].plot(J @ estimate-delta_y)
axes_voltage[2].set(xlabel="Measurement index", ylabel="Fit residual (V)")
fig_voltage.savefig(output_dir / "voltages_single_circle.svg", transparent=True)
fig_voltage.savefig(output_dir / "voltages_single_circle.png", dpi=160)

# %% Conductivity truth and reconstruction
xy, triangles, dofs = result["coordinates"], result["triangles"], result["parameter_dofs"]
fig_fields, axes_fields = plt.subplots(1, 2, layout="constrained", figsize=(8, 4))
color_limit = max(float(np.max(np.abs(result["delta_sigma_true_at_centers"]))),
                  float(np.max(np.abs(result["delta_sigma_estimate"]))), np.finfo(float).eps)
for axis, key, title in zip(axes_fields, ["delta_sigma_true_at_centers", "delta_sigma_estimate"],
                            ["True change at cell centers", "Reconstructed change"]):
    image = axis.tripcolor(xy[:, 0], xy[:, 1], triangles,
                           facecolors=result[key][dofs], shading="flat", cmap="RdBu_r",
                           vmin=-color_limit, vmax=color_limit)
    axis.add_patch(Circle(config.inclusion_center, config.inclusion_radius,
                          fill=False, color="black", linewidth=1))
    axis.set(aspect="equal", title=title)
    axis.axis("off")
    fig_fields.colorbar(image, ax=axis, label="Conductivity change / S m⁻¹")
fig_fields.savefig(output_dir / "tikhonov_single_circle.svg", transparent=True)
fig_fields.savefig(output_dir / "tikhonov_single_circle.png", dpi=150)

plt.show()
