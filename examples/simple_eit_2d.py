"""2D EIT with independent meshes, two protocols, L2(0) and TV reconstruction.

From the updated project root: python -m pip install -e ".[fem,plot,cvxpy]"
Then run this script or its # %% cells in an IDE/notebook. All variables remain
in the namespace. Figures are saved in results/simple_eit_2d/. No added noise.
"""

# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eitlab import (
    ElectrodeDisk, ElectricalModel, ElectricalLinearization, JacobianAssembler,
    MeasurementPlan, MeasurementRunner, ProtocolFactory, CurrentProtocol,
    AdjacentDifference, PotentialReference, ElementwiseConductivity,
)
from eitlab import plotting as viz
from eit_inverse import (
    InverseProblem, Regularizer, SciPySolver, CVXPYSolver,
)

# %% Parameters
background = 1.0
inclusion = 0.7
center = (0.35, 0.2)
radius = 0.25
n_electrodes = 16
maxh_simulation = 0.07
maxh_jacobian = 0.16
current_amplitude = 1.0
contact_impedance = 0.01
alpha_l2 = 3e-3
alpha_tv = 5e-5
output_dir = Path("results/simple_eit_2d")
output_dir.mkdir(parents=True, exist_ok=True)

# %% 1. Two independently generated meshes
# The fine mesh resolves the inclusion; use it for BOTH simulated data sets so
# their voltage difference is not contaminated by a baseline mesh difference.
simulation_geometry = ElectrodeDisk(radius=1.0, n_electrodes=n_electrodes)
simulation_geometry.add_disk_inclusion(name="inclusion", p=center, r=radius)
simulation_mesh = simulation_geometry.make_mesh(maxh=maxh_simulation)

# This coarser homogeneous disk has no inclusion region or inclusion boundary.
jacobian_geometry = ElectrodeDisk(radius=1.0, n_electrodes=n_electrodes)
jacobian_mesh = jacobian_geometry.make_mesh(maxh=maxh_jacobian)
electrodes = tuple(simulation_geometry.electrode_names)
assert electrodes == tuple(jacobian_geometry.electrode_names)
parameter_space = ElementwiseConductivity(jacobian_mesh)
sigma0 = parameter_space.create(background)

# %% 2. Two current protocols, with differential voltage readings
adjacent = ProtocolFactory.adjacent(electrodes, amplitude=current_amplitude)
skip3 = ProtocolFactory.jump_and_skip(
    electrodes, electrodes[0], amplitude=current_amplitude, jump=1, skip=3)
protocol = CurrentProtocol([*adjacent, *skip3], name="adjacent + skip-3")
# Every injection measures U[k] - U[k+1], including wrap-around. This includes
# pairs touching current electrodes. These 16 channels sum to zero per injection.
plan = MeasurementPlan(protocol, AdjacentDifference)
protocol_names = ["Adjacent injection", "Skip-3 injection"]
protocol_slices = [slice(0, n_electrodes**2), slice(n_electrodes**2, 2*n_electrodes**2)]

# %% 3. Fine-mesh simulation, without and with inclusion
simulation_space = ElementwiseConductivity(simulation_mesh)
sigma_without = simulation_space.create(background)
sigma_true = simulation_space.create({"body": background, "inclusion": inclusion})
simulation_model = ElectricalModel(
    simulation_mesh, electrodes, sigma=sigma_without, order=2,
    z_contact=contact_impedance, potential_reference=PotentialReference.MEAN_ZERO)
baseline = MeasurementRunner(simulation_model, plan).run(store_solutions=False)
simulation_model.set_sigma(sigma_true)
measured = MeasurementRunner(simulation_model, plan).run(store_solutions=False)
voltage_without = baseline.vector
voltage_with = measured.vector
delta_voltage = voltage_with - voltage_without

# %% 4. Jacobian at the coarse homogeneous model
jacobian_model = ElectricalModel(
    jacobian_mesh, electrodes, sigma=sigma0, order=2,
    z_contact=contact_impedance, potential_reference=PotentialReference.MEAN_ZERO)
jacobian_baseline = MeasurementRunner(jacobian_model, plan).run()
linearization = ElectricalLinearization(jacobian_model, parameter_space=parameter_space)
J = JacobianAssembler(linearization, jacobian_baseline).assemble_total_jacobian()
# Difference imaging uses the fine simulated homogeneous reference above, NOT
# measured - jacobian_baseline: that would add a coarse/fine baseline mismatch.

# %% 5. Request spatial operators from the conductivity discretization
L0, Dtv = parameter_space.regularization_operators()
# ||L0*x||_2^2 = sum_T area(T)*x_T^2: zero-order L2 Tikhonov.
# ||Dtv*x||_1 = sum_internal_edges length(edge)*|x_left-x_right|: P0 TV.
# The adapter owns areas, adjacency and DOF ordering. The inverse framework
# receives only sparse matrices. No mesh traversal or native FEM fields here.

# %% 6. Solve both inverse problems through the common inverse framework
# Normalize the data term once; alpha values below apply to this scaled problem.
data_scale = np.linalg.norm(J, ord=2)
A = J / data_scale
b = delta_voltage / data_scale
problem_l2 = InverseProblem(A, b, [Regularizer(L0, alpha_l2, norm="l2", name="L2(0)")])
problem_tv = InverseProblem(A, b, [Regularizer(Dtv, alpha_tv, norm="l1", name="TV")])
result_l2 = SciPySolver().solve(problem_l2)
result_tv = CVXPYSolver(solver="CLARABEL").solve(problem_tv)
delta_sigma_l2 = result_l2.parameters
delta_sigma_tv = result_tv.parameters
# Both estimates are conductivity CHANGES; add the known background to display.
# Signed/unconstrained estimates may be plotted; the forward model still
# rejects nonpositive physical conductivity when accepting a parameter.
sigma_l2 = parameter_space.create(background + delta_sigma_l2, positive=False)
sigma_tv = parameter_space.create(background + delta_sigma_tv, positive=False)
print(f"Simulation: {simulation_mesh.ne} triangles; Jacobian: {jacobian_mesh.ne} triangles")
print(f"J: {J.shape}; L2: {result_l2.status}; TV: {result_tv.status}")
for name, estimate in [("L2(0)", delta_sigma_l2), ("TV", delta_sigma_tv)]:
    residual = np.linalg.norm(J @ estimate - delta_voltage) / np.linalg.norm(delta_voltage)
    print(f"{name} relative linearized voltage residual: {residual:.3%}")

# %% 7. Plot the two meshes using the new plotting utility
fig_meshes, mesh_axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
for ax, mesh, title in zip(mesh_axes, [simulation_mesh, jacobian_mesh], [
    f"Simulation mesh ({simulation_mesh.ne} triangles)",
    f"Homogeneous Jacobian mesh ({jacobian_mesh.ne} triangles)",
]):
    viz.plot_mesh(mesh, backend="matplotlib", ax=ax, color="whitesmoke", title=title)
    viz.plot_boundaries(mesh, {name: "tab:red" for name in electrodes},
                        backend="matplotlib", ax=ax, line_width=2)
fig_meshes.savefig(output_dir / "simple_eit_2d_meshes.png", dpi=180)

# %% 8. Conductivity images using the same utility and a shared color scale
fields = [sigma_without, sigma_true, sigma_l2, sigma_tv]
meshes = [simulation_mesh, simulation_mesh, jacobian_mesh, jacobian_mesh]
values = [viz.field_data(mesh, field).values for mesh, field in zip(meshes, fields)]
clim = (min(v.min() for v in values), max(v.max() for v in values))
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), layout="constrained")
plot_results = []
for ax, mesh, field, title in zip(axes, meshes, fields, [
    "Without inclusion", "With inclusion (truth)", "L2(0) / Tikhonov", "Total variation",
]):
    plot_results.append(viz.plot_field(
        mesh, field, backend="matplotlib", ax=ax, cmap="viridis", clim=clim,
        colorbar=False, show_edges=False, title=title))
fig.colorbar(plot_results[-1].artist, ax=axes, label="Conductivity")
fig.savefig(output_dir / "simple_eit_2d.png", dpi=180)

# %% 9. Plot signed differential voltages for each current protocol
# x is injection-major: 16 electrode-pair channels for each of 16 injections.
fig_voltages, voltage_axes = plt.subplots(2, 2, figsize=(12, 6), layout="constrained")
for row, (name, block) in enumerate(zip(protocol_names, protocol_slices)):
    x = np.arange(n_electrodes**2)
    voltage_axes[row, 0].plot(x, voltage_without[block], label="Without inclusion")
    voltage_axes[row, 0].plot(x, voltage_with[block], "--", label="With inclusion")
    voltage_axes[row, 0].set(title=name, ylabel="U[k] - U[k+1] (V)")
    voltage_axes[row, 0].legend()
    voltage_axes[row, 1].plot(x, delta_voltage[block], color="black", label="Simulated change")
    voltage_axes[row, 1].plot(x, (J @ delta_sigma_l2)[block], "--", label="L2 prediction")
    voltage_axes[row, 1].plot(x, (J @ delta_sigma_tv)[block], ":", label="TV prediction")
    voltage_axes[row, 1].set(title=f"{name}: inclusion-induced change", ylabel="Delta voltage (V)")
    voltage_axes[row, 1].legend()
    for ax in voltage_axes[row]:
        ax.set_xlabel("Measurement index (injection-major)")
        ax.grid(alpha=0.2)
fig_voltages.savefig(output_dir / "simple_eit_2d_voltages.png", dpi=180)
plt.show()
