"""Joint conductivity/contact-impedance reconstruction in 2D.

Top-level # %% cells, independent simulation/Jacobian meshes, two injection
protocols, and adjacent differential voltages. No native NGSolve construction.
Run from the installed project: python -m examples.joint_eit_contact_2d
A few regularized Gauss-Newton steps illustrate the joint inverse problem.
This is a noiseless teaching example, not a calibrated experimental protocol.
Conductivity is real P0.
"""

# %% Imports
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from eitlab import (
    ElectrodeDisk, ElectricalModel, ElectricalLinearization, JacobianAssembler,
    MeasurementPlan, MeasurementRunner, ProtocolFactory, CurrentProtocol,
    AdjacentDifference, PotentialReference, ElementwiseConductivity,
    FiniteDifferenceJacobian,
)
from eitlab import plotting as viz
from eit_inverse import InverseProblem, Regularizer, SciPySolver

# %% Physical and reconstruction settings
background = 1.0
inclusion = 0.9                     # modest contrast for this teaching example
center = (0.35, 0.2)
radius = 0.25
n_electrodes = 16
maxh_simulation = 0.07
maxh_jacobian = 0.16
current_amplitude = 1.0
contact_impedance = 0.01            # specific CEM contact impedance, z
# Independent weights: increasing one suppresses that block's variations.
# These are illustrative choices for the scaled, noiseless data below.
alpha_sigma = 3e-3                  # area-weighted relative conductivity changes
alpha_contact = 1e-7               # per-electrode relative contact changes
contact_fd_step = 1e-4              # dimensionless relative contact perturbation
refinement_steps = 3                # set to 0 to inspect the one-step result
output_dir = Path("results/joint_eit_contact_2d")
output_dir.mkdir(parents=True, exist_ok=True)
# One unknown per electrode, always in the explicit electrode-name order.
angles = 2*np.pi*np.arange(n_electrodes)/n_electrodes
z_reference = np.full(n_electrodes, contact_impedance)
relative_z_true = .12*np.cos(2*angles)
relative_z_true[3] += .18
z_true = z_reference*(1 + relative_z_true)

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
simulation_model.set_contact_impedances(dict(zip(electrodes, z_true)))
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

# %% 5. Add the contact-impedance Jacobian block
# Unknown x = [relative conductivity change; relative contact-impedance change].
# Scaling removes the different parameter units from the optimizer.
n_sigma = parameter_space.ndof
n_contact = len(electrodes)
J_sigma = J * background

# This callback uses only the model's public update/measurement API.
# Contact derivatives use finite differences; conductivity uses the adjoint.
def contact_measurements(relative_change):
    values = z_reference*(1 + relative_change)
    jacobian_model.set_contact_impedances(dict(zip(electrodes, values)))
    return MeasurementRunner(jacobian_model, plan).run(store_solutions=False).vector

J_contact = FiniteDifferenceJacobian(
    contact_measurements, step=contact_fd_step, scheme="central"
).jacobian(np.zeros(n_contact))
# Every contact change invalidates old FEM states/linearizations. J_sigma is an
# independent numerical snapshot, already assembled before those updates.
jacobian_model.set_contact_impedances(dict(zip(electrodes, z_reference)))
J_joint = np.column_stack([J_sigma, J_contact])

# %% 6. Regularize the two parameter blocks separately
L0 = parameter_space.l2_operator()
L_sigma = sparse.hstack([L0, sparse.csr_matrix((n_sigma, n_contact))], format="csr")
L_contact = sparse.hstack([sparse.csr_matrix((n_contact, n_sigma)),
                           sparse.eye(n_contact)], format="csr")
regularizers = [
    Regularizer(L_sigma, alpha_sigma, norm="l2", name="Conductivity L2(0)"),
    Regularizer(L_contact, alpha_contact, norm="l2", name="Contact prior"),
]
data_scale = np.linalg.norm(J_sigma, ord=2)
# x=0 is the homogeneous reference, so zero priors mean unchanged sigma and z.
# Bounds keep reconstructed physical values positive and limit extrapolation.
problem_joint = InverseProblem(
    J_joint/data_scale, delta_voltage/data_scale, regularizers,
    bounds=(-.8, 1.0))
result_joint = SciPySolver().solve(problem_joint)
x_joint = result_joint.parameters
one_step_parameters = x_joint.copy()

# %% 7. Optional nonlinear refinement, still using the inverse framework
# The convex solver handles each local subproblem; backtracking checks the
# actual nonlinear voltage misfit PLUS both priors before accepting an update.
def joint_prediction(relative_parameters):
    jacobian_model.set_sigma(parameter_space.create(
        background*(1 + relative_parameters[:n_sigma])))
    jacobian_model.set_contact_impedances(dict(zip(
        electrodes, z_reference*(1 + relative_parameters[n_sigma:]))))
    return (MeasurementRunner(jacobian_model, plan).run(
        store_solutions=False).vector - jacobian_baseline.vector)


def joint_objective(relative_parameters, prediction):
    residual = (prediction-delta_voltage)/data_scale
    return float(residual @ residual) + sum(r.value(relative_parameters) for r in regularizers)

objective_history = []
for iteration in range(refinement_steps):
    prediction = joint_prediction(x_joint)
    objective = joint_objective(x_joint, prediction)
    objective_history.append(objective)
    states = MeasurementRunner(jacobian_model, plan).run()
    tangent = ElectricalLinearization(jacobian_model, parameter_space=parameter_space)
    current_J_sigma = JacobianAssembler(tangent, states).assemble_total_jacobian()*background
    current_J_contact = FiniteDifferenceJacobian(
        contact_measurements, step=contact_fd_step, scheme="central"
    ).jacobian(x_joint[n_sigma:])
    current_J = np.column_stack([current_J_sigma, current_J_contact])
    # Solve for the next absolute relative-parameter vector, not an unpenalized
    # increment: the prior always refers to the original homogeneous reference.
    local_data = delta_voltage-prediction + current_J @ x_joint
    local_problem = InverseProblem(current_J/data_scale, local_data/data_scale,
                                   regularizers, bounds=(-.8, 1.0))
    local_result = SciPySolver().solve(local_problem)
    direction = local_result.parameters-x_joint
    accepted = False
    for fraction in (1., .5, .25, .125, .0625):
        candidate = x_joint + fraction*direction
        candidate_prediction = joint_prediction(candidate)
        if joint_objective(candidate, candidate_prediction) < objective:
            x_joint = candidate
            accepted = True
            break
    if not accepted:
        break
objective_history.append(joint_objective(x_joint, joint_prediction(x_joint)))
relative_sigma_joint = x_joint[:n_sigma]
relative_z_estimate = x_joint[n_sigma:]
z_estimate = z_reference*(1 + relative_z_estimate)
sigma_joint = parameter_space.create(background*(1 + relative_sigma_joint))

# %% 8. Compare with the same inversion while assuming reference contacts
problem_fixed = InverseProblem(
    J_sigma/data_scale, delta_voltage/data_scale,
    [Regularizer(L0, alpha_sigma, norm="l2", name="Conductivity L2(0)")],
    bounds=(-.8, 1.0))
result_fixed = SciPySolver().solve(problem_fixed)
relative_sigma_fixed = result_fixed.parameters.copy()
one_step_fixed = relative_sigma_fixed.copy()
# Give the fixed-contact comparison the same nonlinear refinement budget.
fixed_objective_history = []
for iteration in range(refinement_steps):
    fixed_vector = np.r_[relative_sigma_fixed, np.zeros(n_contact)]
    prediction = joint_prediction(fixed_vector)
    objective = joint_objective(fixed_vector, prediction)
    fixed_objective_history.append(objective)
    states = MeasurementRunner(jacobian_model, plan).run()
    tangent = ElectricalLinearization(jacobian_model, parameter_space=parameter_space)
    fixed_J = JacobianAssembler(tangent, states).assemble_total_jacobian()*background
    local_problem_fixed = InverseProblem(
        fixed_J/data_scale,
        (delta_voltage-prediction+fixed_J @ relative_sigma_fixed)/data_scale,
        [Regularizer(L0, alpha_sigma, norm="l2")], bounds=(-.8, 1.0))
    local_result_fixed = SciPySolver().solve(local_problem_fixed)
    direction = local_result_fixed.parameters-relative_sigma_fixed
    accepted = False
    for fraction in (1., .5, .25, .125, .0625):
        candidate = relative_sigma_fixed + fraction*direction
        fixed_vector = np.r_[candidate, np.zeros(n_contact)]
        candidate_prediction = joint_prediction(fixed_vector)
        if joint_objective(fixed_vector, candidate_prediction) < objective:
            relative_sigma_fixed = candidate
            accepted = True
            break
    if not accepted:
        break
fixed_vector = np.r_[relative_sigma_fixed, np.zeros(n_contact)]
fixed_objective_history.append(joint_objective(fixed_vector, joint_prediction(fixed_vector)))
sigma_fixed = parameter_space.create(background*(1 + relative_sigma_fixed))
linear_prediction_joint = J_joint @ one_step_parameters
linear_prediction_fixed = J_sigma @ one_step_fixed
# Also evaluate both reconstructed models through the nonlinear forward solver.
jacobian_model.set_sigma(sigma_joint)
jacobian_model.set_contact_impedances(dict(zip(electrodes, z_estimate)))
nonlinear_prediction_joint = (MeasurementRunner(jacobian_model, plan).run(
    store_solutions=False).vector - jacobian_baseline.vector)
jacobian_model.set_sigma(sigma_fixed)
jacobian_model.set_contact_impedances(dict(zip(electrodes, z_reference)))
nonlinear_prediction_fixed = (MeasurementRunner(jacobian_model, plan).run(
    store_solutions=False).vector - jacobian_baseline.vector)
# Restore the reference model explicitly; stale FEM snapshots are not reused.
jacobian_model.set_sigma(sigma0)

linear_residual_joint = np.linalg.norm(linear_prediction_joint-delta_voltage)/np.linalg.norm(delta_voltage)
linear_residual_fixed = np.linalg.norm(linear_prediction_fixed-delta_voltage)/np.linalg.norm(delta_voltage)
nonlinear_residual_joint = np.linalg.norm(nonlinear_prediction_joint-delta_voltage)/np.linalg.norm(delta_voltage)
nonlinear_residual_fixed = np.linalg.norm(nonlinear_prediction_fixed-delta_voltage)/np.linalg.norm(delta_voltage)
contact_change_relative_error = np.linalg.norm(z_estimate-z_true)/np.linalg.norm(z_true-z_reference)
print(f"Joint Jacobian: {J_joint.shape}; {n_sigma} conductivity + {n_contact} contact unknowns")
print(f"One-step linearized voltage residual: fixed {linear_residual_fixed:.3%}; joint {linear_residual_joint:.3%}")
print(f"Nonlinear voltage residual: fixed {nonlinear_residual_fixed:.3%}; joint {nonlinear_residual_joint:.3%}")
print(f"Contact-change relative error: {contact_change_relative_error:.3%}")

# %% 9. Conductivity images, including contact/conductivity cross-talk
fields = [sigma_true, sigma_fixed, sigma_joint]
meshes = [simulation_mesh, jacobian_mesh, jacobian_mesh]
values = [viz.field_data(mesh, field).values for mesh, field in zip(meshes, fields)]
clim = (min(v.min() for v in values), max(v.max() for v in values))
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
for ax, mesh, field, title in zip(axes, meshes, fields, [
    "True conductivity", "Assuming fixed contacts", "Joint conductivity + contacts",
]):
    plotted = viz.plot_field(mesh, field, backend="matplotlib", ax=ax,
                             cmap="viridis", clim=clim, colorbar=False,
                             show_edges=False, title=title)
fig.colorbar(plotted.artist, ax=axes, label="Conductivity")
fig.savefig(output_dir / "joint_eit_contact_2d.png", dpi=180)

# %% 10. True and estimated contacts in electrode order
fig_contacts, ax_contacts = plt.subplots(figsize=(9, 3.5), layout="constrained")
indices = np.arange(1, n_contact+1)
ax_contacts.plot(indices, z_true, "o-", label="True contact impedance")
ax_contacts.plot(indices, z_estimate, "s--", label="Joint estimate")
ax_contacts.plot(indices, z_reference, ":", label="Reference / fixed assumption")
ax_contacts.set(xlabel="Electrode index", ylabel="Specific CEM contact impedance z",
                title="One contact-impedance unknown per electrode", xticks=indices)
ax_contacts.grid(alpha=.2)
ax_contacts.legend()
fig_contacts.savefig(output_dir / "joint_eit_contact_2d_contacts.png", dpi=180)

# %% 11. Measured and nonlinear predicted voltage changes
fig_voltages, voltage_axes = plt.subplots(2, 1, figsize=(11, 5), layout="constrained")
for ax, name, block in zip(voltage_axes, protocol_names, protocol_slices):
    ax.plot(delta_voltage[block], color="black", label="Simulated change")
    ax.plot(nonlinear_prediction_fixed[block], "--", label="Fixed contacts")
    ax.plot(nonlinear_prediction_joint[block], ":", label="Joint estimate")
    ax.set(title=name, xlabel="Differential-voltage index", ylabel="Delta voltage (V)")
    ax.legend()
    ax.grid(alpha=.2)
fig_voltages.savefig(output_dir / "joint_eit_contact_2d_voltages.png", dpi=180)
plt.show()
