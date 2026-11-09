"""Homogeneous EIT -> noisy voltages -> repeated reconstructions -> target gain."""

# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples._common import load_backend, make_plan
from eit_forward.adapters import NGSolveConductivityForward, NGSolveAdjointJacobian, triangles_in_parameter_order
from eit_inverse import (triangle_pixel_space, MappedForward, DataSpace, InverseProblem,
    Regularizer, grid_operator, SciPySolver, CVXPYSolver, NoiseSimulationConfig,
    NoiseAmplificationExperiment, tune_noise_amplification)

# %% Settings — edit these before running the following cells
solver_backend = 'scipy'  # ['scipy', 'cvxpy']
norm = 'l2'  # ['l2', 'l1']
order = 0  # [0, 1, 2]
target = 0.5
repeats = 50
validation_repeats = 100
noise_level = 0.001
noise_mode = 'relative_rms'  # ['relative_rms', 'absolute', 'relative_channel']
noise_floor = 0.0
statistic = 'rms'  # ['rms', 'std']
weight_min = 1e-06
weight_max = 100.0
n_weights = 9
method = 'brent'  # ['brent', 'bounded', 'polynomial', 'grid']
polynomial_order = 2
no_refine = False
target_rtol = 0.05
log_weight_tolerance = 0.001
max_iterations = 50
seed = 42
validation_seed = None
sigma0 = 1.0
output_dir = Path("results/noise_amplification_tuning")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "noise_amplification_tuning.npz"
plot_file = output_dir / "noise_amplification_tuning.png"

# %% Compute the demonstration
backend = load_backend()
geometry = backend.geometry.ElectrodeDisk(radius=1., n_electrodes=8)
mesh = geometry.make_mesh(maxh=.45, curve_order=1)
electrodes = tuple(geometry.electrode_names)
raw = NGSolveConductivityForward(backend, mesh, electrodes, make_plan(backend, electrodes, exclude_current=True))
edges = np.linspace(-1, 1, 5)
space = triangle_pixel_space(triangles_in_parameter_order(raw), edges, edges,
                             offset=np.full(raw.n_parameters, sigma0))
forward = MappedForward(raw, NGSolveAdjointJacobian(raw), space)
linearization = forward.linearize(np.zeros(space.ndof))
v0, J = linearization.prediction, linearization.jacobian()
# Common declared data normalization, not an experimental noise covariance.
model_scale = np.linalg.norm(J.toarray(), 2)
problem = InverseProblem(J, np.zeros(len(v0)),
    [Regularizer(grid_operator(space.shape, order, mask=space.mask), 1., norm)],
    data_space=DataSpace(len(v0), standard_deviation=model_scale))
# Noise-level-sensitive nonsmooth solves need tighter absolute tolerances.
solver = SciPySolver(abs_tol=1e-10, rel_tol=1e-7) if solver_backend == "scipy" else CVXPYSolver(tol_gap_abs=1e-11, tol_feas=1e-10)
config = NoiseSimulationConfig(repeats=repeats, noise_level=noise_level,
                               noise_mode=noise_mode, seed=seed,
                               statistic=statistic, store_images=True, noise_floor=noise_floor)
experiment = NoiseAmplificationExperiment(problem, solver, v0, config=config,
                    image_scale=sigma0, image_weights=space.pixel_areas)
result = tune_noise_amplification(experiment, target,
    weight_span=(weight_min, weight_max), n_weights=n_weights,
    method=method, polynomial_order=polynomial_order,
    refine_polynomial=not no_refine, target_rtol=target_rtol,
    validation_repeats=validation_repeats, validation_seed=validation_seed,
    log_weight_tolerance=log_weight_tolerance, max_iterations=max_iterations)
print(f"Homogeneous sigma={sigma0:g}; voltages={len(v0)}; pixels={space.ndof}")
print(f"Metric: normalized noise gain ({statistic}); {repeats} repeats/weight")
print(f"Target={result.target_gain:.5g}; selected weight={result.weight:.6g}; matched={result.matched}")
print(f"Tuning gain={result.estimate.mean_gain:.5g} +/- {result.estimate.gain_standard_error:.3g} SEM")
if result.validation is not None:
    print(f"Independent gain={result.validation.mean_gain:.5g} +/- {result.validation.gain_standard_error:.3g} SEM; matched={result.validation_matched}")
for message in result.messages:
    print(message)
# %% Save numerical results
if output_file:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    saved = dict(weight=result.weight, target=result.target_gain, matched=result.matched,
        scan_weights=result.scan_weights, scan_gains=result.scan_gains,
        scan_standard_errors=result.scan_standard_errors,
        evaluated_weights=result.diagnostics["evaluated_weights"],
        evaluated_gains=result.diagnostics["evaluated_gains"],
        gains=result.estimate.gains, noise_images=result.estimate.noise_images,
        input_noise=result.estimate.input_noise, output_noise=result.estimate.output_noise,
        homogeneous_voltages=v0, noise_samples=experiment.noise_samples,
        sigma0=sigma0, image_scale=experiment.image_scale, input_scale=experiment.input_scale,
        mask=space.mask, pixel_areas=space.pixel_areas, x_edges=edges, y_edges=edges,
        J=J.toarray(), data_scale=model_scale, method=method,
        polynomial_order=polynomial_order, noise_level=noise_level,
        noise_floor=noise_floor,
        noise_mode=noise_mode, statistic=statistic, seed=seed,
        repeats=repeats, norm=norm, order=order, backend=solver_backend)
    saved.update(validation_repeats=validation_repeats,
                 validation_seed=seed+1 if validation_seed is None else validation_seed,
                 target_rtol=target_rtol, log_weight_tolerance=log_weight_tolerance,
                 max_iterations=max_iterations, polynomial_refinement=not no_refine,
                 clean_parameters=result.estimate.clean_parameters)
    if result.validation is not None:
        saved.update(validation_gains=result.validation.gains,
                     validation_noise_images=result.validation.noise_images,
                     validation_matched=result.validation_matched)
    np.savez(output_file, **saved)
# %% Plot the results
from matplotlib.colors import TwoSlopeNorm
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].errorbar(result.scan_weights, result.scan_gains, yerr=2*result.scan_standard_errors, fmt="o-", label="mean +/- 2 SEM")
axes[0].set_xscale("log")
axes[0].axhline(target, color="k", ls="--", label="target")
axes[0].plot(result.weight, result.estimate.mean_gain, "r*", ms=13, label="selected")
if result.validation is not None:
    axes[0].plot(result.weight, result.validation.mean_gain, "gx", ms=10, label="independent check")
axes[0].set(xlabel="regularization weight", ylabel="normalized noise gain")
axes[0].legend(fontsize=8)
image = space.to_image(result.estimate.noise_images[0])
maximum = max(np.nanmax(np.abs(image)), np.finfo(float).eps)
artist = axes[1].imshow(image, origin="lower", extent=[-1, 1, -1, 1], cmap="RdBu_r",
                       norm=TwoSlopeNorm(vmin=-maximum, vcenter=0., vmax=maximum))
axes[1].set(title="One noise-only reconstruction", xlabel="x", ylabel="y")
fig.colorbar(artist, ax=axes[1], label="Conductivity error (S/m)")
fig.tight_layout(); plot_file.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(plot_file, dpi=160)

plt.show()
