#!/usr/bin/env python3
"""Three-mesh EIT simulation and reconstruction, using separated modules.

Reusable numerical pipeline; importing this helper performs no solves or plotting.
The fine homogeneous/inhomogeneous models generate data; a coarse homogeneous
model provides the conductivity parameterization and chosen Jacobian.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ngsolve as ng

import eit_forward as forward_api
from eit_forward import geometries, measurement, stimulation
from eit_forward.backend import load_backend
from eit_inverse import TikhonovSolver
from eitlab.differentiation import FiniteDifferenceJacobian
from eit_forward.adapters import NGSolveConductivityForward, NGSolveAdjointJacobian


@dataclass(frozen=True)
class ExampleConfig:
    n_electrodes: int = 16
    radius: float = 1.0
    order: int = 2
    sigma_background: float = 1.0
    sigma_inclusion: float = 0.1
    noise_level: float = 1e-3
    rng_seed: int = 100000
    regularization_lambda: float = 1e-4
    z_contact: float = 1e-3
    maxh_jacobian: float = 0.1
    maxh_forward: float = 0.05
    inverse: str = "sparsecholesky"
    finite_difference_step: float = 1e-4

    def __post_init__(self):
        if self.n_electrodes < 2 or self.order < 1:
            raise ValueError("Need at least two electrodes and positive FEM order")
        values = [self.radius, self.sigma_background, self.sigma_inclusion,
                  self.z_contact, self.maxh_jacobian, self.maxh_forward,
                  self.finite_difference_step]
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Geometry, conductivity, contact and mesh scales must be positive")
        if self.noise_level < 0 or self.regularization_lambda < 0:
            raise ValueError("Noise level and regularization must be nonnegative")

    @property
    def inclusion_center(self):
        return (0.6 * self.radius * np.cos(2 * np.pi / 3),
                0.6 * self.radius * np.sin(2 * np.pi / 3))

    @property
    def inclusion_radius(self):
        return 0.3 * self.radius


def build_geometry(config, *, inclusion=False):
    geometry = geometries.ElectrodeDisk(radius=config.radius, n_electrodes=config.n_electrodes)
    if inclusion:
        # Geometry specifies material labels. The model receives physical values separately.
        geometry.add_disk_inclusion(name="circle1", p=config.inclusion_center,
                                    r=config.inclusion_radius)
    return geometry


def make_electrical_model(mesh, electrode_names, sigma, config=None):
    config = ExampleConfig() if config is None else config
    return forward_api.ElectricalModel(
        mesh, electrode_names, sigma=sigma, numerical_ground_el=electrode_names[0],
        z_contact=config.z_contact, order=config.order, inverse=config.inverse,
        potential_reference=forward_api.PotentialReference.CURRENT_ELECTRODES_MEAN,
        complex_mode=False)


def generate_measurements(config, electrode_names, plan):
    """Independent finer meshes for the homogeneous and inclusion data."""
    voltages = []
    for inclusion in (False, True):
        geometry = build_geometry(config, inclusion=inclusion)
        mesh = geometry.make_mesh(maxh=config.maxh_forward)
        if tuple(geometry.electrode_names) != tuple(electrode_names):
            raise ValueError("Electrode ordering differs between simulation meshes")
        sigma = {"body": config.sigma_background}
        if inclusion:
            sigma["circle1"] = config.sigma_inclusion
        model = make_electrical_model(mesh, electrode_names, sigma, config)
        result = measurement.MeasurementRunner(model, plan).run(store_solutions=False)
        voltages.append(result.vector)
    baseline, inclusion = voltages
    rng = np.random.default_rng(config.rng_seed)
    noise_sd = config.noise_level * np.std(inclusion)
    noise = noise_sd * rng.standard_normal(inclusion.shape)
    return baseline, inclusion, noise, noise_sd


def calc_abs_correlation(reference_img, reconstructed_img):
    """Absolute Pearson score on valid samples; undefined for constant images."""
    reference, reconstructed = np.asarray(reference_img), np.asarray(reconstructed_img)
    valid = np.isfinite(reference) & np.isfinite(reconstructed)
    if valid.sum() < 2 or np.std(reference[valid]) == 0 or np.std(reconstructed[valid]) == 0:
        return float("nan")
    return float(abs(np.corrcoef(reference[valid], reconstructed[valid])[0, 1]))


def reconstruction_mesh_arrays(mesh, parameter_space):
    """Straight-edged display triangles with face values ordered by parameter DOF."""
    vertices = list(mesh.vertices)
    coordinates = np.array([mesh[v].point[:2] for v in vertices])
    lookup = {vertex.nr: i for i, vertex in enumerate(vertices)}
    triangles, dofs = [], []
    for element in mesh.Elements(ng.VOL):
        triangles.append([lookup[v.nr] for v in element.vertices])
        dofs.append(parameter_space.GetDofNrs(element)[0])
    return coordinates, np.asarray(triangles), np.asarray(dofs)


def run(config=None, *, jacobian_method="adjoint"):
    """Return numerical results and the coarse display mesh; no file side effects."""
    config = ExampleConfig() if config is None else config
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
    homogeneous, inclusion, noise, noise_sd = generate_measurements(config, electrodes, plan)
    delta_y = inclusion + noise - homogeneous
    # The inverse solver consumes arrays only. No mesh or FEM state is passed.
    estimate = TikhonovSolver(config.regularization_lambda, J).solve(delta_y)
    coordinates, triangles, dofs = reconstruction_mesh_arrays(mesh, forward.parameter_space)
    centers = coordinates[triangles].mean(axis=1)
    inside = np.linalg.norm(centers - config.inclusion_center, axis=1) <= config.inclusion_radius
    true_change = np.zeros(forward.n_parameters)
    true_change[dofs] = np.where(inside, config.sigma_inclusion - config.sigma_background, 0.0)
    return dict(J=J, reference_parameters=parameters, delta_sigma_estimate=estimate,
                delta_sigma_true_at_centers=true_change, V_homo=homogeneous,
                V_inho=inclusion, noise=noise, noise_sd=noise_sd, delta_y=delta_y,
                regularization_lambda=config.regularization_lambda,
                correlation=calc_abs_correlation(true_change, estimate),
                coordinates=coordinates, triangles=triangles, parameter_dofs=dofs,
                electrode_names=np.asarray(electrodes), jacobian_method=jacobian_method)

