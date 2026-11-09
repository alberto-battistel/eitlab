"""Test fixtures for the installed EITLab packages."""
from eit_forward.backend import load_backend


def make_disk(backend):
    geometry = backend.geometry.ElectrodeDisk(radius=1.0, n_electrodes=8)
    mesh = geometry.make_mesh(maxh=0.45, curve_order=2)
    # Use geometry ordering, not lexicographic sorting or a mesh-boundary set.
    return mesh, tuple(geometry.electrode_names)


def make_model(backend, mesh, electrodes, sigma=1.0, ground=None):
    return backend.fem.ElectricalModel(
        mesh=mesh,
        electrode_names=list(electrodes),
        sigma=sigma,
        numerical_ground_el=electrodes[0] if ground is None else ground,
        z_contact=0.02,
        potential_reference=backend.fem.PotentialReference.MEAN_ZERO,
        order=1,
        inverse="sparsecholesky",
        complex_mode=False,
    )


def make_pattern(backend, electrodes, source=1, sink=3, amplitude=1.0):
    return backend.stimulation.CurrentPattern(
        {electrodes[source]: amplitude, electrodes[sink]: -amplitude}
    )


def make_plan(backend, electrodes, exclude_current=False):
    protocol = backend.stimulation.ProtocolFactory.adjacent(list(electrodes), amplitude=1.0)
    observation = (
        backend.measurement.SingleEndedNoCurrent
        if exclude_current else backend.measurement.SingleEnded
    )
    return backend.measurement.MeasurementPlan(protocol, observation)
