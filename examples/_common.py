"""Shared model setup and plotting helpers used by the cell-based examples."""
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


def plot_parameter_maps(mesh, parameter_space, values, titles):
    """Comparable elementwise maps, using the actual parameter DOF ordering."""
    import matplotlib.pyplot as plt
    import numpy as np
    from eitlab import plotting as viz
    limit = max(*(float(np.max(np.abs(v))) for v in values), np.finfo(float).eps)
    fig, axes = plt.subplots(1, len(values), figsize=(4.3*len(values), 4),
                             squeeze=False, layout="constrained")
    for ax, value, title in zip(axes.ravel(), values, titles):
        field = viz.numpy_to_gridfunction(mesh, value, space=parameter_space)
        rendered = viz.plot_field(mesh, field, ax=ax, backend="matplotlib",
                                  cmap="RdBu_r", clim=(-limit, limit),
                                  title=title, colorbar=False)
        fig.colorbar(rendered.artist, ax=ax, label="Conductivity change (S/m)")
    return fig, axes.ravel()


def plot_pixel_maps(space, values, titles, *, extent=(-1, 1, -1, 1)):
    """Active-pixel maps with one common symmetric conductivity scale."""
    import matplotlib.pyplot as plt
    import numpy as np
    columns = min(4, len(values))
    rows = int(np.ceil(len(values)/columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4*columns, 3.7*rows),
                             squeeze=False, layout="constrained")
    limit = max(*(float(np.max(np.abs(v))) for v in values), np.finfo(float).eps)
    for ax, value, title in zip(axes.ravel(), values, titles):
        artist = ax.imshow(space.to_image(value), origin="lower", extent=extent,
                           cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    for ax in axes.ravel()[len(values):]:
        ax.set_visible(False)
    fig.colorbar(artist, ax=list(axes.ravel()[:len(values)]),
                 label="Conductivity change (S/m)", shrink=.85)
    return fig, axes
