"""Compare adjacent and opposite injection in a homogeneous CEM disk.

Run with ``python -m examples.disk_forward`` or execute the # %% cells.
This physics demonstration solves two individual current patterns. See
``simple_eit_2d.py`` for a full measurement plan and reconstruction.
"""

# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eitlab import (CurrentPattern, ElectricalModel, ElectrodeDisk,
                    ElementwiseConductivity, PotentialReference)
from eitlab import plotting as viz


# %% Settings -- edit before running the following cells
radius = 1.0                    # m
n_electrodes = 8                # even, so an exactly opposite electrode exists
electrode_angle = 15.0          # degrees; smaller than 360 / n_electrodes
maxh = 0.25                     # m; maximum mesh size
curve_order = 3
fem_order = 1
conductivity = 1.0              # S/m
current_amplitude = 1.0         # A per metre of out-of-plane depth
z_contact = 0.02                # ohm m^2, using the per-unit-depth 2D CEM
source_index = 0                # zero-based index in geometry electrode order
output_dir = Path("results/disk_forward")

# Lengths are in metres. This 2D model uses current per unit out-of-plane
# depth; multiplying by a physical slice thickness gives the total current.


# %% Build the geometry, conductivity and complete electrode model explicitly
if n_electrodes < 4 or n_electrodes % 2:
    raise ValueError("Use an even number of electrodes, at least four")
if not 0 <= source_index < n_electrodes:
    raise ValueError("source_index must identify an electrode")

geometry = ElectrodeDisk(radius=radius, n_electrodes=n_electrodes,
                         electrode_angle=electrode_angle)
mesh = geometry.make_mesh(maxh=maxh, curve_order=curve_order)
electrodes = tuple(geometry.electrode_names)
conductivity_space = ElementwiseConductivity(mesh)
sigma = conductivity_space.create(conductivity)
model = ElectricalModel(
    mesh=mesh,
    electrode_names=electrodes,
    sigma=sigma,
    z_contact=z_contact,
    numerical_ground_el=electrodes[0],
    potential_reference=PotentialReference.MEAN_ZERO,
    order=fem_order,
    inverse="sparsecholesky",
    complex_mode=False,
)


# %% Define and solve just two balanced current patterns
source = electrodes[source_index]
adjacent_sink = electrodes[(source_index + 1) % n_electrodes]
opposite_sink = electrodes[(source_index + n_electrodes // 2) % n_electrodes]
adjacent_pattern = CurrentPattern(
    {source: current_amplitude, adjacent_sink: -current_amplitude},
    name="Adjacent injection",
)
opposite_pattern = CurrentPattern(
    {source: current_amplitude, opposite_sink: -current_amplitude},
    name="Opposite injection",
)
adjacent_solution = model.solve(adjacent_pattern)
opposite_solution = model.solve(opposite_pattern)
U_adjacent = adjacent_solution.electrode_potentials
U_opposite = opposite_solution.electrode_potentials

# Electrode potentials are mean-zero, while solution.field retains the
# numerical ground. Shift both fields to the same mean-zero gauge.
adjacent_field = adjacent_solution.field - np.mean(
    adjacent_solution.raw_electrode_potentials)
opposite_field = opposite_solution.field - np.mean(
    opposite_solution.raw_electrode_potentials)

print(f"Elements: {mesh.ne}; electrodes: {len(electrodes)}")
print("Electrode order:", electrodes)
print(f"Adjacent: +I at {source}, -I at {adjacent_sink}")
print(f"Opposite: +I at {source}, -I at {opposite_sink}")
print("Mean-zero adjacent potentials (V):", U_adjacent)
print("Mean-zero opposite potentials (V):", U_opposite)


# %% Compare potential fields on a common scale and all electrode voltages
fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
adjacent_plot = viz.plot_field(
    mesh, adjacent_field, ax=axes[0, 0], backend="matplotlib",
    title=f"Adjacent: {source} (+I) to {adjacent_sink} (-I)",
    cmap="RdBu_r", colorbar=False,
)
opposite_plot = viz.plot_field(
    mesh, opposite_field, ax=axes[0, 1], backend="matplotlib",
    title=f"Opposite: {source} (+I) to {opposite_sink} (-I)",
    cmap="RdBu_r", colorbar=False,
)
potential_limit = max(
    float(np.max(np.abs(adjacent_plot.data.values))),
    float(np.max(np.abs(opposite_plot.data.values))),
    np.finfo(float).eps,
)
adjacent_plot.artist.set_clim(-potential_limit, potential_limit)
opposite_plot.artist.set_clim(-potential_limit, potential_limit)
for ax, sink in ((axes[0, 0], adjacent_sink), (axes[0, 1], opposite_sink)):
    viz.plot_boundaries(mesh, {source: "red", sink: "blue"}, ax=ax,
                        backend="matplotlib", line_width=4, legend=False)
    ax.set(xlabel="x (m)", ylabel="y (m)")
fig.colorbar(adjacent_plot.artist, ax=list(axes[0, :]),
             label="Potential relative to electrode mean (V)")

viz.plot_mesh(mesh, ax=axes[1, 0], backend="matplotlib",
              title=f"Homogeneous disk: conductivity {conductivity:g} S/m")
axes[1, 0].set(xlabel="x (m)", ylabel="y (m)")

indices = np.arange(n_electrodes)
axes[1, 1].plot(indices, U_adjacent, "o-", label="Adjacent injection")
axes[1, 1].plot(indices, U_opposite, "s--", label="Opposite injection")
axes[1, 1].axhline(0, color="0.6", linewidth=0.8)
axes[1, 1].set(
    title="Electrode potentials (including current electrodes)",
    xlabel="Electrode", ylabel="Mean-zero potential (V)",
    xticks=indices, xticklabels=electrodes,
)
axes[1, 1].tick_params(axis="x", labelrotation=45)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.25)

output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "disk_forward.png", dpi=160)
plt.show()
