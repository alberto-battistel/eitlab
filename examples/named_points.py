"""Named vertices, point-electrode rings, and a point-grounded CEM solve.

Run from the project root: python -m examples.named_points
Point-electrode locations are geometry features; the CEM solve below continues
to use finite electrode patches for current injection.
"""
# %% Imports
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from eitlab import plotting as viz

from eitlab import (ElectrodeDisk, CylinderModel, ElectricalModel,
                    CurrentPattern, PotentialReference)


# %% Settings
output_dir = Path("results/named_points")
output_dir.mkdir(parents=True, exist_ok=True)
maxh_disk = 0.25
maxh_cylinder = 0.35
maxh_cem = 0.3

# %% 2D geometry with a ring consisting entirely of point features.
disk = ElectrodeDisk(radius=1.0, n_electrodes=0)
disk.add_point(name="reference", x=0.0, y=0.0)
disk.add_point_electrode_ring(n=16, start_angle_deg=0.0, group="ring")
disk_mesh = disk.make_mesh(maxh=maxh_disk)
print("Disk point electrodes:", disk.point_electrode_names)
print("Reference vertex:", disk.point_vertex("reference"))

# %% 3D geometry: two point-electrode rings and a connected interior vertex.
cylinder = CylinderModel(radius=1.0, height=1.0)
cylinder.add_point(name="reference", x=0.0, y=0.0, z=0.5)
cylinder.add_point_electrode_ring(n=8, z=0.25, group="lower")
cylinder.add_point_electrode_ring(n=8, z=0.75, start_angle_deg=22.5, group="upper")
cylinder_mesh = cylinder.make_mesh(maxh=maxh_cylinder)
print("Cylinder point electrodes:", cylinder.point_electrode_names)
print("Reference vertex:", cylinder.point_vertex("reference"))

# %% Finite CEM electrodes can coexist with separately named geometry points.
geometry = ElectrodeDisk(radius=1.0, n_electrodes=8)
geometry.add_point(name="ground", x=0.1, y=0.1)
mesh = geometry.make_mesh(maxh=maxh_cem)
model = ElectricalModel(
    mesh, geometry.electrode_names, sigma=1.0,
    numerical_ground_point="ground",
    potential_reference=PotentialReference.DO_NOTHING,
)
pattern = CurrentPattern({geometry.electrode_names[0]: 1.0,
                          geometry.electrode_names[1]: -1.0})
solution = model.solve(pattern)
ground_voltage = solution.field(0.1, 0.1)
print("Potential at ground:", ground_voltage)
print("Adjacent electrode differences:", np.diff(solution.electrode_potentials))

# %% Named points in 2D and 3D, and the point-grounded CEM potential
fig_geometry = plt.figure(figsize=(15, 4.5), layout="constrained")
ax_disk = fig_geometry.add_subplot(131)
ax_cylinder = fig_geometry.add_subplot(132, projection="3d")
ax_potential = fig_geometry.add_subplot(133)
viz.plot_mesh(disk_mesh, ax=ax_disk, backend="matplotlib", title="2D named point electrodes")
viz.plot_mesh(cylinder_mesh, ax=ax_cylinder, backend="matplotlib", opacity=.2,
              title="3D named point electrodes", color="lightgray")
for ax, shape, dimension in [(ax_disk, disk, 2), (ax_cylinder, cylinder, 3)]:
    for name, point in shape.points.items():
        position = point.position[:dimension]
        color = "tab:red" if point.kind == "point" else "tab:blue"
        ax.scatter(*position, color=color, s=18)
        if point.kind == "point":
            ax.text(*position, name, fontsize=8)
field_plot = viz.plot_field(mesh, solution.field, ax=ax_potential, backend="matplotlib",
                            title="CEM potential with named ground", colorbar=False)
ax_potential.plot(.1, .1, "rx", label="Ground point")
ax_potential.legend(fontsize=8)
fig_geometry.colorbar(field_plot.artist, ax=ax_potential, label="Potential (V)")
fig_geometry.savefig(output_dir / "named_points_geometry.png", dpi=160)
fig_voltage, ax_voltage = plt.subplots(figsize=(7, 3.5), layout="constrained")
ax_voltage.plot(solution.electrode_potentials, "o-")
ax_voltage.set(xlabel="Electrode index", ylabel="Ground-referenced potential (V)", title="Finite CEM electrode voltages")
fig_voltage.savefig(output_dir / "named_points_voltages.png", dpi=160)

plt.show()
