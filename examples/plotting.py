"""Plot curved 2D/3D meshes and scalar data with Matplotlib and PyVista.

Run from the project root: python -m examples.plotting
Install .[fem,plot]; enable run_pyvista below after installing .[plot-pv].

The FEM meshes use curve_order=3 by default. EITLab's plotting adapter keeps
only corner vertices: BOTH backends draw straight edges and flat 3D faces.
Mesh-based field plots sample once per volume element, without interpolation
or averaging. Pixel images and 3D slices instead evaluate the original field
at each pixel center, using the curved mesh for domain lookup.
"""

# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
import ngsolve as ng
import numpy as np

from eitlab import CylinderModel, ElectrodeDisk
from eitlab import plotting as viz


# %% Settings -- edit before running the following cells
output_dir = Path("results/plotting")
output_dir.mkdir(parents=True, exist_ok=True)

radius = 0.1                 # m; shared by the disk and cylinder
height = 0.2                 # m
maxh_2d = 0.04               # m; coarse meshes make the plotting facets visible
maxh_3d = 0.06               # m
curve_order = 3              # geometry order used by NGSolve
conductivity_limits = (1.0, 2.0)  # S/m; shared color scale
image_resolution = 128       # pixels along each image/slice axis
slice_z = 0.5 * height       # m; XY slice position
slice_y = 0.0                # m; XZ slice position
slice_x = 0.0                # m; YZ slice position

run_pyvista = True          # True: also create the optional four-panel scene
pyvista_off_screen = False   # True: save the scene without opening a window


# %% Curved meshes and prescribed scalar fields -- no forward solve needed
disk = ElectrodeDisk(radius=radius, n_electrodes=0)
mesh_2d = disk.make_mesh(maxh=maxh_2d, curve_order=curve_order)
cylinder = CylinderModel(radius=radius, height=height)
mesh_3d = cylinder.make_mesh(maxh=maxh_3d, curve_order=curve_order)

# A smooth conductivity distribution, evaluated by the plotting adapter.
sigma_2d = 1.5 + 0.5 * ng.x / radius
# An off-center smooth inclusion and a vertical gradient make all three
# slice orientations visible. These NGSolve expressions only generate test
# fields; a solved model can supply solution.field directly to sample_slice.
sigma_3d = 1.0 + 0.4 * ng.z / height + 0.6 * ng.exp(
    -((ng.x - 0.25 * radius) / (0.4 * radius))**2
    -((ng.y + 0.15 * radius) / (0.5 * radius))**2
    -((ng.z - 0.6 * height) / (0.25 * height))**2
)

# Prepared data can be reused by either backend. Array entries must follow
# mesh.Elements(ng.VOL), NOT an arbitrary finite-element space's DOF ordering.
data_2d = viz.field_data(mesh_2d, sigma_2d)
data_3d = viz.field_data(mesh_3d, sigma_3d)
conductivity_per_element = data_2d.values.copy()

print(f"Disk: {mesh_2d.ne} elements; cylinder: {mesh_3d.ne} elements")
print(f"FEM geometry order: {curve_order}")
print("Plotting limitation (Matplotlib AND PyVista, 2D AND 3D):")
print("  Only corner vertices are displayed: edges are straight; faces are flat.")
print("  Mesh-based field plots use one sample per volume element.")
print("  Pixel images/slices evaluate the original field at every pixel center.")


# %% Matplotlib: 2D mesh, native scalar field, and element-ordered array
fig_2d, axes_2d = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
viz.plot_mesh(mesh_2d, ax=axes_2d[0], backend="matplotlib", color="none",
              title="Disk mesh: straight displayed edges")
field_2d = viz.plot_field(mesh_2d, sigma_2d, ax=axes_2d[1],
                         backend="matplotlib", clim=conductivity_limits,
                         colorbar=False, title="NGSolve scalar field")
array_2d = viz.plot_field(mesh_2d, conductivity_per_element, ax=axes_2d[2],
                         backend="matplotlib", clim=conductivity_limits,
                         colorbar=False, title="One array entry per element")
fig_2d.colorbar(field_2d.artist, ax=list(axes_2d[1:]),
                label="Conductivity (S/m)", shrink=0.8)

# Dashed reference curves show the ideal geometry; they are not mesh edges.
theta = np.linspace(0.0, 2.0 * np.pi, 401)
circle_x = radius * np.cos(theta)
circle_y = radius * np.sin(theta)
for ax in axes_2d:
    ax.plot(circle_x, circle_y, "--", color="tab:red", linewidth=1.4,
            label="Ideal circular boundary")
    ax.set(xlabel="x (m)", ylabel="y (m)")
axes_2d[0].legend(loc="lower center", fontsize=8)
fig_2d.suptitle(f"2D FEM geometry order {curve_order}; plots connect corner vertices")
fig_2d.savefig(output_dir / "matplotlib_2d.png", dpi=180)


# %% Boundary close-up: curved FEM geometry versus the displayed chords
fig_boundary, ax_boundary = plt.subplots(figsize=(6, 4.5), layout="constrained")
viz.plot_mesh(mesh_2d, region="boundary", ax=ax_boundary,
              backend="matplotlib", color="black", line_width=2.0,
              title=f"Boundary close-up (FEM geometry order {curve_order})")
ax_boundary.plot(circle_x, circle_y, "--", color="tab:red", linewidth=1.5,
                 label="Ideal circle (reference)")
ax_boundary.plot([], [], color="black", linewidth=2.0,
                 label="EITLab display: straight chords")
ax_boundary.set(xlim=(0.8 * radius, 1.04 * radius),
                 ylim=(-0.45 * radius, 0.45 * radius),
                 xlabel="x (m)", ylabel="y (m)")
ax_boundary.legend(fontsize=8, loc="upper left")
fig_boundary.savefig(output_dir / "boundary_closeup.png", dpi=180)


# %% Matplotlib: 3D mesh and scalar data on the exterior faces
fig_3d = plt.figure(figsize=(12, 6))
fig_3d.subplots_adjust(left=0.06, right=0.96, bottom=0.08, top=0.84, wspace=0.18)
ax_mesh_3d = fig_3d.add_subplot(121, projection="3d")
ax_field_3d = fig_3d.add_subplot(122, projection="3d")
viz.plot_mesh(mesh_3d, ax=ax_mesh_3d, backend="matplotlib", color="none",
              line_width=0.6, title="Cylinder mesh: flat displayed faces")
# MeshData already holds the sampled values; no second field argument needed.
field_3d = viz.plot_field(data_3d, ax=ax_field_3d, backend="matplotlib",
                         clim=conductivity_limits, colorbar=False,
                         title="Conductivity on exterior faces")
fig_3d.colorbar(field_3d.artist, ax=ax_field_3d,
                label="Conductivity (S/m)", shrink=0.65)
for ax in (ax_mesh_3d, ax_field_3d):
    for z_reference in (0.0, height):
        ax.plot(circle_x, circle_y, np.full_like(theta, z_reference),
                "--", color="tab:red", linewidth=1.4)
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    ax.set_xticks(np.linspace(-radius, radius, 3))
    ax.set_yticks(np.linspace(-radius, radius, 3))
    ax.set_zticks(np.linspace(0.0, height, 5))
    ax.view_init(elev=22, azim=35)
fig_3d.suptitle(f"3D FEM geometry order {curve_order}; dashed red rings show ideal circles")
fig_3d.savefig(output_dir / "matplotlib_3d.png", dpi=180)


# %% Pixel images: a 2D field and three planes through the original 3D field
# No NGSolve/Netgen calls are needed for sampling or displaying these fields.
# For a computed potential, pass solution.field instead of sigma_3d and use
# component="real" if it is complex; label the colorbar as Potential (V).
image_2d, extent_2d = viz.sample_image(
    sigma_2d, mesh_2d, resolution=image_resolution,
    extent=(-radius, radius, -radius, radius),
)
image_xy, extent_xy = viz.sample_slice(
    sigma_3d, mesh_3d, plane="xy", position=slice_z,
    resolution=image_resolution, extent=(-radius, radius, -radius, radius),
)
image_xz, extent_xz = viz.sample_slice(
    sigma_3d, mesh_3d, plane="xz", position=slice_y,
    resolution=image_resolution, extent=(-radius, radius, 0.0, height),
)
image_yz, extent_yz = viz.sample_slice(
    sigma_3d, mesh_3d, plane="yz", position=slice_x,
    resolution=image_resolution, extent=(-radius, radius, 0.0, height),
)

# Outside-volume pixels remain NaN. Use nearest-neighbor display to show the
# actual sampled pixels, with one color scale shared by all four panels.
fig_slices, axes_slices = plt.subplots(2, 2, figsize=(10, 9), layout="constrained")
slice_panels = [
    (image_2d, extent_2d, "2D disk field", "x (m)", "y (m)"),
    (image_xy, extent_xy, f"XY slice: z = {slice_z:.3f} m", "x (m)", "y (m)"),
    (image_xz, extent_xz, f"XZ slice: y = {slice_y:.3f} m", "x (m)", "z (m)"),
    (image_yz, extent_yz, f"YZ slice: x = {slice_x:.3f} m", "y (m)", "z (m)"),
]
for ax, (image, extent, title, xlabel, ylabel) in zip(axes_slices.flat, slice_panels):
    image_artist = viz.plot_image(
        image, ax=ax, extent=extent, interpolation="nearest", cmap="viridis",
        vmin=conductivity_limits[0], vmax=conductivity_limits[1], colorbar=False,
    )
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.set_aspect("equal")
fig_slices.colorbar(image_artist, ax=list(axes_slices.flat),
                    label="Conductivity (S/m)", shrink=0.85)
fig_slices.suptitle("Pixel-center sampling of original fields; outside-volume pixels are blank")
fig_slices.savefig(output_dir / "pixel_images_and_slices.png", dpi=180)


# %% Optional PyVista: 2D and 3D meshes and data in one scene
# Install: python -m pip install -e ".[plot-pv]"
# Set run_pyvista=True above. Interactive mode opens a rotatable 3D view.
# Off-screen mode still requires a working VTK rendering backend.
if run_pyvista:
    import pyvista as pv

    plotter = pv.Plotter(shape=(2, 2), off_screen=pyvista_off_screen,
                         window_size=(1400, 1000))
    plotter.set_background("white")

    plotter.subplot(0, 0)
    mesh_view_2d = viz.plot_mesh(
        mesh_2d, backend="pyvista", plotter=plotter, color="black",
        title=f"2D mesh, order {curve_order}: straight displayed edges",
        backend_options={"style": "wireframe"}, line_width=1.0,
    )
    plotter.subplot(0, 1)
    field_view_2d = viz.plot_field(
        data_2d, backend="pyvista", plotter=plotter,
        title="2D conductivity: one sample per element",
        clim=conductivity_limits,
        backend_options={"scalar_bar_args": {
            "title": "2D conductivity (S/m)", "vertical": True,
            "position_x": 0.86, "position_y": 0.2,
            "width": 0.08, "height": 0.5, "fmt": "%.2f",
            "title_font_size": 12, "label_font_size": 12,
        }},
    )

    plotter.subplot(1, 0)
    mesh_view_3d = viz.plot_mesh(
        mesh_3d, backend="pyvista", plotter=plotter, color="black",
        title=f"3D mesh, order {curve_order}: flat displayed faces",
        backend_options={"style": "wireframe"}, line_width=1.0,
    )
    plotter.view_isometric()
    plotter.subplot(1, 1)
    field_view_3d = viz.plot_field(
        data_3d, backend="pyvista", plotter=plotter,
        title="3D conductivity: tetrahedral cell data",
        clim=conductivity_limits,
        backend_options={"scalar_bar_args": {
            "title": "3D conductivity (S/m)", "vertical": True,
            "position_x": 0.86, "position_y": 0.2,
            "width": 0.08, "height": 0.5, "fmt": "%.2f",
            "title_font_size": 12, "label_font_size": 12,
        }},
    )
    plotter.view_isometric()

    # Each renderer has its own camera; preserve equal physical axis scales.
    for row in range(2):
        for column in range(2):
            plotter.subplot(row, column)
            plotter.show_bounds(
                xtitle="x (m)", ytitle="y (m)", ztitle="z (m)",
                location="outer", ticks="outside", fmt="%.2f", font_size=12,
                n_xlabels=3, n_ylabels=3, n_zlabels=3,
            )
            plotter.reset_camera()
            plotter.camera.zoom(0.8)

    # The 3D actor retains volume cells, available for further inspection:
    # volume_grid = field_view_3d.artist.mapper.dataset
    # middle_slice = volume_grid.slice(normal="z", origin=(0, 0, height / 2))
    if pyvista_off_screen:
        plotter.show(auto_close=False)
        plotter.screenshot(str(output_dir / "pyvista_2d_3d.png"))
        plotter.close()
    else:
        plotter.show(screenshot=str(output_dir / "pyvista_2d_3d.png"))


# %% Display Matplotlib figures; saved PNGs are also available in output_dir
plt.show()
