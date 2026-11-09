# Plotting and boundary inspection

Use `from eitlab import plotting as viz`. Install `.[plot]` for Matplotlib
or `.[plot-pv]` for PyVista. Importing the module does not load a plotting backend.

Run `python -m examples.plotting` for a cell-based demonstration of 2D/3D
meshes, scalar fields, pixel images and XY/XZ/YZ slices. Figures are saved under
`results/plotting/`. Set `run_pyvista=True` in its settings for the optional
PyVista scene, or also set `pyvista_off_screen=True` to save it without a window.

## Meshes, fields and boundaries

```python
from eitlab import plotting as viz

view = viz.plot_mesh(mesh, backend="matplotlib", color="lightgray")
view.save("mesh.svg")
view.close()

view = viz.plot_field(mesh, conductivity, backend="matplotlib",
                      cmap="viridis", clim=(0., 2.), title="Conductivity")
view.show()

view = viz.plot_mesh(mesh, backend="pyvista", opacity=.15)
viz.plot_boundaries(mesh, {"el_0": "red", "el_1": "blue"},
                    backend="pyvista", plotter=view.plotter, legend=True)
view.show()
```

These functions return `PlotResult`: `figure`, `ax`, `plotter`, `artist`,
`data`, `show()`, `save()` and `close()`. No window is displayed unless
`show=True` or `show()` is called. Supply `ax=` for a Matplotlib subplot
or `plotter=` for a PyVista scene.

`backend="auto"` selects Matplotlib in 2D and PyVista in 3D. Choose explicitly
when dependencies matter. Both dimensions support Matplotlib static exports.
For a wireframe use `plot_mesh(mesh, backend="matplotlib", color="none")`.

## Field data and coordinates

`field_data` accepts a native FEM scalar field, a `ConductivityParameter`, or
an element-ordered array. Parameters must belong to the mesh. Arrays follow
`mesh_data` element order, which need not equal an arbitrary FEM space's DOF
order. Use a parameter object or `numpy_to_gridfunction(mesh, values, space=space)`
for explicit DOF mapping.

For complex fields choose `component="real"`, `"imag"`, `"magnitude"` or `"phase"`.
For vectors choose an integer component or `"magnitude"`. Omitted component
selections on complex/vector fields raise an error.

Geometry is linearized at vertices. Fields are sampled at one interior
quadrature point per volume element, without averaging or smoothing. Supported
cells are triangles, quadrilaterals and tetrahedra. Curved/high-order geometry
is rendered approximately: **both Matplotlib and PyVista draw straight edges
and flat faces, even for meshes with `curve_order=3`**. Geometry order does not
change that corner-vertex representation. The plotting example includes ideal
circle overlays and a boundary close-up to show the difference. Matplotlib shows 3D exterior faces; PyVista can
retain volume cells for slices and clipping.

## Boundary inspector

```python
inspector = viz.inspect_boundaries(mesh, show=False, show_names=True)
inspector.show()
```

The PyVista view lists named boundaries as checkboxes. Selection updates the
scene immediately, with optional names at boundary locations and matching
colors in the list. It does not rebuild geometry or run FEM solves. Consult
`inspect_boundaries` and `BoundaryInspector` docstrings for programmatic control.
A graphical PyVista environment is required.

## 2D image sampling

```python
image, extent = viz.sample_image(conductivity, mesh, resolution=128)
artist = viz.plot_image(image, extent=extent)
artist.figure.savefig("conductivity.png", dpi=200)
```

Sampling uses mesh bounds unless an extent is supplied. Outside-domain pixels
are NaN. `plot_image` is a Matplotlib helper returning its image artist, rather
than a `PlotResult`. Pass the sampled extent to preserve physical coordinates.
Supply `ax=` for a subplot. Configure publication fonts in the calling script;
`plt.rcParams["svg.fonttype"] = "none"` preserves editable SVG text.

`sample_image` evaluates the original field at each pixel center using the
native mesh, including curved geometry. It does not sample the one-value-per-cell
representation produced by `field_data`. Increasing resolution refines the
display grid, not the FEM solution. Discontinuous fields evaluated exactly on
an interface can depend on which adjacent element is selected.

## Pixel slices through a 3D field

```python
# For an existing 3D solve; no direct NGSolve/Netgen calls are needed.
image, extent = viz.sample_slice(
    solution.field, mesh, plane="xy", position=0.10,
    resolution=128, component="real",
)
artist = viz.plot_image(image, extent=extent, interpolation="nearest",
                        cmap="viridis")
artist.axes.set(xlabel="x (m)", ylabel="y (m)", title="Potential at z = 0.10 m")
artist.figure.savefig("potential_slice.png", dpi=200)
```

`sample_slice` requires a 3D mesh and returns the same `(image, extent)` pair
as `sample_image`. It evaluates the original field at pixel centers, keeping
within-element variation and using the curved mesh for domain lookup.

| Plane | Image horizontal / vertical axes | Fixed coordinate |
| --- | --- | --- |
| `"xy"` | x / y | z = `position` |
| `"xz"` | x / z | y = `position` |
| `"yz"` | y / z | x = `position` |

Coordinates and `position` use the mesh's physical units. The default plane is
`"xy"` at `position=0.0`. Supply `extent=(umin, umax, vmin, vmax)` for explicit
pixel-edge bounds along the two displayed coordinates; otherwise their bounds
come from the projected mesh vertices. `resolution` is an integer of at least
two and gives the pixel count along each axis. Rows follow the second plane
coordinate and columns the first, so `plot_image` displays the orientation
correctly with its lower origin.

Outside-volume pixels become NaN; a plane missing the volume gives an entirely
NaN image. Field evaluation errors propagate. Scalar fields, conductivity
parameters belonging to the mesh, complex components and vector magnitudes or
component indices use the same rules as `sample_image`/`field_data`. Raw DOF
arrays must first be converted using the appropriate parameter space or
`numpy_to_gridfunction`. The same interface ambiguity for discontinuous fields
applies to slices. This helper supports coordinate planes; oblique planes are
not part of its interface.

## Verification

```bash
MPLBACKEND=Agg python -m pytest tests/test_plotting.py -q
```

Checks cover real 2D/3D meshes, scalar/vector/complex sampling, P0 mapping,
image/slice extents, all slice orientations, native GridFunction evaluation,
curved-domain masking, outside-domain pixels and Matplotlib exports. Interactive PyVista
requires a graphics environment and separate manual inspection.

[Documentation index](../README.md)
