"""Unified optional plotting for EITLab (NumPy is the only eager dependency).

    plot_mesh(mesh, backend="matplotlib")
    plot_field(mesh, sigma_cf, backend="pyvista", component="real")
    plot_boundaries(mesh, {"el1": "red"}, backend="pyvista")

Every plot_* returns PlotResult; nothing is displayed unless show=True.
Use result.figure/result.ax or result.plotter for backend-specific control.
Geometry is linearized at mesh vertices. Fields are sampled once per volume
element, at an interior quadrature point (not averaged or smoothed). Supported
volume cells: triangles, quadrilaterals and tetrahedra. Curved/high-order
visualization needs a future tessellation adapter. See docs/guides/plotting.md.
Image and slice sampling instead evaluate the native field at pixel centers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .parameters import ConductivityParameter

__all__ = ["MeshData", "PlotResult", "mesh_data", "field_data", "plot_mesh",
           "plot_boundaries", "plot_field", "numpy_to_gridfunction",
           "sample_image", "sample_slice", "plot_image",
           "BoundaryInspector", "inspect_boundaries"]


def _dependency(name):
    from importlib import import_module
    try:
        return import_module(name)
    except ImportError as exc:
        raise ImportError(f"This operation requires {name}; install it in your active environment.") from exc


@dataclass
class MeshData:
    """Backend-neutral geometry. Cells use local point indices; IDs retain source order.

    kind is 'lines', 'surface', or 'volume'; dimension is the physical dimension.
    Values, when supplied, are one scalar per cell. No implicit point smoothing.
    """
    points: np.ndarray
    cells: tuple
    dimension: int
    kind: str
    element_ids: np.ndarray
    labels: tuple = ()
    values: np.ndarray | None = None

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)
        self.cells = tuple(np.asarray(c, dtype=int) for c in self.cells)
        self.element_ids = np.asarray(self.element_ids, dtype=int)
        self.labels = tuple(self.labels)
        if self.dimension not in (2, 3) or self.kind not in ("lines", "surface", "volume"):
            raise ValueError("Invalid geometry dimension/kind.")
        if self.points.ndim != 2 or self.points.shape[1] != 3 or not len(self.cells):
            raise ValueError("Expected nonempty cells and points with shape (n, 3).")
        sizes = {"lines": {2}, "surface": {3, 4}, "volume": {4}}[self.kind]
        for c in self.cells:
            if c.ndim != 1 or len(c) not in sizes or np.any(c < 0) or np.any(c >= len(self.points)):
                raise ValueError(f"Unsupported or invalid {self.kind} cell connectivity.")
        if len(self.element_ids) != len(self.cells):
            raise ValueError("One element ID is required per cell.")
        if self.labels and len(self.labels) != len(self.cells):
            raise ValueError("One label is required per cell.")
        if self.values is not None:
            self.values = np.asarray(self.values)
            if self.values.shape != (len(self.cells),) or np.iscomplexobj(self.values):
                raise ValueError("Values must contain one real scalar per cell.")


@dataclass
class PlotResult:
    backend: str
    data: MeshData
    figure: object = None
    ax: object = None
    plotter: object = None
    artist: object = None

    def show(self):
        if self.backend == "matplotlib":
            _dependency("matplotlib.pyplot").show()
        else:
            self.plotter.show()
        return self

    def save(self, filename, **kwargs):
        """Save a Matplotlib figure or a PyVista screenshot. Returns the path."""
        path = Path(filename)
        if self.backend == "matplotlib":
            self.figure.savefig(path, **kwargs)
        else:
            self.plotter.screenshot(str(path), **kwargs)
        return path

    def close(self):
        if self.backend == "matplotlib":
            _dependency("matplotlib.pyplot").close(self.figure)
        else:
            self.plotter.close()


def _geometry(mesh, elements, *, kind, labels=()):
    points, cells, ids, vertices = [], [], [], {}
    for el in elements:
        cell = []
        for v in el.vertices:
            # Identity, rather than coordinates, preserves coincident distinct nodes.
            key = v.nr
            if key not in vertices:
                vertices[key] = len(points)
                xyz = np.zeros(3)
                p = np.asarray(mesh[v].point, dtype=float)
                xyz[:len(p)] = p
                points.append(xyz)
            cell.append(vertices[key])
        cells.append(cell)
        ids.append(el.nr)
    return MeshData(points, tuple(cells), mesh.dim, kind, ids, tuple(labels))


def mesh_data(mesh, *, region="surface"):
    """Extract visible surface, volume, or named boundary elements from NGSolve."""
    ng = _dependency("ngsolve")
    if mesh.dim not in (2, 3):
        raise ValueError("Only 2D and 3D meshes are supported.")
    if region not in ("surface", "volume", "boundary"):
        raise ValueError("region must be surface, volume, or boundary.")
    boundary = region == "boundary" or (region == "surface" and mesh.dim == 3)
    elements = list(mesh.Elements(ng.BND if boundary else ng.VOL))
    names = mesh.GetBoundaries() if boundary else mesh.GetMaterials()
    kind = "lines" if boundary and mesh.dim == 2 else ("volume" if mesh.dim == 3 and not boundary else "surface")
    return _geometry(mesh, elements, kind=kind, labels=[names[e.index] for e in elements])


def _scalar(values, component):
    v = np.asarray(values)
    if isinstance(component, (int, np.integer)) and not isinstance(component, bool):
        if v.ndim != 2 or not 0 <= component < v.shape[1]:
            raise ValueError("A vector component requires a valid nonnegative index.")
        v = v[:, component]
        component = "real" if not np.iscomplexobj(v) else None
    if v.ndim == 2:
        if component != "magnitude":
            raise ValueError("Vector fields require component=<index> or 'magnitude'. Select a complex vector component in NGSolve before plotting real/imaginary/phase.")
        return np.linalg.norm(v, axis=1)
    if v.ndim != 1:
        raise ValueError("Expected a scalar or vector per cell.")
    if component is None:
        if np.iscomplexobj(v):
            raise ValueError("Complex fields require component='real', 'imag', 'magnitude' or 'phase'.")
        return v.astype(float)
    operations = {"real": np.real, "imag": np.imag, "magnitude": np.abs, "phase": np.angle}
    if component not in operations:
        raise ValueError("Unknown field component.")
    return operations[component](v)


def field_data(mesh, field, *, component=None):
    """Sample an NGSolve field inside each volume element, or accept cell arrays.

    Array rows follow list(mesh.Elements(VOL)), NOT a parameter/DOF vector.
    Use numpy_to_gridfunction(..., space=parameter_space) for reconstruction DOFs.
    Interior evaluation preserves discontinuous element fields. The sample is
    a representative value, not an element integral or an exact cell average.
    """
    ng = _dependency("ngsolve")
    data = mesh_data(mesh, region="volume")
    if isinstance(field, ConductivityParameter):
        field = field.to_backend_field(mesh=mesh)
    if callable(field):
        values = []
        for e in mesh.Elements(ng.VOL):
            # Python integration points borrow storage from their rule. Keep
            # both the rule and transformation alive throughout evaluation.
            rule = ng.IntegrationRule(e.type, 0)
            ip = next(iter(rule))
            trafo = mesh.GetTrafo(e)
            values.append(field(trafo(ip)))
        values = np.asarray(values)
    else:
        values = np.asarray(field)
    if values.ndim == 0 or values.shape[0] != len(data.cells):
        raise ValueError("Expected one field sample per volume element in Elements(VOL) order.")
    data.values = _scalar(values, component)
    return data


def _outside(data):
    if data.kind != "volume":
        return data
    faces = {}
    for i, c in enumerate(data.cells):
        for face in ((c[0], c[2], c[1]), (c[0], c[1], c[3]),
                     (c[1], c[2], c[3]), (c[2], c[0], c[3])):
            key = tuple(sorted(face))
            if key in faces:
                faces[key] = None
            else:
                faces[key] = (face, i)
    exterior = [x for x in faces.values() if x is not None]
    if not exterior:
        raise ValueError("No exterior volume faces found.")
    owners = [x[1] for x in exterior]
    return MeshData(data.points, tuple(x[0] for x in exterior), data.dimension,
                    "surface", data.element_ids[owners],
                    tuple(data.labels[i] for i in owners) if data.labels else (),
                    data.values[owners] if data.values is not None else None)


def _select(data, mask):
    idx = np.flatnonzero(mask)
    if not len(idx):
        raise ValueError("No cells matched the selection.")
    return MeshData(data.points, tuple(data.cells[i] for i in idx), data.dimension,
                    data.kind, data.element_ids[idx], tuple(data.labels[i] for i in idx),
                    data.values[idx] if data.values is not None else None)


def _backend(backend, dimension, ax, plotter):
    if backend == "auto":
        backend = "matplotlib" if ax is not None or (dimension == 2 and plotter is None) else "pyvista"
    if backend not in ("matplotlib", "pyvista"):
        raise ValueError("backend must be 'auto', 'matplotlib' or 'pyvista'.")
    if backend == "matplotlib" and plotter is not None:
        raise ValueError("plotter= requires backend='pyvista'.")
    if backend == "pyvista" and ax is not None:
        raise ValueError("ax= requires backend='matplotlib'.")
    return backend


def _render(data, *, backend="auto", ax=None, plotter=None, show=False,
            color="lightgray", opacity=1., show_edges=True, edge_color="black",
            line_width=.4, cmap="viridis", clim=None, colorbar=True,
            title=None, off_screen=False, boundary_colors=None, legend=False,
            backend_options=None, show_axes=True):
    backend = _backend(backend, data.dimension, ax, plotter)
    options = dict(backend_options or {})
    if backend == "matplotlib":
        if off_screen:
            raise ValueError("Use MPLBACKEND=Agg for headless Matplotlib; off_screen is PyVista-only.")
        if options:
            raise ValueError("backend_options is reserved for PyVista add_mesh; use PlotResult.artist for Matplotlib styling.")
        plt = _dependency("matplotlib.pyplot")
        from matplotlib.collections import PolyCollection, LineCollection
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
        from matplotlib.colors import Normalize
        visible = _outside(data)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d" if data.dimension == 3 else None)
        else:
            fig = ax.figure
            if (hasattr(ax, "add_collection3d")) != (data.dimension == 3):
                raise ValueError("Axes dimension must match mesh dimension.")
        polygons = [visible.points[c, :visible.dimension] for c in visible.cells]
        colors = [boundary_colors[n] for n in visible.labels] if boundary_colors else color
        args = dict(linewidths=line_width, alpha=opacity)
        if visible.kind == "lines":
            artist = LineCollection(polygons, colors=colors, **args)
        elif (visible.dimension == 3 and visible.values is None and
              isinstance(color, str) and color == "none"):
            # An empty face-color array makes Poly3DCollection fail during
            # projection on some Matplotlib versions. Render actual edges.
            segments = [polygon[[i, (i+1) % len(polygon)]]
                        for polygon in polygons for i in range(len(polygon))]
            artist = Line3DCollection(segments, colors=edge_color, **args)
        else:
            cls = Poly3DCollection if visible.dimension == 3 else PolyCollection
            artist = cls(polygons, facecolors=colors,
                         edgecolors=edge_color if show_edges else "none", **args)
        if visible.values is not None:
            artist.set_array(visible.values)
            artist.set_cmap(cmap)
            artist.set_norm(Normalize(*(clim or (None, None))))
        if visible.dimension == 3:
            ax.add_collection3d(artist)
        else:
            ax.add_collection(artist)
        p = np.vstack(polygons)
        center = (p.min(axis=0) + p.max(axis=0)) / 2
        radius = max(float(np.ptp(p, axis=0).max()) / 2, 1e-12) * 1.05
        ax.set(xlim=(center[0]-radius, center[0]+radius),
               ylim=(center[1]-radius, center[1]+radius), xlabel="x", ylabel="y")
        if visible.dimension == 3:
            ax.set_zlim(center[2]-radius, center[2]+radius)
            ax.set_zlabel("z")
            ax.set_box_aspect((1, 1, 1))
        else:
            ax.set_aspect("equal", adjustable="box")
        if title:
            ax.set_title(title)
        if not show_axes:
            ax.set_axis_off()
        if visible.values is not None and colorbar:
            fig.colorbar(artist, ax=ax)
        if boundary_colors and legend:
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(color=c, label=n) for n,c in boundary_colors.items()])
        result = PlotResult(backend, data, figure=fig, ax=ax, artist=artist)
    else:
        pv = _dependency("pyvista")
        if plotter is None:
            plotter = pv.Plotter(off_screen=off_screen)
        elif not isinstance(plotter, pv.Plotter):
            raise TypeError("plotter must be a pyvista.Plotter.")
        def add(d, facecolor, label=None):
            connectivity = np.concatenate([np.r_[len(c), c] for c in d.cells])
            if d.kind == "volume":
                grid = pv.UnstructuredGrid(connectivity, np.full(len(d.cells), 10, dtype=np.uint8), d.points)
            else:
                grid = pv.PolyData(d.points, **{"lines" if d.kind == "lines" else "faces": connectivity})
            args = dict(opacity=opacity, line_width=line_width, lighting=False,
                        show_edges=show_edges, edge_color=edge_color)
            if d.values is not None:
                grid.cell_data["field"] = d.values
                args.update(scalars="field", preference="cell", cmap=cmap, clim=clim,
                            show_scalar_bar=colorbar)
            else:
                args["color"] = facecolor
            if label:
                args["label"] = label
            args.update(options)
            return plotter.add_mesh(grid, **args)
        if boundary_colors:
            artist = [add(_select(data, np.array(data.labels) == n), c, n if legend else None)
                      for n,c in boundary_colors.items() if n in data.labels]
            if legend:
                plotter.add_legend()
        else:
            artist = add(data, color)
        if data.dimension == 2:
            plotter.view_xy()
            plotter.enable_parallel_projection()
        if title:
            plotter.add_text(title)
        if show_axes:
            plotter.add_axes()
        plotter.reset_camera()
        result = PlotResult(backend, data, plotter=plotter, artist=artist)
    if show:
        result.show()
    return result


def plot_mesh(mesh, *, region="surface", **kwargs):
    """Render NGSolve mesh or MeshData using either backend."""
    data = mesh if isinstance(mesh, MeshData) else mesh_data(mesh, region=region)
    return _render(data, **kwargs)


def plot_field(mesh, field=None, *, component=None, **kwargs):
    """Plot a scalar component; 3D Matplotlib shows the exterior only.

    PyVista keeps the tetrahedral volume in result.data and the actor dataset;
    use it for backend-specific slices/clip operations. No automatic averaging.
    Passing MeshData with values avoids importing NGSolve at rendering time.
    """
    if isinstance(mesh, MeshData):
        if field is not None or component is not None:
            raise ValueError("MeshData already contains prepared scalar values.")
        data = mesh
    else:
        data = field_data(mesh, field, component=component)
    if data.values is None:
        raise ValueError("Field data has no values.")
    return _render(data, **kwargs)


def plot_boundaries(mesh, boundary_colors, **kwargs):
    """Render selected named boundaries; duplicate names select all matching IDs."""
    if not boundary_colors:
        raise ValueError("Provide at least one boundary color.")
    data = mesh if isinstance(mesh, MeshData) else mesh_data(mesh, region="boundary")
    if data.kind == "volume" or not data.labels:
        raise ValueError("Expected labeled boundary data.")
    unknown = set(boundary_colors) - set(data.labels)
    if unknown:
        raise ValueError(f"Unknown or empty boundaries: {sorted(unknown)}; available: {sorted(set(data.labels))}")
    data = _select(data, np.isin(data.labels, list(boundary_colors)))
    return _render(data, boundary_colors=boundary_colors, colorbar=False, **kwargs)


class BoundaryInspector:
    """Native PyVista boundary checklist, with live actor/label visibility.

    Construct through inspect_boundaries(). UI is deliberately optional and
    outside the solver. A mesh is extracted once; checkbox changes do not
    rebuild geometry or run FEM solves. Keyboard: a=all, n=none, l=labels,
    bracket keys [ and ] change checklist pages.
    """

    def __init__(self, data, *, selected=None, show_names=False, colors=None,
                 opacity=.8, window_size=(1100, 760), page_size=16,
                 off_screen=False, label_positions=None):
        pv = _dependency("pyvista")
        if data.kind == "volume" or not data.labels:
            raise ValueError("Inspector requires named boundary geometry.")
        if not isinstance(page_size, int) or isinstance(page_size, bool) or page_size < 1:
            raise ValueError("page_size must be a positive integer.")
        width, height = window_size
        if width < 800 or height < 400:
            raise ValueError("Use window_size at least (800, 400) for readable controls.")
        self.data = data
        self.names = tuple(dict.fromkeys(data.labels))
        initial = set(self.names if selected is None else selected)
        unknown = initial - set(self.names)
        if unknown:
            raise ValueError(f"Unknown selected boundaries: {sorted(unknown)}")
        colors = dict(colors or {})
        label_positions = dict(label_positions or {})
        if (set(colors) | set(label_positions)) - set(self.names):
            raise ValueError("Color/label-position keys must be boundary names.")
        self.page_size = min(page_size, max(1, (height-230)//28))
        self.page = 0
        self.show_names = bool(show_names)
        self.actors, self.label_actors, self.widgets = {}, {}, []
        self._row_text, self._row_names = [], []
        self._height = height
        self._max_name_chars = max(10, int((width*.32-55)/12))
        self.plotter = pv.Plotter(shape=(1, 2), col_weights=[.32, .68],
                                  border=False, window_size=window_size,
                                  off_screen=off_screen, notebook=False)
        self.plotter.subplot(0, 1)
        self.plotter.set_background("white", all_renderers=False)
        palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
                   "#56B4E9", "#8B4513", "#6A5ACD", "#777777"]
        self.colors = {n: colors.get(n, palette[i % len(palette)]) for i,n in enumerate(self.names)}
        for name in self.names:
            part = _select(data, np.array(data.labels) == name)
            cells = np.concatenate([np.r_[len(c), c] for c in part.cells])
            grid = pv.PolyData(part.points, **{"lines" if part.kind == "lines" else "faces": cells})
            actor = self.plotter.add_mesh(grid, color=self.colors[name], opacity=opacity,
                                          show_edges=data.dimension == 3, line_width=3,
                                          lighting=False, reset_camera=False, render=False)
            self.actors[name] = actor
            centers = np.array([part.points[c].mean(axis=0) for c in part.cells])
            # Anchor on an actual cell, rather than the center of a ring/void.
            anchor = centers[np.argmin(np.linalg.norm(centers-centers.mean(axis=0), axis=1))]
            anchor = np.asarray(label_positions.get(name, anchor), dtype=float)
            if anchor.shape != (3,) or not np.isfinite(anchor).all():
                self.plotter.close()
                raise ValueError("Label positions must be finite (x, y, z) triples.")
            label = self.plotter.add_point_labels([anchor], [name], show_points=False,
                                                  always_visible=True, font_size=14,
                                                  text_color=self.colors[name],
                                                  shape_color="white", shape_opacity=.8,
                                                  reset_camera=False, render=False)
            self.label_actors[name] = label
            actor.SetVisibility(name in initial)
            label.SetVisibility(name in initial and self.show_names)
        if data.dimension == 2:
            self.plotter.view_xy()
            self.plotter.enable_parallel_projection()
        self.plotter.add_axes()
        bounds = np.column_stack((data.points.min(axis=0), data.points.max(axis=0))).ravel()
        self.plotter.reset_camera(bounds=bounds)
        self.plotter.subplot(0, 0)
        self.plotter.set_background("#F2F4F7", all_renderers=False)
        self.plotter.add_text("Boundary inspector", position=(12, height-38),
                              font_size=15, color="black")
        self._status = self.plotter.add_text("", position=(12, height-65), font_size=10, color="black")
        self.name_widget = self.plotter.add_checkbox_button_widget(
            lambda flag: self.set_names(flag), value=self.show_names,
            position=(12, height-110), size=20, border_size=2, color_on="#0072B2")
        self.plotter.add_text("Names in scene", position=(42, height-109), font_size=11, color="black")
        for row in range(self.page_size):
            y = height-155-row*28
            # Capture the row number, not the loop's mutable final value.
            widget = self.plotter.add_checkbox_button_widget(
                lambda flag, row=row: self._toggle_row(row, flag),
                position=(12,y), size=18, border_size=2, value=False,
                color_on="#0072B2")
            self.widgets.append(widget)
            self._row_text.append(self.plotter.add_text("", position=(40,y), font_size=10, color="black"))
        self._page_text = self.plotter.add_text("", position=(12,72), font_size=10, color="black")
        self.plotter.add_text("[ / ] previous / next page\na: all   n: none   l: names\nHover here before using keys",
                              position=(12,12), font_size=9, color="black")
        self.plotter.add_key_event("a", lambda: self.select_all(True))
        self.plotter.add_key_event("n", lambda: self.select_all(False))
        self.plotter.add_key_event("l", lambda: self.set_names(not self.show_names))
        self.plotter.add_key_event("bracketleft", lambda: self.set_page(self.page-1))
        self.plotter.add_key_event("bracketright", lambda: self.set_page(self.page+1))
        self._refresh_controls()
        self.plotter.subplot(0, 1)

    @property
    def selected(self):
        """Currently visible boundary names, preserving mesh order."""
        return tuple(n for n in self.names if self.actors[n].GetVisibility())

    @property
    def page_count(self):
        return (len(self.names)+self.page_size-1)//self.page_size

    def _refresh_controls(self):
        self._row_names = list(self.names[self.page*self.page_size:(self.page+1)*self.page_size])
        for row, (widget, text) in enumerate(zip(self.widgets, self._row_text)):
            present = row < len(self._row_names)
            if present:
                name = self._row_names[row]
                text.SetInput(name if len(name) <= self._max_name_chars else name[:self._max_name_chars-3]+"...")
                text.GetTextProperty().SetColor(_dependency("pyvista").Color(self.colors[name]).float_rgb)
                widget.GetRepresentation().SetState(int(name in self.selected))
            else:
                text.SetInput("")
            widget.SetEnabled(present)
            widget.GetRepresentation().SetVisibility(present)
        self.name_widget.GetRepresentation().SetState(int(self.show_names))
        self._status.SetInput(f"{len(self.selected)} / {len(self.names)} boundaries visible")
        self._page_text.SetInput(f"Page {self.page+1} / {self.page_count}")

    def _toggle_row(self, row, visible):
        if row < len(self._row_names):
            self.set_visible(self._row_names[row], visible)

    def set_visible(self, name, visible=True):
        if name not in self.actors:
            raise ValueError(f"Unknown boundary: {name}")
        self.actors[name].SetVisibility(bool(visible))
        self.label_actors[name].SetVisibility(bool(visible) and self.show_names)
        # During construction VTK calls initial callbacks before controls exist.
        if hasattr(self, "_page_text"):
            self._refresh_controls()
        self.plotter.render()
        return self

    def set_names(self, visible=True):
        self.show_names = bool(visible)
        for name, label in self.label_actors.items():
            label.SetVisibility(self.show_names and bool(self.actors[name].GetVisibility()))
        if hasattr(self, "_page_text"):
            self._refresh_controls()
        self.plotter.render()
        return self

    def select_all(self, visible=True):
        for name, actor in self.actors.items():
            actor.SetVisibility(bool(visible))
            self.label_actors[name].SetVisibility(bool(visible) and self.show_names)
        self._refresh_controls()
        self.plotter.render()
        return self

    def set_page(self, page):
        self.page = max(0, min(int(page), self.page_count-1))
        self._refresh_controls()
        self.plotter.render()
        return self

    def show(self, **kwargs):
        """Open a native interactive window. Static notebook output has no live controls."""
        self.plotter.show(**kwargs)
        return self

    def save(self, filename, **kwargs):
        return self.plotter.screenshot(str(filename), **kwargs)

    def close(self):
        self.plotter.close()


def inspect_boundaries(mesh, *, show=True, **kwargs):
    """List/tick boundary names in a native PyVista window (2D or 3D).

    Examples: inspect_boundaries(mesh, show_names=True); or
    inspector = inspect_boundaries(mesh, selected=['el1'], show=False).
    Other arguments: colors, opacity, window_size, page_size, off_screen,
    label_positions={name: (x,y,z)}. Duplicate names are grouped together.
    PyVista is the GUI backend regardless of the static plotting default.
    """
    data = mesh if isinstance(mesh, MeshData) else mesh_data(mesh, region="boundary")
    inspector = BoundaryInspector(data, **kwargs)
    if show:
        inspector.show()
    return inspector


def numpy_to_gridfunction(mesh, values, *, space=None):
    """Convert a DOF vector through its actual L2(0) space, preserving ordering."""
    ng = _dependency("ngsolve")
    values = np.asarray(values)
    if space is None:
        space = ng.L2(mesh, order=0, complex=np.iscomplexobj(values))
    if space.mesh != mesh:
        raise ValueError("The parameter space belongs to another mesh.")
    if values.shape != (space.ndof,):
        raise ValueError(f"Expected a DOF vector of length {space.ndof}.")
    gf = ng.GridFunction(space)
    target = gf.vec.FV().NumPy()
    if np.iscomplexobj(values) and not np.iscomplexobj(target):
        raise ValueError("Complex values require a complex parameter space.")
    target[:] = values
    return gf


def _sample_grid(field, mesh, *, axes, extent, resolution, component, fixed=None):
    """Evaluate a native field on a pixel-center grid in physical coordinates."""
    if isinstance(field, ConductivityParameter):
        field = field.to_backend_field(mesh=mesh)
    if not isinstance(resolution, (int, np.integer)) or resolution < 2:
        raise ValueError("resolution must be an integer >= 2.")
    if extent is None:
        vertices = np.asarray([v.point for v in mesh.vertices])
        a, b = axes
        extent = (vertices[:, a].min(), vertices[:, a].max(),
                  vertices[:, b].min(), vertices[:, b].max())
    try:
        bounds = np.asarray(extent, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("extent must be (umin, umax, vmin, vmax).") from exc
    if (bounds.shape != (4,) or not np.isfinite(bounds).all()
            or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]):
        raise ValueError("extent must be finite (umin, umax, vmin, vmax) with increasing bounds.")
    image = np.full((resolution, resolution), np.nan)
    # Pixel centers match imshow's edge-based extent.
    u = np.linspace(bounds[0], bounds[1], resolution+1)
    v = np.linspace(bounds[2], bounds[3], resolution+1)
    coordinates = np.zeros(mesh.dim)
    if fixed is not None:
        coordinates[fixed[0]] = fixed[1]
    for j, vv in enumerate((v[:-1]+v[1:])/2):
        for i, uu in enumerate((u[:-1]+u[1:])/2):
            coordinates[axes[0]], coordinates[axes[1]] = uu, vv
            try:
                point = mesh(*map(float, coordinates))
            except ValueError:
                continue
            if point.nr < 0:
                continue
            # Keep evaluation outside the lookup handler: field errors are not
            # outside-domain pixels and must propagate to the caller.
            image[j,i] = _scalar(np.asarray([field(point)]), component)[0]
    return image, tuple(bounds)


def sample_image(field, mesh, *, extent=None, resolution=64, component=None):
    """Return image and extent. Only points outside the mesh become NaN.

    A 2D image is a separate resampling operation, not the default mesh plot.
    Boundary evaluation of discontinuous fields has a side-selection ambiguity.
    """
    if mesh.dim != 2:
        raise ValueError("Image sampling requires a 2D mesh.")
    return _sample_grid(field, mesh, axes=(0, 1), extent=extent,
                        resolution=resolution, component=component)


def sample_slice(field, mesh, *, plane="xy", position=0.0, extent=None,
                 resolution=64, component=None):
    """Sample a coordinate plane through a 3D field; return (image, extent).

    plane='xy' fixes z=position; 'xz' fixes y; 'yz' fixes x. Image columns
    follow the first plane coordinate and rows the second, both increasing.
    extent=(umin, umax, vmin, vmax) gives pixel-edge bounds in those coordinates;
    the default is the projected mesh-vertex bounding box. resolution is the
    number of pixels along each axis. Pass the returned extent to plot_image.

    Evaluate the original field at pixel centers using the native curved mesh.
    Outside-volume pixels are NaN; a plane missing the volume returns all NaNs.
    Field evaluation errors propagate. Discontinuous fields on interfaces have
    the same side-selection ambiguity as sample_image. ConductivityParameter
    objects and scalar/vector/complex component selections are supported.
    """
    if mesh.dim != 3:
        raise ValueError("Slice sampling requires a 3D mesh.")
    planes = {"xy": ((0, 1), 2), "xz": ((0, 2), 1), "yz": ((1, 2), 0)}
    if not isinstance(plane, str) or plane not in planes:
        raise ValueError("plane must be 'xy', 'xz' or 'yz'.")
    if not np.isscalar(position) or np.iscomplexobj(position):
        raise ValueError("position must be a finite real coordinate.")
    try:
        position = float(position)
    except (TypeError, ValueError) as exc:
        raise ValueError("position must be a finite real coordinate.") from exc
    if not np.isfinite(position):
        raise ValueError("position must be a finite real coordinate.")
    axes, fixed_axis = planes[plane]
    return _sample_grid(field, mesh, axes=axes, extent=extent,
                        resolution=resolution, component=component,
                        fixed=(fixed_axis, position))


# Compatibility helpers retain the original return types. New code uses plot_*.

def plot_image(img, ax=None, *, extent=(-1, 1, -1, 1), colorbar=None, **kwargs):
    plt = _dependency("matplotlib.pyplot")
    own = ax is None
    if own:
        _, ax = plt.subplots()
    image = ax.imshow(img, origin="lower", extent=extent, **kwargs)
    if colorbar if colorbar is not None else own:
        ax.figure.colorbar(image, ax=ax)
    return image

