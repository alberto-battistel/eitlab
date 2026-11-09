"""Named, conforming geometry vertices shared by the disk and cylinder builders."""
from dataclasses import dataclass
from types import MappingProxyType
import re

import numpy as np
from netgen.occ import Box, Glue, Vertex
from ngsolve import BBND, BBBND
from eitlab.bibliography import BibliographyMixin


@dataclass(frozen=True)
class NamedPoint:
    name: str
    position: tuple[float, float, float]
    group: str = ""
    kind: str = "point"


class NamedPoints(BibliographyMixin):
    """Geometry description and name-based mesh lookup; no electrical physics."""
    bibliography_component = "netgen"

    def _init_named_points(self, dimension):
        self._point_dimension = dimension
        self._points = {}

    @property
    def points(self):
        return MappingProxyType(self._points)

    @property
    def point_names(self):
        return tuple(self._points)

    @property
    def point_electrode_names(self):
        return tuple(p.name for p in self._points.values() if p.kind == "electrode")

    def _point_tolerance(self):
        return 1e-8 * max(abs(self.radius), abs(getattr(self, "height", 0))) + 1e-12

    def _point_record(self, *, name, x, y, z, group, kind):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Point name must be a nonempty string")
        if name in self._points:
            raise ValueError(f"Duplicate point name: {name!r}")
        position = np.asarray([x, y, z], dtype=float)
        if not np.all(np.isfinite(position)):
            raise ValueError("Point coordinates must be finite")
        if self._point_dimension == 2 and position[2] != 0:
            raise ValueError("Disk points must have z=0")
        tol = self._point_tolerance()
        if any(np.linalg.norm(position - p.position) <= tol for p in self._points.values()):
            raise ValueError("Coincident named points are not supported; reuse the existing name")
        return NamedPoint(name, tuple(float(v) for v in position), str(group), kind)

    def add_point(self, *, name, x, y, z=0.0, group=""):
        """Add a named vertex, using the same keyword/fluent style as add_patch."""
        point = self._point_record(name=name, x=x, y=y, z=z, group=group, kind="point")
        self._points[name] = point
        self._invalidate()
        return self

    def add_point_electrode(self, *, name, x, y, z=0.0, group=""):
        """Register an idealized electrode location, separately from CEM patches."""
        point = self._point_record(name=name, x=x, y=y, z=z, group=group, kind="electrode")
        self._points[name] = point
        self._invalidate()
        return self

    def add_point_electrode_ring(self, n, z=0.0, start_angle_deg=0.0,
                                 group="", prefix=None):
        """Equally spaced points on r=self.radius, with persistent unique names.

        Default names are el_point_0, el_point_1, ...; subsequent rings continue
        the numbering. A custom prefix can distinguish rings explicitly.
        """
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")
        if not np.isfinite(start_angle_deg):
            raise ValueError("start_angle_deg must be finite")
        prefix = f"{self.electrode_prefix}_point" if prefix is None else prefix
        if not isinstance(prefix, str) or not prefix.strip():
            raise ValueError("prefix must be a nonempty string")
        # Validate the entire ring before mutating the builder.
        staged = []
        next_index = 0
        for angle in np.linspace(0, 2*np.pi, n, endpoint=False) + np.deg2rad(start_angle_deg):
            while f"{prefix}_{next_index}" in self._points:
                next_index += 1
            name = f"{prefix}_{next_index}"
            staged.append(self._point_record(
                name=name, x=self.radius*np.cos(angle), y=self.radius*np.sin(angle),
                z=z, group=group, kind="electrode"))
            next_index += 1
        self._points.update((point.name, point) for point in staged)
        self._invalidate()
        return self

    def _embed_points(self, shape):
        """Embed vertices in OCC topology, partitioning volume interiors as needed.

        Floating OCC vertices inside 3D solids are otherwise left disconnected
        by Netgen. A conforming horizontal material-preserving partition puts
        each such point onto a meshed internal face before adding the vertex.
        """
        vertices = []
        tol = self._point_tolerance()
        for point in self._points.values():
            vertex = Vertex(point.position)
            containers = shape.solids if self._point_dimension == 3 else shape.faces
            if not any(part.Distance(vertex) <= tol and len((part * vertex).vertices)
                       for part in containers):
                raise ValueError(f"Point {point.name!r} lies outside the meshed domain or inside a void")
            if self._point_dimension == 3 and min(face.Distance(vertex) for face in shape.faces) > tol:
                low, high = (np.array(tuple(v)) for v in shape.bounding_box)
                margin = max(np.max(high-low), tol) * 2
                cutter = Box(tuple(low-margin),
                             (float(high[0]+margin), float(high[1]+margin), point.position[2]))
                cutter.faces.name = "__point_partition__"
                shape = Glue([shape*cutter, shape-cutter])
            vertex.name = point.name
            vertices.append(vertex)
        return Glue([shape, *vertices]) if vertices else shape

    def _point_mesh(self, mesh):
        mesh = self.mesh if mesh is None else mesh
        if mesh is None:
            raise ValueError("Generate a mesh before retrieving named points")
        if mesh.dim != self._point_dimension:
            raise ValueError("Mesh dimension does not match the geometry")
        return mesh

    def point_region(self, name, *, mesh=None):
        """Return the native zero-dimensional region for constraints or loads."""
        if name not in self._points:
            raise KeyError(name)
        mesh = self._point_mesh(mesh)
        names = mesh.GetBBoundaries() if mesh.dim == 2 else mesh.GetBBBoundaries()
        if name not in names:
            raise ValueError(f"Mesh does not contain point {name!r}")
        return (mesh.BBoundaries if mesh.dim == 2 else mesh.BBBoundaries)(re.escape(name))

    def point_vertex(self, name, *, mesh=None):
        """Return the current zero-based vertex number, resolved by its name."""
        mesh = self._point_mesh(mesh)
        region = self.point_region(name, mesh=mesh)
        nodes = {v.nr for element in mesh.Elements(BBND if mesh.dim == 2 else BBBND)
                 if region.Mask()[element.index] for v in element.vertices}
        if len(nodes) != 1:
            raise RuntimeError(f"Point {name!r} must identify exactly one mesh vertex")
        return nodes.pop()

    def point_dofs(self, space, name):
        """Resolve point DOFs for an NGSolve space, without node/DOF assumptions."""
        mask = space.GetDofs(self.point_region(name, mesh=space.mesh))
        return tuple(i for i, selected in enumerate(mask) if selected)

    def _verify_points(self, mesh):
        """Reject shifted or disconnected points instead of returning an unsafe mesh."""
        for point in self._points.values():
            node = mesh.vertices[self.point_vertex(point.name, mesh=mesh)]
            if not np.allclose(node.point, point.position[:mesh.dim], rtol=0,
                               atol=self._point_tolerance()):
                raise RuntimeError(f"Mesher moved named point {point.name!r}")
            if not len(node.elements):
                raise RuntimeError(f"Named point {point.name!r} is disconnected from the volume mesh")
