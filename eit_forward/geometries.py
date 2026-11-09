"""Disk/cylinder geometry builders with named material regions and mesh points."""
from dataclasses import dataclass
from math import degrees, radians
from netgen.occ import (WorkPlane, Pnt, Box, Circle, Sphere, Rectangle, Cylinder,
                        Glue, Fuse, OCCGeometry, Axis, Z)
from ngsolve import Mesh
import numpy as np
from .points import NamedPoints

boundary_feature_dtype = np.dtype([
    ("kind", "U16"),
    ("name", "U64"),
    ("group", "U64"),
    ("shape", "U16"),     # "sphere", "cylinder", "box"

    ("x", "f8"),
    ("y", "f8"),
    ("z", "f8"),

    ("r", "f8"),
    ("h", "f8"),

    ("ax", "f8"),
    ("ay", "f8"),
    ("az", "f8"),

    ("x2", "f8"),
    ("y2", "f8"),
    ("z2", "f8"),

    ("width", "f8"),
    ("height", "f8"),
    ("depth", "f8"),
    ("angle", "f8"),
])


internal_object_dtype = np.dtype([
    ("mode", "U16"),      # "inclusion" or "void"
    ("name", "U64"),
    ("shape", "U16"),     # "sphere", "cylinder", "box"
    ("x", "f8"),
    ("y", "f8"),
    ("z", "f8"),
    ("r", "f8"),
    ("h", "f8"),
    ("ax", "f8"),
    ("ay", "f8"),
    ("az", "f8"),
    ("x2", "f8"),         # for box max corner
    ("y2", "f8"),
    ("z2", "f8"),
    ("width", "f8"),
    ("length", "f8"),
])


@dataclass
class ElectrodeDisk(NamedPoints):
    radius: float
    n_electrodes: int

    electrode_angle: float | None = None   # degrees
    electrode_width: float | None = None   # arc length

    electrode_prefix: str = "el"
    insulation_name: str = "ins"

    body_name: str = "body"

    angular_shift: float = 0.0

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("radius must be positive")

        if isinstance(self.n_electrodes, (bool, np.bool_)) or not isinstance(self.n_electrodes, (int, np.integer)) or self.n_electrodes < 0:
            raise ValueError("n_electrodes must be a nonnegative integer")

        if not np.isfinite(self.angular_shift):
            raise ValueError("angular_shift must be a finite angle in degrees")

        if self.n_electrodes == 0:
            if self.electrode_angle is not None or self.electrode_width is not None:
                raise ValueError("Electrode dimensions require finite electrodes")
            self.electrode_angle = 0.0
        elif (
            self.electrode_angle is not None
            and self.electrode_width is not None
        ):
            raise ValueError(
                "Specify either electrode_angle or electrode_width, not both."
            )

        if (
            self.electrode_angle is None
            and self.electrode_width is None
        ):
            self.electrode_angle = 180.0 / self.n_electrodes

        if self.electrode_width is not None:
            self.electrode_angle = degrees(
                self.electrode_width / self.radius
            )

        sector_angle = self.sector_angle

        if self.n_electrodes and not 0 < self.electrode_angle < sector_angle:
            raise ValueError(
                "electrode angle must be smaller than "
                f"{sector_angle:.3f} degrees"
            )

        self.electrode_names = []
        self.objects = np.zeros(0, dtype=internal_object_dtype)
        self.face = None
        self.geo = None
        self.mesh = None
        self._init_named_points(2)

    @property
    def sector_angle(self) -> float:
        """Angular pitch between neighbouring electrodes."""
        return 360.0 / self.n_electrodes if self.n_electrodes else 360.0

    @property
    def insulation_angle(self) -> float:
        """Angular width of each insulating gap."""
        return self.sector_angle - self.electrode_angle

    @property
    def actual_electrode_width(self) -> float:
        """Electrode arc length along the disk boundary."""

        return self.radius * radians(self.electrode_angle)

    def _invalidate(self):
        self.geo = None
        self.mesh = None
        self.shape = None

    def _append_object(self,
                       *,
                       mode,
                       name,
                       shape,
                       x=0.0, y=0.0,
                       r=0.0,
                       width=0.0,
                       length=0.0,
                       ):
        row = np.zeros(1, dtype=internal_object_dtype)
        row["mode"][0] = mode
        row["name"][0] = name
        row["shape"][0] = shape
        row["x"][0] = x
        row["y"][0] = y
        row["r"][0] = r
        row["width"] = width
        row["length"] = length

        self.objects = np.concatenate([self.objects, row])
        self._invalidate()

    def add_disk_void(self, *, name, p, r):
        self._append_object(
            mode="void",
            name=name,
            shape="circle",
            x=p[0], y=p[1],
            r=r, 
        )
        return self

    def add_box_void(self, *, name, p, width, length):
        self._append_object(
            mode="void",
            name=name,
            shape="box",
            x=p[0], y=p[1],
            width=width,
            length=length,
        )
        return self

    def add_disk_inclusion(self, *, name, p, r):
        self._append_object(
            mode="inclusion",
            name=name,
            shape="circle",
            x=p[0], y=p[1],
            r=r,
        )
        return self

    def add_box_inclusion(self, *, name, p, width, length):
        self._append_object(
            mode="inclusion",
            name=name,
            shape="box",
            x=p[0], y=p[1],
            width=width,
            length=length,
        )
        return self

    def _make_object_shape(self, rec):
        shape = str(rec["shape"]).lower()

        if shape == "circle":
            return Circle(
                (float(rec["x"]), float(rec["y"])),
                float(rec["r"])
                ).Face()

        elif shape == "box":
            center = (float(rec["length"]/2), float(rec["width"]/2))
            return Rectangle(
                l=float(rec["length"]),
                w=float(rec["width"]),
            ).Face().Move((-center[0],-center[1],0)).Move((float(rec["x"]),float(rec["y"]),0))

        else:
            raise ValueError(f"Unsupported object shape: {shape}")

    def make_geo(self) -> OCCGeometry:
        shift_deg = self.angular_shift % 360.0
        shift_rad = radians(shift_deg)

        self.wp = (
            WorkPlane()
            .MoveTo(
                self.radius * np.cos(shift_rad),
                self.radius * np.sin(shift_rad),
            )
            .Rotate(90.0 + shift_deg)
        )

        self.electrode_names = []

        for i in range(self.n_electrodes):
            electrode_name = f"{self.electrode_prefix}_{i}"
            self.electrode_names.append(electrode_name)
            self.wp.Arc(
                self.radius, self.electrode_angle, name=electrode_name,
            )
            self.wp.Arc(
                self.radius, self.insulation_angle, name=self.insulation_name,
            )

        if self.n_electrodes:
            working_body = self.wp.Face()
        else:
            working_body = Circle((0, 0), self.radius).Face()
            working_body.edges.name = self.insulation_name

        inclusion_parts = []
        self.object_names = []

        for rec in self.objects:
            tool = self._make_object_shape(rec)
            mode = str(rec["mode"])
            name = str(rec["name"])

            if mode == "void":
                working_body = working_body - tool

            elif mode == "inclusion":
                tool.edges.name = name
                tool.faces.name = name
                tool.faces.col = (1, 0, 1)
                working_body = working_body - tool

                inclusion_parts.append(tool)
                self.object_names.append(name)

            else:
                raise ValueError(f"Unsupported object mode: {mode}")

        working_body.faces.name = self.body_name

        shape = (
            Glue([working_body, *inclusion_parts])
            if inclusion_parts else working_body
        )

        shape = self._embed_points(shape)
        self.shape = shape
        self.geo = OCCGeometry(shape, dim=2)
        return self.geo

    def make_mesh(self, maxh=0.1, curve_order=3, **kwargs):
        if self.geo is None:
            self.make_geo()
        mesh = Mesh(
            self.geo.GenerateMesh(maxh=maxh, **kwargs)
        ).Curve(curve_order)
        self._verify_points(mesh)
        self.mesh = mesh
        return mesh


class CylinderModel(NamedPoints):
    def __init__(
        self,
        radius=1.0,
        height=1.0,
        electrode_prefix="el",
        body_name="body",
    ):
        self.radius = radius
        self.height = height
        self.electrode_prefix = electrode_prefix
        self.body_name = body_name

        self.features = np.zeros(0, dtype=boundary_feature_dtype)
        self.objects = np.zeros(0, dtype=internal_object_dtype)

        self.electrode_names = []
        self.patch_names = []
        self.object_names = []

        self.shape = None
        self.geo = None
        self.mesh = None
        self._init_named_points(3)

    def _invalidate(self):
        self.geo = None
        self.mesh = None
        self.shape = None

    def _append_feature(
        self,
        *,
        kind,
        name="",
        group="",
        shape,
        x=0.0,
        y=0.0,
        z=0.0,
        r=0.0,
        h=0.0,
        axis=(0.0, 0.0, 1.0),
        width=0.0,
        depth=0.0,
        angle=0.0,
    ):
        row = np.zeros(1, dtype=boundary_feature_dtype)
    
        row["kind"][0] = kind
        row["name"][0] = name
        row["group"][0] = group
        row["shape"][0] = shape
    
        row["x"][0] = x
        row["y"][0] = y
        row["z"][0] = z
    
        row["r"][0] = r
        row["h"][0] = h
    
        row["ax"][0] = axis[0]
        row["ay"][0] = axis[1]
        row["az"][0] = axis[2]
    
        row["width"][0] = width
        row["depth"][0] = depth
        row["angle"][0] = angle
    
        self.features = np.concatenate([self.features, row])
        self._invalidate()
        
    def _append_object(self,
                       *,
                       mode,
                       name,
                       shape,
                       x=0.0, y=0.0, z=0.0,
                       r=0.0,
                       h=0.0,
                       axis=(0.0, 0.0, 1.0),
                       pmax=(0.0, 0.0, 0.0),
                       ):
        row = np.zeros(1, dtype=internal_object_dtype)
        row["mode"][0] = mode
        row["name"][0] = name
        row["shape"][0] = shape
        row["x"][0] = x
        row["y"][0] = y
        row["z"][0] = z
        row["r"][0] = r
        row["h"][0] = h
        row["ax"][0] = axis[0]
        row["ay"][0] = axis[1]
        row["az"][0] = axis[2]
        row["x2"][0] = pmax[0]
        row["y2"][0] = pmax[1]
        row["z2"][0] = pmax[2]
    
        self.objects = np.concatenate([self.objects, row])
        self._invalidate()

    def add_round_electrode_ring(
        self,
        n,
        z,
        r,
        start_angle_deg=0.0,
        shape="sphere",
        group="",
        h=0.0,
        axis=(0.0, 0.0, 1.0),
    ):
        theta = np.linspace(0, 2*np.pi, n, endpoint=False) + np.deg2rad(start_angle_deg)

        for t in theta:
            x = self.radius * np.cos(t)
            y = self.radius * np.sin(t)
            self.add_round_electrode(
                group=group,
                shape=shape,
                x=x, y=y, z=z,
                r=r,
                h=h,
                axis=axis
                )
        return self

    def add_round_electrode(
        self,
        *,
        x,
        y,
        z,
        r,
        shape="sphere",
        group="",
        h=0.0,
        axis=(0.0, 0.0, 1.0),
    ):
        self._append_feature(
            kind="electrode",
            name="",                  # assigned later
            group=group,
            shape=shape,
            x=x, y=y, z=z,
            r=r,
            h=h,
            axis=axis,
        )
        return self

    def add_rectangular_electrode_ring(
        self,
        *,
        n,
        z,
        width,
        height,
        start_angle_deg=0.0,
        group="",
    ):
        angles = (
            np.linspace(0.0, 360.0, n, endpoint=False)
            + start_angle_deg
        )
    
        for angle in angles:
            self.add_rectangular_electrode(
                angle_deg=angle,
                z=z,
                width=width,
                height=height,
                group=group,
            )
    
        return self
    
    def add_rectangular_electrode(
        self,
        *,
        angle_deg,
        z,
        width,
        height,
        group="",
    ):
        self._append_feature(
            kind="electrode",
            name="",
            group=group,
            shape="box",
            z=z,
            width=width,
            h=height,
            angle=angle_deg,
        )
        return self

    def add_patch(
        self,
        *,
        name,
        x,
        y,
        z,
        r,
        shape="sphere",
        group="",
        h=0.0,
        axis=(0.0, 0.0, 1.0),
    ):
        self._append_feature(
            kind="patch",
            name=name,                # explicit user name
            group=group,
            shape=shape,
            x=x, y=y, z=z,
            r=r,
            h=h,
            axis=axis,
        )
        return self   
    
    def add_rectangular_patch(
        self,
        *,
        name,
        angle_deg,
        z,
        width,
        height,
        group="",
    ):
        self._append_feature(
            kind="patch",
            name=name,
            group=group,
            shape="box",
            z=z,
            width=width,
            h=height,
            angle=angle_deg,
        )
        return self
    
    def add_spherical_inclusion(self, *, name, center, r):
        self._append_object(
            mode="inclusion",
            name=name,
            shape="sphere",
            x=center[0], y=center[1], z=center[2],
            r=r,
        )
        return self
    
    def add_cylindrical_inclusion(self, *, name, p, axis, r, h):
        self._append_object(
            mode="inclusion",
            name=name,
            shape="cylinder",
            x=p[0], y=p[1], z=p[2],
            r=r, h=h,
            axis=axis,
        )
        return self
    
    def add_box_inclusion(self, *, name, pmin, pmax):
        self._append_object(
            mode="inclusion",
            name=name,
            shape="box",
            x=pmin[0], y=pmin[1], z=pmin[2],
            pmax=pmax,
        )
        return self
    
    def add_spherical_void(self, *, name, center, r):
        self._append_object(
            mode="void",
            name=name,
            shape="sphere",
            x=center[0], y=center[1], z=center[2],
            r=r,
        )
        return self
    
    def add_cylindrical_void(self, *, name, p, axis, r, h):
        self._append_object(
            mode="void",
            name=name,
            shape="cylinder",
            x=p[0], y=p[1], z=p[2],
            r=r, h=h,
            axis=axis,
        )
        return self
    
    def add_box_void(self, *, name, pmin, pmax):
        self._append_object(
            mode="void",
            name=name,
            shape="box",
            x=pmin[0], y=pmin[1], z=pmin[2],
            pmax=pmax,
        )
        return self

    def _assign_electrode_names(self):
        electrode_counter = 0
        names = []

        for i in range(len(self.features)):
            if str(self.features["kind"][i]) == "electrode":
                name = f"{self.electrode_prefix}_{electrode_counter}"
                self.features["name"][i] = name
                names.append(name)
                electrode_counter += 1

        self.electrode_names = [str(x) for x in names]
        self.patch_names = [
            str(self.features["name"][i])
            for i in range(len(self.features))
            if str(self.features["kind"][i]) == "patch"
        ]

    def _make_feature_shape(self, rec):
        shape = str(rec["shape"]).lower()
    
        if shape == "sphere":
            return Sphere(
                (
                    float(rec["x"]),
                    float(rec["y"]),
                    float(rec["z"]),
                ),
                float(rec["r"]),
            )
    
        elif shape == "cylinder":
            return Cylinder(
                (
                    float(rec["x"]),
                    float(rec["y"]),
                    float(rec["z"]),
                ),
                (
                    float(rec["ax"]),
                    float(rec["ay"]),
                    float(rec["az"]),
                ),
                r=float(rec["r"]),
                h=float(rec["h"]),
            )
    
        elif shape == "box":
            width = float(rec["width"])
            height = float(rec["h"])
            angle = float(rec["angle"])
            z = float(rec["z"])
        
            depth = max(0.05 * self.radius, 1e-3)
        
            box = Box(
                Pnt(
                    self.radius - depth,
                    -width / 2,
                    -height / 2,
                ),
                Pnt(
                    self.radius + depth,
                    width / 2,
                    height / 2,
                ),
            )
        
            return (
                box
                .Move((0, 0, z))
                .Rotate(Axis((0, 0, 0), Z), angle)
            )
        else:
            raise ValueError(f"Unsupported feature shape: {shape}")

    def _make_object_shape(self, rec):
        shape = str(rec["shape"]).lower()
    
        if shape == "sphere":
            return Sphere(
                (float(rec["x"]), float(rec["y"]), float(rec["z"])),
                float(rec["r"])
            )
    
        elif shape == "cylinder":
            return Cylinder(
                (float(rec["x"]), float(rec["y"]), float(rec["z"])),
                (float(rec["ax"]), float(rec["ay"]), float(rec["az"])),
                r=float(rec["r"]),
                h=float(rec["h"]),
            )
    
        elif shape == "box":
            return Box(
                Pnt(float(rec["x"]),  float(rec["y"]),  float(rec["z"])),
                Pnt(float(rec["x2"]), float(rec["y2"]), float(rec["z2"])),
            )
    
        else:
            raise ValueError(f"Unsupported object shape: {shape}")


    def make_geo(self):
        self._assign_electrode_names()

        host = Cylinder((0, 0, 0), Z, r=self.radius, h=self.height)
        
        bulk = host
        boundary_parts = []

        for rec in self.features:
            tool = self._make_feature_shape(rec)

            bulk = bulk - tool
            patch = host * tool

            name = str(rec["name"])
            kind = str(rec["kind"])

            patch.faces.name = name
            if kind == "electrode":
                patch.faces.col = (1, 0, 0)
            else:
                patch.faces.col = (0, 0, 1)

            boundary_parts.append(patch)
            
        bulk.solids.name = self.body_name
        bulk.faces.name = self.body_name
        
        patched_body = Fuse([bulk, *boundary_parts])

        working_body = patched_body
        inclusion_parts = []
        self.object_names = []
        
        for rec in self.objects:
            tool = self._make_object_shape(rec)
            mode = str(rec["mode"])
            name = str(rec["name"])
        
            if mode == "void":
                working_body = working_body - tool
        
            elif mode == "inclusion":
                inclusion = working_body * tool
                working_body = working_body - tool
        
                inclusion.solids.name = name
                inclusion.faces.name = name
                inclusion.faces.col = (1, 0, 1)
        
                inclusion_parts.append(inclusion)
                self.object_names.append(name)
        
            else:
                raise ValueError(f"Unsupported object mode: {mode}")

        working_body.solids.name = self.body_name
        
        shape = Glue([working_body, *inclusion_parts]) if inclusion_parts else working_body
        
        shape.faces.Max(Z).name = 'top'
        shape.faces.Min(Z).name = 'bottom'

        shape = self._embed_points(shape)
        
        self.shape = shape
        self.geo = OCCGeometry(shape)
        return self.geo
        
    def make_mesh(self, maxh=0.1, curve_order=3, **kwargs):
        if self.geo is None:
            self.make_geo()
        mesh = Mesh(self.geo.GenerateMesh(maxh=maxh, **kwargs)).Curve(curve_order)
        self._verify_points(mesh)
        self.mesh = mesh
        return mesh
