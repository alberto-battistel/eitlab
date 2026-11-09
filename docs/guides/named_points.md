# Named mesh points and point-electrode rings

Both `ElectrodeDisk` and `CylinderModel` now provide the same fluent interface
as their patch/object constructors:

```python
geometry.add_point(name="ground", x=0.0, y=0.0, z=0.5)
geometry.add_point_electrode(name="sensor", x=1.0, y=0.0, z=0.5)
geometry.add_point_electrode_ring(
    n=16, z=0.25, start_angle_deg=11.25, group="lower",
)
geometry.add_point_electrode_ring(n=16, z=0.75, group="upper")
mesh = geometry.make_mesh(maxh=0.2)
vertex_number = geometry.point_vertex("ground")
```

Points have explicit names, coordinates, groups and a feature kind. A point
electrode is a named point with `kind="electrode"`; assigning that role does not
add a finite contact area. Disk points must have `z=0`; omit `z` in 2D.

The ring uses the builder's radius and the existing cylinder ring convention:
angles start on the positive x axis, increase towards positive y, and the
optional angular shift is in degrees. Names default to `el_point_0`,
`el_point_1`, etc., with numbering continuing across subsequent rings. Supply
`prefix="upper"` for names such as `upper_0`. No duplicate endpoint is added.

Use `ElectrodeDisk(radius=1, n_electrodes=0)` to build a disk without finite
electrode arcs before adding a ring of point electrodes. Existing disk
construction with finite electrodes remains available.

## Retrieve points after meshing

- `geometry.point_names`: all registered names in insertion order.
- `geometry.point_electrode_names`: only point electrode names.
- `geometry.points[name]`: immutable `NamedPoint` metadata.
- `geometry.point_vertex(name)`: current zero-based vertex number.
- `geometry.point_region(name)`: native NGSolve zero-dimensional region.
- `geometry.point_dofs(space, name)`: corresponding DOF indices for a space.

Lookup uses names in the mesh, not a stored node number or nearest-node
approximation. Adding points invalidates cached geometry and mesh. Resolve
vertices again after remeshing; node numbers may change. Pass `mesh=old_mesh`
to `point_vertex` or `point_region` to inspect an earlier mesh explicitly.
Point groups are metadata, not automatically generated region names.

Empty/duplicate names, coincident points, nonfinite coordinates, and invalid
ring counts are rejected. Points outside the remaining domain, including in
voids, are rejected when geometry is constructed. Mesh generation verifies
that every named point remains at its requested location and belongs to a
volume element. Invalid meshes are not returned as successful results.

## Use a named point as the numerical zero-voltage reference

```python
from eitlab import ElectricalModel, PotentialReference

model = ElectricalModel(
    mesh, geometry.electrode_names, sigma=1.0,
    numerical_ground_point="ground",
    potential_reference=PotentialReference.DO_NOTHING,
)
```

This pins the field DOF at the named point to zero and leaves the electrode
potential DOFs free. Select either `numerical_ground_point` or an explicit
`numerical_ground_el`, not both. With neither specified, the original default
electrode ground is retained. `duplicate()` preserves the chosen point.

The numerical ground fixes the additive constant. The existing
`potential_reference` option separately controls how returned electrode
voltages are referenced; use `DO_NOTHING` to retain their raw point-referenced
values. Balanced-current solutions have the same voltage differences and
field gradients under electrode and point grounding, within solver precision.

## Point electrodes versus finite CEM electrodes

`point_electrode_names` deliberately remains separate from `electrode_names`,
which continues to list finite electrode boundaries for the complete-electrode
model. This change supplies point-electrode **geometry and lookup**. It does
not implement point-current injection or point-electrode observations in
`ElectricalModel`. Passing point names as its CEM electrode list is rejected.
A point-electrode electrical formulation can use the new regions/DOFs in a
subsequent extension; it must handle point-source singularities explicitly.

## Meshing details

Netgen's OCC geometry supports named vertices on 2D faces and 3D surfaces.
An isolated vertex inside a 3D solid can remain disconnected. For an interior
3D point, the builder therefore partitions the solid horizontally through the
point and embeds the vertex on the resulting internal face. The partition is
conforming; it introduces no physical hole or finite electrode. Material names
are preserved, but a material may occupy multiple volume regions with the same
name. Conductivity mapping handles repeated material names in mesh order.
Partition surfaces are named `__point_partition__`; they are internal interfaces,
not external electrodes or insulating outer surfaces. This can increase the
mesh size, especially with points at many distinct heights.

The existing builders generate triangular/quadrilateral elements in 2D and
tetrahedra in 3D. The point API does not introduce a new element family.

Validation: exact coordinates/connectivity at two mesh sizes, mixed 2D
quadrilateral meshing, two 3D rings, internal conforming interfaces, material
preservation across partitions, rejection of void/exterior points, and
point-ground invariance in 2D/3D. Run `python -m pytest tests/test_named_points.py -q`.
Run the example with `python -m examples.named_points`.

Implementation references:

- [NGSolve maintainers: named and embedded OCC vertices](https://forum.ngsolve.org/t/defining-a-subdomain-consisting-of-a-single-vertex/2497)
- [NGSolve maintainers: point Dirichlet conditions](https://forum.ngsolve.org/t/setting-a-point-value/1906)

[Documentation index](../README.md)
