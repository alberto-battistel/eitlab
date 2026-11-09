"""Named points must be connected FE vertices, not disconnected mesh markers."""
from collections import Counter

import numpy as np
import pytest

ng = pytest.importorskip("ngsolve")
from eitlab import CylinderModel, ElectrodeDisk, ElectricalModel, CurrentPattern, PotentialReference

pytestmark = pytest.mark.fem


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("maxh", [.6, .4])
def test_ring_and_interior_point_are_exact_connected_vertices(dimension, maxh):
    g = ElectrodeDisk(1, 0) if dimension == 2 else CylinderModel()
    z = 0 if dimension == 2 else .43
    assert g.add_point(name="ground.center", x=.12, y=.17, z=z) is g
    assert g.add_point_electrode_ring(6, z=z, start_angle_deg=13, group="sensor") is g
    mesh = g.make_mesh(maxh=maxh)
    space = ng.H1(mesh, order=2)
    assert len(g.point_electrode_names) == 6
    assert not g.electrode_names  # Point features must not masquerade as CEM patches.
    for p in g.points.values():
        node = mesh.vertices[g.point_vertex(p.name)]
        np.testing.assert_allclose(node.point, p.position[:dimension], atol=1e-10)
        assert len(node.elements) > 0
        assert len(g.point_dofs(space, p.name)) == 1
    assert all(g.points[n].group == "sensor" for n in g.point_electrode_names)
    if dimension == 3:
        np.testing.assert_allclose(ng.Integrate(1, mesh), np.pi, rtol=1e-4)


def test_multiple_rings_names_regeneration_and_quad_mesh():
    g = CylinderModel().add_point_electrode_ring(4, z=.25)
    old_mesh = g.make_mesh(maxh=.6)
    g.add_point_electrode_ring(4, z=.75, start_angle_deg=20)
    assert g.mesh is None and g.geo is None
    assert g.point_electrode_names == tuple(f"el_point_{i}" for i in range(8))
    mesh = g.make_mesh(maxh=.5)
    assert mesh is not old_mesh
    assert g.point_vertex("el_point_0", mesh=old_mesh) >= 0
    g2 = ElectrodeDisk(1, 0).add_point(name="center", x=0, y=0)
    g2.make_mesh(maxh=.4, quad_dominated=True)
    assert len(g2.mesh.vertices[g2.point_vertex("center")].elements) > 0


@pytest.mark.parametrize("dimension", [2, 3])
def test_outside_and_void_points_are_rejected(dimension):
    g = ElectrodeDisk(1, 0) if dimension == 2 else CylinderModel()
    g.add_point(name="outside", x=2, y=0, z=0 if dimension == 2 else .5)
    with pytest.raises(ValueError, match="outside"):
        g.make_mesh(maxh=.6)
    assert g.mesh is None
    g = ElectrodeDisk(1, 0) if dimension == 2 else CylinderModel()
    if dimension == 2:
        g.add_disk_void(name="hole", p=(0, 0), r=.2)
    else:
        g.add_spherical_void(name="hole", center=(0, 0, .5), r=.2)
    g.add_point(name="in_void", x=0, y=0, z=0 if dimension == 2 else .5)
    with pytest.raises(ValueError, match="void"):
        g.make_mesh(maxh=.6)


def test_validation_and_failed_ring_are_atomic():
    g = CylinderModel().add_point(name="existing", x=1, y=0, z=.5)
    for kwargs in [dict(name="existing", x=0, y=0), dict(name="", x=0, y=0),
                   dict(name="bad", x=np.nan, y=0),
                   dict(name="coincident", x=1, y=0, z=.5)]:
        with pytest.raises(ValueError):
            g.add_point(**kwargs)
    for n in [0, -1, True, 2.5]:
        with pytest.raises(ValueError):
            g.add_point_electrode_ring(n)
    with pytest.raises(ValueError, match="Coincident"):
        g.add_point_electrode_ring(4, z=.5)
    assert g.point_names == ("existing",)
    with pytest.raises(ValueError, match="Generate"):
        g.point_vertex("existing")
    with pytest.raises(ValueError, match="z=0"):
        ElectrodeDisk(1, 0).add_point(name="bad", x=0, y=0, z=1)


@pytest.mark.parametrize("dimension", [2, 3])
def test_partition_preserves_material_values(dimension):
    g = ElectrodeDisk(1, 4) if dimension == 2 else CylinderModel()
    if dimension == 2:
        g.add_disk_inclusion(name="target", p=(.3, 0), r=.2)
        z = 0
    else:
        g.add_round_electrode_ring(4, z=.5, r=.15)
        g.add_spherical_inclusion(name="target", center=(.3, 0, .5), r=.2)
        z = .4
    g.add_point(name="ground", x=.1, y=.05, z=z)
    mesh = g.make_mesh(maxh=.5)
    model = ElectricalModel(mesh, g.electrode_names, sigma={"body": 1, "target": 2},
                            numerical_ground_point="ground")
    expected = ng.CoefficientFunction([1 if n == "body" else 2 for n in mesh.GetMaterials()])
    assert ng.Integrate((model.sigma_cf - expected)**2, mesh) < 1e-20
    assert set(mesh.GetMaterials()) == {"body", "target"}


@pytest.mark.parametrize("dimension", [2, 3])
def test_point_reference_preserves_voltage_differences_and_field_gradients(dimension):
    g = ElectrodeDisk(1, 4) if dimension == 2 else CylinderModel()
    if dimension == 3:
        g.add_round_electrode_ring(4, z=.5, r=.15)
    g.add_point(name="ground.center", x=.11, y=.13, z=0 if dimension == 2 else .43)
    mesh = g.make_mesh(maxh=.5)
    pattern = CurrentPattern({g.electrode_names[0]: 1, g.electrode_names[1]: -1})
    common = dict(sigma=1., potential_reference=PotentialReference.DO_NOTHING)
    old = ElectricalModel(mesh, g.electrode_names, **common)
    model = ElectricalModel(mesh, g.electrode_names, numerical_ground_point="ground.center", **common)
    u, v = old.solve(pattern), model.solve(pattern)
    assert abs(v.field(*g.points["ground.center"].position[:dimension])) < 1e-12
    np.testing.assert_allclose(np.diff(u.raw_electrode_potentials),
                               np.diff(v.raw_electrode_potentials), atol=1e-10)
    assert ng.Integrate(ng.Norm(ng.grad(u.field) - ng.grad(v.field))**2, mesh) < 1e-18
    assert model.duplicate().numerical_ground_point == "ground.center"
    with pytest.raises(ValueError, match="either"):
        ElectricalModel(mesh, g.electrode_names, sigma=1,
                        numerical_ground_el=g.electrode_names[0], numerical_ground_point="ground.center")
    with pytest.raises(ValueError, match="point"):
        ElectricalModel(mesh, g.electrode_names, sigma=1, numerical_ground_point="missing")


def test_interior_partition_is_conforming():
    g = CylinderModel().add_point(name="one", x=.12, y=.17, z=.31)
    g.add_point(name="two", x=-.2, y=.05, z=.67)
    mesh = g.make_mesh(maxh=.5)
    faces = Counter()
    for tet in mesh.Elements(ng.VOL):
        nodes = [v.nr for v in tet.vertices]
        assert len(nodes) == 4
        for i in range(4):
            faces[tuple(sorted(nodes[:i] + nodes[i+1:]))] += 1
    assert max(faces.values()) == 2
    for face in mesh.Elements(ng.BND):
        if face.mat == "__point_partition__":
            assert faces[tuple(sorted(v.nr for v in face.vertices))] == 2
