"""Parameter ownership, FEM assembly and application/backend boundaries."""
import ast
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def test_contract_import_does_not_load_fem():
    subprocess.run([sys.executable, "-c", '''
import sys
from eitlab import ConductivitySpace, ConductivityParameter
assert "ngsolve" not in sys.modules
assert "eit_inverse" not in sys.modules
'''], check=True)


@pytest.mark.fem
def test_parameter_ownership_and_model_update(backend, disk, pattern):
    from eitlab import ElementwiseConductivity, ConductivityParameter
    from eit_forward.model import ElectricalModel
    mesh, names = disk
    space = ElementwiseConductivity(mesh)
    supplied = np.full(space.ndof, 1.2)
    sigma = space.create(supplied)
    assert isinstance(sigma, ConductivityParameter)
    supplied[:] = 9
    detached = sigma.values
    detached[:] = 7
    np.testing.assert_array_equal(sigma.values, 1.2)
    model = ElectricalModel(mesh, names, sigma=sigma)
    first = model.solve(pattern).electrode_potentials.copy()
    newer = sigma.with_values(2.4)
    np.testing.assert_allclose(model.solve(pattern).electrode_potentials, first)
    model.set_sigma(newer)
    assert not np.allclose(model.solve(pattern).electrode_potentials, first)
    revision = model.revision
    with pytest.raises(ValueError, match="positive"):
        model.set_sigma(space.create(-1, positive=False))
    assert model.revision == revision


@pytest.mark.fem
def test_parameter_materials_and_validation(backend):
    from eitlab import ElementwiseConductivity
    geometry = backend.geometry.ElectrodeDisk(radius=1, n_electrodes=4)
    geometry.add_disk_inclusion(name="target", p=(0, 0), r=.2)
    mesh = geometry.make_mesh(maxh=.4)
    space = ElementwiseConductivity(mesh)
    sigma = space.create({"body": 1., "target": .6})
    field = sigma.to_backend_field()
    material = {"body": 1., "target": .6}
    for element in mesh.Elements(backend.ng.VOL):
        assert sigma.values[space.space.GetDofNrs(element)[0]] == material[element.mat]
    expected = backend.ng.CoefficientFunction([material[n] for n in mesh.GetMaterials()])
    assert backend.ng.Integrate((field-expected)**2, mesh) < 1e-20
    for invalid in ([1.], np.full(space.ndof, np.nan), 0, -1):
        with pytest.raises(ValueError):
            space.create(invalid)
    with pytest.raises(TypeError, match="real"):
        space.create(1+1j)
    with pytest.raises(ValueError, match="materials"):
        space.create({"body": 1.})
    assert np.all(space.create(-1, positive=False).values == -1)


@pytest.mark.fem
def test_wrong_mesh_rejected_by_model_linearization_and_plotting(backend, disk):
    from eitlab import ElementwiseConductivity, ElectricalModel, ElectricalLinearization
    from eitlab.plotting import field_data, sample_image
    mesh, names = disk
    other = backend.geometry.ElectrodeDisk(radius=1, n_electrodes=8).make_mesh(maxh=.4)
    wrong_space = ElementwiseConductivity(other)
    sigma = wrong_space.create(1.)
    model = ElectricalModel(mesh, names, sigma=1.)
    with pytest.raises(ValueError, match="mesh"):
        model.set_sigma(sigma)
    with pytest.raises(ValueError, match="mesh"):
        ElectricalLinearization(model, parameter_space=wrong_space)
    with pytest.raises(ValueError, match="mesh"):
        field_data(mesh, sigma)
    with pytest.raises(ValueError, match="mesh"):
        sample_image(sigma, mesh)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.fem
def test_l2_integral_tv_nullspace_and_plotting(backend, dimension):
    from eitlab import ElementwiseConductivity
    from eitlab.plotting import field_data, sample_image
    from netgen.geom2d import unit_square
    from netgen.csg import unit_cube
    geo = unit_square if dimension == 2 else unit_cube
    mesh = backend.ng.Mesh(geo.GenerateMesh(maxh=.5))
    space = ElementwiseConductivity(mesh)
    mass, jumps = space.regularization_operators()
    values = np.linspace(.5, 1.5, space.ndof)
    sigma = space.create(values)
    field = sigma.to_backend_field()
    expected_energy = backend.ng.Integrate(field**2, mesh)
    assert np.linalg.norm(mass @ values)**2 == pytest.approx(expected_energy)
    np.testing.assert_allclose(jumps @ np.ones(space.ndof), 0, atol=1e-15)
    assert np.abs(jumps @ values).sum() > 0
    np.testing.assert_allclose(space.l2_operator().toarray(), mass.toarray())
    np.testing.assert_allclose(space.tv_operator().toarray(), jumps.toarray())
    expected = [values[space.space.GetDofNrs(e)[0]] for e in mesh.Elements(backend.ng.VOL)]
    np.testing.assert_allclose(field_data(mesh, sigma).values, expected)
    if dimension == 2:
        image, _ = sample_image(space.create(2.), mesh, resolution=8)
        np.testing.assert_allclose(image, 2.)


@pytest.mark.fem
def test_2d_penalties_preserve_previous_example(backend, disk):
    from eitlab import ElementwiseConductivity
    from eitlab.plotting import mesh_data
    from eit_inverse import triangle_regularization_operators
    mesh, _ = disk
    space = ElementwiseConductivity(mesh)
    data = mesh_data(mesh)
    cells = np.empty((space.ndof, 3), dtype=int)
    for element, vertices in zip(mesh.Elements(backend.ng.VOL), data.cells):
        cells[space.space.GetDofNrs(element)[0]] = vertices
    old_l2, old_tv = triangle_regularization_operators(data.points[:, :2], cells)
    new_l2, new_tv = space.regularization_operators()
    values = np.linspace(-2, 3, space.ndof)
    np.testing.assert_allclose(new_l2 @ values, old_l2 @ values, atol=1e-14)
    assert np.abs(new_tv @ values).sum() == pytest.approx(np.abs(old_tv @ values).sum())
    np.testing.assert_allclose((new_tv.T @ new_tv).toarray(),
                               (old_tv.T @ old_tv).toarray(), atol=1e-14)


@pytest.mark.fem
def test_jacobian_accepts_conductivity_space(backend, model, disk):
    from eitlab import ElementwiseConductivity, ElectricalLinearization, JacobianAssembler
    from tests.support import make_plan
    mesh, names = disk
    space = ElementwiseConductivity(mesh)
    model.set_sigma(space.create(1.))
    results = backend.measurement.MeasurementRunner(model, make_plan(backend, names)).run()
    wrapped = ElectricalLinearization(model, parameter_space=space)
    native = ElectricalLinearization(model, parameter_space=space.space)
    assert wrapped.mapping is space
    np.testing.assert_allclose(JacobianAssembler(wrapped, results).assemble_total_jacobian(),
                               JacobianAssembler(native, results).assemble_total_jacobian())


def test_teaching_example_has_no_native_fem_construction():
    path = Path(__file__).resolve().parents[1] / "examples" / "simple_eit_2d.py"
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(item.name != "ngsolve" for item in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module != "ngsolve"
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"GetDofNrs", "GridFunction", "Elements", "CoefficientFunction"}


def test_pixel_geometry_import_without_fem():
    subprocess.run([sys.executable, "-c", '''
import sys
from eit_inverse import triangle_pixel_space
assert "ngsolve" not in sys.modules
assert "eit_forward.parameters" not in sys.modules
'''], check=True)


@pytest.mark.fem
def test_fem_pixel_interface_order_and_conservation(backend, disk):
    from eitlab import ElementwiseConductivity
    from eitlab.plotting import mesh_data
    from eit_inverse import triangle_pixel_space
    mesh, _ = disk
    space = ElementwiseConductivity(mesh)
    # The adapter must produce the same map as explicitly ordered triangles.
    data = mesh_data(mesh)
    triangles = np.empty((space.ndof, 3, 2))
    for e, cell in zip(mesh.Elements(backend.ng.VOL), data.cells):
        triangles[space.space.GetDofNrs(e)[0]] = data.points[cell, :2]
    edges = np.linspace(-1, 1, 9)
    actual, expected = space.pixel_space(edges, edges), triangle_pixel_space(triangles, edges, edges)
    np.testing.assert_allclose(actual.P.toarray(), expected.P.toarray())
    np.testing.assert_allclose(actual.R.toarray(), expected.R.toarray())
    np.testing.assert_allclose(actual.R @ np.ones(space.ndof), 1)
    values = np.linspace(-1., 2., space.ndof)
    assert actual.pixel_areas @ (actual.R @ values) == pytest.approx(actual.fem_areas @ values)
    # No inverse solver is imported to assemble a FEM pixel map.
    subprocess.run([sys.executable, "-c", '''
import sys, numpy as np
from eitlab import ElectrodeDisk, ElementwiseConductivity
mesh=ElectrodeDisk(radius=1., n_electrodes=4).make_mesh(maxh=.5)
space=ElementwiseConductivity(mesh)
space.pixel_space(np.linspace(-1,1,5), np.linspace(-1,1,5))
assert "eit_inverse" not in sys.modules
'''], check=True)
