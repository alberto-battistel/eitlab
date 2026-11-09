"""Real NGSolve integration checks; run with pytest in an NGSolve environment."""
import sys
import numpy as np
import pytest
import eitlab.plotting as p
ng = pytest.importorskip("ngsolve")
from netgen.geom2d import unit_square
from netgen.csg import unit_cube


@pytest.fixture(params=[2, 3])
def mesh(request):
    geo = unit_square if request.param == 2 else unit_cube
    return ng.Mesh(geo.GenerateMesh(maxh=.5))


def test_geometry_and_real_sampling(mesh):
    d = p.field_data(mesh, ng.x)
    centers = np.array([d.points[c].mean(axis=0) for c in d.cells])
    np.testing.assert_allclose(d.values, centers[:,0], atol=1e-12)
    np.testing.assert_array_equal(d.element_ids, [e.nr for e in mesh.Elements(ng.VOL)])


def test_discontinuous_dof_mapping(mesh):
    space = ng.L2(mesh, order=0)
    values = np.arange(space.ndof, dtype=float)
    gf = p.numpy_to_gridfunction(mesh, values, space=space)
    d = p.field_data(mesh, gf)
    expected = [values[space.GetDofNrs(e)[0]] for e in mesh.Elements(ng.VOL)]
    np.testing.assert_array_equal(d.values, expected)


def test_complex_and_vector_sampling(mesh):
    with pytest.raises(ValueError, match="Complex"):
        p.field_data(mesh, ng.x + 2j*ng.y)
    np.testing.assert_allclose(p.field_data(mesh, ng.x+2j*ng.y, component="imag").values,
                               2*p.field_data(mesh, ng.y).values)
    vec = ng.CoefficientFunction((ng.x, ng.y))
    np.testing.assert_allclose(p.field_data(mesh, vec, component=1).values,
                               p.field_data(mesh, ng.y).values)
    with pytest.raises(ValueError, match="Vector"):
        p.field_data(mesh, vec)


def test_mesh_field_wireframe_and_boundaries(mesh, tmp_path):
    r = p.plot_mesh(mesh, backend="matplotlib", color="red", show_axes=False)
    assert r.backend == "matplotlib"
    np.testing.assert_allclose(r.artist.get_facecolor()[0][:3], [1,0,0])
    r.save(tmp_path / "mesh.svg")
    r.close()
    r = p.plot_field(mesh, ng.x, backend="matplotlib", clim=(0,1))
    assert len(r.data.values) == mesh.ne
    r.save(tmp_path / "field.png")
    r.close()
    wire = p.plot_mesh(mesh, backend="matplotlib", color="none")
    fig, ax = wire.figure, wire.ax
    if mesh.dim == 3:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        assert isinstance(ax.collections[-1], Line3DCollection)
    fig.savefig(tmp_path / "wire.png")
    import matplotlib.pyplot as plt
    plt.close(fig)
    name = mesh.GetBoundaries()[0]
    r = p.plot_boundaries(mesh, {name:"red"}, backend="matplotlib", legend=True)
    assert set(r.data.labels) == {name}
    r.close()


def test_image_extent_axis_and_outside():
    mesh = ng.Mesh(unit_square.GenerateMesh(maxh=.5))
    image, extent = p.sample_image(ng.x, mesh, extent=(-.5,1.5,-.5,1.5), resolution=8)
    assert np.isnan(image[0]).all()
    np.testing.assert_allclose(image[2,2:6], [.125,.375,.625,.875])
    assert extent == (-.5,1.5,-.5,1.5)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    image, bounds = p.sample_image(ng.x, mesh, resolution=8)
    artist = p.plot_image(image, ax=ax, extent=bounds)
    assert artist.axes is ax
    np.testing.assert_allclose(artist.get_extent(), [0,1,0,1])
    plt.close(fig)
    def broken(point):
        raise RuntimeError("field evaluation failed")
    with pytest.raises(RuntimeError, match="field evaluation failed"):
        p.sample_image(broken, mesh)


def test_validation(mesh):
    with pytest.raises(ValueError):
        p.numpy_to_gridfunction(mesh, np.zeros(1))
    with pytest.raises(ValueError):
        p.plot_boundaries(mesh, {"missing":"red"})
    with pytest.raises(ValueError):
        p.plot_mesh(mesh, backend="invalid")
    with pytest.raises(ValueError):
        p.field_data(mesh, np.zeros(1))
    with pytest.raises(ValueError):
        p.plot_mesh(mesh, backend="matplotlib", plotter=object())


@pytest.fixture
def volume_mesh():
    return ng.Mesh(unit_cube.GenerateMesh(maxh=.5))


@pytest.mark.parametrize("plane", ["xy", "xz", "yz"])
def test_slice_coordinates_and_orientation(volume_mesh, plane):
    image, extent = p.sample_slice(
        ng.x + 10*ng.y + 100*ng.z, volume_mesh,
        plane=plane, position=.3, resolution=4,
    )
    u, v = np.meshgrid([.125, .375, .625, .875], [.125, .375, .625, .875])
    expected = {"xy": u + 10*v + 30,
                "xz": u + 3 + 100*v,
                "yz": .3 + 10*u + 100*v}[plane]
    assert extent == (0, 1, 0, 1)
    np.testing.assert_allclose(image, expected, atol=1e-12)


def test_slice_samples_original_gridfunction(volume_mesh, tmp_path):
    # Quadratic variation within elements must survive pixel sampling.
    field = ng.GridFunction(ng.H1(volume_mesh, order=2))
    field.Set(ng.x**2 + 3*ng.y + 7*ng.z)
    image, extent = p.sample_slice(field, volume_mesh, plane="xz", position=.4,
                                   resolution=8)
    u, v = np.meshgrid(np.arange(8)/8 + 1/16, np.arange(8)/8 + 1/16)
    np.testing.assert_allclose(image, u**2 + 1.2 + 7*v, atol=1e-11)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    artist = p.plot_image(image, ax=ax, extent=extent, interpolation="nearest")
    assert tuple(artist.get_extent()) == extent
    np.testing.assert_allclose(artist.get_array(), image)
    fig.savefig(tmp_path / "slice.png")
    plt.close(fig)


def test_slice_masks_only_outside_points(volume_mesh):
    image, extent = p.sample_slice(
        ng.x, volume_mesh, position=.5, extent=(-.5, 1.5, -.5, 1.5), resolution=8,
    )
    expected_mask = np.ones((8, 8), dtype=bool)
    expected_mask[2:6, 2:6] = False
    np.testing.assert_array_equal(np.isnan(image), expected_mask)
    np.testing.assert_allclose(image[2:6, 2:6], np.tile([.125, .375, .625, .875], (4, 1)))
    assert extent == (-.5, 1.5, -.5, 1.5)
    missing, _ = p.sample_slice(ng.x, volume_mesh, position=2, resolution=4)
    assert np.isnan(missing).all()

    def broken(point):
        raise ValueError("field evaluation failed")

    with pytest.raises(ValueError, match="field evaluation failed"):
        p.sample_slice(broken, volume_mesh, position=.5, resolution=4)


def test_slice_components_and_conductivity_parameter(volume_mesh):
    from eit_forward.parameters import ElementwiseConductivity
    space = ElementwiseConductivity(volume_mesh)
    parameter = space.create(np.full(space.ndof, 1.7))
    image, _ = p.sample_slice(parameter, volume_mesh, position=.5, resolution=4)
    np.testing.assert_allclose(image, 1.7)
    other_mesh = ng.Mesh(unit_cube.GenerateMesh(maxh=.5))
    with pytest.raises(ValueError, match="another mesh"):
        p.sample_slice(parameter, other_mesh, resolution=4)

    x, y = np.meshgrid([.125, .375, .625, .875], [.125, .375, .625, .875])
    field = ng.x + 2j*ng.y
    for component, expected in [
        ("real", x), ("imag", 2*y),
        ("magnitude", np.abs(x + 2j*y)), ("phase", np.angle(x + 2j*y)),
    ]:
        image, _ = p.sample_slice(field, volume_mesh, position=.5,
                                   resolution=4, component=component)
        np.testing.assert_allclose(image, expected)
    with pytest.raises(ValueError, match="Complex"):
        p.sample_slice(field, volume_mesh, position=.5, resolution=4)

    vector = ng.CoefficientFunction((ng.x, ng.y, ng.z))
    image, _ = p.sample_slice(vector, volume_mesh, position=.5, resolution=4, component=2)
    np.testing.assert_allclose(image, .5)
    image, _ = p.sample_slice(vector, volume_mesh, position=.5,
                               resolution=4, component="magnitude")
    np.testing.assert_allclose(image, np.sqrt(x*x + y*y + .25))


def test_slice_curved_domain():
    from eitlab import CylinderModel
    mesh = CylinderModel(radius=1, height=1).make_mesh(maxh=.5, curve_order=3)
    # Between polygon vertices, but inside the cubic curved cylinder.
    point = (.998*np.cos(.1), .998*np.sin(.1))
    image, _ = p.sample_slice(
        ng.x + ng.y + ng.z, mesh, position=.5, resolution=2,
        extent=(point[0] - .0001, point[0] + .0001,
                point[1] - .0001, point[1] + .0001),
    )
    assert np.isfinite(image).all()
    # Outside the circular cross section remains masked even inside its box.
    image, _ = p.sample_slice(ng.z, mesh, position=.5, resolution=16,
                               extent=(-1.1, 1.1, -1.1, 1.1))
    assert np.isnan(image[0, 0])
    assert np.isfinite(image[8, 8])


@pytest.mark.parametrize("options", [
    {"plane": "zy"}, {"plane": None},
    {"position": np.nan}, {"position": np.inf},
    {"position": [0]}, {"position": 1j}, {"position": "bad"},
    {"resolution": 0}, {"resolution": 1.5},
    {"extent": (0, 0, 0, 1)}, {"extent": (0, 1, 2, 1)},
    {"extent": (0, 1, 0, np.inf)}, {"extent": (0, 1)},
])
def test_slice_rejects_invalid_arguments(volume_mesh, options):
    with pytest.raises(ValueError):
        p.sample_slice(ng.x, volume_mesh, **options)


def test_image_and_slice_mesh_dimensions(volume_mesh):
    mesh_2d = ng.Mesh(unit_square.GenerateMesh(maxh=.5))
    with pytest.raises(ValueError, match="3D"):
        p.sample_slice(ng.x, mesh_2d)
    with pytest.raises(ValueError, match="2D"):
        p.sample_image(ng.x, volume_mesh)
