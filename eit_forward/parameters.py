"""NGSolve conductivity discretization, field factory and spatial operators."""
from collections.abc import Mapping
from itertools import combinations
from math import factorial

import numpy as np
import ngsolve as ng
from scipy import sparse

from eitlab.parameters import ConductivityParameter


class ElementwiseConductivity:
    """One real conductivity per L2(order=0) DOF, in NGSolve DOF order.

    Reconstruction bases/pixels are composed outside this mapping using the
    backend-independent ParameterSpace already provided by eit_inverse.
    """
    def __init__(self, mesh, space=None):
        self.mesh = mesh
        self.space = ng.L2(mesh, order=0) if space is None else space
        if self.space.mesh is not mesh:
            raise ValueError("Parameter space must use the model mesh")
        seen = set()
        for element in mesh.Elements(ng.VOL):
            dofs = self.space.GetDofNrs(element)
            if len(dofs) != 1 or dofs[0] < 0 or dofs[0] in seen:
                raise NotImplementedError("Expected one unique parameter DOF per volume element")
            seen.add(dofs[0])
        if seen != set(range(self.space.ndof)):
            raise ValueError("Parameter DOFs must cover the complete volume")

    @property
    def ndof(self):
        return self.space.ndof

    @property
    def coordinates(self):
        """Representative interior points in conductivity DOF order.

        Supports mathematical target construction without exposing native FEM
        elements or GridFunctions. These are mapped order-zero quadrature
        points, not exact geometric centroids of curved cells.
        """
        points = np.empty((self.ndof, self.mesh.dim))
        for element in self.mesh.Elements(ng.VOL):
            rule = ng.IntegrationRule(element.type, 0)
            ip = next(iter(rule))
            trafo = self.mesh.GetTrafo(element)
            points[self.space.GetDofNrs(element)[0]] = trafo(ip).point[:self.mesh.dim]
        return points

    def create(self, values, *, positive=True):
        """Create conductivity from a scalar, DOF vector or material dictionary.

        No GridFunction construction is needed in application code. Dictionary
        keys must cover exactly the mesh materials. Values are real and finite;
        positive=False is reserved for signed perturbations/image estimates.
        """
        if isinstance(values, Mapping):
            if set(values) != set(self.mesh.GetMaterials()):
                raise ValueError("Conductivity dictionary must cover exactly the mesh materials")
            coefficients = np.empty(self.ndof)
            for element in self.mesh.Elements(ng.VOL):
                value = values[element.mat]
                if np.iscomplexobj(value):
                    raise TypeError("This conductivity map uses real parameters")
                coefficients[self.space.GetDofNrs(element)[0]] = value
        elif np.ndim(values) == 0:
            coefficients = np.full(self.ndof, values)
        else:
            coefficients = values
        return ConductivityParameter(self, coefficients, positive=positive)

    def pixel_space(self, x_edges, y_edges, *, offset=None):
        """Area-overlap element/pixel mapping, assembled on the FEM side.

        R maps element coefficients to pixel area averages; P maps pixels to
        element area averages. They preserve constants and integrals but are
        not inverse/transpose matrices. Supports triangular 2D P0 only, with
        vertex-linearized geometry; use affine meshes for exact overlap.
        """
        from eitlab.discretization import triangle_pixel_space
        if self.mesh.dim != 2:
            raise NotImplementedError("Pixel overlap currently supports triangular 2D conductivity")
        points = {vertex.nr: np.asarray(vertex.point[:2]) for vertex in self.mesh.vertices}
        triangles = np.empty((self.ndof, 3, 2))
        for element in self.mesh.Elements(ng.VOL):
            if len(element.vertices) != 3:
                raise NotImplementedError("Pixel overlap currently requires triangular cells")
            triangles[self.space.GetDofNrs(element)[0]] = [points[v.nr] for v in element.vertices]
        return triangle_pixel_space(triangles, x_edges, y_edges, offset=offset)

    def regularization_operators(self):
        """Return L2(0) and P0 TV matrices in this space's DOF order.

        Supports 2D triangles and 3D tetrahedra. L2 uses cell area/volume;
        TV uses interior edge length/face area, without a domain-boundary
        penalty. Geometry is linearized at vertices, just as in the previous
        example. Curved cell/facet integration is not claimed here.
        No inverse or plotting module is needed to assemble these operators.
        """
        dim = self.mesh.dim
        if dim not in (2, 3):
            raise NotImplementedError("P0 spatial operators require triangles or tetrahedra")
        points = {vertex.nr: np.asarray(vertex.point[:dim])
                  for vertex in self.mesh.vertices}
        measures = np.empty(self.ndof)
        owners = {}
        for element in self.mesh.Elements(ng.VOL):
            vertices = tuple(vertex.nr for vertex in element.vertices)
            if len(vertices) != dim + 1:
                raise NotImplementedError("P0 spatial operators require triangles or tetrahedra")
            dof = self.space.GetDofNrs(element)[0]
            coordinates = np.array([points[i] for i in vertices])
            measures[dof] = abs(np.linalg.det(coordinates[1:] - coordinates[0])) / factorial(dim)
            for facet in combinations(vertices, dim):
                owners.setdefault(tuple(sorted(facet)), []).append(dof)
        if not np.all(np.isfinite(measures)) or np.any(measures <= 0):
            raise ValueError("Cells must have positive finite area/volume")
        rows, columns, values = [], [], []
        n_facets = 0
        for facet, adjacent in owners.items():
            if len(adjacent) > 2:
                raise ValueError("Nonmanifold facet shared by more than two cells")
            if len(adjacent) != 2:
                continue
            coordinates = np.array([points[i] for i in facet])
            if dim == 2:
                measure = np.linalg.norm(coordinates[1] - coordinates[0])
            else:
                measure = np.linalg.norm(np.cross(coordinates[1] - coordinates[0],
                                                  coordinates[2] - coordinates[0])) / 2
            rows.extend((n_facets, n_facets))
            columns.extend(adjacent)
            values.extend((measure, -measure))
            n_facets += 1
        mass = sparse.diags(np.sqrt(measures), format="csr")
        jumps = sparse.csr_matrix((values, (rows, columns)), shape=(n_facets, self.ndof))
        return mass, jumps

    def l2_operator(self):
        """L satisfying ||L*x||_2^2 = sum(cell_measure*x_cell**2)."""
        return self.regularization_operators()[0]

    def tv_operator(self):
        """D satisfying ||D*x||_1 = sum(interface_measure*abs(jump))."""
        return self.regularization_operators()[1]

    def validate(self, parameters, *, positive=True):
        if np.iscomplexobj(parameters):
            raise TypeError("This conductivity map uses real parameters")
        values = np.asarray(parameters, dtype=float)
        if values.shape != (self.ndof,) or not np.all(np.isfinite(values)):
            raise ValueError("Expected one finite value per conductivity DOF")
        if positive and np.any(values <= 0):
            raise ValueError("Conductivity parameters must be positive")
        return values.copy()

    def to_field(self, parameters, *, positive=True):
        values = self.validate(parameters, positive=positive)
        field = ng.GridFunction(self.space)
        field.vec.FV().NumPy()[:] = values
        return field

    def from_coefficient(self, coefficient):
        field = ng.GridFunction(self.space)
        field.Set(coefficient)
        return field.vec.FV().NumPy().copy()
