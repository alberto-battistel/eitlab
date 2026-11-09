"""NGSolve complete-electrode forward model; no reconstruction algorithms."""
from collections.abc import Mapping
from numbers import Number
from types import MappingProxyType
import re

import numpy as np
import ngsolve as ng

from .observations import PotentialReference
from .state import EITSolution, VariationalTerm
from eitlab.parameters import ConductivityParameter
from eitlab.bibliography import BibliographyMixin


def calculate_boundary_dimension(mesh, boundary_name):
    """Electrode boundary measure: length in 2D, area in 3D."""
    return ng.Integrate(1, mesh, ng.BND, definedon=mesh.Boundaries(boundary_name))


class ElectricalModel(BibliographyMixin):
    bibliography_component = "ngsolve"
    def __init__(self, mesh, electrode_names, sigma, numerical_ground_el=None,
                 z_contact=0.01, potential_reference=PotentialReference.CURRENT_ELECTRODES_MEAN,
                 order=1, inverse="sparsecholesky", complex_mode=False, *,
                 numerical_ground_point=None):
        self.mesh = mesh
        self.electrode_names = tuple(electrode_names)
        if (len(self.electrode_names) < 2 or
                len(set(self.electrode_names)) != len(self.electrode_names) or
                not set(self.electrode_names).issubset(mesh.GetBoundaries())):
            raise ValueError("Provide at least two unique electrode boundaries present in the mesh")
        self.numerical_ground_point = numerical_ground_point
        if numerical_ground_point is not None:
            if numerical_ground_el is not None:
                raise ValueError("Choose either a numerical ground electrode or point")
            names = mesh.GetBBoundaries() if mesh.dim == 2 else mesh.GetBBBoundaries()
            if numerical_ground_point not in names:
                raise ValueError("Numerical ground point must name a point in the mesh")
            vb = ng.BBND if mesh.dim == 2 else ng.BBBND
            nodes = {v.nr for el in mesh.Elements(vb) if el.mat == numerical_ground_point
                     for v in el.vertices}
            if len(nodes) != 1 or not len(mesh.vertices[next(iter(nodes))].elements):
                raise ValueError("Numerical ground point must identify one connected mesh vertex")
            self.numerical_ground_el = None
        else:
            self.numerical_ground_el = self.electrode_names[0] if numerical_ground_el is None else numerical_ground_el
            if self.numerical_ground_el not in self.electrode_names:
                raise ValueError("Numerical ground must name an electrode")
        if not isinstance(potential_reference, PotentialReference):
            raise TypeError("potential_reference must be a PotentialReference")
        if not isinstance(order, int) or order < 1:
            raise ValueError("H1 order must be a positive integer")
        self.order, self.inverse, self.complex_mode = order, inverse, bool(complex_mode)
        self.potential_reference = potential_reference
        self._identity = object()
        self.revision = 0
        self.invA = None
        self.z_contact = self._contact_values(z_contact)
        self._build_spaces()
        self.set_sigma(sigma)

    def _numeric(self, value, name):
        number = complex(value)
        if not np.isfinite(number) or number.real <= 0:
            raise ValueError(f"{name} must be finite with positive real part")
        if not self.complex_mode:
            if number.imag != 0:
                raise ValueError(f"Complex {name} requires complex_mode=True")
            return float(number.real)
        return number

    def _contact_values(self, value):
        if isinstance(value, Mapping):
            if set(value) != set(self.electrode_names):
                raise ValueError("Contact impedances must cover exactly the model electrodes")
            values = {name: self._numeric(value[name], "contact impedance")
                      for name in self.electrode_names}
        else:
            values = {name: self._numeric(value, "contact impedance") for name in self.electrode_names}
        return MappingProxyType(values)

    def _convert_sigma(self, sigma):
        if isinstance(sigma, ConductivityParameter):
            owned = sigma.to_backend_field(mesh=self.mesh, positive=True)
            return owned, owned
        if isinstance(sigma, Number):
            value = self._numeric(sigma, "conductivity")
            return value, ng.CoefficientFunction(value)
        if isinstance(sigma, Mapping):
            if set(sigma) != set(self.mesh.GetMaterials()):
                raise ValueError("Conductivity dictionary must cover exactly the mesh materials")
            values = {name: self._numeric(sigma[name], "conductivity") for name in self.mesh.GetMaterials()}
            return MappingProxyType(values), ng.CoefficientFunction(
                [values[name] for name in self.mesh.GetMaterials()])
        if isinstance(sigma, ng.GridFunction):
            if sigma.space.mesh is not self.mesh:
                raise ValueError("Conductivity GridFunction must use the model mesh")
            owned = ng.GridFunction(sigma.space)
            owned.vec.data = sigma.vec
            return owned, owned
        if isinstance(sigma, ng.CoefficientFunction):
            # An arbitrary CF may contain externally mutable Parameters. Call
            # reassemble() after changing those; their state cannot be inferred.
            return sigma, sigma
        raise TypeError("Conductivity must be a ConductivityParameter, number, material mapping, GridFunction or CoefficientFunction")

    def _build_spaces(self):
        options = {}
        if self.numerical_ground_point is not None:
            flag = "dirichlet_bbnd" if self.mesh.dim == 2 else "dirichlet_bbbnd"
            options[flag] = re.escape(self.numerical_ground_point)
        field = ng.H1(self.mesh, order=self.order, complex=self.complex_mode, **options)
        electrodes = []
        for name in self.electrode_names:
            options = dict(definedon=self.mesh.Boundaries(name), complex=self.complex_mode)
            if name == self.numerical_ground_el:
                options["dirichlet"] = name
            electrodes.append(ng.NumberSpace(self.mesh, **options))
        self.fes = ng.FESpace([field] + electrodes)
        self.trial, self.test = self.fes.TrialFunction(), self.fes.TestFunction()
        self.u, self.w = self.trial[0], self.test[0]
        self.U, self.W = tuple(self.trial[1:]), tuple(self.test[1:])
        self.electrode_lengths = MappingProxyType({
            name: calculate_boundary_dimension(self.mesh, name) for name in self.electrode_names})
        if any(value <= 0 for value in self.electrode_lengths.values()):
            raise ValueError("Electrode boundaries must have positive measure")
        self.electrode_dofs = MappingProxyType({
            name: next(iter(self.fes.Range(i + 1))) for i, name in enumerate(self.electrode_names)})

    def _assemble(self, sigma_cf):
        terms = [VariationalTerm(sigma_cf * ng.InnerProduct(ng.grad(self.u), ng.grad(self.w)), ng.dx)]
        terms.extend(VariationalTerm((self.u - self.U[i]) * (self.w - self.W[i]) / self.z_contact[name], ng.ds(name))
                     for i, name in enumerate(self.electrode_names))
        form = ng.BilinearForm(self.fes, symmetric=True)
        for term in terms:
            form += term.integrand * term.integration_measure
        form.Assemble()
        return form, tuple(terms)

    def set_sigma(self, sigma):
        """Atomically replace coefficients and assemble; never leave a stale A."""
        owned, coefficient = self._convert_sigma(sigma)
        form, terms = self._assemble(coefficient)
        self.sigma_input, self.sigma_cf = owned, coefficient
        self.a, self.a_terms = form, terms
        self.invA = None
        self.revision += 1

    def set_contact_impedances(self, values):
        old = self.z_contact
        self.z_contact = self._contact_values(values)
        try:
            self.reassemble()
        except Exception:
            self.z_contact = old
            raise

    def reassemble(self):
        """Re-evaluate externally mutable coefficients on the existing spaces."""
        form, terms = self._assemble(self.sigma_cf)
        self.a, self.a_terms = form, terms
        self.invA = None
        self.revision += 1

    @property
    def inverse_operator(self):
        if self.invA is None:
            self.invA = self.a.mat.Inverse(self.fes.FreeDofs(), inverse=self.inverse)
        return self.invA

    @property
    def operator_matrix(self):
        return self.a.mat

    @property
    def free_dofs(self):
        return self.fes.FreeDofs()

    def validate_pattern_electrodes(self, pattern):
        unknown = set(pattern.currents) - set(self.electrode_names)
        if unknown:
            raise ValueError(f"Unknown electrodes: {sorted(unknown)}")

    def full_current_dict(self, pattern):
        self.validate_pattern_electrodes(pattern)
        values = {name: pattern.current_on(name) for name in self.electrode_names}
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("Currents must be finite")
        if not self.complex_mode and any(np.iscomplex(v) for v in values.values()):
            raise ValueError("Complex currents require complex_mode=True")
        if abs(sum(values.values())) > getattr(pattern, "tolerance", 1e-12):
            raise ValueError("Net injected current must be zero")
        return values

    def make_electrode_rhs(self, weights):
        unknown = set(weights) - set(self.electrode_names)
        if unknown:
            raise ValueError(f"Unknown electrodes: {sorted(unknown)}")
        rhs = self.a.mat.CreateColVector()
        rhs[:] = 0
        for name, value in weights.items():
            rhs[self.electrode_dofs[name]] = value
        return rhs

    def solve(self, pattern, potential_reference=None):
        currents = self.full_current_dict(pattern)
        rhs = self.make_electrode_rhs(currents)
        gfu = ng.GridFunction(self.fes)
        gfu.vec.data = self.inverse_operator * rhs
        reference = self.potential_reference if potential_reference is None else potential_reference
        return EITSolution(pattern, currents, gfu, reference, self._identity, self.revision)

    def validate_solution(self, solution):
        if solution.model_id is not self._identity or solution.model_revision != self.revision:
            raise ValueError("Forward state belongs to another model or a stale model revision")
        if len(solution.gfu.vec) != self.fes.ndof:
            raise ValueError("Forward state uses a different finite-element layout")

    def duplicate(self):
        """A fresh solver configuration sharing the mesh, not a deep mesh snapshot."""
        return type(self)(self.mesh, self.electrode_names, self.sigma_input,
                          self.numerical_ground_el, dict(self.z_contact),
                          self.potential_reference, self.order, self.inverse, self.complex_mode,
                          numerical_ground_point=self.numerical_ground_point)

    def __repr__(self):
        return f"ElectricalModel(electrodes={len(self.electrode_names)}, order={self.order}, revision={self.revision})"
