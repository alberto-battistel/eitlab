"""NGSolve conductivity derivatives and protocol Jacobian assembly.

This layer knows about FEM states and operators. Reconstruction algorithms
receive only its numerical matrices and do not call adjoint solves directly.
"""
import numpy as np
import ngsolve as ng

from .parameters import ElementwiseConductivity


class ElectricalLinearization:
    """Real P0 conductivity linearization fixed at one model revision."""
    def __init__(self, model, parameter_space=None, adjoint_inverse=None):
        if model.complex_mode:
            raise NotImplementedError("Complex adjoints need a declared Hermitian/real-parameter convention")
        self.model = model
        self.revision = model.revision
        self.mapping = (parameter_space if isinstance(parameter_space, ElementwiseConductivity)
                        else ElementwiseConductivity(model.mesh, space=parameter_space))
        if self.mapping.mesh is not model.mesh:
            raise ValueError("Parameter space must use the model mesh")
        self.parameter_space = self.mapping.space
        self.adjoint_inverse = adjoint_inverse
        self._dparameter = ng.GridFunction(self.parameter_space)
        self.A_prime_sym = self._build_derivative_form()
        self._invAT = None

    def validate(self):
        if self.model.revision != self.revision:
            raise ValueError("Linearization is stale: create one at the current model revision")

    @property
    def invAT(self):
        self.validate()
        if self._invAT is None:
            if self.adjoint_inverse is None or self.adjoint_inverse == self.model.inverse:
                # Real symmetric CEM on the same free DOFs: reuse A's factor.
                self._invAT = self.model.inverse_operator
            else:
                self._invAT = self.model.a.mat.CreateTranspose().Inverse(
                    self.model.free_dofs, inverse=self.adjoint_inverse)
        return self._invAT

    def _build_derivative_form(self):
        derivative = ng.BilinearForm(self.model.fes)
        for term in self.model.a_terms:
            expression = term.integrand.Diff(self.model.sigma_cf, self._dparameter)
            if str(expression) != "ZeroCoefficientFunction":
                derivative += expression * term.integration_measure
        if not derivative.integrators:
            raise ValueError("All conductivity derivatives are zero")
        return derivative

    def _local_derivative_matrix(self, element_id):
        self.validate()
        element = self.model.fes.GetFE(element_id)
        transform = self.model.mesh.GetTrafo(element_id)
        result = None
        for integrator in self.A_prime_sym.integrators:
            matrix = integrator.CalcElementMatrix(element, transform)
            result = matrix if result is None else result + matrix
        if result is None:
            raise RuntimeError("No conductivity derivative integrators")
        return result

    def solve_adjoint(self, rhs_vectors):
        self.validate()
        rhs = list(rhs_vectors)
        if not rhs:
            raise ValueError("At least one adjoint right-hand side is required")
        rhs_mv = ng.MultiVector(rhs[0], len(rhs))
        for i, vector in enumerate(rhs):
            rhs_mv[i].data = vector
        states = ng.MultiVector(rhs[0], len(rhs))
        states[:] = self.invAT * rhs_mv
        return states

    def derivative_form(self, direction):
        """Assemble A'[direction]; usable by tangent backend extensions."""
        self.validate()
        values = self.mapping.validate(direction, positive=False)
        self._dparameter.vec.FV().NumPy()[:] = values
        self.A_prime_sym.Assemble()
        return self.A_prime_sym

    def assemble_parameter_jacobian(self, solution, adjoint_states):
        self.validate()
        self.model.validate_solution(solution)
        adjoints = list(adjoint_states)
        result = np.zeros((len(adjoints), self.parameter_space.ndof))
        state_vector = solution.gfu.vec
        self._dparameter.vec[:] = 0
        previous = None
        try:
            for element in self.model.mesh.Elements(ng.VOL):
                element_id = ng.ElementId(ng.VOL, element.nr)
                dof = self.parameter_space.GetDofNrs(element_id)[0]
                if previous is not None:
                    self._dparameter.vec[previous] = 0
                self._dparameter.vec[dof] = 1
                previous = dof
                local_matrix = self._local_derivative_matrix(element_id)
                state_dofs = self.model.fes.GetDofNrs(element_id)
                x = ng.Vector(len(state_dofs))
                adjoint_local = ng.Vector(len(state_dofs))
                for local, global_dof in enumerate(state_dofs):
                    x[local] = state_vector[global_dof] if global_dof >= 0 else 0
                action = local_matrix * x
                for j, adjoint in enumerate(adjoints):
                    for local, global_dof in enumerate(state_dofs):
                        adjoint_local[local] = adjoint[global_dof] if global_dof >= 0 else 0
                    result[j, dof] += -ng.InnerProduct(adjoint_local, action)
        finally:
            self._dparameter.vec[:] = 0
        return result


class JacobianAssembler:
    """Compose stored observation maps with valid forward/adjoint states."""
    def __init__(self, linearization, results):
        self.linearization = linearization
        self.model = linearization.model
        self.results = results
        if not self.results.solutions:
            raise ValueError("Jacobian assembly requires run(store_solutions=True)")
        if not (len(self.results.solutions) == len(self.results.patterns) == len(self.results.observations)):
            raise ValueError("Missing forward states or observation definitions")
        self.current_protocol = self.results.patterns

    def adjoint_rhs(self, current_pattern, solution):
        self.model.validate_solution(solution)
        for pattern, state, observation in zip(self.results.patterns, self.results.solutions,
                                                self.results.observations, strict=True):
            if state is solution:
                if pattern != current_pattern:
                    raise ValueError("Current pattern does not match the forward state")
                return observation.adjoint_rhs(self.model, current_pattern)
        raise ValueError("Forward state is not part of these measurement results")

    def assemble_partial_jacobian(self, current_pattern, solution):
        states = self.linearization.solve_adjoint(self.adjoint_rhs(current_pattern, solution))
        return self.linearization.assemble_parameter_jacobian(solution, states)

    def assemble_total_jacobian(self):
        self.linearization.validate()
        if self.results.model_id is not self.model._identity or self.results.model_revision != self.model.revision:
            raise ValueError("Measurements belong to another model or stale revision")
        return np.vstack([self.assemble_partial_jacobian(pattern, solution)
                          for pattern, solution in zip(self.results.patterns, self.results.solutions, strict=True)])
