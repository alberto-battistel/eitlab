"""NGSolve backend adapters; inverse algorithms receive only numerical arrays.

Geometry/current/contact
configuration is fixed in each adapter; conductivity updates reuse its FEM
spaces and boundary metadata. Every derivative is built at a valid revision.
"""
import numpy as np

from eit_forward.parameters import ElementwiseConductivity


class NGSolveConductivityForward:
    def __init__(self, backend, mesh, electrodes, measurement_plan, *, parameter_map=None,
                 z_contact=0.02, order=1, inverse="sparsecholesky", ground=None,
                 potential_reference=None):
        self.backend = backend
        self.mesh = mesh
        self.electrodes = tuple(electrodes)
        self.plan = measurement_plan
        self.parameter_map = ElementwiseConductivity(mesh) if parameter_map is None else parameter_map
        if self.parameter_map.mesh is not mesh:
            raise ValueError("Parameter mapping must use the forward mesh")
        self.parameter_space = self.parameter_map.space
        self.n_parameters = self.parameter_map.ndof
        reference = (backend.fem.PotentialReference.MEAN_ZERO
                     if potential_reference is None else potential_reference)
        self.model = backend.fem.ElectricalModel(
            mesh, self.electrodes, sigma=self.parameter_map.to_field(np.ones(self.n_parameters)),
            numerical_ground_el=self.electrodes[0] if ground is None else ground,
            z_contact=z_contact, potential_reference=reference, order=order,
            inverse=inverse, complex_mode=False)
        self.runner = backend.measurement.MeasurementRunner(self.model, self.plan)
        self._parameters = np.ones(self.n_parameters)
        self._model_revision = self.model.revision

    def _solve(self, parameters, *, store_solutions=True):
        values = self.parameter_map.validate(parameters)
        if not np.array_equal(values, self._parameters) or self.model.revision != self._model_revision:
            self.model.set_sigma(self.parameter_map.to_field(values))
            self._parameters = values.copy()
            self._model_revision = self.model.revision
        results = self.runner.run(store_solutions=store_solutions)
        return self.model, self.runner, results

    def evaluate(self, parameters):
        """Flat observations; no FEM states retained in the returned data."""
        _, _, results = self._solve(parameters, store_solutions=False)
        return results.vector

    def linearize(self, parameters, *, method="adjoint", step=1e-4):
        """Return a numerical prediction/Jacobian snapshot with Jv and J.Tw."""
        from eit_forward.numeric import MatrixLinearization
        values = self.parameter_map.validate(parameters)
        if method == "adjoint":
            provider = NGSolveAdjointJacobian(self)
        elif method in {"forward", "backward", "central"}:
            from eitlab.differentiation import FiniteDifferenceJacobian
            provider = FiniteDifferenceJacobian(self.evaluate, step=step, scheme=method)
        else:
            raise ValueError("Unknown differentiation method")
        matrix = provider.jacobian(values)
        prediction = self.evaluate(values)
        return MatrixLinearization(values, prediction, matrix)


class NGSolveAdjointJacobian:
    """Same jacobian(parameters) interface as FiniteDifferenceJacobian."""
    def __init__(self, forward):
        self.forward = forward

    def jacobian(self, parameters):
        model, _, results = self.forward._solve(parameters)
        backend = self.forward.backend
        linearization = backend.fem.ElectricalLinearization(model, parameter_space=self.forward.parameter_space)
        return backend.fem.JacobianAssembler(linearization, results).assemble_total_jacobian()


def triangles_in_parameter_order(forward):
    """Extract affine triangles in the adapter's L2(order=0) DOF order.

    Curved meshes need curved-element intersection integration and are rejected.
    Use curve_order=1 for the supplied geometric pixel-mapping example.
    """
    mesh = forward.mesh
    if mesh.dim != 2 or mesh.GetCurveOrder() > 1:
        raise ValueError("Exact polygon overlap requires a straight-sided 2D mesh")
    triangles = np.empty((forward.n_parameters, 3, 2))
    seen = np.zeros(forward.n_parameters, dtype=bool)
    for element in mesh.Elements(forward.backend.ng.VOL):
        dofs = forward.parameter_space.GetDofNrs(element)
        if len(element.vertices) != 3 or len(dofs) != 1 or dofs[0] < 0 or seen[dofs[0]]:
            raise ValueError("Expected triangles with one unique L2 DOF per element")
        triangles[dofs[0]] = [mesh[v].point[:2] for v in element.vertices]
        seen[dofs[0]] = True
    if not seen.all():
        raise ValueError("Not every FEM parameter has a triangle")
    return triangles


