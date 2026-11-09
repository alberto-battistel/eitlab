"""Optional interface bridge; neither optimization backend imports NGSolve."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .problem import real_matrix, real_vector


class ForwardOperator(Protocol):
    def evaluate(self, parameters): ...
    def linearize(self, parameters): ...


@dataclass
class MatrixLinearization:
    reference_parameters: np.ndarray
    prediction: np.ndarray
    matrix: object

    def jacobian(self):
        return self.matrix.copy()

    def jvp(self, direction):
        return self.matrix @ real_vector(direction, self.matrix.shape[1], "direction")

    def vjp(self, weights):
        return self.matrix.T @ real_vector(weights, self.matrix.shape[0], "weights")


class MappedForward:
    """FEM function F composed with offset+P p, using either Jacobian provider.

    forward has evaluate(fem_parameters); provider has jacobian(fem_parameters).
    An identity ParameterSpace is sufficient when reconstruction stays on FEM.
    """
    def __init__(self, forward, jacobian_provider, parameter_space):
        self.forward = forward
        self.provider = jacobian_provider
        self.parameter_space = parameter_space

    def evaluate(self, parameters):
        y = np.asarray(self.forward.evaluate(self.parameter_space.to_fem(parameters)))
        if y.ndim != 1 or not np.all(np.isfinite(y)) or np.iscomplexobj(y):
            raise ValueError("This inverse bridge expects a finite real measurement vector")
        return y.copy()

    def linearize(self, parameters):
        p = real_vector(parameters, self.parameter_space.ndof, "parameters")
        prediction = self.evaluate(p)
        J = self.parameter_space.project_jacobian(
            self.provider.jacobian(self.parameter_space.to_fem(p)))
        if J.shape[0] != len(prediction):
            raise ValueError("Prediction and Jacobian measurement layouts differ")
        return MatrixLinearization(p.copy(), prediction, real_matrix(J, "Jacobian"))
