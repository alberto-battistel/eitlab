"""Numerical derivative snapshots: no NGSolve or inverse-solver imports."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatrixLinearization:
    reference_parameters: np.ndarray
    prediction: np.ndarray
    matrix: np.ndarray

    def __post_init__(self):
        for name in ("reference_parameters", "prediction", "matrix"):
            value = np.array(getattr(self, name), copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError("Linearization arrays must be finite")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if (self.reference_parameters.ndim != 1 or self.prediction.ndim != 1 or
                self.matrix.shape != (self.prediction.size, self.reference_parameters.size)):
            raise ValueError("Linearization array dimensions differ")

    def jacobian(self):
        return self.matrix.copy()

    def jvp(self, direction):
        value = np.asarray(direction)
        if value.shape != self.reference_parameters.shape:
            raise ValueError("Parameter direction has the wrong shape")
        return self.matrix @ value

    def vjp(self, weights):
        value = np.asarray(weights)
        if value.shape != self.prediction.shape:
            raise ValueError("Measurement weights have the wrong shape")
        return self.matrix.conj().T @ value
