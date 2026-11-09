"""Finite-difference measurement Jacobian for any numerical forward function.

No NGSolve or symbolic differentiation is required. Parameters are real-valued;
measurements may be real or complex. Each evaluation must compute its result
from the complete supplied parameter vector, with consistent measurement order.
"""
import numpy as np


class FiniteDifferenceJacobian:
    """Provide jacobian(parameters), with shape (n_measurements, n_parameters).

    forward: callable taking a real 1-D parameter vector and returning a 1-D
        measurement vector. Supply forward.evaluate for an object-based model.
    step: positive absolute perturbation, either scalar or one value per parameter.
        Units are those of the parameters. There is no hidden relative scaling.
    scheme: "central" (default), "forward", or "backward".

    Calls per Jacobian: 2*n+1 for central, n+1 for forward/backward. A baseline
    evaluation determines output shape and checks consistency. It is not reused
    between calls to jacobian(), so nonlinear iterates get a fresh baseline.
    """
    def __init__(self, forward, step=1e-4, scheme="central"):
        if not callable(forward):
            raise TypeError("forward must be a callable parameter-to-measurement function")
        if scheme not in {"central", "forward", "backward"}:
            raise ValueError("scheme must be central, forward, or backward")
        self.forward = forward
        self.step = np.array(step, dtype=float, copy=True)
        if self.step.ndim > 1 or not np.all(np.isfinite(self.step)) or np.any(self.step <= 0):
            raise ValueError("step must contain finite positive absolute perturbations")
        self.scheme = scheme

    def _evaluate(self, parameters, expected_shape=None):
        # Copy both sides: callbacks may modify inputs or return a shared buffer.
        measurement = np.array(self.forward(parameters.copy()), copy=True)
        if measurement.ndim != 1 or measurement.size == 0:
            raise ValueError("forward must return a nonempty 1-D measurement vector")
        if not np.issubdtype(measurement.dtype, np.number) or not np.all(np.isfinite(measurement)):
            raise ValueError("forward returned nonnumeric or nonfinite measurements")
        if expected_shape is not None and measurement.shape != expected_shape:
            raise ValueError("forward measurement shape changed during differentiation")
        # Convert integer outputs before subtraction to avoid integer overflow.
        return measurement.astype(np.result_type(measurement.dtype, float), copy=False)

    def jacobian(self, parameters):
        raw = np.asarray(parameters)
        if np.iscomplexobj(raw):
            raise TypeError("parameters must be real; split complex parameters into real/imaginary parts")
        p = np.array(raw, dtype=float, copy=True)
        if p.ndim != 1 or p.size == 0 or not np.all(np.isfinite(p)):
            raise ValueError("parameters must be a nonempty finite 1-D vector")
        if self.step.ndim == 0:
            steps = np.full(p.size, float(self.step))
        elif self.step.shape == p.shape:
            steps = self.step
        else:
            raise ValueError("vector step must have one entry per parameter")

        baseline = self._evaluate(p)
        columns = []
        for index, h in enumerate(steps):
            plus, minus = p.copy(), p.copy()
            plus[index] += h
            minus[index] -= h
            if (self.scheme != "backward" and plus[index] == p[index]) or (
                self.scheme != "forward" and minus[index] == p[index]
            ):
                raise ValueError(f"step for parameter {index} is too small to change its value")

            if self.scheme == "central":
                column = (self._evaluate(plus, baseline.shape) - self._evaluate(minus, baseline.shape)) / (2 * h)
            elif self.scheme == "forward":
                column = (self._evaluate(plus, baseline.shape) - baseline) / h
            else:
                column = (baseline - self._evaluate(minus, baseline.shape)) / h
            columns.append(column)
        return np.column_stack(columns)

