"""Numerical electrode observation definitions; independent of FEM assembly."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np


class PotentialReference(Enum):
    DO_NOTHING = "do-nothing"
    MEAN_ZERO = "mean-zero"
    FIRST_ELECTRODE = "first-electrode"
    CURRENT_ELECTRODES_MEAN = "current-electrodes-mean"

    def matrix(self, U, currents=None, current_tolerance=1e-12):
        values = np.asarray(U)
        if values.ndim != 1 or not values.size:
            raise ValueError("Expected a nonempty electrode vector")
        if not np.isfinite(current_tolerance) or current_tolerance < 0:
            raise ValueError("current_tolerance must be finite and nonnegative")
        n = values.size
        R = np.eye(n)
        if self is PotentialReference.DO_NOTHING:
            return R
        if self is PotentialReference.MEAN_ZERO:
            return R - np.ones((n, n)) / n
        if self is PotentialReference.FIRST_ELECTRODE:
            R[:, 0] -= 1
            return R
        if currents is None or len(currents) != n:
            raise ValueError("Provide one current per electrode in the voltage vector's order")
        indices = [i for i, value in enumerate(currents.values())
                   if abs(value) > current_tolerance]
        if len(indices) != 2:
            raise ValueError("CURRENT_ELECTRODES_MEAN requires exactly two active electrodes")
        weights = np.zeros(n)
        weights[indices] = 0.5
        return R - np.outer(np.ones(n), weights)

    def apply(self, U, *, currents=None, current_tolerance=1e-12):
        return self.matrix(U, currents, current_tolerance) @ np.asarray(U)


class MeasurementProtocol(ABC):
    def __init__(self, electrode_names):
        self.electrode_names = tuple(electrode_names)
        if not self.electrode_names or len(set(self.electrode_names)) != len(self.electrode_names):
            raise ValueError("Electrode names must be nonempty and unique")

    @property
    def n_electrodes(self):
        return len(self.electrode_names)

    @property
    @abstractmethod
    def protocol_name(self): ...

    @abstractmethod
    def matrix(self, current_pattern): ...

    def __repr__(self):
        return f"MeasurementProtocol: {self.protocol_name}"


class SingleEnded(MeasurementProtocol):
    @property
    def protocol_name(self):
        return "single-ended"

    def matrix(self, current_pattern):
        return np.eye(self.n_electrodes)


class SingleEndedNoCurrent(MeasurementProtocol):
    def __init__(self, electrode_names, current_tolerance=1e-12):
        super().__init__(electrode_names)
        if not np.isfinite(current_tolerance) or current_tolerance < 0:
            raise ValueError("current_tolerance must be finite and nonnegative")
        self.current_tolerance = current_tolerance

    @property
    def protocol_name(self):
        return "single-ended no current electrodes"

    def matrix(self, current_pattern):
        keep = [i for i, name in enumerate(self.electrode_names)
                if abs(current_pattern.current_on(name)) <= self.current_tolerance]
        return np.eye(self.n_electrodes)[keep]


class AdjacentDifference(MeasurementProtocol):
    """U[k]-U[k+1], including the wrap-around pair; fixed electrode order."""
    @property
    def protocol_name(self):
        return "adjacent-difference"

    def matrix(self, current_pattern):
        identity = np.eye(self.n_electrodes)
        return identity - np.roll(identity, -1, axis=0)


@dataclass(frozen=True)
class ElectrodeObservation:
    """One authoritative extraction/reference/selection map and its transpose.

    The model owns extraction from state DOFs. This object defines the numerical
    electrode transform and asks the model to lift its rows into state vectors.
    """
    protocol: MeasurementProtocol
    reference: PotentialReference
    current_tolerance: float = 1e-12

    @property
    def electrode_names(self):
        return self.protocol.electrode_names

    def matrix(self, pattern):
        names = self.electrode_names
        currents = {name: pattern.current_on(name) for name in names}
        R = self.reference.matrix(np.zeros(len(names)), currents, self.current_tolerance)
        M = np.asarray(self.protocol.matrix(pattern))
        if M.ndim != 2 or M.shape[1] != len(names) or not np.all(np.isfinite(M)):
            raise ValueError("Invalid observation matrix")
        return M @ R

    def evaluate(self, solution):
        if tuple(solution.currents) != self.electrode_names:
            raise ValueError("Solution and observation electrode orders differ")
        return self.matrix(solution.pattern) @ solution.raw_electrode_potentials

    def adjoint_rhs(self, model, pattern):
        if tuple(model.electrode_names) != self.electrode_names:
            raise ValueError("Model and observation electrode orders differ")
        return [model.make_electrode_rhs(dict(zip(self.electrode_names, row)))
                for row in self.matrix(pattern)]

    def bind(self, pattern):
        return BoundElectrodeObservation(self.electrode_names,
                                         tuple(pattern.current_on(name) for name in self.electrode_names),
                                         self.matrix(pattern))


@dataclass(frozen=True)
class BoundElectrodeObservation:
    """Immutable per-injection observation snapshot used by data and derivatives."""
    electrode_names: tuple
    currents: tuple
    _transform: np.ndarray

    def __post_init__(self):
        transform = np.array(self._transform, copy=True)
        transform.flags.writeable = False
        object.__setattr__(self, "_transform", transform)

    def matrix(self, pattern=None):
        if pattern is not None and tuple(pattern.current_on(name) for name in self.electrode_names) != self.currents:
            raise ValueError("Observation was bound to a different current pattern")
        return self._transform.copy()

    def evaluate(self, solution):
        if tuple(solution.currents) != self.electrode_names:
            raise ValueError("Solution and observation electrode orders differ")
        return self.matrix(solution.pattern) @ solution.raw_electrode_potentials

    def adjoint_rhs(self, model, pattern):
        if tuple(model.electrode_names) != self.electrode_names:
            raise ValueError("Model and observation electrode orders differ")
        return [model.make_electrode_rhs(dict(zip(self.electrode_names, row)))
                for row in self.matrix(pattern)]
