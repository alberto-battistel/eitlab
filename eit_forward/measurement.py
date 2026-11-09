"""Protocol execution and numerical measurement records."""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable

import numpy as np

from .observations import (MeasurementProtocol, SingleEnded, SingleEndedNoCurrent,
                           AdjacentDifference, ElectrodeObservation)
from .stimulation import CurrentPattern


@dataclass(frozen=True)
class MeasurementPlan:
    current_protocol: object
    measurement_protocol: MeasurementProtocol | Callable = SingleEnded

    def __post_init__(self):
        patterns = tuple(self.current_protocol)
        if not patterns or any(not isinstance(p, CurrentPattern) for p in patterns):
            raise ValueError("A plan needs at least one CurrentPattern")
        object.__setattr__(self, "current_protocol", patterns)

    def bind(self, electrode_names):
        protocol = self.measurement_protocol
        if not isinstance(protocol, MeasurementProtocol):
            protocol = protocol(electrode_names)
        if tuple(protocol.electrode_names) != tuple(electrode_names):
            raise ValueError("Plan and model electrode orders differ")
        return protocol


@dataclass(frozen=True)
class MeasurementResults:
    voltage_blocks: tuple
    patterns: tuple
    observations: tuple
    model_id: object
    model_revision: int
    metadata: object
    solutions: tuple = ()

    def __post_init__(self):
        blocks = []
        for value in self.voltage_blocks:
            block = np.array(value, copy=True)
            block.flags.writeable = False
            blocks.append(block)
        object.__setattr__(self, "voltage_blocks", tuple(blocks))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if len(blocks) != len(self.patterns) or len(blocks) != len(self.observations):
            raise ValueError("Measurement block, pattern and observation counts differ")
        if self.solutions and len(self.solutions) != len(blocks):
            raise ValueError("There must be one stored state per injection")

    @property
    def vector(self):
        return np.concatenate(self.voltage_blocks)

    @property
    def offsets(self):
        return np.cumsum([0] + [len(block) for block in self.voltage_blocks])

    @property
    def labels(self):
        return tuple((i, row) for i, block in enumerate(self.voltage_blocks)
                     for row in range(len(block)))

    @property
    def voltages(self):
        """Stack voltage blocks for rectangular protocols; use vector otherwise."""
        if len({len(block) for block in self.voltage_blocks}) > 1:
            raise ValueError("Variable channel counts: use results.vector and results.offsets")
        return np.stack(self.voltage_blocks)


class MeasurementRunner:
    def __init__(self, model, measurement_plan):
        self.model = model
        self.plan = measurement_plan
        self.current_protocol = measurement_plan.current_protocol
        self.measurement_protocol = measurement_plan.bind(model.electrode_names)

    def run(self, *, store_solutions=True):
        """Retain states only when derivatives or field inspection need them."""
        blocks, states, observations = [], [], []
        for pattern in self.current_protocol:
            solution = self.model.solve(pattern)
            observation = ElectrodeObservation(self.measurement_protocol, solution.reference).bind(pattern)
            blocks.append(observation.evaluate(solution))
            observations.append(observation)
            if store_solutions:
                states.append(solution)
        return MeasurementResults(tuple(blocks), tuple(self.current_protocol), tuple(observations),
                                  self.model._identity, self.model.revision,
                                  {"revision": self.model.revision,
                                   "electrode_names": tuple(self.model.electrode_names),
                                   "reference": self.model.potential_reference.value,
                                   "order": self.model.order,
                                   "contact_impedances": tuple(self.model.z_contact.items())},
                                  tuple(states))
