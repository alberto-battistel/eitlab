"""Backend state and immutable evaluation metadata; not inverse data objects."""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .observations import PotentialReference


@dataclass(frozen=True)
class EITSolution:
    pattern: Any
    currents: Any
    gfu: Any
    reference: PotentialReference = PotentialReference.DO_NOTHING
    model_id: Any = None
    model_revision: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "currents", MappingProxyType(dict(self.currents)))

    @property
    def raw_electrode_potentials(self):
        return np.array([component.vec[0] for component in self.gfu.components[1:]])

    @property
    def matrix(self):
        return self.reference.matrix(self.raw_electrode_potentials, self.currents)

    @property
    def electrode_potentials(self):
        return self.matrix @ self.raw_electrode_potentials

    @property
    def field(self):
        return self.gfu.components[0]

    def __str__(self):
        return (f"EITSolution(pattern={self.pattern.name or 'unnamed'}, "
                f"electrodes={len(self.currents)}, revision={self.model_revision}, "
                f"reference={self.reference.value})")


@dataclass(frozen=True)
class VariationalTerm:
    integrand: Any
    integration_measure: Any
