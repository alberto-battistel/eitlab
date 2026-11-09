"""Backend-independent conductivity values and their discretization contract.

Neither this module nor inverse solvers import NGSolve. A concrete FEM adapter
implements the contract and owns the conversion to its native field objects.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ConductivitySpace(Protocol):
    """All matrices and vectors use the same conductivity coefficient order.

    l2_operator returns L with ||L x||_2^2 equal to the discretized L2 integral.
    tv_operator returns D for the P0 penalty ||D x||_1 (interior jumps only).
    Other bases may need a grouped gradient norm, rather than this P0 TV rule.
    """
    mesh: object
    ndof: int

    def create(self, values, *, positive=True): ...
    def validate(self, values, *, positive=True): ...
    def to_field(self, values, *, positive=True): ...
    def l2_operator(self): ...
    def tv_operator(self): ...


@dataclass(frozen=True, eq=False, init=False)
class ConductivityParameter:
    """Owned conductivity coefficients; create through a ConductivitySpace.

    values returns a copy. Changing it cannot silently change an assembled
    model. Create a replacement and call model.set_sigma(new_parameter).
    Signed image estimates may be created with positive=False; forward models
    still validate physical conductivity before accepting such an estimate.
    """
    parameter_space: ConductivitySpace
    _values: np.ndarray

    def __init__(self, parameter_space, values, *, positive=True):
        coefficients = parameter_space.validate(values, positive=positive)
        coefficients.setflags(write=False)
        object.__setattr__(self, "parameter_space", parameter_space)
        object.__setattr__(self, "_values", coefficients)

    @property
    def values(self):
        return self._values.copy()

    @property
    def mesh(self):
        return self.parameter_space.mesh

    def with_values(self, values, *, positive=True):
        return self.parameter_space.create(values, positive=positive)

    def to_backend_field(self, *, mesh=None, positive=False):
        """Adapter/plotting bridge; application scripts need not call this."""
        if mesh is not None and mesh is not self.mesh:
            raise ValueError("Conductivity parameter belongs to another mesh")
        return self.parameter_space.to_field(self._values, positive=positive)
