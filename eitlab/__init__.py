"""Public lazy imports for EITLab; inverse-only use does not import NGSolve."""
from importlib import import_module

__version__ = "0.3.0"

_EXPORTS = {
    "Reference": ("eitlab.bibliography", "Reference"),
    "BibliographyMixin": ("eitlab.bibliography", "BibliographyMixin"),
    "get_reference": ("eitlab.bibliography", "get_reference"),
    "references": ("eitlab.bibliography", "references"),
    "main_article": ("eitlab.bibliography", "main_article"),
    "ElectricalModel": ("eit_forward.model", "ElectricalModel"),
    "ElectricalLinearization": ("eit_forward.linearization", "ElectricalLinearization"),
    "JacobianAssembler": ("eit_forward.linearization", "JacobianAssembler"),
    "ElementwiseConductivity": ("eit_forward.parameters", "ElementwiseConductivity"),
    "ConductivityParameter": ("eitlab.parameters", "ConductivityParameter"),
    "ConductivitySpace": ("eitlab.parameters", "ConductivitySpace"),
    "PotentialReference": ("eit_forward.observations", "PotentialReference"),
    "ElectrodeObservation": ("eit_forward.observations", "ElectrodeObservation"),
    "SingleEnded": ("eit_forward.observations", "SingleEnded"),
    "SingleEndedNoCurrent": ("eit_forward.observations", "SingleEndedNoCurrent"),
    "AdjacentDifference": ("eit_forward.observations", "AdjacentDifference"),
    "MeasurementPlan": ("eit_forward.measurement", "MeasurementPlan"),
    "MeasurementRunner": ("eit_forward.measurement", "MeasurementRunner"),
    "CurrentPattern": ("eit_forward.stimulation", "CurrentPattern"),
    "CurrentProtocol": ("eit_forward.stimulation", "CurrentProtocol"),
    "ProtocolFactory": ("eit_forward.stimulation", "ProtocolFactory"),
    "ElectrodeDisk": ("eit_forward.geometries", "ElectrodeDisk"),
    "CylinderModel": ("eit_forward.geometries", "CylinderModel"),
    "NGSolveConductivityForward": ("eit_forward.adapters", "NGSolveConductivityForward"),
    "NGSolveAdjointJacobian": ("eit_forward.adapters", "NGSolveAdjointJacobian"),
    "FiniteDifferenceJacobian": ("eitlab.differentiation", "FiniteDifferenceJacobian"),
    "TikhonovSolver": ("eit_inverse.tikhonov", "TikhonovSolver"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
