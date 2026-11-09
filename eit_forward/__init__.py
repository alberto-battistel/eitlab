"""Forward modeling, parameterization and observations for electrical tomography.

NGSolve-dependent modules are imported explicitly. Importing the stimulation or
observation modules does not load NGSolve or the independent inverse package.
"""
from importlib import import_module

_EXPORTS = {
    "ElectricalModel": (".model", "ElectricalModel"),
    "calculate_boundary_dimension": (".model", "calculate_boundary_dimension"),
    "EITSolution": (".state", "EITSolution"),
    "VariationalTerm": (".state", "VariationalTerm"),
    "PotentialReference": (".observations", "PotentialReference"),
    "ElectrodeObservation": (".observations", "ElectrodeObservation"),
    "MeasurementProtocol": (".observations", "MeasurementProtocol"),
    "SingleEnded": (".observations", "SingleEnded"),
    "SingleEndedNoCurrent": (".observations", "SingleEndedNoCurrent"),
    "AdjacentDifference": (".observations", "AdjacentDifference"),
    "ElectricalLinearization": (".linearization", "ElectricalLinearization"),
    "JacobianAssembler": (".linearization", "JacobianAssembler"),
    "ElementwiseConductivity": (".parameters", "ElementwiseConductivity"),
    "MeasurementPlan": (".measurement", "MeasurementPlan"),
    "MeasurementRunner": (".measurement", "MeasurementRunner"),
    "MeasurementResults": (".measurement", "MeasurementResults"),
    "CurrentPattern": (".stimulation", "CurrentPattern"),
    "CurrentProtocol": (".stimulation", "CurrentProtocol"),
    "ProtocolFactory": (".stimulation", "ProtocolFactory"),
    "ElectrodeDisk": (".geometries", "ElectrodeDisk"),
    "CylinderModel": (".geometries", "CylinderModel"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value
