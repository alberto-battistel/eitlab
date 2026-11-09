"""Load the installed NGSolve backend without source-path manipulation."""
from importlib import import_module
from types import SimpleNamespace


def load_backend():
    """Return the modules used by electrical numerical adapters.

    NGSolve and Netgen are imported only when this function is called.
    """
    return SimpleNamespace(
        ng=import_module("ngsolve"),
        fem=import_module("eit_forward"),
        geometry=import_module("eit_forward.geometries"),
        measurement=import_module("eit_forward.measurement"),
        stimulation=import_module("eit_forward.stimulation"),
    )
