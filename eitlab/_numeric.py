"""Shared real-valued numerical validation; no FEM or inverse imports."""
import numpy as np
from scipy import sparse


def real_vector(value, size, name):
    if np.iscomplexobj(value):
        raise TypeError(f"{name} must be real; stack real/imaginary data explicitly")
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return a.copy()


def real_matrix(value, name):
    if np.iscomplexobj(value):
        raise TypeError(f"{name} must be real")
    a = sparse.csr_matrix(value, dtype=float).copy()
    if not np.all(np.isfinite(a.data)):
        raise ValueError(f"{name} must be finite")
    return a

