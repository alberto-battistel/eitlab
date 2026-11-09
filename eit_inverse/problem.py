"""A common mathematical problem, independent of optimization frameworks."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import sparse

from eitlab._numeric import real_vector, real_matrix


class DataSpace:
    """Whitening W: data fidelity is ||W(A x - b)||_2^2.

    Use standard_deviation for independent noise, or whitening for a general
    precomputed inverse square root of covariance. Omit both for identity.
    """
    def __init__(self, n_data, *, standard_deviation=None, whitening=None):
        if n_data < 1:
            raise ValueError("n_data must be positive")
        if standard_deviation is not None and whitening is not None:
            raise ValueError("Choose standard_deviation or whitening")
        self.n_data = n_data
        if standard_deviation is not None:
            sd = np.broadcast_to(np.asarray(standard_deviation, dtype=float), (n_data,))
            if not np.all(np.isfinite(sd)) or np.any(sd <= 0):
                raise ValueError("Noise standard deviations must be finite and positive")
            self.W = sparse.diags(1 / sd, format="csr")
        else:
            self.W = sparse.eye(n_data, format="csr") if whitening is None else real_matrix(whitening, "W")
            if self.W.shape != (n_data, n_data):
                raise ValueError("Whitening must have shape (n_data, n_data)")


class Regularizer:
    """Backend-neutral mathematical regularizer ``weight * phi(L(x-prior))``.

    ``l1`` and squared ``l2`` are supported by both bundled backends. ``huber``
    and ``group_l2`` are represented independently of any optimization package;
    CVXPY currently implements them while SciPy may add implementations later.
    """
    def __init__(self, operator, weight, norm="l2", prior=None, name="", *,
                 huber_M=1.0, group_size=None):
        self.L = real_matrix(operator, "regularization operator")
        if norm not in ("l1", "l2", "huber", "group_l2"):
            raise ValueError("norm must be 'l1', 'l2', 'huber', or 'group_l2'")
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("weight must be finite and nonnegative")
        self.norm, self.weight, self.name = norm, float(weight), name
        self.prior = np.zeros(self.L.shape[1]) if prior is None else real_vector(prior, self.L.shape[1], "prior")
        self.huber_M = float(huber_M)
        self.group_size = None if group_size is None else int(group_size)
        if norm == "huber" and (not np.isfinite(self.huber_M) or self.huber_M <= 0):
            raise ValueError("huber_M must be finite and positive")
        if norm == "group_l2":
            if self.group_size is None or self.group_size < 1 or self.L.shape[0] % self.group_size:
                raise ValueError("group_size must be positive and divide the number of operator rows")

    @property
    def n_parameters(self):
        return self.L.shape[1]

    def value(self, x):
        v = np.asarray(self.L @ (x - self.prior)).ravel()
        if self.norm == "l2":
            phi = v @ v
        elif self.norm == "l1":
            phi = np.abs(v).sum()
        elif self.norm == "huber":
            a = np.abs(v)
            phi = np.where(a <= self.huber_M, a*a,
                           2*self.huber_M*a-self.huber_M**2).sum()
        else:
            phi = np.linalg.norm(v.reshape(-1, self.group_size), axis=1).sum()
        return self.weight * float(phi)


class BackendRegularizer:
    """Base class for an explicit backend-specific regularizer extension.

    Third-party solver packages can follow this pattern with their own backend
    key and payload. Such extensions are intentionally outside the portable
    mathematical regularizer API.
    """
    backend = None

    def __init__(self, n_parameters, weight=1.0, name=""):
        if not isinstance(n_parameters, (int, np.integer)) or n_parameters < 1:
            raise ValueError("n_parameters must be positive")
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("weight must be finite and nonnegative")
        self._n_parameters = int(n_parameters)
        self.weight, self.name = float(weight), name
        # Compatibility with InverseProblem.active_regularizers.
        self.L = sparse.csr_matrix((1, self._n_parameters))

    @property
    def n_parameters(self):
        return self._n_parameters


class CVXPYExpressionRegularizer(BackendRegularizer):
    """Escape hatch for arbitrary convex CVXPY regularization expressions.

    ``expression(variable, cp)`` returns an unweighted scalar CVXPY expression.
    ``value(parameters)`` must return the matching unweighted numeric value so
    :class:`InverseProblem` can report a backend-independent objective value.
    CVXPY itself remains lazily imported by :class:`CVXPYSolver`.
    """
    backend = "cvxpy"

    def __init__(self, n_parameters, expression, value, weight=1.0, name=""):
        super().__init__(n_parameters, weight, name)
        if not callable(expression) or not callable(value):
            raise TypeError("expression and value must be callable")
        self.expression = expression
        self._value = value

    def value(self, x):
        return self.weight * float(self._value(np.asarray(x)))


class InverseProblem:
    """One fixed linear problem. x is whichever variable A's columns describe.

    For difference EIT use A=J(reference), b=y_measured-y_reference, x=delta_p.
    Bounds and priors refer to x, not implicitly to absolute conductivity.
    """
    def __init__(self, matrix, data, regularizers=(), *, data_space=None, bounds=None):
        self.matrix = real_matrix(matrix, "matrix")
        self.n_data, self.n_parameters = self.matrix.shape
        if min(self.matrix.shape) < 1:
            raise ValueError("matrix must have nonempty dimensions")
        self.data = real_vector(data, self.n_data, "data")
        self.regularizers = tuple(regularizers)
        if any(r.n_parameters != self.n_parameters for r in self.regularizers):
            raise ValueError("Every regularization operator must have n_parameters columns")
        self.data_space = DataSpace(self.n_data) if data_space is None else data_space
        if self.data_space.n_data != self.n_data:
            raise ValueError("DataSpace has the wrong dimension")
        self.A = (self.data_space.W @ self.matrix).tocsr()
        self.b = np.asarray(self.data_space.W @ self.data)
        if bounds is None:
            self.lower = np.full(self.n_parameters, -np.inf)
            self.upper = np.full(self.n_parameters, np.inf)
        else:
            self.lower, self.upper = [np.broadcast_to(np.asarray(v, dtype=float), (self.n_parameters,)).copy() for v in bounds]
            if np.any(np.isnan(self.lower)) or np.any(np.isnan(self.upper)) or np.any(self.lower >= self.upper):
                raise ValueError("Bounds require lower < upper and no NaNs")

    @property
    def active_regularizers(self):
        return tuple(r for r in self.regularizers if r.weight > 0 and r.L.shape[0] > 0)

    def objective(self, x):
        residual = self.A @ x - self.b
        return float(residual @ residual) + sum(r.value(x) for r in self.regularizers)


@dataclass
class InverseResult:
    parameters: np.ndarray
    objective: float
    weighted_residual_norm: float
    status: str
    backend: str
    iterations: int | None
    diagnostics: dict


class InverseSolver(Protocol):
    def solve(self, problem: InverseProblem) -> InverseResult: ...


def make_result(problem, x, status, backend, iterations=None, **diagnostics):
    return InverseResult(np.asarray(x).copy(), problem.objective(x),
                         float(np.linalg.norm(problem.A @ x - problem.b)),
                         status, backend, iterations, diagnostics)
