"""Small real-valued zero-order Tikhonov solver; no FEM imports."""
import numpy as np


class TikhonovSolver:
    """Minimize ||J x - y||² + lambda ||x||² with a reusable SVD.

    One lambda returns (n_parameters,); multiple lambdas return one column each.
    Lambda=0 gives the numerical minimum-norm least-squares solution.
    """
    def __init__(self, lambdas, matrix):
        if np.iscomplexobj(matrix) or np.iscomplexobj(lambdas):
            raise TypeError("This solver uses real-valued matrices and penalties")
        values = np.asarray(matrix, dtype=float)
        self.lambdas = np.atleast_1d(np.asarray(lambdas, dtype=float))
        if values.ndim != 2 or not all(values.shape) or not np.all(np.isfinite(values)):
            raise ValueError("matrix must be a nonempty finite 2-D array")
        if (self.lambdas.ndim != 1 or not self.lambdas.size or
                not np.all(np.isfinite(self.lambdas)) or np.any(self.lambdas < 0)):
            raise ValueError("lambdas must be finite and nonnegative")
        self.n_parameters = values.shape[1]
        self.n_measurements = values.shape[0]
        self.U, self.s, self.Vt = np.linalg.svd(values, full_matrices=False)
        self._cutoff = np.finfo(float).eps * max(values.shape) * self.s[0]

    def solve(self, data):
        if np.iscomplexobj(data):
            raise TypeError("This solver uses real-valued data")
        y = np.asarray(data, dtype=float)
        if y.shape != (self.n_measurements,) or not np.all(np.isfinite(y)):
            raise ValueError("data must be a finite vector matching the matrix rows")
        projected = self.U.T @ y
        solutions = []
        for penalty in self.lambdas:
            denominator = self.s**2 + penalty
            valid = denominator > 0 if penalty > 0 else self.s > self._cutoff
            weights = np.divide(self.s, denominator, out=np.zeros_like(self.s), where=valid)
            solutions.append(self.Vt.T @ (weights * projected))
        return solutions[0] if len(solutions) == 1 else np.column_stack(solutions)
