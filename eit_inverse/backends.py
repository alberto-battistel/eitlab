"""SciPy common-objective solver and optional extended CVXPY convex backend."""
import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import factorized, lsmr

from .problem import make_result
from eitlab.bibliography import BibliographyMixin


class SciPySolver(BibliographyMixin):
    """L2: augmented least squares. L1: generalized-lasso ADMM using SciPy.

    L1 is nonsmooth: it is not replaced by a smooth approximation or sent to
    ordinary least_squares. ADMM handles multiple L1/L2 terms and box bounds.
    """
    bibliography_component = "scipy"

    def __init__(self, *, rho=1.0, abs_tol=1e-7, rel_tol=1e-6, max_iter=20000, adaptive_rho=True):
        if rho <= 0 or not np.isfinite(rho) or abs_tol <= 0 or rel_tol <= 0 or max_iter < 1:
            raise ValueError("rho, tolerances, and max_iter must be positive")
        self.rho, self.abs_tol, self.rel_tol, self.max_iter = rho, abs_tol, rel_tol, max_iter
        self.adaptive_rho = adaptive_rho

    def solve(self, problem):
        p = problem
        regs = p.active_regularizers
        unsupported = [r for r in regs if getattr(r, "norm", None) not in ("l1", "l2")]
        if unsupported:
            names = ", ".join(r.name or getattr(r, "norm", type(r).__name__) for r in unsupported)
            raise NotImplementedError(
                "SciPySolver currently supports portable L1/L2 regularizers only; "
                f"unsupported regularizer(s): {names}")
        matrices, targets = [p.A], [p.b]
        for r in regs:
            if r.norm == "l2":
                matrices.append(np.sqrt(r.weight) * r.L)
                targets.append(np.sqrt(r.weight) * (r.L @ r.prior))
        B = sparse.vstack(matrices, format="csr")
        d = np.concatenate(targets)
        l1 = [r for r in regs if r.norm == "l1"]
        if not l1:
            sol = lsq_linear(B, d, bounds=(p.lower, p.upper), tol=self.abs_tol,
                             lsq_solver="lsmr", lsmr_tol=min(self.abs_tol * 0.01, 1e-12),
                             max_iter=self.max_iter, lsmr_maxiter=max(1000, 5*p.n_parameters))
            if not sol.success:
                raise RuntimeError(f"SciPy least squares did not converge: {sol.message}")
            return make_result(p, sol.x, "optimal", "scipy", sol.nit, method="lsq_linear", optimality=sol.optimality)

        blocks = [r.L for r in l1]
        offsets = [r.L @ r.prior for r in l1]
        weights = [np.full(r.L.shape[0], r.weight) for r in l1]
        l1_rows = sum(r.L.shape[0] for r in l1)
        bounded = np.any(np.isfinite(p.lower)) or np.any(np.isfinite(p.upper))
        if bounded:
            blocks.append(sparse.eye(p.n_parameters, format="csr"))
            offsets.append(np.zeros(p.n_parameters))
        K = sparse.vstack(blocks, format="csr")
        c = np.concatenate(offsets)
        weights = np.concatenate(weights)
        rho = self.rho
        # If unidentifiable null spaces make H singular, solve the augmented
        # least-squares x subproblem directly; no hidden ridge is introduced.
        def factorize(current_rho):
            H = (2 * (B.T @ B) + current_rho * (K.T @ K)).tocsc()
            try:
                return factorized(H), None
            except RuntimeError:
                augmented = sparse.vstack([np.sqrt(2)*B, np.sqrt(current_rho)*K], format="csr")
                return None, augmented
        solve_x, augmented = factorize(rho)
        x = np.zeros(p.n_parameters)
        z = K @ x - c
        u = np.zeros_like(z)
        rhs_data = 2 * (B.T @ d)
        for iteration in range(1, self.max_iter + 1):
            target = c + z - u
            if solve_x is None:
                ls = lsmr(augmented, np.concatenate([np.sqrt(2)*d, np.sqrt(rho)*target]),
                          atol=1e-12, btol=1e-12, maxiter=max(1000, 5*p.n_parameters))
                if ls[1] not in (0, 1, 2, 4, 5):
                    raise RuntimeError("ADMM least-squares subproblem did not converge")
                x = ls[0]
            else:
                x = solve_x(rhs_data + rho * (K.T @ target))
            shifted = K @ x - c
            previous = z.copy()
            v = shifted + u
            z[:l1_rows] = np.sign(v[:l1_rows]) * np.maximum(np.abs(v[:l1_rows]) - weights/rho, 0)
            if bounded:
                z[l1_rows:] = np.clip(v[l1_rows:], p.lower, p.upper)
            u += shifted - z
            primal = np.linalg.norm(shifted - z)
            dual = np.linalg.norm(rho * (K.T @ (z - previous)))
            eps_primal = np.sqrt(K.shape[0])*self.abs_tol + self.rel_tol*max(np.linalg.norm(shifted), np.linalg.norm(z))
            eps_dual = np.sqrt(p.n_parameters)*self.abs_tol + self.rel_tol*np.linalg.norm(rho*(K.T @ u))
            if primal <= eps_primal and dual <= eps_dual:
                # Feasible output for finite box bounds; ADMM residuals document
                # the pre-projection iterate's convergence accuracy.
                x = np.clip(x, p.lower, p.upper)
                return make_result(p, x, "optimal", "scipy", iteration, method="admm",
                                   primal_residual=float(primal), dual_residual=float(dual), final_rho=rho)
            if self.adaptive_rho and iteration % 25 == 0:
                updated = rho*2 if primal > 10*dual else rho/2 if dual > 10*primal else rho
                updated = np.clip(updated, 1e-8, 1e8)
                if updated != rho:
                    u *= rho/updated  # keep the unscaled multiplier unchanged
                    rho = float(updated)
                    solve_x, augmented = factorize(rho)
        raise RuntimeError(f"SciPy ADMM did not converge after {self.max_iter} iterations; "
                           f"primal={primal:.3g}, dual={dual:.3g}. Adjust rho/scaling/tolerances.")


class CVXPYSolver(BibliographyMixin):
    """CVXPY is optional and imported only when solve() is invoked."""
    bibliography_component = "cvxpy"
    def __init__(self, solver="CLARABEL", **solver_options):
        self.solver = solver
        self.solver_options = solver_options

    def solve(self, problem):
        try:
            import cvxpy as cp
        except ImportError as exc:
            raise ImportError("Install eitlab[cvxpy] for the CVXPY backend") from exc
        if self.solver not in cp.installed_solvers():
            raise RuntimeError(f"CVXPY solver {self.solver!r} is not installed; available: {cp.installed_solvers()}")
        p = problem
        x = cp.Variable(p.n_parameters)
        objective = cp.sum_squares(p.A @ x - p.b)
        for r in p.active_regularizers:
            # Backend expressions own their parameterization; L and prior
            # belong to the portable mathematical regularizer interface.
            if getattr(r, "backend", None) == "cvxpy":
                objective += r.weight * r.expression(x, cp)
                continue
            elif getattr(r, "backend", None) is not None:
                raise NotImplementedError(
                    f"Regularizer {r.name or type(r).__name__!r} targets backend {r.backend!r}")
            v = r.L @ (x-r.prior)
            if r.norm == "l2":
                penalty = cp.sum_squares(v)
            elif r.norm == "l1":
                penalty = cp.norm1(v)
            elif r.norm == "huber":
                penalty = cp.sum(cp.huber(v, r.huber_M))
            elif r.norm == "group_l2":
                grouped = cp.reshape(v, (r.L.shape[0] // r.group_size, r.group_size), order="C")
                penalty = cp.sum(cp.norm(grouped, 2, axis=1))
            else:
                raise NotImplementedError(f"Unsupported portable penalty {r.norm!r}")
            objective += r.weight * penalty
        constraints = []
        lo, hi = np.isfinite(p.lower), np.isfinite(p.upper)
        if lo.any():
            constraints.append(x[lo] >= p.lower[lo])
        if hi.any():
            constraints.append(x[hi] <= p.upper[hi])
        model = cp.Problem(cp.Minimize(objective), constraints)
        model.solve(solver=self.solver, **self.solver_options)
        if model.status != cp.OPTIMAL or x.value is None:
            raise RuntimeError(f"CVXPY did not reach optimal status: {model.status}")
        return make_result(p, np.asarray(x.value), model.status, "cvxpy",
                           model.solver_stats.num_iters, solver=self.solver)
