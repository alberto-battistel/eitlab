"""Homogeneous, noise-only Monte Carlo tuning of a regularization weight.

The default metric is dimensionless relative output/input noise gain.
The forward PDE is evaluated once by the caller; each repeat solves the same
linearized inverse problem with another noisy homogeneous voltage vector.
"""
from dataclasses import dataclass, replace

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import brentq, minimize_scalar

from .problem import real_matrix, real_vector
from .selection import _with_weight


@dataclass(frozen=True)
class NoiseSimulationConfig:
    repeats: int = 50
    noise_level: float = 1e-3
    noise_mode: str = "relative_rms"  # or absolute / relative_channel
    seed: int = 42
    statistic: str = "rms"           # or spatial std
    ddof: int = 0
    store_images: bool = False
    noise_floor: float = 0.0         # absolute voltage SD added in quadrature


@dataclass
class NoiseAmplificationEstimate:
    weight: float
    mean_gain: float
    gain_std: float
    gain_standard_error: float
    gains: np.ndarray
    output_noise: np.ndarray
    input_noise: np.ndarray
    clean_parameters: np.ndarray
    noise_images: np.ndarray | None


@dataclass
class NoiseTuningResult:
    weight: float
    target_gain: float
    estimate: NoiseAmplificationEstimate
    validation: NoiseAmplificationEstimate | None
    matched: bool
    validation_matched: bool | None
    method: str
    scan_weights: np.ndarray
    scan_gains: np.ndarray
    scan_standard_errors: np.ndarray
    diagnostics: dict
    messages: tuple


def _positive_scale(value, size, name):
    if np.iscomplexobj(value):
        raise TypeError(f"{name} must be real")
    a = np.broadcast_to(np.asarray(value, dtype=float), (size,)).copy()
    if not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return a


class NoiseAmplificationExperiment:
    """Fixed noise draws, homogeneous reference, and one inverse configuration.

    problem describes difference reconstruction. Its original data are ignored:
    clean data are zero; noisy data are epsilon/difference_scale. Whitening,
    priors, bounds, parameter map and all other penalties are preserved.
    Output noise subtracts the clean reconstruction at the SAME weight.

    input_scale defaults to RMS homogeneous voltage; image_scale should be
    homogeneous conductivity in the output's physical units. image_operator
    optionally maps parameters to a FEM/pixel evaluation field; image_weights
    optionally supplies its physical areas/volumes. Both metrics are normalized
    before computing one output/input gain per sample, then averaging the gains.
    """
    def __init__(self, problem, solver, homogeneous_voltages, *, config=None,
                 regularizer_index=0, input_scale=None, image_scale=1.,
                 image_operator=None, image_weights=None, difference_scale=1.,
                 noise_covariance=None, noise_samples=None):
        self.problem, self.solver = problem, solver
        self.config = NoiseSimulationConfig() if config is None else config
        cfg = self.config
        if int(cfg.repeats) != cfg.repeats or cfg.repeats < 2:
            raise ValueError("repeats must be an integer >=2")
        if not np.isfinite(cfg.noise_level) or cfg.noise_level <= 0:
            raise ValueError("noise_level must be finite and positive")
        if not np.isfinite(cfg.noise_floor) or cfg.noise_floor < 0:
            raise ValueError("noise_floor must be finite and nonnegative")
        if cfg.noise_mode not in ("relative_rms", "absolute", "relative_channel"):
            raise ValueError("Unknown noise_mode")
        if cfg.statistic not in ("rms", "std") or cfg.ddof not in (0, 1):
            raise ValueError("statistic must be rms/std and ddof 0/1")
        if int(cfg.seed) != cfg.seed or cfg.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if int(regularizer_index) != regularizer_index or not 0 <= regularizer_index < len(problem.regularizers):
            raise ValueError("regularizer_index is out of range")
        self.regularizer_index = int(regularizer_index)
        self.homogeneous_voltages = real_vector(homogeneous_voltages, problem.n_data, "homogeneous_voltages")
        self.voltage_rms = float(np.sqrt(np.mean(self.homogeneous_voltages**2)))
        self.input_scale = _positive_scale(self.voltage_rms if input_scale is None else input_scale,
                                          problem.n_data, "input_scale")
        self.difference_scale = _positive_scale(difference_scale, problem.n_data, "difference_scale")
        self.image_operator = None if image_operator is None else real_matrix(image_operator, "image_operator")
        if self.image_operator is not None and self.image_operator.shape[1] != problem.n_parameters:
            raise ValueError("image_operator columns must match reconstruction parameters")
        self.n_image = problem.n_parameters if self.image_operator is None else self.image_operator.shape[0]
        self.image_scale = _positive_scale(image_scale, self.n_image, "image_scale")
        self.image_weights = np.ones(self.n_image) if image_weights is None else real_vector(image_weights, self.n_image, "image_weights")
        if np.any(self.image_weights < 0) or self.image_weights.sum() <= 0:
            raise ValueError("image_weights must be nonnegative with a positive sum")
        if cfg.statistic == "std" and (problem.n_data < 2 or np.count_nonzero(self.image_weights) < 2):
            raise ValueError("Spatial std needs at least two data and image entries")
        if noise_covariance is not None and noise_samples is not None:
            raise ValueError("Choose noise_covariance or explicit noise_samples")
        self.noise_covariance = None
        self.explicit_samples = noise_samples is not None
        rng = np.random.default_rng(int(cfg.seed))
        if noise_samples is not None:
            if np.iscomplexobj(noise_samples):
                raise TypeError("noise_samples must be real")
            noise = np.asarray(noise_samples, dtype=float)
            if noise.ndim != 2 or noise.shape[1] != problem.n_data or len(noise) < 2 or not np.isfinite(noise).all():
                raise ValueError("noise_samples must have finite shape (n>=2, n_data)")
            noise = noise.copy()
        elif noise_covariance is not None:
            if np.iscomplexobj(noise_covariance):
                raise TypeError("noise_covariance must be real")
            covariance = np.asarray(noise_covariance, dtype=float)
            if covariance.shape != (problem.n_data, problem.n_data) or not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=0):
                raise ValueError("noise_covariance must be a finite symmetric n_data square matrix")
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            scale = np.max(np.abs(eigenvalues))
            if scale == 0 or np.min(eigenvalues) < -1e-10*scale:
                raise ValueError("noise_covariance must be nonzero positive semidefinite")
            factor = eigenvectors*np.sqrt(np.maximum(eigenvalues, 0))
            noise = rng.standard_normal((int(cfg.repeats), problem.n_data)) @ factor.T
            self.noise_covariance = covariance.copy()
        else:
            sd = cfg.noise_level
            if cfg.noise_mode == "relative_rms":
                sd *= self.voltage_rms
            elif cfg.noise_mode == "relative_channel":
                sd *= np.abs(self.homogeneous_voltages)
            sd = np.sqrt(np.asarray(sd)**2+cfg.noise_floor**2)
            noise = rng.standard_normal((int(cfg.repeats), problem.n_data))*sd
        self.noise_samples = noise
        self.noise_samples.setflags(write=False)
        self.noisy_voltages = self.homogeneous_voltages[None, :] + noise
        self.input_noise = np.array([self._measure(e/self.input_scale, np.ones(problem.n_data)) for e in noise])
        if not np.isfinite(self.input_noise).all() or np.any(self.input_noise <= 0):
            raise ValueError("Every input-noise draw must have positive measured magnitude")
        self._cache = {}

    def _measure(self, values, weights):
        weights = weights/weights.sum()
        if self.config.statistic == "rms":
            return float(np.sqrt(weights @ (values**2)))
        center = weights @ values
        variance = weights @ ((values-center)**2)
        effective_n = 1/np.sum(weights**2)
        correction = effective_n/(effective_n-self.config.ddof)
        return float(np.sqrt(max(variance*correction, 0.)))

    def evaluate(self, weight):
        """Return mean of repeated gains; cached evaluations use identical noise."""
        weight = float(weight)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("weight must be finite and nonnegative")
        if weight in self._cache:
            return self._cache[weight]
        clean_problem = _with_weight(self.problem, weight, self.regularizer_index,
                                    data=np.zeros(self.problem.n_data))
        clean = self.solver.solve(clean_problem).parameters
        output_noise, images = [], []
        for e in self.noise_samples:
            # Algebraically equal to noisy_voltages - homogeneous_voltages.
            # Use the known additive draw to avoid small-noise cancellation.
            difference = e/self.difference_scale
            noisy_problem = _with_weight(self.problem, weight, self.regularizer_index, data=difference)
            estimate = self.solver.solve(noisy_problem).parameters
            image = estimate-clean
            if self.image_operator is not None:
                image = np.asarray(self.image_operator @ image)
            normalized = image/self.image_scale
            output_noise.append(self._measure(normalized, self.image_weights))
            if self.config.store_images:
                images.append(image.copy())
        output_noise = np.asarray(output_noise)
        gains = output_noise/self.input_noise
        if not np.isfinite(gains).all():
            raise RuntimeError("Reconstruction produced nonfinite noise gains")
        std = float(np.std(gains, ddof=1))
        result = NoiseAmplificationEstimate(weight, float(gains.mean()), std, std/np.sqrt(len(gains)),
                    gains, output_noise, self.input_noise.copy(), clean.copy(),
                    np.stack(images) if images else None)
        self._cache[weight] = result
        return result

    def independent(self, repeats, seed, noise_samples=None):
        """Same experiment/model/settings; different noise draws or held-out draws."""
        return NoiseAmplificationExperiment(self.problem, self.solver, self.homogeneous_voltages,
            config=replace(self.config, repeats=repeats, seed=seed),
            regularizer_index=self.regularizer_index, input_scale=self.input_scale,
            image_scale=self.image_scale, image_operator=self.image_operator,
            image_weights=self.image_weights, difference_scale=self.difference_scale,
            noise_covariance=self.noise_covariance if noise_samples is None else None,
            noise_samples=noise_samples)


def tune_noise_amplification(experiment, target_gain, *, weights=None,
        weight_span=(1e-6, 1e2), n_weights=12, method="brent", polynomial_order=2,
        refine_polynomial=True, target_rtol=.05, log_weight_tolerance=.001,
        max_iterations=50, validation_repeats=100, validation_seed=None,
        validation_noise_samples=None):
    """Scan, then match desired Monte Carlo mean gain within the sampled span.

    brent: root search in log10(weight), first sampled target crossing.
    bounded: bounded optimization of relative gain error near the best grid point.
    polynomial: fit log10(gain) vs log10(weight), evaluate in-span real roots;
        optionally refine against actual reconstructions in a sampled bracket.
    grid: use the closest sampled mean gain only.

    Zero-gain samples are excluded from log-polynomial fits. The surrogate is
    never accepted without actual reconstruction at the proposed weight.
    Unreachable targets return matched=False, not extrapolated fitted guesses.
    Independent validation is reported separately from the tuning tolerance.
    """
    if not np.isfinite(target_gain) or target_gain <= 0 or not np.isfinite(target_rtol) or target_rtol <= 0:
        raise ValueError("target_gain and target_rtol must be finite and positive")
    if method not in ("brent", "bounded", "polynomial", "grid"):
        raise ValueError("Unknown search method")
    if log_weight_tolerance <= 0 or not np.isfinite(log_weight_tolerance) or int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("Search tolerance/iteration limit must be positive")
    if int(polynomial_order) != polynomial_order or polynomial_order < 1:
        raise ValueError("polynomial_order must be a positive integer")
    if int(validation_repeats) != validation_repeats or validation_repeats < 0 or validation_repeats == 1:
        raise ValueError("validation_repeats must be 0 or an integer >=2")
    polynomial_order, max_iterations, validation_repeats = int(polynomial_order), int(max_iterations), int(validation_repeats)
    if weights is None:
        span = np.asarray(weight_span, dtype=float)
        if span.shape != (2,) or not np.isfinite(span).all() or span[0] <= 0 or span[1] <= span[0] or int(n_weights) != n_weights or n_weights < 3:
            raise ValueError("Use a positive increasing weight_span and n_weights>=3")
        weights = np.logspace(np.log10(span[0]), np.log10(span[1]), int(n_weights))
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1 or len(weights) < 3 or not np.isfinite(weights).all() or np.any(weights <= 0) or np.any(np.diff(weights) <= 0):
            raise ValueError("weights must be at least three positive increasing values")
    weights = weights.copy()
    scan = [experiment.evaluate(w) for w in weights]
    gains = np.array([r.mean_gain for r in scan])
    standard_errors = np.array([r.gain_standard_error for r in scan])
    logs = np.log10(weights)
    errors = np.abs(gains-target_gain)/target_gain
    best = int(np.argmin(errors))
    candidates = [scan[best]]
    messages, diagnostics = [], {}
    brackets = [(j, j+1) for j in range(len(weights)-1)
                if (gains[j]-target_gain)*(gains[j+1]-target_gain) < 0]
    diagnostics["crossing_brackets"] = np.array([(weights[a], weights[b]) for a, b in brackets]).reshape(-1, 2)
    reachable = gains.min() <= target_gain <= gains.max()

    def mismatch(log_weight):
        return experiment.evaluate(10.**float(log_weight)).mean_gain-target_gain

    def refine(bracket):
        a, b = bracket
        root, info = brentq(mismatch, logs[a], logs[b], xtol=log_weight_tolerance,
                           maxiter=max_iterations, full_output=True, disp=False)
        if not info.converged:
            raise RuntimeError("Noise-gain bracketed search did not converge")
        diagnostics["root_iterations"] = info.iterations
        candidates.append(experiment.evaluate(10.**root))

    if not reachable:
        messages.append("Target is outside the sampled gain range; change the weight span or noise/model settings")
    elif method == "brent" and brackets:
        refine(brackets[0])
    elif method == "bounded":
        a, b = max(best-1, 0), min(best+1, len(weights)-1)
        result = minimize_scalar(lambda u: (mismatch(u)/target_gain)**2,
                  bounds=(logs[a], logs[b]), method="bounded",
                  options=dict(xatol=log_weight_tolerance, maxiter=max_iterations))
        if not result.success:
            raise RuntimeError("Noise-gain bounded search did not converge")
        diagnostics["optimizer_iterations"] = result.nit
        candidates.append(experiment.evaluate(10.**result.x))
    elif method == "polynomial":
        positive = gains > 0
        if np.count_nonzero(positive) < polynomial_order+1:
            raise ValueError("Too few positive gains for the requested polynomial order")
        polynomial = Polynomial.fit(logs[positive], np.log10(gains[positive]), polynomial_order)
        roots = (polynomial-np.log10(target_gain)).roots()
        real_roots = np.sort(np.array([r.real for r in roots if abs(r.imag) < 1e-8
                                      and logs[0] <= r.real <= logs[-1]]))
        diagnostics.update(polynomial_coefficients=polynomial.coef.copy(),
                           polynomial_domain=polynomial.domain.copy(),
                           polynomial_window=polynomial.window.copy(),
                           polynomial_log_rmse=float(np.sqrt(np.mean((polynomial(logs[positive])-np.log10(gains[positive]))**2))),
                           surrogate_weights=10.**real_roots)
        for root in real_roots:
            candidates.append(experiment.evaluate(10.**root))
        if not len(real_roots):
            messages.append("Polynomial has no in-span target root; retained measured candidates")
        if refine_polynomial and brackets:
            proposed = min(candidates, key=lambda r: abs(r.mean_gain-target_gain))
            nearest = min(brackets, key=lambda ab: abs((logs[ab[0]]+logs[ab[1]])/2-np.log10(proposed.weight)))
            refine(nearest)
            diagnostics["polynomial_refined"] = True
    chosen = min(candidates, key=lambda r: abs(r.mean_gain-target_gain))
    matched = abs(chosen.mean_gain-target_gain)/target_gain <= target_rtol
    if len(brackets) > 1:
        messages.append("Multiple sampled target crossings; search is not assumed to be globally monotone")
    if chosen.weight in (weights[0], weights[-1]):
        messages.append("Selected weight is at the search boundary")
    if not matched:
        messages.append("The measured tuning gain did not meet target_rtol")
    validation = None
    if validation_noise_samples is not None or validation_repeats:
        if experiment.explicit_samples and validation_noise_samples is None:
            messages.append("Explicit tuning samples require separate validation_noise_samples for independent validation")
        else:
            seed = experiment.config.seed+1 if validation_seed is None else validation_seed
            repeats = len(validation_noise_samples) if validation_noise_samples is not None else validation_repeats
            validation = experiment.independent(repeats, seed, validation_noise_samples).evaluate(chosen.weight)
    validation_matched = None if validation is None else abs(validation.mean_gain-target_gain)/target_gain <= target_rtol
    if validation_matched is False:
        messages.append("Independent validation missed target_rtol; inspect uncertainty or increase repeats")
    diagnostics.update(evaluated_weights=np.array(sorted(experiment._cache)),
                       evaluated_gains=np.array([experiment._cache[w].mean_gain for w in sorted(experiment._cache)]),
                       target_rtol=target_rtol, tuning_repeats=len(experiment.noise_samples))
    return NoiseTuningResult(chosen.weight, float(target_gain), chosen, validation,
                            bool(matched), validation_matched, method, weights, gains,
                            standard_errors, diagnostics, tuple(messages))
