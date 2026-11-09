"""Monte Carlo gain and target-weight searches with independent analytic checks."""
import numpy as np
import pytest

from eit_inverse import (InverseProblem, Regularizer, SciPySolver, CVXPYSolver,
    NoiseSimulationConfig, NoiseAmplificationExperiment, tune_noise_amplification)


def identity_experiment(solver=None, statistic="rms", **kwargs):
    p = InverseProblem(np.eye(5), np.zeros(5), [Regularizer(np.eye(5), 1.)])
    cfg = NoiseSimulationConfig(repeats=6, noise_level=.03, statistic=statistic, store_images=True)
    return NoiseAmplificationExperiment(p, solver or SciPySolver(), np.ones(5), config=cfg, **kwargs)


@pytest.mark.parametrize("backend", ["scipy", "cvxpy"])
@pytest.mark.parametrize("statistic", ["rms", "std"])
def test_identity_ridge_gain_is_analytic_for_every_sample(backend, statistic):
    solver = SciPySolver()
    if backend == "cvxpy":
        pytest.importorskip("cvxpy")
        solver = CVXPYSolver()
    experiment = identity_experiment(solver, statistic)
    result = experiment.evaluate(3.)
    np.testing.assert_allclose(result.gains, .25, atol=1e-7)
    np.testing.assert_allclose(result.noise_images, experiment.noise_samples/4, atol=1e-8)
    assert experiment.evaluate(3.) is result
    np.testing.assert_allclose(experiment.noisy_voltages, 1+experiment.noise_samples)


@pytest.mark.parametrize("method", ["brent", "bounded", "polynomial"])
def test_search_matches_analytic_target_weight_and_independent_validation(method):
    experiment = identity_experiment()
    result = tune_noise_amplification(experiment, .25, weight_span=(.01, 100),
        n_weights=7, method=method, log_weight_tolerance=1e-6, target_rtol=1e-4,
        validation_repeats=8)
    np.testing.assert_allclose(result.weight, 3., rtol=1e-4)
    np.testing.assert_allclose(result.estimate.mean_gain, .25, rtol=1e-4)
    assert result.matched and result.validation_matched
    assert len(result.validation.gains) == 8


def test_clean_reconstruction_is_subtracted_with_nonzero_prior():
    p = InverseProblem(np.eye(4), np.full(4, 99.),
        [Regularizer(np.eye(4), 1., prior=np.ones(4))], bounds=(.1, 2.))
    e = NoiseAmplificationExperiment(p, SciPySolver(), np.ones(4),
        config=NoiseSimulationConfig(repeats=4, noise_level=.01))
    r = e.evaluate(1.)
    np.testing.assert_allclose(r.clean_parameters, .5)
    np.testing.assert_allclose(r.gains, .5, atol=1e-7)
    # Original data=99 is intentionally ignored for homogeneous-only tuning.
    assert np.all(p.data == 99.)


def test_image_mapping_physical_area_weights_and_difference_scale():
    A = np.diag([1., 2., 3.])
    noise = np.array([[.1, -.2, .3], [-.3, .1, .2], [.1, .1, -.2]])
    P = np.array([[1., .5, 0.], [0., .5, 1.]])
    p = InverseProblem(A, np.zeros(3), [Regularizer(np.eye(3), 1.)])
    e = NoiseAmplificationExperiment(p, SciPySolver(), np.ones(3), noise_samples=noise,
        image_operator=P, image_weights=[1., 3.], image_scale=2., difference_scale=2.)
    r = e.evaluate(.5)
    x = (noise/2)*np.array([1., 2., 3.])/(np.array([1., 4., 9.])+.5)
    images = x @ P.T / 2
    expected = np.sqrt((images[:, 0]**2+3*images[:, 1]**2)/4)/np.sqrt(np.mean(noise**2, axis=1))
    np.testing.assert_allclose(r.gains, expected)


def test_fixed_seed_correlated_draws_and_standard_error():
    p = InverseProblem(np.diag([1., 2., 3.]), np.zeros(3), [Regularizer(np.eye(3), 1.)])
    covariance = np.array([[2., .4, 0.], [.4, 1., 0.], [0., 0., .5]])*.001
    cfg = NoiseSimulationConfig(repeats=25, seed=12)
    first = NoiseAmplificationExperiment(p, SciPySolver(), np.ones(3), config=cfg, noise_covariance=covariance)
    second = NoiseAmplificationExperiment(p, SciPySolver(), np.ones(3), config=cfg, noise_covariance=covariance)
    np.testing.assert_array_equal(first.noise_samples, second.noise_samples)
    r = first.evaluate(.4)
    assert r.gain_std > 0
    np.testing.assert_allclose(r.gain_standard_error, np.std(r.gains, ddof=1)/5)
    independent = first.independent(25, 13)
    assert not np.array_equal(first.noise_samples, independent.noise_samples)


def test_l1_noise_gain_depends_on_noise_level_and_suppresses_noise():
    p = InverseProblem(np.eye(4), np.zeros(4), [Regularizer(np.eye(4), .1, "l1")])
    solver = SciPySolver(abs_tol=1e-10, rel_tol=1e-9)
    low = NoiseAmplificationExperiment(p, solver, np.ones(4),
        config=NoiseSimulationConfig(repeats=4, noise_level=.01))
    high = NoiseAmplificationExperiment(p, solver, np.ones(4),
        config=NoiseSimulationConfig(repeats=4, noise_level=.2))
    assert low.evaluate(.1).mean_gain < .01
    assert high.evaluate(.1).mean_gain > .1
    assert high.evaluate(1.).mean_gain < high.evaluate(.01).mean_gain
    result = tune_noise_amplification(high, .25, weight_span=(.001, 1.),
                                     n_weights=7, validation_repeats=0)
    assert result.matched


def test_channel_relative_noise_floor_preserves_near_zero_channels():
    p = InverseProblem(np.eye(3), np.zeros(3), [Regularizer(np.eye(3), 1.)])
    cfg = NoiseSimulationConfig(repeats=6, noise_level=.01, noise_mode="relative_channel", noise_floor=.02)
    e = NoiseAmplificationExperiment(p, SciPySolver(), [0., 1., 2.], config=cfg)
    sd = np.sqrt(np.array([0., .01, .02])**2+.02**2)
    expected = np.random.default_rng(cfg.seed).standard_normal((6, 3))*sd
    np.testing.assert_allclose(e.noise_samples, expected)
    assert np.any(e.noise_samples[:, 0] != 0)


def test_unreachable_gain_is_reported_without_extrapolation():
    e = identity_experiment()
    r = tune_noise_amplification(e, 2., weight_span=(.1, 10.), n_weights=5,
                                method="polynomial", validation_repeats=0)
    assert not r.matched
    assert r.weight == .1
    assert any("outside" in m for m in r.messages)
    assert r.validation is None


def test_unrefined_polynomial_is_checked_with_actual_reconstruction():
    e = identity_experiment()
    r = tune_noise_amplification(e, .25, weight_span=(.01, 100.), n_weights=7,
        method="polynomial", polynomial_order=1, refine_polynomial=False,
        target_rtol=1e-5, validation_repeats=0)
    np.testing.assert_allclose(r.estimate.mean_gain, 1/(1+r.weight))
    assert not r.matched  # polynomial prediction alone is not accepted
    assert len(r.diagnostics["polynomial_coefficients"]) == 2


def test_explicit_noise_requires_separate_validation_samples():
    noise = np.array([[.1, -.2, .3, -.4, .5], [-.1, .3, -.2, .5, -.4]])
    e = identity_experiment(noise_samples=noise)
    r = tune_noise_amplification(e, .25, weights=[.1, 3., 10.], validation_repeats=4)
    assert r.validation is None
    assert any("separate" in m for m in r.messages)
    r = tune_noise_amplification(e, .25, weights=[.1, 3., 10.],
                                validation_noise_samples=noise[::-1]*2)
    assert r.validation_matched


def test_invalid_settings_and_zero_input_noise_are_rejected():
    p = InverseProblem(np.eye(3), np.zeros(3), [Regularizer(np.eye(3), 1.)])
    with pytest.raises(ValueError, match="repeats"):
        NoiseAmplificationExperiment(p, SciPySolver(), np.ones(3), config=NoiseSimulationConfig(repeats=1))
    with pytest.raises(ValueError, match="positive"):
        NoiseAmplificationExperiment(p, SciPySolver(), np.ones(3), noise_samples=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="semidefinite"):
        NoiseAmplificationExperiment(p, SciPySolver(), np.ones(3), noise_covariance=-np.eye(3))
    e = identity_experiment()
    with pytest.raises(ValueError):
        tune_noise_amplification(e, 0.)


@pytest.mark.fem
def test_real_homogeneous_eit_voltages_noise_and_weight_matching(backend, disk):
    from eit_forward.adapters import NGSolveConductivityForward, NGSolveAdjointJacobian
    from tests.support import make_plan
    mesh, electrodes = disk
    forward = NGSolveConductivityForward(backend, mesh, electrodes, make_plan(backend, electrodes, exclude_current=True))
    sigma0 = np.ones(forward.n_parameters)
    voltages = forward.evaluate(sigma0)
    J = NGSolveAdjointJacobian(forward).jacobian(sigma0)
    problem = InverseProblem(J, np.zeros(len(voltages)), [Regularizer(np.eye(len(sigma0)), 1.)])
    e = NoiseAmplificationExperiment(problem, SciPySolver(), voltages,
                                    config=NoiseSimulationConfig(repeats=4, noise_level=.001))
    target = e.evaluate(.1).mean_gain
    result = tune_noise_amplification(e, target, weights=[.01, .1, 1.], validation_repeats=0)
    assert result.matched
    np.testing.assert_allclose(result.weight, .1)
    assert np.isfinite(result.estimate.gains).all()
