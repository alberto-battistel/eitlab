# Tune regularization to a desired noise amplification

This implements the homogeneous workflow:

1. Simulate noise-free homogeneous voltages once, and build the reference Jacobian.
2. Generate several independent additive voltage-noise samples.
3. For each regularization weight, reconstruct every noisy difference frame.
4. Measure each reconstructed noise image and its input noise.
5. Average the per-frame amplification factors.
6. Search for the weight matching the requested mean amplification.
7. Check the chosen weight with independent noise samples.

The forward model stays outside the inverse module. The real EIT example
performs the homogeneous FEM simulation; the generic tuner consumes its voltage
array, Jacobian/problem, reconstruction backend and evaluation map.
It supports the existing SciPy/CVXPY solvers, L1/L2 penalties, priors and bounds.

## Explicit noise-gain definition

With homogeneous conductivity sigma0 and homogeneous voltage RMS V0_RMS,
for repeat i define:

$$
s_{V,i}=\operatorname{RMS}\left(\frac{\epsilon_i}{V_{0,\mathrm{RMS}}}\right),
\qquad
s_{\sigma,i}=
\sqrt{\frac{\sum_p a_p(\delta\sigma_{i,p}/\sigma_0)^2}{\sum_p a_p}},
\qquad G_i=\frac{s_{\sigma,i}}{s_{V,i}}.
$$

The reported gain is the arithmetic mean of G_i over repeats. Pixel areas or
FEM volumes may supply a_p; otherwise entries are equally weighted.
Output errors are measured relative to the noise-free reconstruction at the
**same weight**. This avoids treating prior/bound-induced clean-image bias as
random noise. That clean reconstruction is also returned for inspection.

Default gain is **dimensionless relative noise amplification**: the ratio
of normalized reconstruction noise to normalized input voltage noise. The
experiment uses homogeneous noise realizations without a point-target signal.

Choose a target from tolerable image noise and expected input noise, e.g.
target_gain = allowed relative conductivity error / relative voltage noise.
The example uses 0.5 as an illustrative adjustable target.

RMS is the default because it includes uniform offsets in a noise image.
Spatial std is an alternative: it removes the mean within each frame and can
miss uniform conductivity noise. Repeats estimate variability; **do not average
the noise images first**, since that would cancel noise rather than measure it.
The implementation averages per-image gains, not the ratio of separately
averaged noise magnitudes.

## Minimal API

~~~python
from eit_inverse import (
    NoiseSimulationConfig, NoiseAmplificationExperiment,
    tune_noise_amplification, SciPySolver,
)

# problem.matrix is the homogeneous Jacobian; variables are conductivity changes.
# homogeneous_voltages came from the forward model at sigma0.
config = NoiseSimulationConfig(
    repeats=50,
    noise_level=1e-3,            # SD = 0.1% of homogeneous voltage RMS
    noise_mode="relative_rms",
    seed=42,
    statistic="rms",
)
experiment = NoiseAmplificationExperiment(
    problem, SciPySolver(), homogeneous_voltages,
    config=config,
    image_scale=sigma0,
    image_weights=pixel_areas,
)
result = tune_noise_amplification(
    experiment,
    target_gain=0.5,
    weight_span=(1e-6, 1e2),
    n_weights=12,
    method="brent",
    validation_repeats=100,
)
regularization_weight = result.weight
print(result.estimate.mean_gain, result.matched)
print(result.validation.mean_gain, result.validation_matched)
~~~

The experiment ignores the problem's original data: the homogeneous difference
frame is zero, and each noisy frame is epsilon. It retains the matrix,
whitening, priors, bounds and other regularizers. It varies one penalty weight
via regularizer_index, default zero. This is a fixed linearized reconstruction,
not repeated nonlinear PDE optimization.

Weights still multiply squared L2 or unsquared L1, as in the existing package.
There is no hidden squaring of lambda or factor 1/2.

## Sensible adjustable parameters

| Parameter | API default | Suggested use |
| --- | --- | --- |
| repeats | 50 | Start with 20–50; use 200–500 for a final estimate if needed |
| validation_repeats | 100 | Fresh independent check; consider 200–500 for tighter Monte Carlo precision |
| noise_level | 0.001 | Start at 0.1% RMS only if plausible for the instrument; then use measured noise |
| noise_mode | relative_rms | Uniform additive noise SD based on homogeneous voltage RMS |
| noise_floor | 0 V | Add an absolute floor, especially for near-zero channels |
| seed | 42 | Reproducible tuning; independent validation defaults to seed+1 |
| target_gain | Caller-specified | Example 0.5; derive it from the desired conductivity-noise budget |
| weight_span | 1e-6 to 1e2 | Broad exploratory range; adapt to matrix/operator scaling |
| n_weights | 12 | Try 9–15 initially; 20–30 for a detailed scan |
| method | brent | Bracketed root search with measured reconstructions |
| polynomial_order | 2 | Order 2 or 3 is usually a sensible low-order starting point |
| refine_polynomial | True | Refine the fitted suggestion using actual reconstructions |
| target_rtol | 0.05 | 5% mean-gain tolerance; use 1–2% only with enough repeats |
| log_weight_tolerance | 0.001 decades | About 0.23% weight resolution; search accuracy is separate from Monte Carlo precision |
| max_iterations | 50 | Scalar root/optimization iteration limit |
| statistic | rms | Use std deliberately if removing spatial mean is desired |
| ddof | 0 | Population spatial std; 1 enables effective-sample-size correction |
| store_images | False | Save every reconstructed noise image only when needed |

The CLI example uses nine coarse scan points to keep the demonstration fast.
For squared L2, a useful exploratory scaling is lambda_scale = ||WJ||² / ||L||²
when L is nonzero, and then span several decades around lambda_scale.
L1 weight scales also depend on the expected noise/parameter amplitude.
The default numeric span is not universal.

Increase repeat count when the gain standard error is appreciable relative to
the requested tolerance. This standard error falls approximately as
1/sqrt(repeats) for independent samples. It measures Monte Carlo uncertainty,
not model or discretization error.

## Search methods

Every method starts with a log-spaced scan, reusing **identical fixed noise draws
at every weight**. Common draws make comparisons less noisy and the search
function reproducible. Candidate evaluations are cached.

- **brent:** bracket a target crossing and use scipy.optimize.brentq in
  log10(weight). The first crossing in ascending weight is used.
- **bounded:** minimize squared relative gain mismatch in log10(weight), in
  the neighboring scan interval around the closest sampled point.
- **polynomial:** fit log10(mean gain) against log10(weight), using a numerically
  scaled NumPy Polynomial. Solve the polynomial for the requested target, retain
  real roots inside the span, and evaluate them with actual reconstructions.
  By default, refine with a measured bracketed root near the proposed candidate.
- **grid:** choose the closest measured candidate, with no continuous search.

~~~python
result = tune_noise_amplification(
    experiment, 0.5,
    method="polynomial", polynomial_order=3,
    refine_polynomial=True,
)
~~~

Zero gains are omitted from log-polynomial fitting. High-order polynomial
oscillations or poor fit can produce misleading roots; no extrapolation outside
the span is allowed. The actual reconstructed gain is always checked, even when
refine_polynomial=False. Fit coefficients, domain/window, fitted log error and
candidate roots are returned in diagnostics.

No global monotonicity is assumed. Multiple sampled target crossings are
reported; the grid and diagnostics allow inspection. If the desired gain is
outside the sampled range, the closest measured candidate is returned with
messages and normally matched=False, rather than a fabricated extrapolated
solution. matched always reflects the measured relative target tolerance.
Search failures raise instead of quietly claiming convergence.

First-/second-order penalties have null spaces. Their unpenalized modes can
leave a nonzero noise floor even at large weight. Some requested gains are
therefore unattainable with that penalty; a zeroth-order term or a different
measurement definition may be needed if physically appropriate.

## Noise model and repeated measurements

Generated samples are additive zero-mean Gaussian voltage noise:

- relative_rms: SD_j = noise_level * RMS(homogeneous voltages).
- absolute: SD_j = noise_level in voltage units.
- relative_channel: SD_j = noise_level * abs(homogeneous voltage_j).
- An absolute noise_floor is added in quadrature to these SDs.

In channel-relative mode, an exactly zero channel has no generated noise unless
a floor is supplied. A real instrument often has both relative and absolute
contributions; use measurements to choose them.

A physical voltage covariance matrix may be supplied as noise_covariance:
finite, symmetric, nonzero positive semidefinite. It determines the draws in
voltage-squared units and overrides the level/mode/floor generation settings.
Full correlated noise and full inverse whitening are supported here because
this is not a row cross-validation split.

Or pass noise_samples of shape (n_samples,n_measurements), using measured
noise-only frames or another distribution. Its row count replaces config.repeats.
With explicit samples, independent validation requires a separate
validation_noise_samples array; the tuner does not assume an unknown sample
distribution is Gaussian.

The homogeneous reference is noise-free. If both a measurement and its
reference are noisy, supply the appropriate **difference-noise covariance**.
For independent reference/measurement errors, their covariances add.
The reference is not resimulated for each frame.

L2 unconstrained linear reconstruction usually has gain independent of a scalar
noise-level change. L1 thresholds and active bounds can make gain depend on the
noise amplitude; tune at the intended operating noise level.
Keep solver tolerances below the errors being measured; the example tightens
tolerances for low-noise/L1 solves. Noise suppressed below numerical accuracy
cannot be estimated reliably by loose solves.

## Parameter/evaluation maps and scaling

Optional image_operator maps the reconstruction vector to an evaluation field,
for example P for pixel-to-FEM prolongation or R for FEM-to-pixel export.
Supply physical areas/volumes in image_weights, with zeros allowed for excluded
entries. image_scale can be a scalar or one positive scale per evaluation entry.
For conductivity changes, use the homogeneous conductivity.

input_scale defaults to homogeneous voltage RMS. Set input_scale=1 and
image_scale=1 only when deliberately requesting a raw, generally dimensioned
output-noise/input-noise ratio. The reported metric depends on these choices.
If homogeneous voltages are all zero, provide a positive explicit input_scale
and an absolute noise/covariance model.

For normalized difference data, set difference_scale to the per-channel
normalization used by the Jacobian. The helper reconstructs epsilon/difference_scale,
while the input-gain denominator still describes the physical voltage noise.
The supplied matrix must use that same normalization. input_scale and
difference_scale have different roles and must not be silently interchanged.

The current inverse interface is real-valued. For complex EIT, declare a
consistent real/imaginary stacking, covariance and physical image statistic
before using these tools.

## Returned diagnostics and independent validation

Each estimate returns mean_gain, gain_std, gain_standard_error, all per-sample
gains, normalized output/input noise magnitudes, clean reconstruction and
optional physical noise images. Gain spread uses sample std across repeats;
SEM is spread divided by sqrt(number of repeats).

The tuning result contains the sampled curve with SEM, chosen estimate, target,
matched flag, evaluated weights, search diagnostics, messages and a separate
independent validation estimate/flag. A tuning match does not imply the validation
mean will match exactly: it is a new Monte Carlo sample.

~~~python
print(result.messages)
print(result.estimate.gain_standard_error)
print(result.validation.gain_standard_error)
~~~

The default real EIT run selected lambda about 0.1387:
tuning gain 0.5000 +/- 0.0150 SEM; independent gain
0.5136 +/- 0.0087 SEM. Both meet the default 5% tolerance.
These results are specific to this mesh, observation protocol, scaling,
noise model and pixel space; they are not recommended weights for other setups.

## Run the examples

~~~bash
# Real NGSolve homogeneous disk; 50 tuning and 100 validation samples:
python -m examples.noise_amplification_tuning

# Polynomial suggestion, followed by measured refinement:
python -m examples.noise_amplification_tuning \
    --method polynomial --polynomial-order 3

# Independent, intentionally larger Monte Carlo evaluation:
python -m examples.noise_amplification_tuning \
    --repeats 200 --validation-repeats 300 --target 0.5 --target-rtol 0.02

# Other backend and penalty:
python -m examples.noise_amplification_tuning --backend cvxpy
python -m examples.noise_amplification_tuning --norm l1 --order 0

# Alternative std metric and bounded optimization:
python -m examples.noise_amplification_tuning --statistic std --method bounded

# Save the curve, paired noise and selected noise images:
python -m examples.noise_amplification_tuning \
    --output results/noise_tuning.npz --plot results/noise_tuning.png

python -m pytest -q tests/test_noise_tuning.py
python -m pytest -q
~~~

CLI options also expose weight-min/max, n-weights, noise-level/mode/floor,
seed/validation-seed, sigma0, no-refine, target-rtol, log-weight-tolerance
and max-iterations. API additionally supports covariance, explicit samples,
custom image maps/weights and difference normalization.
The real example uses the affine-triangle disk and an area-weighted pixel image.
Use --plot and --output to save the figure and numerical results.

Tests check analytic identity gain 1/(1+lambda), weight lambda=3 for gain .25
under all three continuous methods, both backends and RMS/std, clean-image
bias removal, physical mapping/areas, correlated draws, reproducibility/SEM,
L1 noise-level dependence, absolute floors, unreachable targets, independent
explicit samples, unrefined polynomial validation, and real homogeneous EIT.

References:

- [SciPy bracketed root search](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html)
- [SciPy bounded scalar optimization](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html)

[Documentation index](../README.md)
