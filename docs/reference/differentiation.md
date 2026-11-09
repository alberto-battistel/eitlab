# Interchangeable adjoint and finite-difference Jacobians

The derivative of interest is the complete parameter-to-measurement map:

`F(p) -> flat voltage vector`, with `J[j, k] = dF_j / dp_k`.

Differentiate this map rather than an intermediate field. It includes the
conductivity parameterization, stimulation protocol, voltage reference,
measurement selection, and output ordering. Both providers expose
`jacobian(parameters)` and return `(number_of_measurements, number_of_parameters)`.

## A generic fallback with no symbolic derivatives

```python
from eitlab.differentiation import FiniteDifferenceJacobian
from eit_inverse import TikhonovSolver

# forward.evaluate(p) may use NGSolve, another FEM package, or any numerical model.
provider = FiniteDifferenceJacobian(
    forward.evaluate,
    step=1e-4,          # absolute change in each parameter's own units
    scheme="central",
)
J = provider.jacobian(p_reference)
delta_y = measured_voltages - forward.evaluate(p_reference)
delta_p = TikhonovSolver(1e-4, J).solve(delta_y)
```

The function supplied to `FiniteDifferenceJacobian` is the only dependency on
the forward model. The finite-difference module itself imports only NumPy.
It does not construct an adjoint, use `Diff`, or inspect a variational form.
There is no automatic exception-swallowing fallback: select the provider
appropriate to the available model capabilities.

## Select the adjoint through the same interface

```python
from eit_forward.adapters import NGSolveAdjointJacobian

provider = NGSolveAdjointJacobian(forward)
J = provider.jacobian(p_reference)
```

The inverse solver still receives the same matrix. `NGSolveConductivityForward`
in `eit_forward.adapters` is the numerical bridge to the electrical model. It maps
a vector into `L2(order=0)` conductivity coefficients and returns the protocol's
flattened voltages. It preserves one parameter ordering for both providers.

For a different parameterization, write another `evaluate(p)` callback that
maps its parameters into the physical coefficients. A regionwise or
log-conductivity mapping needs no changes to `FiniteDifferenceJacobian`.
Its Jacobian is then with respect to those parameters, not necessarily sigma.

## Formulas and evaluation cost

For parameter index k, unit direction e_k, and absolute step h_k:

| Scheme | Jacobian column k | Full protocol evaluations for n parameters |
| --- | --- | --- |
| Forward | `(F(p + h_k e_k) - F(p)) / h_k` | n + 1 |
| Backward | `(F(p) - F(p - h_k e_k)) / h_k` | n + 1 |
| Central | `(F(p + h_k e_k) - F(p - h_k e_k)) / (2 h_k)` | 2n + 1 |

Central differences have second-order truncation error for sufficiently smooth
maps; one-sided differences have first-order error. This implementation makes
one baseline evaluation even for central differences to check the measurement
layout. Each evaluation means running the entire configured injection protocol,
not solving just one current pattern.

The supplied example has 40 parameters: central differences require 81
protocol evaluations, forward differences 41. Only voltage vectors are retained
for the finite differences. A full dense Jacobian still occupies O(m*n) memory.

Finite differences become expensive for large elementwise parameter spaces.
Use a smaller parameter basis when appropriate; the adjoint remains a useful
efficient option for many conductivity parameters.

## Step sizes, bounds, and state

- `step` is an absolute scalar or a vector containing one positive step per
  parameter. It is not multiplied by the parameter value automatically.
- For parameters with different scales or units, supply per-parameter steps.
- Compare results at h, h/2, and h/4. Look for stable columns or stable Jv
  products. A smaller step can amplify solver tolerances or evaluation noise.
- Central differences require both perturbed points to be admissible. For a
  conductivity parameter, ensure `sigma - h > 0`. Select forward/backward
  differences near a bound, or use a suitable parameterization. This minimal
  implementation does not automatically detect bounds or adjust step direction
  separately for each parameter.
- Keep the mesh, measurement ordering, and protocol fixed while differentiating.
  Remeshing after every perturbation can introduce mesh effects into the result.
- Every callback evaluation must use the complete supplied parameter vector.
  Repeated calls at the same vector should reproduce the same measurements.

The maintained NGSolve adapter now reuses its spaces and safely reassembles the
operator when conductivity changes. Symbolic linearizations are tied to revisions;
ordinary finite differences need only forward solves. The generic finite-difference
class cannot restore hidden state in arbitrary callbacks, so callbacks must evaluate
the complete supplied vector. The provided adapter restores the requested model on
every subsequent evaluation. Input arrays and returned measurement buffers are copied.

The generic implementation accepts **real parameters** and real or complex
measurement vectors. To estimate a complex conductivity, represent its real
and imaginary parts as separate real parameters. The current example adapter
and extracted Tikhonov solver are real-valued; they have not been extended to
complex EIT by this addition.

SciPy also documents central and one-sided numerical differentiation and step
direction near domain boundaries:
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.differentiate.jacobian.html>.

## Run examples and verification

```bash
# Actual fallback: no adjoint/symbolic derivative is called
python -m examples.finite_difference_fallback --method central

# The cheaper one-sided alternative
python -m examples.finite_difference_fallback --method forward

# Optional validation against the adjoint, when available
python -m examples.finite_difference_fallback --method central --compare

# Choose the adjoint through the same provider interface
python -m examples.finite_difference_fallback --method adjoint

# Select an absolute perturbation step
python -m examples.finite_difference_fallback --step 1e-5

# Generic numerical checks; no NGSolve required
python -m pytest -q tests/test_eitlab/differentiation.py

# Includes full finite-difference versus adjoint matrix comparisons
python -m pytest -q -m jacobian
```

The generic checks cover an analytic nonlinear map, zero-valued parameters,
scalar/per-parameter steps, complex measurements, unchanged caller inputs,
callbacks returning shared buffers, and inconsistent measurement sizes.

The FEM comparisons use nonuniform reference conductivities, both complete and
current-excluded measurements, and both forward and central differences. After
calculating an independent adjoint reference, the test deliberately disables
`ElectricalLinearization` before calculating the finite-difference Jacobian. This verifies
that the fallback works when symbolic linearization is unavailable.

On the homogeneous example with absolute step 1e-4, the measured relative
matrix errors against the adjoint were approximately 2.5e-9 for central
differences and 4.8e-5 for forward differences. These are results for this
configuration, not accuracy guarantees for other models or steps.

[Documentation index](../README.md)
