# Joint conductivity and electrode contact impedance

Run from the installed project:

```bash
python -m examples.joint_eit_contact_2d
```

The example uses top-level `# %%` cells and the conductivity parameter
interface. It imports no NGSolve module and constructs no native GridFunction.
Two independently generated meshes, adjacent and skip-3 injections, and
adjacent differential voltage readings match the other 2D teaching example.
The fine mesh contains an inclusion; the Jacobian mesh is homogeneous.

Here the synthetic measurement changes BOTH conductivity and one specific CEM
contact impedance per electrode. The contact vector always follows the explicit
`electrodes` tuple. This is the model's specific contact impedance `z` in the
complete-electrode boundary law, not a separately added lumped series resistor.
Use a consistent 2D depth/unit convention before interpreting physical units.
The homogeneous fine-mesh baseline uses reference contacts; the inclusion
measurement uses the perturbed contacts.

## Parameters and two independent priors

The unknown is dimensionless:

```text
x = [x_sigma, x_contact]
sigma = background * (1 + x_sigma)
z = z_reference * (1 + x_contact)
```

The objective is

\[
\left\|\frac{F(\sigma,z)-F(\sigma_0,z_0)-\Delta V}{s}\right\|_2^2
+\alpha_\sigma\|L_0 x_\sigma\|_2^2
+\alpha_z\|x_z\|_2^2.
\]

`F` is the coarse model; `Delta V` is the change measured with the fine model.
This difference formulation removes a static fine/coarse baseline offset, but
does not remove model error in the response to conductivity/contact changes.
`s` is the spectral norm of the reference relative-conductivity Jacobian block.

- `alpha_sigma` penalizes conductivity variation using the adapter's
  area-weighted zero-order L2 operator.
- `alpha_contact` penalizes the per-electrode contact changes independently.
- Increasing either weight suppresses the associated parameter block. There
  is no constraint that the weights add to one; their numerical magnitudes are
  not directly comparable because the operators and numbers of unknowns differ.

The defaults are illustrative values for noiseless synthetic data, not
automatically selected experimental noise/prior parameters. Relative scaling
removes raw conductivity/impedance units from the unknown vector, but does not
equalize the two blocks' sensitivities. For arbitrary experimental data, scale
the data term using a suitable noise model and select the priors accordingly.

## Derivatives and optimization

The joint matrix is `[J_sigma, J_contact]`. Conductivity derivatives use the
existing adjoint, multiplied by the conductivity reference to apply the chain
rule for relative changes. Contact derivatives use the generic central finite
difference provider, with steps in dimensionless relative-contact coordinates.
The public model setter reassembles the CEM operator after every contact update.
Old states/linearizations are not reused after these changes.

The first solve is a single linearized reconstruction through `InverseProblem`
and `SciPySolver`. `refinement_steps=3` adds regularized Gauss-Newton updates.
Each subproblem solves for the next absolute relative-parameter vector, so the
priors remain centered on the original reference, rather than on a moving
iterate. Backtracking accepts only decreases in the nonlinear objective,
including both priors. Bounds `-0.8 <= x <= 1` keep physical sigma and z positive
and limit extrapolation; they are teaching choices. Set `refinement_steps=0`
to inspect the initial linearized reconstruction.

The comparison solve assumes the contacts stayed at their reference values
and receives the same number of nonlinear refinement steps. Its conductivity
can absorb some actual contact changes. The figures show:

- True conductivity, reconstruction with assumed reference contacts, and joint
  conductivity/contact reconstruction, using a common color scale.
- True, reference and estimated contacts for all electrodes.
- Measured voltage changes and predictions from both reconstructed nonlinear
  forward models, separately for the two injection protocols.

## Interpretation and validation

With the defaults, there are 210 conductivity and 16 contact unknowns and 512
voltage channels. Differential channels include current electrodes, which helps
retain sensitivity to contact effects. They are redundant measurements, not
512 independent data values. No noise is added.

With equal nonlinear refinement budgets, both fits have about 0.76% relative
voltage residual. Their similar data fit illustrates that near-boundary
conductivity can compensate for assumed contacts; fitting voltages alone does
not establish correct parameter recovery. The contact-change
relative error is about 29% (error normalized by the true change, not the full
contact value). Recovery is therefore approximate, despite a small voltage
residual: contact effects, near-boundary conductivity, regularization and
discretization remain coupled. This example does not establish uniqueness or
replace an experimental electrode-calibration procedure.

Tests check both parameter blocks against a combined directional finite
difference, positive reconstructed values, objective descent, nonlinear data fit and
the independent effects of both regularization weights.

[Documentation index](../README.md)
