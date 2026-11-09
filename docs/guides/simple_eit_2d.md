# Two-mesh, two-protocol 2D EIT example

From the project root, with your Python environment active:

```bash
python -m pip install -e ".[fem,plot]"
python -m examples.simple_eit_2d
```

Or open `examples/simple_eit_2d.py` in Spyder/VS Code and run its `# %%` cells.
There is no `main()` wrapper: meshes, models, voltage vectors, `J`, both inverse
problems, results and reconstructed conductivity parameters remain available for inspection.
The script also translates directly to notebook cells. Restart the IDE kernel
if you installed a different version of the package in that environment.

The example uses these defaults:

- Fine simulation mesh: unit disk with an inclusion centered at (0.35, 0.2),
  radius 0.25, background conductivity 1 and inclusion conductivity 0.7.
- Independent coarse Jacobian mesh: homogeneous unit disk, without a predefined
  inclusion region. Real elementwise constant conductivity parameters.
- 16 electrodes, 1 A injection, adjacent and skip-3 current protocols. Skip-3
  means source `k`, sink `k+4`, cycling through every source electrode.
- Differential voltage readings `U[k]-U[k+1]`, with wrap-around, for every
  injection, including voltage pairs touching current electrodes. Thus each
  protocol has 16 x 16 readings. The pairs sum to zero for each injection;
  these are redundant channels, not 512 independent measurements.
- Difference data: fine-mesh inclusion voltages minus fine-mesh homogeneous
  voltages. The coarse homogeneous voltages are used to form the Jacobian,
  not subtracted from the synthetic measurements.

Both reconstructions use `InverseProblem` and `Regularizer` from `eit_inverse`.
Both Tikhonov and TV use `SciPySolver`; its ADMM path handles the L1 TV
penalty. The common inverse API supports these objectives, so the backends can
also be swapped. A common scaling
`A=J/||J||_2`, `b=delta_voltage/||J||_2` is used before applying the displayed
regularization weights. `alpha_l2` and `alpha_tv` are demonstration choices.

The unknown is the conductivity change `x`. Zero-order L2 is
`sum(area(T)*x[T]**2)`. For P0 conductivity the interior element gradient is
zero, so TV must penalize jumps: `sum(length(e)*abs(x[left]-x[right]))` over
internal edges. There is no artificial zero value outside the domain.
`parameter_space.regularization_operators()` requests both sparse operators
from the FEM conductivity adapter in its actual coefficient order. The example
contains no triangle traversal, area calculation or adjacency construction;
the inverse framework has no NGSolve import.
The area/edge weights use straight triangles at mesh vertices, approximating
curved elements. TV boundaries remain constrained by the reconstruction mesh.

The script adds the known background to each reconstructed change and saves under `results/simple_eit_2d/`:

- `simple_eit_2d.png`: homogeneous conductivity, truth, Tikhonov and TV.
- `simple_eit_2d_meshes.png`: simulation and Jacobian meshes, electrodes in red.
- `simple_eit_2d_voltages.png`: both voltage data sets, their difference, and
  linearized voltage predictions from each reconstruction, for both protocols.

Mesh and conductivity panels call the integrated `eitlab.plotting` utility;
voltage traces use ordinary Matplotlib line plots. All outputs go to the working
directory. There is no added noise, no positivity constraint and no nonlinear
iteration. The example demonstrates the workflow, not quantitative experimental
performance or a mesh-convergence study.


## Conductivity parameters without native FEM construction

```python
from eitlab import ElementwiseConductivity

parameter_space = ElementwiseConductivity(jacobian_mesh)
sigma0 = parameter_space.create(background)
L0, Dtv = parameter_space.regularization_operators()
# Individual requests are also available:
# L0 = parameter_space.l2_operator()
# Dtv = parameter_space.tv_operator()

# The model and the linearization accept these application-level objects:
# model = ElectricalModel(jacobian_mesh, electrodes, sigma=sigma0)
# linearization = ElectricalLinearization(model, parameter_space=parameter_space)
sigma_image = parameter_space.create(background + delta_sigma_tv, positive=False)
# viz.plot_field(jacobian_mesh, sigma_image, backend="matplotlib")
```

The representation is selected by its class, `ElementwiseConductivity`, with no
NGSolve `L2`, `GridFunction` or coefficient-vector machinery in the example.
`create` accepts a scalar, a coefficient array or a full material dictionary.
The resulting `ConductivityParameter.values` is an owned copy, not a live
reference into a solver. Use `sigma.with_values(...)` or `parameter_space.create`
and explicitly call `model.set_sigma(...)` to assemble an updated forward model.
Parameters from another mesh are rejected by models and plotting.

`ConductivitySpace` documents the interface independently of NGSolve; the
NGSolve implementation lives in `eit_forward.parameters`. Its P0 spatial
operators support triangular 2D and tetrahedral 3D meshes. They use vertex-based
area/volume and interface length/area, preserving the previous example's
straight-cell approximation for curved geometry. A smooth or higher-order
basis must supply its own operators and declare the appropriate gradient/group
norm: the P0 jump formula must not be reused blindly.

`positive=False` supports signed changes and unconstrained image estimates.
Passing such a parameter into a forward model still checks that all physical
conductivity values are positive. The currently implemented parameter class is
real-valued; complex conductivity remains available through the `ElectricalModel`
API, rather than being silently coerced by this factory. Native FEM inputs and `triangle_regularization_operators` are also supported.

[Documentation index](../README.md)
