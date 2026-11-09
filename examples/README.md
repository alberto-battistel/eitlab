# Examples

Install EITLab from the [main README](../README.md), then run an example with
`python -m examples.NAME` using the same environment. Each script has editable
settings near the top and `# %%` cells for imports, computation and plotting.
Computed models, arrays, results, figures and axes remain available for inspection.

## Run as cells or a script

Install from the [main README](../README.md), select that environment's Python
kernel in your IDE, open an example, and use **Run Cell** / **Run All Cells**.
Execute cells from top to bottom, and rerun the affected computation after
changing settings. VS Code and Spyder understand `# %%` directly.

In Jupyter, select the same environment's kernel. Use:

```python
%matplotlib inline
%run examples/disk_reconstruction.py
```

Run Jupyter from the project root for that relative file path. `%run` executes
the script and leaves its variables available in subsequent cells. To edit and
run individual cells in JupyterLab, optionally install Jupytext in the active
conda environment and convert a script:

```bash
python -m pip install jupytext
jupytext --to notebook examples/disk_reconstruction.py
```

Open the generated `.ipynb` with the EITLab environment's kernel. Jupytext is
optional; it is not needed for the scripts or the IDE cell workflow.

For terminal execution, use `python -m examples.NAME` from the project root or
another working directory after installation. Edit the script's settings cell
first. Figures display with `plt.show()` and save PNG copies under
`results/<example_name>/`, relative to the current working directory. Most
inverse examples also save their numerical arrays as NPZ.

Use `%matplotlib inline` for all Matplotlib figures in a notebook, or
`%matplotlib widget` for interactive notebook axes (requires `ipympl`). An IDE
or terminal can use QtAgg with an installed Qt binding. On headless Linux/macOS,
`MPLBACKEND=Agg python -m examples.NAME` saves the same plots without windows.
The scripts do not force a backend or close their figures. The optional native
PyVista boundary inspector is separate; see the [plotting guide](../docs/guides/plotting.md).

## Example index

Every example needs the `plot` extra. Add `fem` for the FEM examples below;
`cvxpy` is required by examples that demonstrate CVXPY-only convex penalties
or compare the SciPy and CVXPY backends.
Standard L2/LASSO/TV reconstruction uses SciPy without CVXPY.

| Example | Description and figures | Additional extras |
| --- | --- | --- |
| [plotting.py](plotting.py) | Matplotlib and optional PyVista meshes/fields in 2D/3D, pixel images and XY/XZ/YZ slices; highlight straight displayed edges despite curved FEM geometry. | `fem`; optional `plot-pv` |
| [disk_forward.py](disk_forward.py) | Start here: eight-electrode CEM disk, mesh and potential, electrode voltages, current linearity and conductivity update. | `fem` |
| [simple_eit_2d.py](simple_eit_2d.py) | Fine-mesh inclusion data and independent coarse Jacobian; SciPy L2/TV conductivity images and two-protocol voltage fits. | `fem` |
| [linear_inverse.py](linear_inverse.py) | Small numerical problem without FEM; plot parameter recovery and data fits for several Tikhonov weights. | none |
| [disk_reconstruction.py](disk_reconstruction.py) | Same-mesh synthetic reconstruction; truth, estimate and error maps on a common scale, plus voltage fits and residuals. | `fem` |
| [single_circle.py](single_circle.py) | Independent-mesh simulation with configurable Jacobian and Tikhonov reconstruction; conductivity maps and reference/target voltages. | `fem` |
| [named_points.py](named_points.py) | Named vertices and point-electrode rings in 2D/3D; geometry, point-grounded CEM potential and finite-electrode voltages. | `fem` |
| [pixel_inverse.py](pixel_inverse.py) | Area-overlap pixel mapping; plot truth, every selected L1/L2 order/backend reconstruction, and voltage fits. | `fem` |
| [joint_eit_contact_2d.py](joint_eit_contact_2d.py) | Joint conductivity/contact estimation and nonlinear refinement; conductivity, contact-impedance and voltage figures. | `fem` |
| [finite_difference_fallback.py](finite_difference_fallback.py) | Numerical or adjoint Jacobian; sensitivity/error heatmaps and reconstructed conductivity. Adjoint comparison is enabled by default. | `fem` |
| [regularization_orders.py](regularization_orders.py) | Reproducible 1D blur inverse problem; six L1/L2 order-0/1/2 reconstructions compared with truth. | none |
| [regularization_features.py](regularization_features.py) | Compare L2, LASSO and TV with both backends plus a CVXPY-only Huber-gradient reconstruction. | `cvxpy` |
| [regularization_weight_selection.py](regularization_weight_selection.py) | L-curve, truth correlation and held-out CV; selection diagnostics and resulting reconstructions. | none |
| [noise_amplification_tuning.py](noise_amplification_tuning.py) | Homogeneous voltages and repeated noisy inversions; gain curve, target, independent check and noise-only image. | `fem` |

`disk_forward.py` is a forward-physics demonstration with two individual current
patterns. Its settings expose geometry, conductivity, contact impedance, mesh
resolution and current amplitude. Potential maps and electrode voltages use the
same mean-zero reference. The 2D current is specified per unit out-of-plane depth;
the script documents this convention alongside the settings.

For a complete measurement plan combining multiple injections and observation
rules, followed by reconstruction, continue with `simple_eit_2d.py`.

## Settings and interpretation

Typical settings include `solver_backend = "scipy"` (or `"cvxpy"`/`"both"`),
`output_dir`, regularization weights and mesh sizes. `pixel_inverse` uses
`regularization_norm` and `regularization_order`; `single_circle` groups mesh,
conductivity and solver settings in `config = ExampleConfig(...)`.

Spatial conductivity comparisons use a common symmetric color scale. FEM values
are plotted in the actual conductivity DOF order; inactive pixels remain masked.
These coarse synthetic examples illustrate numerical behavior. Same-mesh data
fits do not establish experimental reconstruction accuracy.

`__init__.py`, `_common.py` (shared setup/plotting helpers) and `_single_circle.py`
(reusable numerical pipeline) are support modules. Importing a demonstration
executes its cells; import `examples._single_circle` to reuse its numerical
functions without running the demonstration.

[Documentation index](../docs/README.md)

