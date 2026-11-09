# API changes in EITLab 0.3

Version 0.3 removes compatibility code and the historical test-kit layout.
Install in a fresh environment, or reinstall this checkout and restart Python
after replacing the previous files. The maintained numerical methods remain.

## Imports

| Removed location | Maintained location |
| --- | --- |
| `FEM.ElectricalModel` | `eitlab.ElectricalModel` or `eit_forward.model.ElectricalModel` |
| `FEM.EITAdjoint` | `eitlab.ElectricalLinearization` |
| `FEM.JacobianAssembler` | `eitlab.JacobianAssembler` |
| Root `geometries`, `measurement`, `stimulation` | `eit_forward.geometries`, `.measurement`, `.stimulation`; common types also in `eitlab` |
| `inverse.TikhonovSolver` | `eit_inverse.TikhonovSolver` or `eitlab.TikhonovSolver` |
| `finite_difference` | `eitlab.differentiation`; common classes also in `eitlab` |
| `ngsolve_forward` | `eit_forward.adapters`; common adapters also in `eitlab` |
| `eit_inverse.spaces` | `eitlab.discretization`; classes also exported by `eit_inverse` |
| `eit_single_circle_forward.py` | `examples.single_circle`, run with `python -m examples.single_circle` |

`reference_model`, `eit_test_support`, `--eit-source` and example `--source`
switches are removed. Production adapters use `eit_forward.backend.load_backend()`.
Example setup lives in `examples._common`; tests have their own support module.

## Calls and geometry

- Use `MeasurementPlan` instead of `MeasurementPlanner`.
- Use `ElectricalLinearization` instead of `EITAdjoint`.
- Use `JacobianAssembler(linearization, results)`; the runner argument is removed.
- Use `model.set_sigma(value)` instead of `update_sigma`.
- Use `FiniteDifferenceJacobian(forward.evaluate, step=..., scheme=...)`
  instead of the `PerturbationJacobian` wrapper.
- Geometry no longer accepts `body_value`, inclusion `value` fields or the
  unused cylinder `objects` argument. Supply conductivity through
  `space.create({"body": ..., "target": ...})` or a material map on the model.
- `CylinderModel.get_material_dict()` is removed with the physical-value metadata.

## Plotting

Use `plot_mesh`, `plot_field` and `plot_boundaries`, which return `PlotResult`.
The old `draw_*`, `plot_gf`, `plot_sigma_cf`, `calc_image`, `plot_img` wrappers
and deprecated keywords are removed. Use `sample_image` followed by
`plot_image(image, extent=extent)` for a regular image. Select a backend and
use `ax=` or `plotter=` for existing views.

The SVD `TikhonovSolver` remains as a maintained algorithm.

## Repository

`pyproject.toml` is the single dependency declaration, replacing requirements
text files. Current guides/method references are under `docs/guides` and
`docs/reference`. Historical checklists, integration-kit notes, validation logs
and bundled generated plots/NPZ files are removed. New outputs are written
under the git-ignored `results/` directory. Source archives contain no virtual
environments, build caches or generated results.

[Documentation index](../README.md)
