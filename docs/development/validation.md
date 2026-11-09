# Validation: EITLab 0.3.0

Checked on 6 November 2026 using Python 3.12, the Agg Matplotlib backend
and one OpenMP/OpenBLAS thread.

## Source validation

- Collection: 184 tests, with no collection errors or skips.
- Complete suite: 184 passed, with no failures, errors or skips.
- The retained suite covers CEM physics, adjoint/numerical Jacobians,
  conductivity and pixel maps, SciPy/CVXPY regularization, noise/weight
  selection, named points, plotting, notebook examples and bibliography.
- The registry contains five software references; standalone bibliography and
  inverse/pixel imports remain independent of FEM imports.

## Distribution verification

Wheel and source distribution contents were checked for excluded features and
cached files. The source distribution collects the same 184 test cases.
Both the source tree and source distribution pass local Markdown file-target
checks.

The wheel was installed offline in a new private environment using compatible
existing dependencies. All project packages resolve to that installed wheel,
and its five-reference bibliography resolves under `share/eitlab`.

The installed `examples.disk_forward`, `examples.simple_eit_2d` and
`examples.pixel_inverse` demonstrations all executed successfully outside the
source tree. Six PNGs and one NPZ were verified as nonempty and readable.

## Scope

The retained electrical and inverse implementations were tested on Linux.
Interactive PyVista behavior and external documentation URLs were not checked.

[Documentation index](../README.md)
