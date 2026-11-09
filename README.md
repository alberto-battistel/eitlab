# EITLab

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24084811.svg)](https://doi.org/10.5281/zenodo.24084811)

Finite-element electrical impedance tomography with the complete electrode
model, conductivity Jacobians, and independent inverse solvers.
Reconstruction options include Tikhonov and total variation.

## Installation

Extract the archive and open a terminal in the directory containing this
README and `pyproject.toml`. Python 3.12 is recommended; the package requires
Python 3.10 or newer and compatible dependency wheels.

Use an installed [Miniconda or Anaconda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html)
distribution. Open a conda-enabled terminal (Anaconda Prompt on Windows),
then create and activate an environment:

```bash
conda create --name eitlab python=3.12 pip
conda activate eitlab
```

These commands apply on Linux, macOS and Windows. Conda manages the Python
environment; install EITLab and its dependencies with pip inside that active
environment. From the extracted project directory, run:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[fem,plot,cvxpy]"
python -c "import eitlab, ngsolve, cvxpy; print('EITLab', eitlab.__version__, 'NGSolve', ngsolve.__version__); print(cvxpy.installed_solvers())"
```

In each new terminal, run `conda activate eitlab` before using EITLab.

`-e` makes edits in this checkout available immediately. Use
`python -m pip install ".[fem,plot,cvxpy]"` for a regular installation instead.
NGSolve installs matching Netgen; no separate Netgen installation is needed.
If your platform needs additional system libraries, follow the
[NGSolve installation guide](https://ngsolve.org/installation.html).

For numerical inversion without FEM, install only `python -m pip install -e .`.
All dependencies are declared in `pyproject.toml`:

| Extra | Adds | Needed for |
| --- | --- | --- |
| `fem` | NGSolve and matching Netgen | Geometry and electrical forward models |
| `plot` | Matplotlib | Static plots and image exports |
| `cvxpy` | CVXPY and solver dependencies | Optional convex/reference backend; advanced convex regularizers and backend-comparison examples |
| `plot-pv` | PyVista | Optional 3D viewing and boundary inspector |
| `test` | pytest | Test suite |

## Starting example 1: electrical forward problem

```bash
python -m examples.disk_forward
```

Build a small disk with eight finite electrodes, solve a balanced injection,
print electrode voltages, check current linearity and change the conductivity.
The example also compares measurements with and without current electrodes.
It requires `fem` and `plot`; mesh, potential and voltage plots are displayed
and saved in `results/disk_forward/`.

## Starting example 2: reconstruct a 2D inclusion

```bash
python -m examples.simple_eit_2d
```

This simulates homogeneous and inclusion data on a fine mesh and builds a
Jacobian on an independent coarse homogeneous mesh. It combines adjacent and
skip-3 injections with adjacent differential voltage measurements, then
reconstructs conductivity changes with zero-order L2 and TV penalties.

Mesh, conductivity and voltage figures are saved under
`results/simple_eit_2d/` and displayed when a graphical backend is available.
The example uses `fem`, `plot` and `cvxpy`. Run its `# %%` sections individually
in an IDE if preferred; geometry, solver settings and weights are near the top.
For a Linux/macOS terminal without a display:

```bash
MPLBACKEND=Agg python -m examples.simple_eit_2d
```

The [worked explanation](docs/guides/simple_eit_2d.md) covers measurement
ordering and the distinction between conductivity and its change.

All examples follow the same notebook-friendly `# %%` layout, with editable
settings near the top and computed variables left in the namespace. In a Jupyter
notebook using the EITLab environment's kernel, run:

```python
%matplotlib inline
%run examples/disk_forward.py
```

Use **Run Cell** in VS Code/Spyder for individual sections. See the
[example index](examples/README.md) for Jupytext conversion to `.ipynb`, optional
interactive notebook plots, and the figures provided by each demonstration.

## Where to go next

- [Example index](examples/README.md): every example, dependencies and commands.
- [Documentation index](docs/README.md): guides, mathematical references and development notes.
- [Bibliography](docs/reference/bibliography.md): citations from root `references.bib`.
- [API changes in 0.3](docs/development/api_changes.md): updated imports and removed interfaces.

`eit_forward` owns FEM physics, `eit_inverse` consumes numerical matrices and
data, and `eitlab` supplies convenience imports, parameters, discretization,
differentiation, bibliography and optional plotting.

## Citation

If you use EITLab in your research, please cite the archived software version
used for your study. For the initial release:

> Battistel, A. (2026). *EITLab* (Version 0.1.0) [Computer software].
> Zenodo. https://doi.org/10.5281/zenodo.24084811

The DOI and badge above refer to the archived **v0.1.0** release. For
reproducibility, cite the specific software version used in your study.
See [CITATION.cff](CITATION.cff) for machine-readable citation metadata,
and [GitHub releases](https://github.com/alberto-battistel/eitlab/releases)
for available versions.

## Tests

```bash
python -m pip install -e ".[test,fem,plot,cvxpy]"
python -m pytest -q
```

See [testing](docs/development/testing.md) for focused checks and
[validation](docs/development/validation.md) for this release's verification.
