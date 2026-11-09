# Testing and development

From the repository root, in an active environment:

```bash
python -m pip install -e ".[test,fem,cvxpy,plot]"
python -m pytest -q
```

Tests use maintained package modules. No source-directory flag or reference
snapshot is involved. CI installs the same extras, runs the suite and executes
a small single-circle reconstruction.

## Focused checks

```bash
python -m pytest -q tests/test_inverse.py tests/test_inverse_architecture.py
python -m pytest -q tests/test_inverse.py tests/test_bibliography.py
python -m pytest -q -m jacobian
python -m pytest -q tests/test_named_points.py
```

Missing NGSolve or CVXPY may skip their optional checks. A skip does not validate
that feature; install the full extras above to exercise them. For headless plots:

```bash
MPLBACKEND=Agg python -m pytest -q
```

On PowerShell, use `$env:MPLBACKEND="Agg"`. For timing consistency, optionally
set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`. Interactive PyVista boundary
inspection needs a graphical environment and separate manual verification.

## Wheel validation

```bash
python -m pip wheel . --no-deps --wheel-dir dist
```

Install the wheel in another environment and run examples from outside the
checkout to detect accidental root-file dependencies. The bibliography is
installed under `share/eitlab`; offline lookup resolves it there.

For an inverse-only environment install `.[test]` and run the numerical inverse,
differentiation and bibliography tests. The source distribution includes docs
and tests; the wheel contains runtime packages, examples and bibliography.

[Documentation index](../README.md)
