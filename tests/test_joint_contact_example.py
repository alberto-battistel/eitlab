"""End-to-end joint estimation, parameter blocks and derivative ordering."""
from pathlib import Path
import runpy

import numpy as np
import pytest


@pytest.mark.fem
def test_joint_conductivity_contact_example(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from eit_inverse import InverseProblem, Regularizer, SciPySolver
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: None)
    script = Path(__file__).resolve().parents[1] / "examples/joint_eit_contact_2d.py"
    s = runpy.run_path(str(script))
    try:
        n, nz = s["n_sigma"], s["n_contact"]
        assert nz == 16
        assert s["J_joint"].shape == (512, n+nz)
        assert s["simulation_mesh"] is not s["jacobian_mesh"]
        assert s["nonlinear_residual_joint"] < s["nonlinear_residual_fixed"]
        # This synthetic example remains approximate: discretization and
        # conductivity/contact ambiguity preclude claiming exact calibration.
        assert s["contact_change_relative_error"] < .5
        assert np.all(s["z_estimate"] > 0)
        assert np.all(s["sigma_joint"].values > 0)
        assert np.all(np.diff(s["objective_history"]) <= 1e-14)
        for block, weight in zip(s["regularizers"], [s["alpha_sigma"], s["alpha_contact"]]):
            assert block.weight == weight
        assert s["L_sigma"][:, n:].nnz == 0
        assert s["L_contact"][:, :n].nnz == 0

        # A combined conductivity/contact direction verifies the units,
        # column order and the adjoint/finite-difference block composition.
        direction = np.random.default_rng(7).normal(size=n+nz)
        direction /= np.linalg.norm(direction)
        h = 1e-4
        values = [s["joint_prediction"](sign*h*direction) for sign in (1, -1)]
        numerical = (values[0]-values[1])/(2*h)
        analytic = s["J_joint"] @ direction
        error = np.linalg.norm(numerical-analytic)/np.linalg.norm(numerical)
        assert error < 1e-5

        # Increasing only the contact weight must suppress its fitted changes.
        regs = [s["regularizers"][0], Regularizer(s["L_contact"], 1., norm="l2")]
        tighter = InverseProblem(s["J_joint"]/s["data_scale"],
                                 s["delta_voltage"]/s["data_scale"], regs,
                                 bounds=(-.8, 1.0))
        fitted = SciPySolver().solve(tighter).parameters
        assert np.linalg.norm(fitted[n:]) < .01*np.linalg.norm(s["one_step_parameters"][n:])
        # The conductivity weight can be changed without altering the contact
        # prior; this independently suppresses the area-weighted image norm.
        regs = [Regularizer(s["L_sigma"], 1., norm="l2"), s["regularizers"][1]]
        tighter_sigma = InverseProblem(s["J_joint"]/s["data_scale"],
                                       s["delta_voltage"]/s["data_scale"], regs,
                                       bounds=(-.8, 1.0))
        fitted_sigma = SciPySolver().solve(tighter_sigma).parameters
        assert np.linalg.norm(s["L_sigma"] @ fitted_sigma) < np.linalg.norm(
            s["L_sigma"] @ s["one_step_parameters"])
        assert len(list(s["output_dir"].glob("joint_eit_contact_2d*.png"))) == 3
        print(f"Joint conductivity/contact directional derivative relative error: {error:.3g}")
    finally:
        plt.close("all")
