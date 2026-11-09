"""Execute the updated demonstrations as notebook cells with kernel arguments."""
from pathlib import Path
import re
import sys

import pytest


NUMERICAL = ["linear_inverse", "regularization_orders",
             "regularization_weight_selection"]
FEM = ["disk_forward", "disk_reconstruction", "single_circle", "named_points",
       "pixel_inverse", "finite_difference_fallback", "noise_amplification_tuning"]
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("name", NUMERICAL + [pytest.param(n, marks=pytest.mark.fem) for n in FEM])
def test_example_runs_cell_by_cell(name, tmp_path, monkeypatch):
    plt = pytest.importorskip("matplotlib.pyplot")
    if name in FEM:
        pytest.importorskip("ngsolve")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ipykernel_launcher.py", "-f", "kernel.json"])
    displays = []
    monkeypatch.setattr(plt, "show", lambda: displays.append(tuple(plt.get_fignums())))
    path = EXAMPLES / f"{name}.py"
    namespace = {"__name__": "__main__", "__file__": str(path)}
    try:
        # Each cell is compiled independently, as it is in an IDE/notebook.
        for cell in re.split(r"^# %%.*$", path.read_text(), flags=re.MULTILINE):
            exec(compile(cell, str(path), "exec"), namespace)
        assert displays and any(displays[-1]), "The plotting cell must display its figures"
        images = list(namespace["output_dir"].glob("*.png"))
        assert images and all(image.stat().st_size > 1000 for image in images)
        assert any(hasattr(value, "savefig") for value in namespace.values()), "Figures must remain available"
        if name == "disk_reconstruction":
            assert namespace["relative_residual"] < .1
        if name == "finite_difference_fallback":
            assert namespace["error"] < 1e-5
        if name == "second_order":
            assert namespace["objective_error"] < 2e-4
    finally:
        plt.close("all")
