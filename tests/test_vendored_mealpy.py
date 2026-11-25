"""The repository runs without an installed mealpy; the vendored subset must
behave exactly like mealpy 3.0.3 (compared only where mealpy is installed)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_no_external_mealpy_needed():
    code = ("import sys; sys.modules['mealpy'] = None; sys.path.insert(0, 'src');"
            "from amdabc_fs.optim.registry import OPTIMIZERS; from amdabc_fs.optim.runner import run;"
            "import numpy as np;"
            "r = run('PSO', 10, lambda m: float(m.mean()), 30, 0); assert r.fe_used == 30")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


@pytest.mark.parametrize("name", ["PSO.OriginalPSO", "SSO.OriginalSSO", "HHO.OriginalHHO", "AGTO.OriginalAGTO"])
def test_same_as_mealpy_303(name):
    mealpy = pytest.importorskip("mealpy")
    if mealpy.__version__ != "3.0.3":
        pytest.skip("installed mealpy is not 3.0.3")
    import importlib

    from amdabc_fs.optim import mealpy_core
    mod, cls = name.split(".")
    f = lambda x: float(np.sum((x - 0.3) ** 2))  # noqa: E731
    out = []
    for pkg, var in ((importlib.import_module(f"mealpy.swarm_based.{mod}"), mealpy.FloatVar),
                     (importlib.import_module(f"amdabc_fs.optim.mealpy_core.swarm_based.{mod}"), mealpy_core.FloatVar)):
        m = getattr(pkg, cls)(epoch=20, pop_size=10)
        m.solve({"obj_func": f, "bounds": var(lb=(-1.,) * 8, ub=(1.,) * 8), "minmax": "min", "log_to": None}, seed=5)
        out.append((m.g_best.solution.copy(), m.g_best.target.fitness, m.nfe_counter))
    assert np.array_equal(out[0][0], out[1][0]) and out[0][1] == out[1][1] and out[0][2] == out[1][2]
