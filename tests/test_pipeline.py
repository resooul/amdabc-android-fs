"""Pipeline tests on a small synthetic data set (no real data needed)."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amdabc_fs import config as C
from amdabc_fs import stages as S
from amdabc_fs.checks import check_all
from amdabc_fs.data.prepare import outer_split, prepare
from amdabc_fs.fs.consensus import zscore_consensus
from amdabc_fs.io import read_npz, write_json

ROOT = Path(__file__).resolve().parents[1]
SRC = {"Drebin": 1, "Genome": 1, "CICInvesAndMal2019": 0, "Maldroid2020": 0, "Omer": 0}


def _fake_sources(ddir: Path, n=120, d=30, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"op{j}" for j in range(d)]
    for rep in ("dvm", "jvm", "hybrid"):
        for s, lab in SRC.items():
            r = np.random.default_rng(abs(hash(s)) % 2**32)
            X = (r.random((n, d)) < (0.3 + 0.3 * lab * (np.arange(d) < 5))).astype(int)
            df = pd.DataFrame(X, columns=cols)
            df.insert(0, "uygulama_adi", [f"{s}_{i}" for i in range(n)])
            df["Class"] = lab
            df.to_csv(ddir / f"{rep} {s}.csv", index=False)


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipe")
    (tmp / "raw").mkdir()
    _fake_sources(tmp / "raw")
    cfg = C.load(ROOT / "config" / "smoke.yaml")
    cfg = copy.deepcopy(cfg)
    cfg.update(data_dir=str(tmp / "raw"), results_dir=str(tmp / "res"))
    cfg["optimization"].update(max_fe=20, seeds=3)
    cfg["filters"]["k"] = 10
    prepare(cfg)
    return cfg


def test_leakage_checks_pass(cfg):
    assert check_all(cfg) == ["dvm", "jvm", "hybrid"]


def test_filter_and_fitness_see_training_rows_only(cfg, monkeypatch):
    seen = {}
    real_top_k, real_fit = S.top_k, S.WrapperFitness

    def spy_top_k(method, X, y, k):
        seen["filter_rows"] = len(X)
        return real_top_k(method, X, y, k)

    class SpyFit(real_fit):
        def __init__(self, X, y, **kw):
            seen["fitness_rows"] = len(X)
            super().__init__(X, y, **kw)

    monkeypatch.setattr(S, "top_k", spy_top_k)
    monkeypatch.setattr(S, "WrapperFitness", SpyFit)
    S._candidates.cache_clear()
    out = Path(cfg["results_dir"]) / "spy.npz"
    S.opt_task(cfg, "dvm", 0, "AMDABC", 0, out)
    tr, te = outer_split(S.data(cfg, "dvm"), 0)
    assert seen["filter_rows"] == len(tr) and seen["fitness_rows"] == len(tr)
    z = read_npz(out)
    assert int(z["fe"]) == 20 and len(z["curve"]) == 20


def test_consensus_rule():
    masks = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0]], bool)
    # freq = [1, 2/3, 1/3, 0], mean = 0.5 -> keep features 0 and 1
    assert zscore_consensus(masks).tolist() == [True, True, False, False]


def test_atomic_write_leaves_no_partial_file(tmp_path, monkeypatch):
    import amdabc_fs.io as io

    def boom(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(io.json, "dumps", boom)
    with pytest.raises(KeyboardInterrupt):
        write_json(tmp_path / "x.json", {"a": 1})
    assert not list(tmp_path.iterdir())
