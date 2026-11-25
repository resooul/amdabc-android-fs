"""Read the per-task result files into tidy tables."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..io import read_json, read_npz


def _json_rows(pattern_root: Path, pattern: str) -> pd.DataFrame:
    rows = [read_json(p) for p in sorted(pattern_root.glob(pattern))]
    return pd.DataFrame(rows)


def _current(cfg, df: pd.DataFrame) -> pd.DataFrame:
    """Keep only results of the models/algorithms/representations in this config
    (files of models removed from the config are ignored, not deleted)."""
    if not len(df):
        return df
    keep = df["model"].isin(cfg["models"]["ml"] + cfg["models"]["dl"]) & df["rep"].isin(cfg["representations"])
    if "alg" in df:
        keep &= df["alg"].isin(cfg["optimization"]["algorithms"])
    if "fold" in df:
        keep &= df["fold"] < cfg["split"]["n_folds"]
    return df[keep].reset_index(drop=True)


def base(cfg) -> pd.DataFrame:
    return _current(cfg, _json_rows(Path(cfg["results_dir"]) / "base", "*/*/*/fold*.json"))


def evals(cfg) -> pd.DataFrame:
    return _current(cfg, _json_rows(Path(cfg["results_dir"]) / "eval", "*/*/*/fold*.json"))


def subsets(cfg) -> pd.DataFrame:
    rows = []
    for p in sorted((Path(cfg["results_dir"]) / "subsets").glob("*/fold*/*.json")):
        d = read_json(p)
        rows.append({"rep": p.parts[-3], "fold": int(p.parts[-2][4:]), "alg": p.stem,
                     "size": d["size"], "mean_run_size": float(np.mean(d["run_sizes"])),
                     "n_features_total": d["n_features_total"]})
    return pd.DataFrame(rows)


def opt(cfg):
    """Returns (runs DataFrame, curves dict[(rep, alg)] -> array runs x max_fe)."""
    rows, curves = [], {}
    root = Path(cfg["results_dir"]) / "opt"
    for p in sorted(root.glob("*/fold*/*/seed*.npz")):
        rep, fold, alg, seed = p.parts[-4], int(p.parts[-3][4:]), p.parts[-2], int(p.stem[4:])
        z = read_npz(p)
        rows.append({"rep": rep, "fold": fold, "alg": alg, "seed": seed, "fitness": float(z["best_fit"]),
                     "n_selected": int(z["mask"].sum()), "seconds": float(z["seconds"]),
                     "fe": int(z["fe"]), "n_fits": int(z["n_fits"])})
        curves.setdefault((rep, alg), []).append(z["curve"])
    return pd.DataFrame(rows), {k: np.vstack(v) for k, v in curves.items()}


def scen(cfg):
    root = Path(cfg["results_dir"]) / "scen"
    rows, preds = [], {}
    models = set(cfg["models"]["ml"] + cfg["models"]["dl"])
    for p in sorted((root / "eval").glob("*/*/*/*.json")):
        d = read_json(p)
        if d["model"] not in models:
            continue
        rows.append(d)
        preds[(d["rep"], d["scenario"], d["variant"], d["model"])] = read_npz(p.with_suffix(".npz"))
    sel = []
    for p in sorted((root / "opt").glob("*/*/seed*.npz")):
        z = read_npz(p)
        sel.append({"rep": p.parts[-3], "scenario": p.parts[-2], "seed": int(p.stem[4:]),
                    "seconds": float(z["seconds"]), "fitness": float(z["best_fit"])})
    sizes = {}
    for p in sorted((root / "subsets").glob("*/*.json")):
        d = read_json(p)
        sizes[(p.parts[-2], p.stem)] = (d["n_features_total"], d["size"])
    return pd.DataFrame(rows), preds, pd.DataFrame(sel), sizes
