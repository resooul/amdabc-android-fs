"""Short timing benchmark (a few minutes) and a run-time estimate for the config.

    python run.py --config config/default.yaml --benchmark --workers 20

Measures on this machine, fold 0 of each representation:
  * one wrapper-fitness evaluation (LR, 5-fold inner CV) on 100 candidates;
  * one fit of every ML model on all features;
  * one training epoch of every DL model on the configured device.
Writes results_dir/benchmark.json and prints the estimate. The optimization
estimate is an upper bound: it assumes every evaluation fits a classifier,
while repeated masks are answered from a cache.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from . import stages as S
from .data.prepare import outer_split
from .fs.fitness import WrapperFitness
from .io import write_json

log = logging.getLogger("amdabc")
DL_EPOCHS_ASSUMED = 30          # typical epochs before early stopping


def run(cfg, workers: int, n_eval: int = 15) -> dict:
    reps, folds = cfg["representations"], cfg["split"]["n_folds"]
    o, f = cfg["optimization"], cfg["fitness"]
    rng = np.random.default_rng(0)
    res = {"workers": workers, "per_rep": {}}

    for rep in reps:
        d = S.data(cfg, rep)
        tr, te = outer_split(d, 0)
        cand = S.candidates(cfg, rep, 0)
        fit = WrapperFitness(d["X"][tr][:, cand], d["y"][tr], metric=f["metric"], weight=f["weight"],
                             inner_folds=f["inner_folds"], inner_seed=f["inner_seed"])
        t0 = time.perf_counter()
        for _ in range(n_eval):
            fit(rng.random(len(cand)) > 0.5)
        s_fe = (time.perf_counter() - t0) / n_eval
        ml = {}
        for m in cfg["models"]["ml"]:
            t0 = time.perf_counter()
            S.fit_eval(cfg, m, d["X"][tr], d["y"][tr], d["X"][te], d["y"][te])
            ml[m] = time.perf_counter() - t0
        dl = {}
        for m in cfg["models"]["dl"]:
            from .dl.models import DLClassifier
            dd = cfg["dl"]
            clf = DLClassifier(m, epochs=1, patience=1, batch_size=dd["batch_size"], lr=dd["lr"],
                               val_fraction=dd["val_fraction"], device=dd["device"])
            t0 = time.perf_counter()
            clf.fit(d["X"][tr].astype(np.float32), d["y"][tr])
            dl[m] = time.perf_counter() - t0
            res["dl_device"] = clf.device
        res["per_rep"][rep] = {"seconds_per_fe": s_fe, "ml_fit_seconds": ml, "dl_epoch_seconds": dl}
        log.info("benchmark %s: %.3f s/FE, ML fits %s s, DL epoch %s s", rep, s_fe,
                 {k: round(v, 2) for k, v in ml.items()}, {k: round(v, 1) for k, v in dl.items()})

    # ---- estimate for the whole config ----
    n_alg, n_seed, max_fe = len(o["algorithms"]), o["seeds"], o["max_fe"]
    pr = res["per_rep"]
    opt_cpu = sum(n_alg * folds * n_seed * max_fe * pr[r]["seconds_per_fe"] for r in reps)
    n_var = 1 + len(cfg["filters"]["methods"])
    n_alg_ml = len(o["algorithms"])
    dl_algs = cfg["models"].get("dl_eval_algorithms", "all")
    n_alg_dl = n_alg_ml if dl_algs == "all" else len(dl_algs)
    n_sc = len(cfg["scenarios"]["names"])
    ml_cpu = sum(sum(pr[r]["ml_fit_seconds"].values()) * (folds * (n_var + n_alg_ml) + 2 * n_sc) for r in reps)
    dl_cpu = sum(sum(pr[r]["dl_epoch_seconds"].values()) * DL_EPOCHS_ASSUMED *
                 (folds * (n_var + n_alg_dl) + 2 * n_sc) for r in reps)
    scen_opt = sum(n_sc * cfg["scenarios"]["seeds"] * max_fe * pr[r]["seconds_per_fe"] for r in reps)
    res["estimate_hours"] = {
        "optimization_cpu_hours_upper": opt_cpu / 3600,
        "optimization_wall_hours_upper": opt_cpu / 3600 / workers,
        "ml_wall_hours": ml_cpu / 3600 / workers,
        "scenario_selection_wall_hours": scen_opt / 3600 / workers,
        "dl_hours_single_process": dl_cpu / 3600,
        "dl_assumed_epochs": DL_EPOCHS_ASSUMED,
    }
    write_json(Path(cfg["results_dir"]) / "benchmark.json", res)
    e = res["estimate_hours"]
    log.info("estimate with %d workers: optimization <= %.1f h (%.0f CPU-h), ML %.2f h, scenario selection %.2f h, "
             "DL %.1f h in one process on %s (assumes %d epochs per model)", workers,
             e["optimization_wall_hours_upper"], e["optimization_cpu_hours_upper"], e["ml_wall_hours"],
             e["scenario_selection_wall_hours"], e["dl_hours_single_process"], res.get("dl_device"),
             DL_EPOCHS_ASSUMED)
    return res
