"""Experiment stages. Each stage builds a list of resumable tasks (tasks.py).

Order (as in the paper):
  prepare   -> clean data, folds, scenario splits
  base      -> 7 ML + 4 DL on all features and on FScore/Chi2/Pearson top-100
  opt       -> 15 optimizers x reps x folds x seeds, FE budget
  subsets   -> z-score consensus per (rep, fold, optimizer)
  eval      -> ML + DL on each consensus subset
  scenarios -> AMDABC selection on scenario training sources, full vs selected
Result layout: see README.md.
"""
from __future__ import annotations

import pickle
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .data.prepare import load_rep, outer_split, scenario_indices
from .fs.consensus import zscore_consensus
from .fs.filters import top_k
from .fs.fitness import WrapperFitness
from .io import read_json, read_npz, write_json, write_npz
from .ml import metrics
from .ml.models import make as make_ml
from .optim.runner import run as run_optimizer
from .tasks import Task

# ------------------------------------------------------------------ helpers ----
R = lambda cfg: Path(cfg["results_dir"])            # noqa: E731


@lru_cache(maxsize=8)
def _data(results_dir: str, rep: str) -> dict:
    return load_rep({"results_dir": results_dir}, rep)


def data(cfg, rep):
    return _data(str(cfg["results_dir"]), rep)


def make_model(cfg, name):
    if name in cfg["models"]["dl"]:
        from .dl.models import DLClassifier
        d = cfg["dl"]
        return DLClassifier(name, epochs=d["epochs"], patience=d["patience"], batch_size=d["batch_size"],
                            lr=d["lr"], val_fraction=d["val_fraction"], device=d["device"])
    return make_ml(name)


def fit_eval(cfg, name, Xtr, ytr, Xte, yte, keep_pred=False) -> dict:
    model = make_model(cfg, name)
    Xtr, Xte = np.asarray(Xtr, np.float32), np.asarray(Xte, np.float32)
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    score = model.predict_proba(Xte)[:, 1]
    t_pred = time.perf_counter() - t0
    pred = (score >= 0.5).astype(int) if name in cfg["models"]["dl"] else model.predict(Xte)
    out = metrics.compute(yte, pred, score)
    try:
        size = len(pickle.dumps(model.net.state_dict() if hasattr(model, "net") else model))
    except Exception:
        size = -1
    out.update(n_features=int(Xtr.shape[1]), fit_seconds=t_fit,
               predict_us_per_sample=1e6 * t_pred / max(len(Xte), 1), model_bytes=size)
    if keep_pred:
        out["_pred"] = {"y": yte, "pred": pred, "score": score}
    return out


def models(cfg):
    return cfg["models"]["ml"] + cfg["models"]["dl"]


# ---------------------------------------------------------------- base ----------
def base_task(cfg, rep, variant, model, k, out):
    d = data(cfg, rep)
    tr, te = outer_split(d, k)
    X, y = d["X"], d["y"]
    cols = np.arange(X.shape[1]) if variant == "full" else top_k(variant, X[tr], y[tr], cfg["filters"]["k"])
    res = fit_eval(cfg, model, X[tr][:, cols], y[tr], X[te][:, cols], y[te])
    res.update(rep=rep, variant=variant, model=model, fold=k)
    write_json(out, res)


def base_tasks(cfg):
    T = []
    for rep in cfg["representations"]:
        for variant in ["full"] + cfg["filters"]["methods"]:
            for m in models(cfg):
                for k in range(cfg["split"]["n_folds"]):
                    out = R(cfg) / "base" / rep / variant / m / f"fold{k}.json"
                    T.append(Task(f"base/{rep}/{variant}/{m}/f{k}", out, base_task, (cfg, rep, variant, m, k, out)))
    return T


# ---------------------------------------------------------------- opt -----------
@lru_cache(maxsize=16)
def _candidates(results_dir, rep, k, method, kk):
    d = _data(results_dir, rep)
    tr, _ = outer_split(d, k)
    return top_k(method, d["X"][tr], d["y"][tr], kk)


def candidates(cfg, rep, k, train_idx=None):
    if train_idx is None:
        return _candidates(str(cfg["results_dir"]), rep, k, cfg["filters"]["wrapper_filter"], cfg["filters"]["k"])
    d = data(cfg, rep)
    return top_k(cfg["filters"]["wrapper_filter"], d["X"][train_idx], d["y"][train_idx], cfg["filters"]["k"])


def _optimize(cfg, Xtr, ytr, cand, alg, seed, out):
    f = cfg["fitness"]
    fit = WrapperFitness(Xtr[:, cand], ytr, metric=f["metric"], weight=f["weight"],
                         inner_folds=f["inner_folds"], inner_seed=f["inner_seed"])
    r = run_optimizer(alg, len(cand), fit, cfg["optimization"]["max_fe"], seed)
    write_npz(out, curve=r.curve.astype(np.float64), mask=r.best_mask, cand=cand,
              best_fit=np.float64(r.best_fit), seconds=np.float64(r.seconds),
              fe=np.int64(r.fe_used), n_fits=np.int64(fit.n_fits), epochs=np.int64(r.epochs))


def opt_task(cfg, rep, k, alg, seed, out):
    d = data(cfg, rep)
    tr, _ = outer_split(d, k)
    cand = candidates(cfg, rep, k)
    _optimize(cfg, d["X"][tr], d["y"][tr], cand, alg, seed, out)


def opt_path(cfg, rep, k, alg, seed):
    return R(cfg) / "opt" / rep / f"fold{k}" / alg / f"seed{seed:02d}.npz"


def opt_tasks(cfg):
    T = []
    o = cfg["optimization"]
    # seed-major order: an interrupted run leaves every optimizer with similar progress
    for seed in range(o["seeds"]):
        for rep in cfg["representations"]:
            for k in range(cfg["split"]["n_folds"]):
                for alg in o["algorithms"]:
                    out = opt_path(cfg, rep, k, alg, seed)
                    T.append(Task(f"opt/{rep}/f{k}/{alg}/s{seed}", out, opt_task, (cfg, rep, k, alg, seed, out)))
    return T


# ---------------------------------------------------------------- subsets -------
def _consensus(cfg, paths, n_features):
    runs = [read_npz(p) for p in paths]
    cand = runs[0]["cand"]
    masks = np.array([r["mask"] for r in runs])
    keep = zscore_consensus(masks, cfg["consensus"]["sigma"])
    return {"features": cand[keep].tolist(), "size": int(keep.sum()),
            "run_sizes": masks.sum(1).tolist(), "n_candidates": int(len(cand)),
            "n_features_total": int(n_features),
            "best_fit": [float(r["best_fit"]) for r in runs]}


def subset_task(cfg, rep, k, alg, out):
    d = data(cfg, rep)
    paths = [opt_path(cfg, rep, k, alg, s) for s in range(cfg["optimization"]["seeds"])]
    res = _consensus(cfg, paths, d["X"].shape[1])
    res["names"] = [str(d["features"][i]) for i in res["features"]]
    write_json(out, res)


def subset_tasks(cfg):
    T = []
    for rep in cfg["representations"]:
        for k in range(cfg["split"]["n_folds"]):
            for alg in cfg["optimization"]["algorithms"]:
                out = R(cfg) / "subsets" / rep / f"fold{k}" / f"{alg}.json"
                T.append(Task(f"subset/{rep}/f{k}/{alg}", out, subset_task, (cfg, rep, k, alg, out)))
    return T


# ---------------------------------------------------------------- eval ----------
def eval_task(cfg, rep, alg, model, k, out):
    d = data(cfg, rep)
    tr, te = outer_split(d, k)
    cols = np.array(read_json(R(cfg) / "subsets" / rep / f"fold{k}" / f"{alg}.json")["features"])
    X, y = d["X"], d["y"]
    res = fit_eval(cfg, model, X[tr][:, cols], y[tr], X[te][:, cols], y[te])
    res.update(rep=rep, alg=alg, model=model, fold=k)
    write_json(out, res)


def eval_tasks(cfg):
    T = []
    dl_algs = cfg["models"].get("dl_eval_algorithms", "all")
    for rep in cfg["representations"]:
        for alg in cfg["optimization"]["algorithms"]:
            for m in models(cfg):
                if m in cfg["models"]["dl"] and dl_algs != "all" and alg not in dl_algs:
                    continue
                for k in range(cfg["split"]["n_folds"]):
                    out = R(cfg) / "eval" / rep / alg / m / f"fold{k}.json"
                    T.append(Task(f"eval/{rep}/{alg}/{m}/f{k}", out, eval_task, (cfg, rep, alg, m, k, out)))
    return T


# ---------------------------------------------------------------- scenarios -----
def scen_opt_task(cfg, rep, sc, seed, out):
    d = data(cfg, rep)
    tr, _ = scenario_indices(cfg, rep, sc)
    cand = candidates(cfg, rep, None, train_idx=tr)
    _optimize(cfg, d["X"][tr], d["y"][tr], cand, "AMDABC", seed, out)


def scen_opt_path(cfg, rep, sc, seed):
    return R(cfg) / "scen" / "opt" / rep / sc / f"seed{seed:02d}.npz"


def scen_subset_task(cfg, rep, sc, out):
    d = data(cfg, rep)
    paths = [scen_opt_path(cfg, rep, sc, s) for s in range(cfg["scenarios"]["seeds"])]
    res = _consensus(cfg, paths, d["X"].shape[1])
    res["names"] = [str(d["features"][i]) for i in res["features"]]
    write_json(out, res)


def scen_eval_task(cfg, rep, sc, variant, model, out):
    d = data(cfg, rep)
    tr, te = scenario_indices(cfg, rep, sc)
    X, y = d["X"], d["y"]
    if variant == "full":
        cols = np.arange(X.shape[1])
    else:
        cols = np.array(read_json(R(cfg) / "scen" / "subsets" / rep / f"{sc}.json")["features"])
    res = fit_eval(cfg, model, X[tr][:, cols], y[tr], X[te][:, cols], y[te], keep_pred=True)
    pred = res.pop("_pred")
    res.update(rep=rep, scenario=sc, variant=variant, model=model)
    write_npz(out.with_suffix(".npz"), y=pred["y"], pred=pred["pred"], score=pred["score"])
    write_json(out, res)                        # written last: marks the task as done


def scenario_tasks(cfg):
    s = cfg["scenarios"]
    opt, sub, ev = [], [], []
    for rep in cfg["representations"]:
        for sc in s["names"]:
            for seed in range(s["seeds"]):
                out = scen_opt_path(cfg, rep, sc, seed)
                opt.append(Task(f"scen-opt/{rep}/{sc}/s{seed}", out, scen_opt_task, (cfg, rep, sc, seed, out)))
            out = R(cfg) / "scen" / "subsets" / rep / f"{sc}.json"
            sub.append(Task(f"scen-subset/{rep}/{sc}", out, scen_subset_task, (cfg, rep, sc, out)))
            for variant in ["full", "AMDABC"]:
                for m in models(cfg):
                    out = R(cfg) / "scen" / "eval" / rep / sc / variant / f"{m}.json"
                    ev.append(Task(f"scen-eval/{rep}/{sc}/{variant}/{m}", out, scen_eval_task,
                                   (cfg, rep, sc, variant, m, out)))
    return opt, sub, ev
