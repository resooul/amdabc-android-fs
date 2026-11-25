"""Leakage checks on the prepared data. run.py calls them before every stage
that trains anything; a failed check stops the run."""
from __future__ import annotations

import numpy as np

from .data.prepare import load_rep, scenario_indices


class LeakageError(AssertionError):
    pass


def _check(cond, msg):
    if not cond:
        raise LeakageError(msg)


def check_all(cfg) -> list[str]:
    done = []
    for rep in cfg["representations"]:
        d = load_rep(cfg, rep)
        X, y, fold = d["X"], d["y"], d["fold"]
        # 1. no identical vectors left -> no exact train/test duplicate in any split
        _check(len(np.unique(X, axis=0)) == len(X), f"{rep}: duplicate feature vectors remain")
        # 2. outer folds form a partition, every fold has both classes
        _check(set(np.unique(fold)) == set(range(cfg["split"]["n_folds"])), f"{rep}: fold ids")
        for k in range(cfg["split"]["n_folds"]):
            _check(len(np.unique(y[fold == k])) == 2, f"{rep}: fold {k} lacks a class")
        # 3. no identifier-like column (a feature unique per row)
        _check(X.shape[1] == len(d["features"]) and "uygulama_adi" not in set(d["features"]),
               f"{rep}: identifier column present")
        # 4. scenarios: disjoint rows; held-out benign sources never in training
        for name in cfg["scenarios"]["names"]:
            tr, te = scenario_indices(cfg, rep, name)
            _check(not set(tr) & set(te), f"{rep} {name}: train/test rows overlap")
            sdef = cfg["scenarios"]["definitions"][name]
            held = set(sdef["test_benign"])
            for i in tr:
                if y[i] == 0:
                    s = set(d["sources"][i].split("|"))
                    _check(not s & held, f"{rep} {name}: held-out benign source in training")
                    _check(s <= set(sdef["train_benign"]), f"{rep} {name}: unexpected benign source")
            for i in te:
                if y[i] == 0:
                    _check(set(d["sources"][i].split("|")) & held, f"{rep} {name}: test benign not held out")
        done.append(rep)
    return done
