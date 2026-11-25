"""Wrapper fitness used by all optimizers.

fitness(mask) = w * (1 - CV score) + (1 - w) * |mask| / D, lower is better;
empty mask -> 1.0. CV: StratifiedKFold(5, shuffle=True, random_state=42) on the
outer training fold, LogisticRegression(C=1, solver="liblinear"). The inner
split is fixed, so the same mask always gets the same value; a per-process
cache returns it without refitting (the FE counter still counts the call).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

METRICS = {"accuracy": accuracy_score, "balanced_accuracy": balanced_accuracy_score}


class WrapperFitness:
    def __init__(self, X, y, metric="balanced_accuracy", weight=0.9, inner_folds=5, inner_seed=42):
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.y = np.asarray(y)
        self.score = METRICS[metric]
        self.w = weight
        skf = StratifiedKFold(inner_folds, shuffle=True, random_state=inner_seed)
        self.splits = list(skf.split(self.X, self.y))
        self.cache: dict[bytes, float] = {}
        self.n_fits = 0

    def cv_score(self, mask: np.ndarray) -> float:
        Xs = self.X[:, mask]
        s = []
        for tr, te in self.splits:
            clf = LogisticRegression(C=1.0, solver="liblinear")
            clf.fit(Xs[tr], self.y[tr])
            s.append(self.score(self.y[te], clf.predict(Xs[te])))
        self.n_fits += 1
        return float(np.mean(s))

    def __call__(self, mask: np.ndarray) -> float:
        mask = np.asarray(mask, bool)
        if not mask.any():
            return 1.0
        key = np.packbits(mask).tobytes()
        if key not in self.cache:
            self.cache[key] = self.w * (1.0 - self.cv_score(mask)) + (1.0 - self.w) * mask.mean()
        return self.cache[key]
