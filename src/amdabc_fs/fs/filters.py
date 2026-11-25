"""Filter methods (FScore, Chi2, Pearson); always fitted on training rows only."""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import chi2, f_classif


def _pearson(X, y):
    X = X.astype(float)
    yc = y - y.mean()
    Xc = X - X.mean(0)
    den = np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (Xc * yc[:, None]).sum(0) / den
    return np.nan_to_num(np.abs(r))


def scores(method: str, X, y) -> np.ndarray:
    if method == "fscore":
        with np.errstate(divide="ignore", invalid="ignore"):
            s = f_classif(X, y)[0]           # columns constant in this fold give nan -> 0
    elif method == "chi2":
        s = chi2(X, y)[0]
    elif method == "pearson":
        s = _pearson(X, y)
    else:
        raise ValueError(method)
    return np.nan_to_num(s, nan=0.0)


def top_k(method: str, X, y, k: int) -> np.ndarray:
    """Column indices of the k best features (stable order, ties by index)."""
    s = scores(method, X, y)
    k = min(k, X.shape[1])
    order = np.lexsort((np.arange(len(s)), -s))     # descending score, then index
    return np.sort(order[:k])
