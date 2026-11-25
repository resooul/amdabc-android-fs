"""Statistical tests.

Optimizers (paired blocks = (fold, seed)):
  * Wilcoxon signed-rank, AMDABC vs each competitor, two-sided; raw and
    Holm-adjusted p-values (Holm over the competitors of one table and representation);
  * Friedman test and mean ranks (rank 1 = lowest fitness);
  * Nemenyi critical difference for the CD diagram;
  * Vargha-Delaney A12 = P(AMDABC < other) + 0.5 P(tie)  (> 0.5 favours AMDABC);
  * bootstrap 95% CI of the mean fitness.
Classifiers (paired outer folds): Nadeau-Bengio corrected resampled t-test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

Q_ALPHA_005 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
               10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391}


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def a12(x, y) -> float:
    """Probability that a value of x is lower (better) than a value of y."""
    x, y = np.asarray(x), np.asarray(y)
    lt = (x[:, None] < y[None, :]).mean()
    eq = (x[:, None] == y[None, :]).mean()
    return float(lt + 0.5 * eq)


def bootstrap_ci(x, n=2000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def optimizer_tests(runs: pd.DataFrame, algs: list[str], rep: str, ref="AMDABC", n_boot=2000, seed=42):
    """runs: columns rep, fold, seed, alg, fitness. Returns one row per algorithm."""
    w = (runs[(runs.rep == rep) & runs.alg.isin(algs)]
         .pivot_table(index=["fold", "seed"], columns="alg", values="fitness")
         .dropna()[algs])
    ranks = w.rank(axis=1, method="average").mean()
    fr = st.friedmanchisquare(*[w[a].to_numpy() for a in algs]) if len(algs) > 2 else None
    others = [a for a in algs if a != ref]
    pw = []
    for a in others:
        d = w[ref] - w[a]
        pw.append(1.0 if np.allclose(d, 0) else st.wilcoxon(w[ref], w[a]).pvalue)
    p_holm = dict(zip(others, holm(pw)))
    p_raw = dict(zip(others, pw))
    k, n = len(algs), len(w)
    cd = Q_ALPHA_005.get(k, np.nan) * np.sqrt(k * (k + 1) / (6 * n))
    rows = []
    for a in algs:
        lo, hi = bootstrap_ci(w[a], n_boot, seed)
        rows.append({"rep": rep, "alg": a, "mean": w[a].mean(), "std": w[a].std(ddof=1),
                     "median": w[a].median(), "ci_low": lo, "ci_high": hi, "rank": ranks[a],
                     "p_wilcoxon": p_raw.get(a, np.nan), "p_holm": p_holm.get(a, np.nan),
                     "a12_vs_ref": a12(w[ref], w[a]) if a != ref else np.nan,
                     "friedman_chi2": fr.statistic if fr else np.nan,
                     "friedman_p": fr.pvalue if fr else np.nan, "cd": cd, "n_blocks": n})
    return pd.DataFrame(rows)


def nadeau_bengio(a, b, n_train: int, n_test: int):
    """Corrected resampled t-test over paired folds (a, b: per-fold scores)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * st.t.sf(abs(t), k - 1)
    return float(t), float(p)


def paired_bootstrap_diff(y, p_a, p_b, metric, n=2000, seed=42):
    """CI of metric(y, p_b) - metric(y, p_a) by resampling test rows."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        diffs.append(metric(y[i], p_b[i]) - metric(y[i], p_a[i]))
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
