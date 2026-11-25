"""Z-score consensus over the runs of one (representation, fold, optimizer).

Rule:
freq_j = share of runs selecting feature j; keep j if
freq_j >= mean(freq) + sigma * std(freq), with sigma = 0.
"""
from __future__ import annotations

import numpy as np


def zscore_consensus(masks: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    masks = np.asarray(masks, bool)
    freq = masks.mean(0)
    keep = freq >= freq.mean() + sigma * freq.std()
    if not keep.any():                       # degenerate: all runs empty
        keep = freq == freq.max()
    return keep
