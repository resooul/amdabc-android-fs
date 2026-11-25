"""Metrics. Paper columns: ACC, AUC, PRE, REC, F1 (weighted averages, in %).
Additional columns: balanced accuracy, MCC, macro F1, malware recall, FPR."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)


def compute(y_true, y_pred, y_score) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    return {
        "ACC": 100 * accuracy_score(y_true, y_pred),
        "AUC": 100 * auc,
        "PRE": 100 * precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "REC": 100 * recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": 100 * f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "BACC": 100 * balanced_accuracy_score(y_true, y_pred),
        "MCC": 100 * matthews_corrcoef(y_true, y_pred),
        "F1_macro": 100 * f1_score(y_true, y_pred, average="macro", zero_division=0),
        "REC_malware": 100 * tp / max(tp + fn, 1),
        "FPR": 100 * fp / max(fp + tn, 1),
    }
