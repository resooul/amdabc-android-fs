"""Settings and data tables, generated from the code and the data card so the
manuscript always states the values that were actually used.

  tab:ml_hyperparameters   ML models (read from the instantiated estimators)
  tab:dl_hyperparameters   DL models (architecture + training settings)
  tab:optimizer_params     optimizers (read from the instantiated optimizers)
  tab:dataset_cleaning     rows/columns per cleaning step (data card)
"""
from __future__ import annotations

import pandas as pd

from ..ml.models import make as make_ml
from ..optim.registry import OPTIMIZERS
from .layout import REP_LABEL

ML_KEYS = {
    "CAT": ["depth", "iterations", "learning_rate"],
    "GBM": ["max_depth", "n_estimators"],
    "LDA": ["solver", "shrinkage"],
    "LR": ["C", "solver"],
    "MLP": ["hidden_layer_sizes", "activation", "solver", "max_iter"],
    "RF": ["max_depth", "n_estimators", "criterion"],
    "XGB": ["tree_method", "max_depth", "n_estimators", "learning_rate"],
}
DL_ARCH = {
    "DL-MLP": "dense 128-64, ReLU, dropout 0.2",
    "CNN1D": "conv1d 32 (k=5), max-pool 2, conv1d 64 (k=5), global max-pool, dropout 0.2",
    "BiLSTM": "feature tokens (d=16), BiLSTM 32 units per direction, mean pooling, dropout 0.2",
    "PatchTransformer": "patches of 8 features as tokens (d=64), [CLS], 2 encoder layers, 4 heads",
    "FTTransformer": "one token per feature (d=32), [CLS], 2 encoder layers, 4 heads",
}


def _tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_")


def _tabular(label, caption, header, rows, spec="ll", scale="0.75"):
    body = " \\\\\n".join("\t\t\t" + " & ".join(_tex_escape(c) for c in r) for r in rows)
    head = " & ".join("\\textbf{" + h + "}" for h in header)
    return (f"\\begin{{table}} [h!]\n\t\\centering\n\t\\caption{{{caption}}}\n\t\\label{{{label}}}\n"
            f"\t\\scalebox{{{scale}}}{{\n\t\t\\begin{{tabular}}{{{spec}}}\n"
            f"\t\t\t{head} \\\\\n\t\t\t\\hline\n{body}\n"
            "\t\t\\end{tabular}\n\t}\n\\end{table}\n")


def ml_hyperparameters(models):
    rows = []
    for m in sorted(models):
        p = make_ml(m).get_params()
        rows.append([m, ", ".join(f"{k}: {p[k]}" for k in ML_KEYS[m])])
    return _tabular("tab:ml_hyperparameters", "Hyperparameter settings of the ML algorithms",
                    ["Model", "Hyperparameters"], rows), pd.DataFrame(rows, columns=["Model", "Hyperparameters"])


def dl_hyperparameters(cfg):
    d = cfg["dl"]
    train = (f"Adam (lr {d['lr']}), binary cross-entropy, batch {d['batch_size']}, at most {d['epochs']} epochs, "
             f"early stopping (patience {d['patience']}) on a {int(100 * d['val_fraction'])}\\% stratified "
             "validation split of the training fold")
    rows = [[m, DL_ARCH[m]] for m in cfg["models"]["dl"]]
    tex = _tabular("tab:dl_hyperparameters", "Architecture of the DL baselines. Training for all models: " + train,
                   ["Model", "Architecture"], rows)
    return tex, pd.DataFrame(rows, columns=["Model", "Architecture"])


def optimizer_params(cfg):
    o = cfg["optimization"]
    rows = []
    for name in o["algorithms"]:
        spec = OPTIMIZERS[name]
        m = spec.cls(epoch=50, **spec.kwargs)
        kw = spec.kwargs
        if "n_limits" in kw:            # ABC family: colony of pop_size bees = pop_size/2 food sources
            sn = getattr(m, "NS", None)
            lim = kw["n_limits"] if kw["n_limits"] is not None else f"SN x D (= {sn} x D)"
            extra = f", n_best: {kw['n_best']}" if "n_best" in kw else ""
            par = f"colony: {kw['pop_size']}, SN: {sn}, L: {lim}{extra}"
        else:
            if hasattr(m, "initialize_variables"):
                try:
                    m.initialize_variables()
                except Exception:
                    pass
            keys = [k for k in ("c1", "c2", "w", "p1", "p2", "beta", "FADS", "P") if hasattr(m, k)]
            par = ", ".join([f"pop: {m.pop_size}"] + [f"{k}: {getattr(m, k)}" for k in keys])
        rows.append([name, par])
    cap = (f"The specific parameter settings of the optimization algorithms (every run stops after "
           f"{o['max_fe']} fitness evaluations; {o['seeds']} runs per outer fold)")
    return _tabular("tab:optimizer_params", cap, ["Algorithm", "Parameter Settings"], rows, scale="0.9"), \
        pd.DataFrame(rows, columns=["Algorithm", "Parameters"])


def dataset_cleaning(card, reps):
    steps = [("rows_raw", "Rows (processed tables)"), ("zero_rows_dropped", "Empty vectors removed"),
             ("conflict_rows_dropped", "Label-conflicting rows removed"),
             ("duplicate_rows_collapsed", "Duplicate rows collapsed"), ("rows_clean", "Unique rows used"),
             ("malware", "  malware"), ("benign", "  benign"),
             ("cols_raw", "Columns (incl. identifier and class)"), ("dup_cols_dropped", "Copied columns (.1) removed"),
             ("const_cols_dropped", "Constant columns removed"),
             ("const_cols_after_cleaning", "Constant after row cleaning"),
             ("identical_cols_dropped", "Identical columns removed"), ("n_features", "Features used")]
    rows = [[lab] + [f"{card[r][k]:,}" for r in reps] for k, lab in steps]
    tex = _tabular("tab:dataset_cleaning", "Rows and columns at each data-preparation step",
                   ["Step"] + [REP_LABEL[r] for r in reps], rows, spec="l" + "r" * len(reps), scale="0.9")
    return tex, pd.DataFrame(rows, columns=["Step"] + [REP_LABEL[r] for r in reps])
