"""One-page summary of a finished run: results/report/SUMMARY.md.

Built from the CSV files the report stage writes, so it can be rebuilt at any
time with ``python run.py --stage report``. Sections are skipped when the
corresponding stage has not produced results yet.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .layout import REP_LABEL, TABLE_ORDER


def _read(p: Path):
    return pd.read_csv(p) if p.exists() else None


def _md(df: pd.DataFrame, floatfmt="{:.2f}") -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        rows.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(rows)


def write(cfg, out: Path) -> Path:
    root = Path(cfg["results_dir"])
    st, tb = out / "stats", out / "tables"
    reps = cfg["representations"]
    L = [f"# Run summary ({root})", ""]

    env = sorted((root / "environment").glob("*.json"))
    if env:                      # every start of run.py leaves a record; list the distinct machines
        seen = []
        for p in env:
            e = json.loads(p.read_text(encoding="utf-8"))
            m = (f"{e['cpu']}, {e['logical_cores']} threads, {e['ram_gb']} GB RAM; {e['os']}; "
                 f"Python {e['python']}; torch {e['packages'].get('torch')}")
            if m not in seen:
                seen.append(m)
        L += ["Machine(s) recorded in results/environment: " + " | ".join(seen) + ".", ""]
    o = cfg["optimization"]
    L += [f"Protocol: {cfg['split']['n_folds']}-fold outer CV; {len(o['algorithms'])} optimizers x "
          f"{o['seeds']} seeds, {o['max_fe']} FE each; fitness {cfg['fitness']['weight']} x (1 - "
          f"{cfg['fitness']['metric']}) + {1 - cfg['fitness']['weight']:.1f} x |S|/D.", ""]

    # progress: result files present vs expected
    n_f, n_rep = cfg["split"]["n_folds"], len(reps)
    n_models = len(cfg["models"]["ml"]) + len(cfg["models"]["dl"])
    n_var = 1 + len(cfg["filters"]["methods"])
    dl_algs = cfg["models"].get("dl_eval_algorithms", "all")
    n_dl_algs = len(o["algorithms"]) if dl_algs == "all" else len(dl_algs)
    sc = cfg["scenarios"]
    expected = {
        "base": (n_models * n_rep * n_var * n_f, "base/*/*/*/fold*.json"),
        "opt": (len(o["algorithms"]) * n_rep * n_f * o["seeds"], "opt/*/fold*/*/seed*.npz"),
        "subsets": (len(o["algorithms"]) * n_rep * n_f, "subsets/*/fold*/*.json"),
        "eval": ((len(cfg["models"]["ml"]) * len(o["algorithms"]) + len(cfg["models"]["dl"]) * n_dl_algs)
                 * n_rep * n_f, "eval/*/*/*/fold*.json"),
        "scenarios": (n_rep * len(sc["names"]) * (sc["seeds"] + 2 * n_models), None),
    }
    prog = []
    for k, (n, pat) in expected.items():
        if pat:
            done = sum(1 for _ in root.glob(pat))
        else:
            done = (sum(1 for _ in root.glob("scen/opt/*/*/seed*.npz"))
                    + sum(1 for _ in root.glob("scen/eval/*/*/*/*.json")))
        prog.append({"Stage": k, "Done": done, "Total": n, "%": f"{100 * min(done, n) / max(n, 1):.0f}"})
    complete = all(p["Done"] >= p["Total"] for p in prog)
    L += ["## Progress" + ("" if complete else " (INTERIM: results below use the runs finished so far)"), "",
          _md(pd.DataFrame(prog)), ""]

    card = _read(tb / "dataset_cleaning.csv")
    if card is not None:
        keep = card[card["Step"].str.strip().isin(["Rows (processed tables)", "Unique rows used", "malware",
                                                   "benign", "Features used"])]
        L += ["## Data", "", _md(keep), ""]

    base = _read(tb / "result_base_model_mean_accuracy.csv")
    if base is not None:
        piv = base.pivot_table(index="Model", columns="Dataset", values="ACC").round(2)
        piv = piv[[REP_LABEL[r] for r in reps if REP_LABEL[r] in piv.columns]]
        piv["Mean"] = piv.mean(1).round(2)
        L += ["## Base learners, all features (accuracy %, mean of outer folds)", "",
              _md(piv.sort_values("Mean", ascending=False).reset_index()), ""]

    ot = _read(st / "optimizer_tests.csv")
    if ot is not None:
        L += ["## Optimizers (final fitness, lower is better)", "",
              "Mean over folds x seeds; rank = Friedman mean rank; p = Wilcoxon vs AMDABC, Holm-adjusted; "
              "A12 > 0.5 means AMDABC tends to be better.", ""]
        for group, algs in TABLE_ORDER.items():
            g = ot[ot.group == group]
            if not len(g):
                continue
            rows = []
            for a in algs:
                r = {"Algorithm": a}
                for rep in reps:
                    x = g[(g.rep == rep) & (g.alg == a)]
                    if len(x):
                        x = x.iloc[0]
                        p = "-" if pd.isna(x.p_holm) else f"{x.p_holm:.1e}"
                        r[REP_LABEL[rep]] = f"{x['mean']:.4f} (rank {x['rank']:.2f}, p {p})"
                rows.append(r)
            L += [f"### {group}", "", _md(pd.DataFrame(rows)), ""]
            best = g.loc[g.groupby("rep")["rank"].idxmin(), ["rep", "alg"]]
            L += ["Best mean rank: " + ", ".join(f"{REP_LABEL[r]} {a}" for r, a in best.itertuples(index=False)), ""]

    for group in ("abc_variants", "metaheuristics"):
        w = _read(tb / f"result_wrapper_{group}_dataset_mean_accuracy.csv")
        if w is None:
            continue
        w = w[["Dataset", "FS Method", "ACC", "F1", "FS"]].copy()
        L += [f"## Wrapper subsets vs all features ({group}; mean of the 7 ML models)", "", _md(w), ""]

    eff = _read(st / "efficiency_full_vs_amdabc.csv")
    if eff is not None:
        e2 = eff.groupby("rep")[["full_n_features", "amdabc_n_features", "fit_speedup", "predict_speedup"]].mean()
        e2.index = [REP_LABEL[r] for r in e2.index]
        L += ["## Efficiency, AMDABC subset vs all features (mean over models)", "",
              _md(e2.reset_index().rename(columns={"index": "Dataset"})), ""]

    f1 = _read(tb / "rev_f1.csv")
    if f1 is not None:
        f1["Dataset"] = f1["rep"].map(REP_LABEL)
        L += ["## Source-shift scenarios (LR, weighted F1 %)", "",
              "CI = 95% bootstrap interval of (AMDABC - all features) in percentage points.", "",
              _md(f1[["scenario", "Dataset", "n_full", "n_selected", "full_F1", "amdabc_F1", "ci_low", "ci_high"]]), ""]

    path = out / "SUMMARY.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
