"""Stage 'report': statistics, tables and figures from the stored results.

Pure function of the result files: delete results/report/ and rerun to
rebuild everything. Outputs:
  report/figures/*.png      (manuscript file names)
  report/tables/*.tex|.csv  (manuscript labels and layout)
  report/stats/*.csv        (all tests, incl. Holm, A12, bootstrap CI, CD)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ..io import read_json, write_csv
from . import collect, figures, settings_tables, stats, summary, tables
from .layout import REP_LABEL

log = logging.getLogger("amdabc")


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def report(cfg, parts=("base", "opt", "eval", "scenarios")):
    root = Path(cfg["results_dir"])
    out = root / "report"
    fig_dir, tab_dir, st_dir = out / "figures", out / "tables", out / "stats"
    reps = cfg["representations"]
    ml = cfg["models"]["ml"]
    wmodels = ml if cfg["models"]["wrapper_table_models"] == "ml" else ml + cfg["models"]["dl"]
    card = read_json(root / "data" / "data_card.json")
    write_csv(st_dir / "data_card.csv", pd.DataFrame(card).T.drop(columns=["input_sha256", "scenarios"], errors="ignore"))

    figures.alpha_behaviour(cfg, fig_dir)
    for name, (tex, csv) in {
        "ml_hyperparameters": settings_tables.ml_hyperparameters(ml),
        "dl_hyperparameters": settings_tables.dl_hyperparameters(cfg),
        "optimizer_params": settings_tables.optimizer_params(cfg),
        "dataset_cleaning": settings_tables.dataset_cleaning(card, reps),
    }.items():
        _write(tab_dir / f"{name}.tex", tex)
        write_csv(tab_dir / f"{name}.csv", csv, index=False)

    def safe(name, fn):
        """Interim reports run on partial results: a section that cannot be
        built yet is skipped with a warning instead of stopping the report."""
        try:
            return fn()
        except Exception as e:                       # noqa: BLE001
            log.warning("report: %s skipped (results incomplete? %s: %s)", name, type(e).__name__, e)
            return None

    base = collect.base(cfg) if "base" in parts else None

    def _base():
        figures.base_models(base, cfg, fig_dir)
        figures.filters(base, cfg, fig_dir, ml)
        tex, csv = tables.base_models(base, reps, sorted(base.model.unique()))
        _write(tab_dir / "result_base_model_mean_accuracy.tex", tex)
        write_csv(tab_dir / "result_base_model_mean_accuracy.csv", csv, index=False)
        tex, csv = tables.filters(base, reps, cfg["filters"]["methods"], ml)
        _write(tab_dir / "result_filter_dataset_mean_accuracy.tex", tex)
        write_csv(tab_dir / "result_filter_dataset_mean_accuracy.csv", csv, index=False)
        write_csv(st_dir / "base_all_metrics.csv", base, index=False)
    if base is not None and len(base):
        safe("base learners", _base)

    runs, curves = collect.opt(cfg) if "opt" in parts else (pd.DataFrame(), {})
    all_st = []
    if len(runs):
        write_csv(st_dir / "opt_runs.csv", runs, index=False)
        o = cfg["optimization"]
        for group, algs in cfg["groups"].items():
            def _group(group=group, algs=algs):
                st = pd.concat([stats.optimizer_tests(runs, algs, rep, n_boot=cfg["stats"]["bootstrap"],
                                                      seed=cfg["stats"]["seed"]) for rep in reps])
                st["group"] = group
                all_st.append(st)
                _write(tab_dir / f"result_opt_{group}.tex", tables.opt_results(st, group, reps))
                tex, tot = tables.opt_statistics(st, group, reps)
                _write(tab_dir / f"result_opt_statistics_{group}.tex", tex)
                write_csv(tab_dir / f"result_opt_statistics_{group}_total.csv", tot, index=False)
                for rep in reps:
                    F = figures.convergence_table(curves, rep, algs, o["fe_per_iteration"], o["max_fe"])
                    write_csv(st_dir / "convergence" / f"{group}_{rep}.csv", F, index_label="Iteration")
                    figures.convergence(F, group, rep, cfg, fig_dir)
                    figures.boxplot(runs, group, rep, cfg, fig_dir)
                    figures.cd_diagram(st[st.rep == rep], group, rep, cfg, fig_dir)
            safe(f"optimizers/{group}", _group)
        if all_st:
            write_csv(st_dir / "optimizer_tests.csv", pd.concat(all_st), index=False)
        # runtime per optimizer (efficiency, same FE budget)
        write_csv(st_dir / "optimizer_runtime.csv",
                  runs.groupby(["rep", "alg"])[["seconds", "n_fits", "n_selected"]].mean().reset_index(), index=False)

    if "eval" in parts and (root / "eval").exists():
        def _eval():
            ev, sizes = collect.evals(cfg), collect.subsets(cfg)
            b = base if base is not None else collect.base(cfg)
            n_feat = {r: card[r]["n_features"] for r in reps}
            write_csv(st_dir / "eval_all_metrics.csv", ev, index=False)
            write_csv(st_dir / "subset_sizes.csv", sizes, index=False)
            figures.feature_size(sizes, cfg, fig_dir)
            for group in ("abc_variants", "metaheuristics"):
                tex, csv = tables.wrapper(b, ev, sizes, n_feat, group, reps, wmodels)
                _write(tab_dir / f"result_wrapper_{group}_dataset_mean_accuracy.tex", tex)
                write_csv(tab_dir / f"result_wrapper_{group}_dataset_mean_accuracy.csv", csv, index=False)
                for rep in reps:
                    figures.wrapper_accuracy(b, ev, group, rep, cfg, fig_dir, wmodels)
            _classifier_tests(cfg, b, ev, card, st_dir)
            _efficiency(b, ev, st_dir, tab_dir)
        safe("wrapper subsets", _eval)

    if "scenarios" in parts and (root / "scen" / "eval").exists():
        safe("scenarios", lambda: _scenarios(cfg, tab_dir, st_dir))
    path = summary.write(cfg, out)
    log.info("report written to %s; summary: %s", out, path)


def _classifier_tests(cfg, base, ev, card, st_dir):
    """Nadeau-Bengio: subset vs all features, per model and representation."""
    rows = []
    k = cfg["split"]["n_folds"]
    for rep in cfg["representations"]:
        n = card[rep]["rows_clean"]
        n_te, n_tr = n / k, n - n / k
        for alg in cfg["optimization"]["algorithms"]:
            for m in sorted(ev.model.unique()):
                a = base[(base.rep == rep) & (base.variant == "full") & (base.model == m)].sort_values("fold")
                b = ev[(ev.rep == rep) & (ev.alg == alg) & (ev.model == m)].sort_values("fold")
                if len(a) != len(b) or not len(a):
                    continue
                for metric in ("ACC", "F1", "BACC", "MCC"):
                    t, p = stats.nadeau_bengio(b[metric], a[metric], n_tr, n_te)
                    rows.append({"rep": rep, "alg": alg, "model": m, "metric": metric,
                                 "full_mean": a[metric].mean(), "subset_mean": b[metric].mean(),
                                 "diff": b[metric].mean() - a[metric].mean(), "t": t, "p": p})
    write_csv(st_dir / "classifier_nadeau_bengio.csv", pd.DataFrame(rows), index=False)


def _efficiency(base, ev, st_dir, tab_dir):
    """Fit time, per-sample inference time and model size: all features vs AMDABC subset."""
    cols = ["n_features", "fit_seconds", "predict_us_per_sample", "model_bytes"]
    a = base[base.variant == "full"].groupby(["rep", "model"])[cols].mean().add_prefix("full_")
    b = ev[ev.alg == "AMDABC"].groupby(["rep", "model"])[cols].mean().add_prefix("amdabc_")
    df = a.join(b).reset_index()
    df["fit_speedup"] = df["full_fit_seconds"] / df["amdabc_fit_seconds"]
    df["predict_speedup"] = df["full_predict_us_per_sample"] / df["amdabc_predict_us_per_sample"]
    write_csv(st_dir / "efficiency_full_vs_amdabc.csv", df, index=False)
    rows = [[REP_LABEL[r.rep], r.model, f"{r.full_n_features:.0f}/{r.amdabc_n_features:.0f}",
             f"{r.full_fit_seconds:.4f}", f"{r.amdabc_fit_seconds:.4f}",
             f"{r.full_predict_us_per_sample:.2f}", f"{r.amdabc_predict_us_per_sample:.2f}"]
            for r in df.itertuples()]
    _write(tab_dir / "efficiency_full_vs_amdabc.tex", tables.simple(
        "tab:efficiency", "Training and inference cost with all features and with the AMDABC subset "
        "(mean over outer folds; inference in microseconds per sample).",
        ["Dataset", "Model", "Features", "Fit (s) full", "Fit (s) AMDABC", "Infer full", "Infer AMDABC"], rows))


def _scenarios(cfg, tab_dir, st_dir):
    df, preds, sel, sizes = collect.scen(cfg)
    write_csv(st_dir / "scenario_all_metrics.csv", df, index=False)
    wf1 = lambda y, p: 100 * f1_score(y, p, average="weighted")   # noqa: E731
    rows_f1, rows_dl, rows_t, csv = [], [], [], []
    for sc in cfg["scenarios"]["names"]:
        for rep in cfg["representations"]:
            case = f"{sc} {REP_LABEL[rep]}"
            full = preds[(rep, sc, "full", "LR")]
            amd = preds[(rep, sc, "AMDABC", "LR")]
            lo, hi = stats.paired_bootstrap_diff(full["y"], full["pred"], amd["pred"], wf1,
                                                 cfg["stats"]["bootstrap"], cfg["stats"]["seed"])
            def g(v, m, c):
                hit = df[(df.rep == rep) & (df.scenario == sc) & (df.variant == v) & (df.model == m)][c]
                return hit.iloc[0] if len(hit) else float("nan")
            nf, ns = sizes[(rep, sc)]
            rows_f1.append([case, f"{nf}/{ns}", f"{g('full', 'LR', 'F1'):.2f}", f"{g('AMDABC', 'LR', 'F1'):.2f}",
                            f"[{lo:.2f}, {hi:.2f}]"])
            csv.append({"scenario": sc, "rep": rep, "n_full": nf, "n_selected": ns,
                        "full_F1": g("full", "LR", "F1"), "amdabc_F1": g("AMDABC", "LR", "F1"), "ci_low": lo, "ci_high": hi})
            mlp = "DL-MLP" if "DL-MLP" in df.model.unique() else "MLP"
            rows_dl.append([case] + [f"{g('full', m, 'F1'):.2f}" for m in cfg["models"]["dl"]] +
                           [f"{g('AMDABC', 'LR', 'F1'):.2f}", f"{g('AMDABC', 'LR', 'REC_malware'):.2f}"])
            s = sel[(sel.rep == rep) & (sel.scenario == sc)]["seconds"].mean()
            rows_t.append([case, f"{s:.2f}", f"{g('full', 'LR', 'fit_seconds'):.4f}",
                           f"{g('AMDABC', 'LR', 'fit_seconds'):.4f}", f"{g('full', mlp, 'fit_seconds'):.2f}"])
    write_csv(tab_dir / "rev_f1.csv", pd.DataFrame(csv), index=False)
    _write(tab_dir / "rev_f1.tex", tables.simple(
        "tab:rev_f1", "Weighted F1 and selected-feature counts. Difference intervals are in percentage points "
        "and compare AMDABC with full-feature LR.",
        ["Case", "Full/selected", "Full F1", "AMDABC F1", r"Difference 95\% CI"], rows_f1))
    _write(tab_dir / "rev_dl.tex", tables.simple(
        "tab:rev_dl", "DL baselines on all features versus LR on the AMDABC subset (weighted F1 and malware recall, %).",
        ["Case"] + [f"{m} F1" for m in cfg["models"]["dl"]] + ["AMDABC F1", "AMDABC recall"], rows_dl))
    _write(tab_dir / "rev_time.tex", tables.simple(
        "tab:rev_time", "Mean selection time per run and classifier fit times in seconds.",
        ["Case", "Selection", "Full LR fit", "Selected LR fit", "MLP fit"], rows_t))
    sizes_rows = []
    for sc in cfg["scenarios"]["names"]:
        card = read_json(Path(cfg["results_dir"]) / "data" / "data_card.json")
        r0 = cfg["representations"][0]
        sizes_rows.append([sc, card[r0]["scenarios"][sc]["train"], card[r0]["scenarios"][sc]["test"]])
    _write(tab_dir / "rev_sources.tex", tables.simple(
        "tab:rev_sources", f"Source-partition sizes ({REP_LABEL[cfg['representations'][0]]}, unique vectors).",
        ["Scenario", "Training rows", "Test rows"], sizes_rows))
