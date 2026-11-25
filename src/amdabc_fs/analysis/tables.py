"""LaTeX tables in the manuscript's tblr layout (labels and captions unchanged).

Every table is also written as CSV with full precision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .layout import FILTER_LABEL, REP_LABEL, TABLE_ORDER, WRAPPER_ORDER

METRICS = ["ACC", "AUC", "PRE", "REC", "F1"]
HEAD = " & ".join(rf"\textbf{{{m}(\%)}}" for m in METRICS)


def _e(x):
    return f"{x:.2E}"


def _p(x):
    return "-" if not np.isfinite(x) else f"{x:.2E}"


def _f(x, d=2):
    return f"{x:.{d}f}"


def _bold(s, on):
    return rf"\textbf{{{s}}}" if on else s


def _wrap(label, caption, spec, body, scale="0.8"):
    return (f"\\begin{{table}} [h!]\n\t\\centering\n\t\\caption{{{caption}}}\n\t\\label{{{label}}}\n"
            f"\t\\scalebox{{{scale}}}{{\n\t\\begin{{tblr}}{{\n{spec}\t\t}}\n{body}\n\t\\end{{tblr}}\n\t}}\n\\end{{table}}\n")


def _rowspans(start, n_groups, size):
    return "".join(f"\t\t\tcell{{{start + i * size}}}{{1}} = {{r={size}}}{{}},\n" for i in range(n_groups))


# ------------------------------------------------------------------ base -------
def base_models(base: pd.DataFrame, reps, models):
    full = base[base.variant == "full"]
    m = full.groupby(["model", "rep"])[METRICS].mean()
    models = sorted(models)
    lines, csv = [], []
    for mod in models:
        for i, rep in enumerate(reps):
            v = m.loc[(mod, rep)]
            lines.append(f"\t\t{mod if i == 0 else ''} & {REP_LABEL[rep]} & " + " & ".join(_f(v[c]) for c in METRICS))
            csv.append({"Model": mod, "Dataset": REP_LABEL[rep], **{c: v[c] for c in METRICS}})
    n = len(reps)
    spec = ("\t\t\tcells = {c},\n" + _rowspans(2, len(models), n) + "\t\t\thline{2} = {-}{},\n"
            f"\t\t\thline{{{','.join(str(2 + n * i) for i in range(1, len(models)))}}} = {{-}}{{dashed}},\n")
    body = rf"		\textbf{{Model}} & \textbf{{Dataset}} & {HEAD} \\" + "\n" + " \\\\\n".join(lines)
    tex = _wrap("tab:result_base_model_mean_accuracy",
                "Results of the ML models on the original malware detection datasets", spec, body)
    return tex, pd.DataFrame(csv)


def filters(base: pd.DataFrame, reps, filters_, models):
    b = base[base.model.isin(models)]
    m = b.groupby(["rep", "variant"])[METRICS].mean()
    variants = ["full"] + list(filters_)
    lines, csv = [], []
    for rep in reps:
        for i, v in enumerate(variants):
            r = m.loc[(rep, v)]
            lines.append(f"\t\t{REP_LABEL[rep] if i == 0 else ''} & {FILTER_LABEL[v]} & " + " & ".join(_f(r[c]) for c in METRICS))
            csv.append({"Dataset": REP_LABEL[rep], "Filter": FILTER_LABEL[v], **{c: r[c] for c in METRICS}})
    n = len(variants)
    spec = ("\t\t\tcells = {c},\n" + _rowspans(2, len(reps), n) + "\t\t\thline{2} = {-}{},\n"
            f"\t\t\thline{{{','.join(str(2 + n * i) for i in range(1, len(reps)))}}} = {{-}}{{dashed}},\n")
    body = (rf"		\textbf{{\textbf{{Dataset}}}} & \textbf{{Filter}} & {HEAD} \\" + "\n" + " \\\\\n".join(lines))
    tex = _wrap("tab:result_filter_dataset_mean_accuracy",
                "Results of the ML models on the original and filtered Android malware detection datasets", spec, body)
    return tex, pd.DataFrame(csv)


# ------------------------------------------------------------------ optimisation
OPT_CAPTION = {
    "ablation": ("tab:result_opt_ablation", "Optimization results of the original ABC, AMDABC, and AMDABC components on the Android malware detection datasets"),
    "abc_variants": ("tab:result_opt_abcvariants", "Optimization results of the ABC variants on the Android malware detection datasets"),
    "metaheuristics": ("tab:result_opt_metaheuristics", "Optimization results of the metaheuristics on the Android malware detection datasets"),
}
STAT_CAPTION = {
    "ablation": ("tab:result_opt_statistics_ablation", "Statistical results of the original ABC, AMDABC, and AMDABC components on the Android malware detection datasets"),
    "abc_variants": ("tab:result_opt_statistics_abcvariants", "Statistical results of the ABC variants on the Android malware detection datasets"),
    "metaheuristics": ("tab:result_opt_statistics_metaheuristics", "Statistical results of the metaheuristics on the Android malware detection datasets"),
}


def opt_results(st: pd.DataFrame, group, reps):
    algs = TABLE_ORDER[group]
    hdr = "Algorithm & " + " & ".join(f"{REP_LABEL[r]} & " for r in reps) + r" \\" + "\n\t\t& " + \
          " & ".join("Mean & Std." for _ in reps) + r" \\"
    lines = []
    for a in algs:
        cells = []
        for r in reps:
            v = st[(st.rep == r) & (st.alg == a)].iloc[0]
            cells += [_e(v["mean"]), _e(v["std"])]
        lines.append(f"\t\t{a} & " + " & ".join(cells))
    nc = 1 + 2 * len(reps)
    spec = ("\t\t\tcells = {c},\n\t\t\trow{1} = {font=\\bfseries},\n\t\t\trow{2} = {font=\\bfseries},\n"
            "\t\t\tcell{1}{1} = {r=2}{},\n" +
            "".join(f"\t\t\tcell{{1}}{{{2 + 2 * i}}} = {{c=2}}{{}},\n" for i in range(len(reps))) +
            "".join(f"\t\t\tcell{{3}}{{{j}}} = {{font=\\bfseries}},\n" for j in range(2, nc + 1)) +
            f"\t\t\tvline{{{','.join(str(2 + 2 * i) for i in range(len(reps)))}}} = {{-}}{{dashed}},\n"
            f"\t\t\thline{{2}} = {{2-{nc}}}{{dashed}},\n\t\t\thline{{3}} = {{1-{nc}}}{{}},\n")
    lab, cap = OPT_CAPTION[group]
    return _wrap(lab, cap, spec, "\t\t" + hdr.replace("& \\\\", "\\\\").replace("&  &", "& &") + "\n" + " \\\\\n".join(lines))


def opt_statistics(st: pd.DataFrame, group, reps, p_col="p_holm"):
    algs = TABLE_ORDER[group]
    rank = {r: st[st.rep == r].set_index("alg")["rank"] for r in reps}
    total = pd.Series({a: np.mean([rank[r][a] for r in reps]) for a in algs})
    overall = total.rank(method="min").astype(int)
    lines = []
    for a in algs:
        cells = []
        for r in reps:
            v = st[(st.rep == r) & (st.alg == a)].iloc[0]
            cells += [_p(v[p_col]), _f(v["rank"])]
        cells += [_f(total[a]), str(overall[a])]
        lines.append(f"\t\t{a} & " + " & ".join(cells))
    hdr = ("Algorithm & " + " & ".join(f"{REP_LABEL[r]} & " for r in reps) + " & Total & " + r" \\" + "\n\t\t& " +
           " & ".join("$p-$Value & Rank" for _ in reps) + r" & Score & Rank \\")
    nc = 3 + 2 * len(reps)
    spec = ("\t\t\tcells = {c},\n\t\t\trow{1} = {font=\\bfseries},\n\t\t\trow{2} = {font=\\bfseries},\n"
            "\t\t\tcell{1}{1} = {r=2}{},\n" +
            "".join(f"\t\t\tcell{{1}}{{{2 + 2 * i}}} = {{c=2}}{{}},\n" for i in range(len(reps) + 1)) +
            "".join(f"\t\t\tcell{{3}}{{{j}}} = {{font=\\bfseries}},\n" for j in range(2, nc + 1)) +
            f"\t\t\tvline{{{','.join(str(2 + 2 * i) for i in range(len(reps) + 1))}}} = {{-}}{{dashed}},\n"
            f"\t\t\thline{{2}} = {{2-{nc}}}{{dashed}},\n\t\t\thline{{3}} = {{-}}{{}},\n")
    lab, cap = STAT_CAPTION[group]
    hdr = hdr.replace("&  & Total", "& & Total").replace("& \\\\", "\\\\")
    return _wrap(lab, cap, spec, "\t\t" + hdr + "\n" + " \\\\\n".join(lines)), \
        pd.DataFrame({"alg": algs, "total_score": total.values, "overall_rank": overall.values})


# ------------------------------------------------------------------ wrapper ----
WRAP_CAPTION = {
    "abc_variants": ("tab:result_wrapper_abcvariants_dataset_mean_accuracy",
                     "Predictive performance of Android malware detection: Baseline vs. ABC-variant wrapper feature selection"),
    "metaheuristics": ("tab:result_wrapper_metaheuristics_dataset_mean_accuracy",
                       "Predictive performance of Android malware detection: Baseline vs. metaheuristic wrapper feature selection"),
}


def wrapper(base, ev, sizes, n_feat, group, reps, models):
    b = base[(base.variant == "full") & base.model.isin(models)].groupby("rep")[METRICS].mean()
    e = ev[ev.model.isin(models)].groupby(["rep", "alg"])[METRICS].mean()
    sz = sizes.groupby(["rep", "alg"])["size"].mean()
    order = WRAPPER_ORDER[group]
    lines, csv = [], []
    for rep in reps:
        lines.append(f"\t\t{REP_LABEL[rep]} & - & " + " & ".join(_f(b.loc[rep, c]) for c in METRICS) + f" & {n_feat[rep]}")
        csv.append({"Dataset": REP_LABEL[rep], "FS Method": "-", **b.loc[rep].to_dict(), "FS": n_feat[rep]})
        for a in order:
            r = e.loc[(rep, a)]
            is_ref = a == "AMDABC"
            cells = [_bold(_f(r[c]), is_ref) for c in METRICS] + [_bold(f"{sz[(rep, a)]:.0f}", is_ref)]
            lines.append(f"\t\t & {a} & " + " & ".join(cells))
            csv.append({"Dataset": REP_LABEL[rep], "FS Method": a, **r.to_dict(), "FS": sz[(rep, a)]})
    n = 1 + len(order)
    spec = ("\t\t\tcells = {c},\n" + _rowspans(2, len(reps), n) + "\t\t\thline{2} = {-}{},\n"
            f"\t\t\thline{{{','.join(str(2 + n * i) for i in range(1, len(reps)))}}} = {{-}}{{dashed}},\n")
    body = (rf"		\textbf{{Dataset}} & \textbf{{FS Method}} & {HEAD} & \textbf{{FS}} \\" + "\n" + " \\\\\n".join(lines))
    lab, cap = WRAP_CAPTION[group]
    return _wrap(lab, cap, spec, body, scale="0.80"), pd.DataFrame(csv)


# ------------------------------------------------------------------ simple -----
def simple(label, caption, cols, rows, align=None):
    align = align or "l" + "r" * (len(cols) - 1)
    body = " \\\\\n".join(" & ".join(map(str, r)) for r in rows)
    return (f"\\begin{{table}}[htbp]\n\\centering\\small\n\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{align}}}\n\\hline\n{' & '.join(cols)} \\\\ \\hline\n{body} \\\\\n\\hline\n"
            "\\end{tabular}\n\\end{table}\n")
