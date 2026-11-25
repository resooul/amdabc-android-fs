"""Figures, reproducing the archived notebooks' plotting code (sizes, styles,
labels, legends, file names and savefig arguments).

Sources: 5-FS-OPT-Results*.ipynb (convergence, boxplots, alpha) and
ML-Results-2.ipynb (bar plots). The convergence/boxplot notebooks ran with
plt.style.use('fivethirtyeight') followed by sns.set_style('whitegrid'); the
bar-plot notebook used matplotlib defaults. Each figure re-creates its style
in a context so figures do not influence each other.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .layout import ACC_BAR_ORDER, FEATURE_BAR_ORDER, FIGURE_ORDER, GROUP_STEM, REP_LABEL  # noqa: E402

SAVE = dict(bbox_inches="tight", facecolor="white", edgecolor="black", transparent=True)


@contextmanager
def opt_style():
    with matplotlib.rc_context():
        plt.style.use("fivethirtyeight")
        sns.set_style("whitegrid")
        yield


@contextmanager
def bar_style():
    with matplotlib.rc_context():
        plt.style.use("default")
        yield


def _save(fig, out: Path, cfg):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=cfg["figures"]["dpi"], **SAVE)
    plt.close(fig)


# ------------------------------------------------------------ convergence -----
def convergence_table(curves: dict, rep: str, algs: list[str], fpi: int, max_fe: int) -> pd.DataFrame:
    """Mean best-so-far fitness per 'iteration' (= fpi FE), with the notebook's baseline row."""
    T = max_fe // fpi
    cols = {}
    for a in algs:
        c = curves[(rep, a)]                         # runs x max_fe
        idx = np.arange(1, T + 1) * fpi - 1          # best after fpi, 2*fpi, ... FE
        cols[a] = c[:, idx].mean(0)
    df = pd.DataFrame(cols)
    base = pd.DataFrame({a: df.iloc[0].max() for a in algs}, index=[-1])   # notebook: max of first row
    df = pd.concat([base, df])
    df.index = range(len(df))
    return df


def convergence(F: pd.DataFrame, group: str, rep: str, cfg, out_dir: Path):
    algs = FIGURE_ORDER[group]
    with opt_style():
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(F["AMDABC"], ls="-", color="black", lw=3, label="AMDABC")
        for a in algs[1:]:
            ax.plot(F[a], ls="-.", lw=2, label=a)
        ax.tick_params(axis="y", which="major", color="black", labelsize=20)
        ax.tick_params(axis="x", which="major", color="black", labelsize=20)
        plt.ylabel("Average Fitness Values", fontsize=20)
        plt.xlabel("Iterations", fontsize=20)
        plt.yscale("log")
        ax.legend(fontsize=15)
        _save(fig, out_dir / f"result_opt_convergence_{GROUP_STEM[group]}_{rep}_fscore.png", cfg)


# ------------------------------------------------------------ box plots -------
def boxplot(runs: pd.DataFrame, group: str, rep: str, cfg, out_dir: Path):
    algs = FIGURE_ORDER[group]
    df = pd.DataFrame({a: pd.Series(runs[(runs.rep == rep) & (runs.alg == a)].sort_values(["fold", "seed"])
                                    ["fitness"].to_numpy()) for a in algs})       # unequal counts -> NaN (interim)
    df_long = df.melt(var_name="Algorithm", value_name="Fitness").dropna()
    with opt_style():
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.boxplot(x="Algorithm", y="Fitness", data=df_long, ax=ax, hue="Algorithm", legend=False,
                    palette="pastel",
                    flierprops=dict(marker="D", markersize=5, markerfacecolor="gray", markeredgecolor="none"),
                    medianprops=dict(linestyle="-", linewidth=1.5, color="firebrick"),
                    boxprops=dict(linestyle="-", linewidth=1.5, edgecolor="black"),
                    whiskerprops=dict(linestyle="--", linewidth=1.5, color="gray"),
                    capprops=dict(linestyle="-", linewidth=1.5, color="black"))
        ax.set_xlabel("Optimization Algorithms", fontsize=20, labelpad=12)
        ax.set_ylabel("Fitness Values", fontsize=20, labelpad=15)
        ax.tick_params(axis="x", labelsize=15, rotation=45)
        plt.xticks(ticks=range(len(df.columns)), labels=df.columns)
        _save(fig, out_dir / f"result_opt_boxplot_{GROUP_STEM[group]}_{rep}_fscore.png", cfg)


# ------------------------------------------------------------ bar plots -------
def _bars(df_plot, x, order, xlabel, out, cfg, ylabel="Mean Accuracy"):
    with bar_style():
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.barplot(data=df_plot, x=x, y="ACC", order=order, hue=x, hue_order=order, legend=False,
                    palette="pastel", edgecolor="black", linewidth=1.0, ax=ax)
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha="center", va="center", xytext=(0, 10), textcoords="offset points",
                            fontweight="bold", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=16, labelpad=15)
        ax.set_xlabel(xlabel, fontsize=16, labelpad=15)
        ax.tick_params(axis="x", labelsize=14, rotation=45)
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        sns.despine()
        ax.set_ylim(0, df_plot["ACC"].max() * 1.1)
        plt.tight_layout()
        _save(fig, out, cfg)


def base_models(base: pd.DataFrame, cfg, out_dir: Path):
    full = base[base.variant == "full"]
    df_plot = full.groupby("model", as_index=False)["ACC"].mean().sort_values("ACC", ascending=False)
    df_plot = df_plot.rename(columns={"model": "Model"})
    _bars(df_plot, "Model", list(df_plot["Model"]), "Model", out_dir / "result_base_model_mean_accuracy.png", cfg)


def filters(base: pd.DataFrame, cfg, out_dir: Path, models):
    b = base[base.model.isin(models)]
    lab = {"full": "", "fscore": " FScore", "chi2": " Chi2", "pearson": " Pearson"}
    rows, order = [], []
    for rep in cfg["representations"]:
        for v in ["full"] + cfg["filters"]["methods"]:
            name = REP_LABEL[rep] + lab[v]
            order.append(name)
            rows.append({"Dataset": name, "ACC": b[(b.rep == rep) & (b.variant == v)]["ACC"].mean()})
    _bars(pd.DataFrame(rows), "Dataset", order, "Dataset", out_dir / "result_filter_dataset_mean_accuracy.png", cfg)


def wrapper_accuracy(base, ev, group, rep, cfg, out_dir, models):
    b = base[(base.rep == rep) & (base.variant == "full") & base.model.isin(models)]
    e = ev[(ev.rep == rep) & ev.model.isin(models)]
    order = [REP_LABEL[rep]] + ACC_BAR_ORDER[group]
    rows = [{"Dataset": order[0], "ACC": b["ACC"].mean()}]
    rows += [{"Dataset": a, "ACC": e[e.alg == a]["ACC"].mean()} for a in ACC_BAR_ORDER[group]]
    _bars(pd.DataFrame(rows), "Dataset", order, "Dataset",
          out_dir / f"result_{rep}_{GROUP_STEM[group]}_dataset_mean_accuracy.png", cfg)


def feature_size(sizes: pd.DataFrame, cfg, out_dir: Path):
    """Mean consensus-subset size over the outer folds."""
    m = sizes.groupby(["rep", "alg"], as_index=False)["size"].mean()
    m["Dataset"] = m["rep"].map(REP_LABEL)
    m = m.rename(columns={"alg": "Algorithm", "size": "Feature Size"})
    for group_name, order in FEATURE_BAR_ORDER.items():
        df_subset = m[m.Algorithm.isin(order)]
        with bar_style():
            fig, ax = plt.subplots(figsize=(15, 10))
            sns.barplot(data=df_subset, x="Algorithm", y="Feature Size", hue="Dataset", order=order,
                        hue_order=["DVM", "JVM", "Hybrid"], palette="pastel", edgecolor="black",
                        linewidth=1.0, ax=ax)
            for p in ax.patches:
                h = p.get_height()
                if h > 0:
                    ax.annotate(f"{int(round(h))}", (p.get_x() + p.get_width() / 2., h), ha="center",
                                va="center", xytext=(0, 10), textcoords="offset points",
                                fontweight="bold", fontsize=11)
            ax.set_ylabel("Feature Size", fontsize=15, labelpad=15)
            ax.set_xlabel("Algorithm", fontsize=15, labelpad=15)
            ax.tick_params(axis="x", labelsize=13)
            ax.tick_params(axis="y", labelsize=13)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            plt.legend(title="Dataset", title_fontsize="13", fontsize="12", loc="upper right", ncol=4)
            sns.despine()
            ax.set_ylim(0, df_subset["Feature Size"].max() * 1.15)
            plt.tight_layout()
            _save(fig, out_dir / f"result_{group_name}_feature_size_barplot.png", cfg)


def alpha_behaviour(cfg, out_dir: Path, max_iterations=50):
    """Cosine schedule of alpha (paper Fig. alpha), unchanged."""
    def calc(t, T):
        return 0.5 * (1.0 - 0.1) * (1 + np.cos((t / T) * np.pi)) + 0.1
    epochs = np.arange(0, max_iterations + 1)
    vals = [calc(e, max_iterations) for e in epochs]
    with bar_style():
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.plot(epochs, vals, ls="-", color="blue", lw=3, label=r"Adaptive $\alpha$ Value")
        ax.scatter(epochs, vals, color="red", s=15, zorder=5)
        ax.tick_params(axis="y", which="major", color="black", labelsize=15)
        ax.tick_params(axis="x", which="major", color="black", labelsize=15)
        plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        plt.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5)
        plt.ylabel(r"Parameter Value ($\alpha$)", fontsize=20)
        plt.xlabel("Iterations", fontsize=15)
        plt.yscale("linear")
        ax.legend(fontsize=15)
        plt.grid(True, linestyle=":", alpha=0.6)
        _save(fig, out_dir / "amdabc_alpha_behaviour.png", cfg)


def cd_diagram(st_rows: pd.DataFrame, group: str, rep: str, cfg, out_dir: Path):
    """Supplementary: mean Friedman ranks with the Nemenyi critical difference."""
    r = st_rows.sort_values("rank")
    cd = float(r["cd"].iloc[0])
    with bar_style():
        fig, ax = plt.subplots(figsize=(10, 2 + 0.35 * len(r)))
        ax.hlines(range(len(r)), 1, r["rank"], color="lightgray")
        ax.plot(r["rank"], range(len(r)), "o", color="black")
        ax.set_yticks(range(len(r)), r["alg"])
        best = r["rank"].min()
        ax.axvspan(best, best + cd, color="tab:blue", alpha=0.15, label=f"CD = {cd:.2f}")
        ax.set_xlabel("Mean rank (Friedman)")
        ax.invert_yaxis()
        ax.legend(loc="lower right")
        sns.despine()
        _save(fig, out_dir / "supplementary" / f"cd_{GROUP_STEM[group]}_{rep}.png", cfg)
