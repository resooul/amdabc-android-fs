# AMDABC: wrapper feature selection for Android malware detection

Code, data and results of the article *AMDABC: An Artificial Bee Colony
Algorithm for Android Malware Detection* (R. Özdemir, R. S. Arslan,
M. Taşyürek, V. Aslantaş). One command rebuilds every table, figure and
statistical test of the article from the raw opcode tables; the results of
the run reported in the article are included.

## Contents

| Path | What |
|---|---|
| `run.py` | Entry point: runs the stages below, resumable |
| `config/default.yaml` | Protocol of the article (all settings in one file) |
| `config/smoke.yaml` | Same pipeline with small numbers, for a quick check |
| `src/amdabc_fs/` | Data preparation, filters, wrapper fitness, optimizers, ML/DL models, statistics, figures, tables |
| `src/amdabc_fs/optim/` | AMDABC, its ablation variants, the ABC variants and MPA (`paper_abc.py`, `paper_mpa.py`), FE budget (`budget.py`) |
| `tests/` | 33 tests: FE budget, determinism, leakage, consensus rule, optimizer correctness |
| `data/raw/` | Opcode presence tables, one file per representation and source, with SHA-256 sums |
| `results/` | Complete output of the run reported in the article (see `results/README.md`) |
| `results/report/` | Figures, LaTeX/CSV tables, statistics and `SUMMARY.md` |

## Setup

Python 3.11.

```bash
conda create -n amdabc python=3.11 -y && conda activate amdabc      # or: python3.11 -m venv .venv
pip install -r requirements.txt
python -m pytest -q
```

`requirements.txt` installs the CPU build of PyTorch. For a GPU, install the
CUDA build afterwards (the article's run used `torch 2.14.1+cu126`):
`pip install --force-reinstall --no-deps torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126`.

## Reproduce

```bash
# rebuild figures, tables and statistics from the stored results (about 2 min)
python run.py --config config/default.yaml --stage report

# rerun everything from the raw tables into a new directory
#   (set results_dir in a copy of config/default.yaml, e.g. results_rerun)
python run.py --config config/my_rerun.yaml --workers 20

python run.py --config config/smoke.yaml --workers 4                 # quick end-to-end check
python run.py --config config/default.yaml --status                  # progress per stage
python run.py --config config/default.yaml --benchmark --workers 20  # machine info + run-time estimate
```

Stages (in the order of the article; `--stage NAME` runs one):

| Stage | What | Tasks |
|---|---|---|
| `prepare` | Clean data, 5 outer folds, source-shift splits, data card | – |
| `base` | 7 ML + 4 DL models on all features and on FScore/Chi2/Pearson top-100 | 660 |
| `opt` | 15 optimizers x 3 representations x 5 folds x 30 seeds, 500 FE each | 6,750 |
| `subsets` | Z-score consensus of the 30 runs of each fold | 225 |
| `eval` | 7 ML models on every subset, 4 DL models on the AMDABC subset | 1,635 |
| `scenarios` | AMDABC selection on the training sources only; all features vs subset | 384 |
| `report` | Tables, figures, statistics, `SUMMARY.md` | – |

The article's run took about 2.5 h on an Intel i9-12900K (24 threads, 20
workers) with an RTX 3080 Ti for the DL models; optimization accounts for
2 h 15 min.

**Interrupting is safe.** Every task writes one result file atomically; run
the same command again and finished tasks are skipped. Runs are deterministic
on the CPU (an interrupted and resumed run gives bit-identical results); DL
models trained on a GPU can differ in the last digits. The first run stores its
configuration in `results_dir/config_used.yaml`, and a later run with different
settings on the same directory stops instead of mixing results. Every start
records the hardware and package versions in `results_dir/environment/`.

## Protocol

- **Data.** DVM (Dalvik), JVM (Java bytecode) and Hybrid (both) opcode
  presence vectors of 13,799 applications; malware from Drebin (which contains
  all Genome samples), benign applications from CICInvesAndMal2019,
  CIC-MalDroid2020 and the Omer collection. Cleaning (`src/amdabc_fs/data/prepare.py`):
  identifier column, copied/constant/identical columns, empty vectors,
  label-conflicting vectors and duplicate rows are removed, leaving 5,204 /
  3,427 / 6,765 unique vectors with 151 / 109 / 260 features. Every step is
  counted in `results/report/tables/dataset_cleaning.csv`.
- **Evaluation.** Stratified 5-fold outer CV. Filters, the wrapper and the
  consensus see only the outer training fold; leakage checks run before every
  training stage.
- **Wrapper.** FScore top-100 candidates; fitness
  `0.9 * (1 - balanced accuracy) + 0.1 * |S| / D` with a 5-fold inner CV of
  logistic regression; position > 0.5 selects a feature; empty subset = 1.0.
- **Budget.** Every optimizer stops after exactly 500 fitness evaluations
  (`optim/budget.py`); best values and convergence curves are taken from the
  evaluated subsets, not from the optimizer's own bookkeeping. Convergence
  plots use one "iteration" = 10 evaluations.
- **Subsets.** For each fold, features selected in at least the mean share of
  the 30 runs are kept (z-score rule with sigma = 0).
- **Models.** XGB, LR, GBM, RF, CatBoost, LDA and MLP with the fixed
  hyperparameters of `tables/ml_hyperparameters.tex`; DL baselines DL-MLP,
  1-D CNN, BiLSTM and a patch Transformer (`tables/dl_hyperparameters.tex`).
- **Source shift.** S1–S4 train on two (S4: one) benign sources and test on
  benign sources never seen in training; malware is split 70/30.
- **Statistics.** Friedman ranks and Nemenyi CD, Wilcoxon vs AMDABC with Holm
  correction, Vargha–Delaney A12, bootstrap 95% intervals, Nadeau–Bengio
  corrected t-tests for classifiers, paired bootstrap intervals for scenario
  differences.

## Main results (from `results/report/SUMMARY.md`)

Final fitness (mean over 5 folds x 30 seeds, lower is better):

| | DVM | JVM | Hybrid |
|---|---|---|---|
| **AMDABC** | **0.0813** | **0.1392** | **0.0730** |
| MPA | 0.0849 | 0.1422 | 0.0763 |
| PSO | 0.0884 | 0.1635 | 0.0807 |
| ABCX | 0.0933 | 0.1759 | 0.0842 |
| canonical ABC | 0.1041 | 0.1985 | 0.0943 |

AMDABC has the best mean Friedman rank in all three comparisons (ablation, ABC
variants, metaheuristics) on all three representations. Against the ABC
variants and the other metaheuristics, Holm-adjusted Wilcoxon p < 1e-10 except
MPA on JVM (p = 0.14). In the ablation, the best-guided search alone (AMDABC2)
comes close (p = 0.01 / 0.06 / 0.33), while the adaptive coefficient and the
restart alone do not improve on canonical ABC.
Its consensus subsets keep 30.6 / 41.6 / 32.8 features (DVM / JVM / Hybrid);
the mean accuracy of the 7 ML models is 94.61 / 94.20 / 95.95% versus
95.79 / 94.40 / 96.71% with all features, and model training is about 2.9 times
faster on DVM and Hybrid. Under source shift the subset loses 0.6–4.0 points
of weighted F1 with logistic regression. Full numbers: `results/report/`.

## Optimizer code

- `optim/paper_abc.py` and `optim/paper_mpa.py` are generated from the authors'
  original implementation by `tools/port_paper_abc.py` and
  `tools/port_paper_mpa.py` (source SHA-1 checked). Class bodies are verbatim;
  the only changes are marked `PATCH`: the AMDABC global-restart index, the
  FE-based schedule of the adaptive coefficient, and the solution/fitness
  pairing in MPA.
- PSO, SSO, HHO and GTO (gorilla troops, mealpy `AGTO`) and the `Optimizer`
  base class are unmodified code from mealpy 3.0.3 (MIT licence), vendored in
  `optim/mealpy_core/` by `tools/vendor_mealpy.py`; mealpy itself is not
  required. All 15 optimizers give bit-identical results to mealpy 3.0.3.

## Citation and licence

Please cite the article (see `CITATION.cff`). Code: MIT licence (`LICENSE`);
`optim/mealpy_core/` keeps the mealpy MIT licence. The opcode tables were
extracted from the public datasets named above; please also cite their
original publications when using them.
