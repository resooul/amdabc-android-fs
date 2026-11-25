# Results of the run reported in the article

Produced by `python run.py --config config/default.yaml --workers 20` on an
Intel i9-12900K / RTX 3080 Ti, Windows 10, Python 3.11.18, 24–25 November 2026
(machine and package records in `environment/`, full log in `run.log`).
`config_used.yaml` is the exact configuration of the run.

| Path | Content |
|---|---|
| `data/` | Cleaned tables (`{rep}.npz`: X, y, feature names, source sets), outer folds (`{rep}_folds.csv`), scenario splits (`{rep}_scenarios.json`), `data_card.json` (counts per cleaning step, SHA-256 of inputs and outputs) |
| `base/{rep}/{feature set}/{model}/fold{k}.json` | Base learners: all metrics, fit time, inference time, model size |
| `opt/{rep}/fold{k}/{optimizer}/seed{s}.npz` | One optimizer run: best-so-far curve per evaluation, best mask, candidate features, best fitness, time, number of classifier fits |
| `subsets/{rep}/fold{k}/{optimizer}.json` | Consensus subset of the 30 runs (indices and opcode names) |
| `eval/{rep}/{optimizer}/{model}/fold{k}.json` | Models trained on a consensus subset |
| `scen/` | Source-shift scenarios: selection runs, subsets, predictions and metrics |
| `report/` | Figures, tables (LaTeX + CSV), statistics, `SUMMARY.md` |
| `benchmark.json` | Timing benchmark used to plan the run |

`python run.py --config config/default.yaml --stage report` rebuilds
`report/` from these files; the rebuilt tables and statistics are identical to
the stored ones.

Two tasks were logged as failed in `run.log` (a file-sync client briefly
locked a temporary marker file after the result had been written); their result
files are complete and were kept, and the stage was resumed.
