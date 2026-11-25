"""Run the experiments. Safe to interrupt: rerun the same command to continue.

Examples
  python run.py --config config/default.yaml --workers 20            # everything
  python run.py --config config/default.yaml --stage opt --workers 20
  python run.py --config config/smoke.yaml --workers 2               # quick check
  python run.py --config config/default.yaml --status                # progress only
  python run.py --config config/default.yaml --benchmark --workers 20 # machine info + time estimate

Stages (in order): prepare, base, opt, subsets, eval, scenarios, report.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from amdabc_fs import config as C  # noqa: E402
from amdabc_fs import stages as S  # noqa: E402
from amdabc_fs.data.prepare import prepare  # noqa: E402
from amdabc_fs import envinfo  # noqa: E402
from amdabc_fs.checks import check_all  # noqa: E402
from amdabc_fs.io import clean_tmp  # noqa: E402
from amdabc_fs.tasks import run_tasks  # noqa: E402

STAGES = ["prepare", "base", "opt", "subsets", "eval", "scenarios", "report"]


def setup_logging(results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    log = logging.getLogger("amdabc")
    log.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(results_dir / "run.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def _only(tasks, models):
    if not models:
        return tasks
    keep = set(models)
    return [t for t in tasks if t.key.split("/")[-2 if t.key.startswith(("base", "eval")) else -1] in keep]


def task_lists(cfg, models=None):
    opt_s, sub_s, ev_s = S.scenario_tasks(cfg)
    return {"base": _only(S.base_tasks(cfg), models), "opt": S.opt_tasks(cfg), "subsets": S.subset_tasks(cfg),
            "eval": _only(S.eval_tasks(cfg), models), "scenarios": opt_s + sub_s + _only(ev_s, models)}


def status(cfg):
    for name, tl in task_lists(cfg).items():
        done = sum(t.done() for t in tl)
        print(f"{name:10s} {done:6d}/{len(tl):<6d} {100 * done / max(len(tl), 1):5.1f}%")


def _guard_config(cfg, root: Path):
    """Results of different settings must never be mixed: the first run stores the
    config; later runs on the same results_dir must use the identical config."""
    import yaml
    stored = root / "config_used.yaml"
    if not stored.exists():
        stored.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
        return
    old = yaml.safe_load(stored.read_text(encoding="utf-8"))
    if old != cfg:
        diff = sorted(k for k in set(old) | set(cfg) if old.get(k) != cfg.get(k))
        raise SystemExit(f"config differs from the one used for {root} (sections: {diff}). "
                         "Use a new results_dir, or delete the old results.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--stage", default="all", choices=["all"] + STAGES)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--benchmark", action="store_true",
                    help="save hardware info, time a few evaluations and estimate the run time, then stop")
    ap.add_argument("--models", default=None,
                    help="comma list: run only these classifiers in base/eval/scenarios (e.g. to run DL "
                         "models separately with fewer workers on the GPU)")
    args = ap.parse_args()

    cfg = C.load(args.config)
    models = args.models.split(",") if args.models else None
    root = Path(cfg["results_dir"])
    if args.status:
        status(cfg)
        return
    log = setup_logging(root)
    _guard_config(cfg, root)
    envinfo.save(root, log)                     # hardware/software record of this start
    removed = clean_tmp(root)
    if removed:
        log.info("removed %d leftover temporary files from an interrupted run", removed)
    if args.benchmark:
        from amdabc_fs import benchmark
        prepare(cfg)
        benchmark.run(cfg, args.workers)
        return
    log.info("config %s, stage %s, workers %d", args.config, args.stage, args.workers)
    todo = STAGES if args.stage == "all" else [args.stage]
    t0 = time.time()
    checked = False
    for st in todo:
        if st != "prepare" and not checked:
            log.info("leakage checks passed: %s", ", ".join(check_all(cfg)))
            checked = True
        if st == "prepare":
            prepare(cfg)
        elif st == "scenarios":
            opt_s, sub_s, ev_s = S.scenario_tasks(cfg)
            run_tasks(opt_s, args.workers, "scenario-opt")
            run_tasks(sub_s, 1, "scenario-subsets")
            run_tasks(_only(ev_s, models), args.workers, "scenario-eval")
        elif st == "report":
            from amdabc_fs.analysis.report import report
            report(cfg)
        else:
            tl = task_lists(cfg, models)[st]
            run_tasks(tl, 1 if st == "subsets" else args.workers, st)
    log.info("finished %s in %.1f min", args.stage, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
