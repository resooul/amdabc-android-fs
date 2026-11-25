"""Resumable task execution.

A stage is a list of ``Task`` objects. Each task knows its output file; tasks
whose output exists are skipped, the rest run in a process pool. Results are
written by the task itself (atomically, see io.py), so the pool can be killed
at any moment and the stage restarted with the same command.
"""
from __future__ import annotations

import logging
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger("amdabc")


@dataclass
class Task:
    key: str                     # human-readable id, e.g. "opt/dvm/f0/AMDABC/s3"
    output: Path                 # file that marks the task as done
    fn: Callable                 # top-level function (picklable)
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)

    def done(self) -> bool:
        return Path(self.output).exists()


def _marker(task: Task) -> Path:
    return Path(str(task.output) + ".started")


def _call(task: Task):
    t0 = time.perf_counter()
    m = _marker(task)
    m.parent.mkdir(parents=True, exist_ok=True)
    try:
        m.touch()
    except OSError:
        pass                              # marker is only a crash hint
    task.fn(*task.args, **task.kwargs)
    try:
        m.unlink(missing_ok=True)
    except OSError:
        pass
    return task.key, time.perf_counter() - t0


def _init_worker():
    # one BLAS/torch thread per process; parallelism comes from the pool
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def run_tasks(tasks: list[Task], workers: int = 1, name: str = "stage") -> None:
    todo = [t for t in tasks if not t.done()]
    log.info("%s: %d tasks, %d done, %d to run", name, len(tasks), len(tasks) - len(todo), len(todo))
    if not todo:
        return
    failed = []
    t0 = time.perf_counter()
    if workers <= 1:
        for i, t in enumerate(todo, 1):
            try:
                _, dt = _call(t)
                log.info("[%s %d/%d] %s (%.1fs)", name, i, len(todo), t.key, dt)
            except Exception:
                failed.append(t.key)
                log.error("[%s] %s failed:\n%s", name, t.key, traceback.format_exc())
    else:
        # A worker killed by the OS (out of memory, ...) breaks the whole pool.
        # Then the pool is rebuilt and the unfinished tasks are resubmitted; a
        # task that was running in a broken pool twice is reported as failed.
        crashes: dict[str, int] = {}
        pending, n_done, n_total = list(todo), 0, len(todo)
        while pending:
            broken = False
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
                futs = {ex.submit(_call, t): t for t in pending}
                for fut in as_completed(futs):
                    t = futs[fut]
                    try:
                        _, dt = fut.result()
                        n_done += 1
                        eta = (time.perf_counter() - t0) / n_done * (n_total - n_done)
                        log.info("[%s %d/%d] %s (%.1fs, ETA %.0f min)", name, n_done, n_total, t.key, dt, eta / 60)
                    except BrokenProcessPool:
                        broken = True
                    except Exception:
                        failed.append(t.key)
                        log.error("[%s] %s failed:\n%s", name, t.key, traceback.format_exc())
            pending = [t for t in pending if not t.done() and t.key not in failed]
            if broken:
                for t in pending:                       # tasks that were running when it broke
                    if _marker(t).exists():
                        crashes[t.key] = crashes.get(t.key, 0) + 1
                        _marker(t).unlink(missing_ok=True)
                gave_up = [t.key for t in pending if crashes.get(t.key, 0) >= 3]
                failed += gave_up
                pending = [t for t in pending if t.key not in gave_up]
                log.warning("[%s] a worker died (memory?); restarting the pool for %d task(s)", name, len(pending))
                if workers > 1 and any(crashes.get(t.key, 0) >= 2 for t in pending):
                    workers = max(1, workers // 2)
                    log.warning("[%s] reducing workers to %d", name, workers)
            else:
                break
    if failed:
        raise RuntimeError(f"{name}: {len(failed)} task(s) failed, e.g. {failed[:3]}; "
                           "rerun the same command to retry only the failed tasks")
