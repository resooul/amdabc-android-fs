"""Run one optimizer for exactly ``max_fe`` evaluations.

Epoch-scheduled algorithms (stock mealpy ones such as MPA, GTO, HHO) compute
their schedule from ``epoch / self.epoch``. To make that schedule end where the
FE budget ends, ``calibrate_epochs`` first runs the algorithm with the same
seed on a cheap stand-in objective and counts how many epochs the budget
covers; that number is passed as ``epoch``. The budget itself is enforced by
``BudgetedProblem`` and does not depend on the calibration.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .budget import BudgetedProblem, BudgetExhausted
from .registry import OPTIMIZERS

BIG_EPOCH = 100_000


@dataclass
class RunResult:
    optimizer: str
    seed: int
    fe_used: int
    best_fit: float
    best_mask: np.ndarray
    curve: np.ndarray          # best-so-far fitness after each FE, length fe_used
    epochs: int                # epoch setting passed to the optimizer
    seconds: float


def _solve(name: str, problem: BudgetedProblem, seed: int, epoch: int):
    spec = OPTIMIZERS[name]
    model = spec.cls(epoch=epoch, **spec.kwargs)
    problem.arm()
    try:
        model.solve(problem, seed=seed)
    except BudgetExhausted:
        pass
    return model


def calibrate_epochs(name: str, n_dims: int, max_fe: int, seed: int) -> int:
    """Number of epochs the budget covers, measured on a cheap stand-in objective."""
    if OPTIMIZERS[name].uses_fe_progress:
        return BIG_EPOCH        # schedule already runs on FE progress
    probe = BudgetedProblem(n_dims, lambda m: float(m.mean()), max_fe)
    model = _solve(name, probe, seed, BIG_EPOCH)
    done = len(model.history.list_epoch_time)          # completed epochs
    return max(1, done + (1 if probe.fe_used >= max_fe else 0))


def run(name: str, n_dims: int, fitness_of_mask, max_fe: int, seed: int) -> RunResult:
    epochs = calibrate_epochs(name, n_dims, max_fe, seed)
    while True:
        problem = BudgetedProblem(n_dims, fitness_of_mask, max_fe)
        t0 = time.perf_counter()
        _solve(name, problem, seed, epochs)
        secs = time.perf_counter() - t0
        # The real objective can need fewer FE per epoch than the stand-in
        # (e.g. fewer scout restarts). Rare; rerun with one more epoch.
        if problem.fe_used >= max_fe or epochs >= BIG_EPOCH:
            break
        epochs += 1
    st = problem.state
    return RunResult(name, seed, st.used, st.best_fit, st.best_mask,
                     np.asarray(st.curve), epochs, secs)
