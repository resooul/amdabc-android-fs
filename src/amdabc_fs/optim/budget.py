"""Function-evaluation (FE) budget for mealpy optimizers.

mealpy checks its stopping rule once per epoch, so a run can overshoot a FE
budget by up to one epoch, and algorithms differ in how many evaluations one
epoch costs (PSO 1 per agent, MPA and AGTO 2, ABC 2 per food source ...).
This module stops every optimizer at exactly ``max_fe`` evaluations instead:

* ``BudgetedProblem`` counts every call of the objective (cache hits included,
  because a cache is an implementation detail, not a saving of the algorithm),
  keeps the best value and mask seen so far, and raises ``BudgetExhausted`` on
  call ``max_fe + 1``. The best result is therefore always a value that was
  really evaluated for that mask, independent of the optimizer's own
  bookkeeping (see the MPA solution/fitness mismatch in the local mealpy copy).
* ``progress()`` returns FE_used / max_fe in [0, 1]. Our own ABC classes use it
  in place of ``epoch / self.epoch`` so their schedules (e.g. AMDABC's cosine
  alpha) run on the FE axis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from .mealpy_core import FloatVar, Problem


class BudgetExhausted(Exception):
    """Raised inside the objective when the FE budget is used up."""


@dataclass
class BudgetState:
    max_fe: int
    used: int = 0
    best_fit: float = np.inf
    best_mask: np.ndarray | None = None
    curve: list = field(default_factory=list)      # best-so-far after each FE


class BudgetedProblem(Problem):
    """Minimisation problem on [0, 1]^D with a hard FE budget.

    Parameters
    ----------
    n_dims : number of candidate features (after the train-fold filter).
    fitness_of_mask : callable(bool mask) -> float, lower is better.
    max_fe : evaluation budget, e.g. 500.
    threshold : binary transfer, feature j is selected if x_j > threshold.
    """

    def __init__(self, n_dims: int, fitness_of_mask: Callable[[np.ndarray], float],
                 max_fe: int, threshold: float = 0.5, **kwargs):
        self.fitness_of_mask = fitness_of_mask
        self.threshold = threshold
        self.state = BudgetState(max_fe=int(max_fe))
        self.armed = False   # objective calls before arm() are rejected
        bounds = FloatVar(lb=np.zeros(n_dims), ub=np.ones(n_dims), name="mask")
        kwargs.setdefault("log_to", None)
        super().__init__(bounds=bounds, minmax="min", **kwargs)
        # mealpy probes obj_func once to detect the number of objectives;
        # declaring it here avoids that probe, so no evaluation goes uncounted.
        self._n_objs, self.obj_weights = 1, np.ones(1)

    # -- control -----------------------------------------------------------
    def arm(self) -> None:
        self.armed = True

    def progress(self) -> float:
        return min(1.0, self.state.used / self.state.max_fe)

    @property
    def fe_used(self) -> int:
        return self.state.used

    # -- objective -----------------------------------------------------------
    def obj_func(self, solution):
        if not self.armed:
            raise RuntimeError("objective called before arm(); an evaluation would go uncounted")
        st = self.state
        if st.used >= st.max_fe:
            raise BudgetExhausted
        st.used += 1
        mask = np.asarray(solution) > self.threshold
        fit = float(self.fitness_of_mask(mask))
        if fit < st.best_fit:
            st.best_fit, st.best_mask = fit, mask.copy()
        st.curve.append(st.best_fit)
        return fit
