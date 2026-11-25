import numpy as np
import pytest

from amdabc_fs.optim import paper_abc as pa
from amdabc_fs.optim.budget import BudgetedProblem, BudgetExhausted
from amdabc_fs.optim.registry import OPTIMIZERS
from amdabc_fs.optim.runner import run

D, FE = 40, 500
W = np.random.default_rng(0).normal(size=D)


def toy_fitness(mask):
    """Cheap FS-like objective: a few useful features, a size penalty."""
    if not mask.any():
        return 1.0
    return 0.9 * (1 / (1 + np.exp(W[mask].sum()))) + 0.1 * mask.mean()


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_exact_budget(name):
    r = run(name, D, toy_fitness, FE, seed=42)
    assert r.fe_used == FE, f"{name} used {r.fe_used} FE"
    assert len(r.curve) == FE
    assert np.all(np.diff(r.curve) <= 0)                  # best-so-far never worsens
    assert r.best_fit == pytest.approx(toy_fitness(r.best_mask))   # value belongs to mask


@pytest.mark.parametrize("name", ["AMDABC", "MPA", "HHO", "GTO"])
def test_deterministic(name):
    a = run(name, D, toy_fitness, FE, seed=7)
    b = run(name, D, toy_fitness, FE, seed=7)
    assert np.array_equal(a.curve, b.curve) and np.array_equal(a.best_mask, b.best_mask)


def test_unarmed_problem_refuses_calls():
    p = BudgetedProblem(D, toy_fitness, FE)
    with pytest.raises(RuntimeError):
        p.obj_func(np.zeros(D))


def test_budget_raises_after_limit():
    p = BudgetedProblem(D, toy_fitness, 3)
    p.arm()
    for _ in range(3):
        p.obj_func(np.ones(D))
    with pytest.raises(BudgetExhausted):
        p.obj_func(np.ones(D))


def _changed_after_restart(legacy):
    """Constant objective: nothing improves, so only restart/scouts move agents."""
    changed = []

    class Probe(pa.AMDABC):
        def evolve(self, epoch):
            before = [a.solution.copy() for a in self.pop]
            stuck_best = self.trials[self.get_index_best(self.pop, "min")] >= self.n_limits - 2
            super().evolve(epoch)
            if stuck_best and not changed:
                changed.append(sum(not np.array_equal(b, a.solution) for b, a in zip(before, self.pop)))

    p = BudgetedProblem(D, lambda m: 0.5, 400)
    m = Probe(epoch=1000, pop_size=10, n_limits=3, legacy_restart=legacy)
    p.arm()
    try:
        m.solve(p, seed=1)
    except BudgetExhausted:
        pass
    return changed[0], m.NS


def test_restart_regenerates_whole_population():
    n_fixed, ns = _changed_after_restart(legacy=False)
    n_legacy, _ = _changed_after_restart(legacy=True)
    assert n_fixed == ns          # every food source replaced
    assert n_legacy <= 2          # archived bug: best slot (+ at most one scout)


def test_alpha_follows_fe_progress():
    p = BudgetedProblem(D, toy_fitness, 100)
    p.arm()
    m = pa.AMDABC(epoch=100_000, pop_size=10)
    m.problem = p
    for _ in range(25):
        p.obj_func(np.ones(D))
    assert m._progress(1) == pytest.approx(0.25)


def test_mpa_solution_matches_fitness():
    """Every agent's stored fitness must belong to its stored solution."""
    from amdabc_fs.optim.paper_mpa import MPA
    from amdabc_fs.optim.mealpy_core import FloatVar
    f = lambda x: float(np.sum(x ** 2))
    m = MPA(epoch=30, pop_size=10)
    m.solve({"obj_func": f, "bounds": FloatVar(lb=(-5.,) * 10, ub=(5.,) * 10), "minmax": "min", "log_to": None}, seed=3)
    assert all(abs(a.target.fitness - f(a.solution)) < 1e-9 for a in m.pop)
    assert abs(m.g_best.target.fitness - f(m.g_best.solution)) < 1e-9
