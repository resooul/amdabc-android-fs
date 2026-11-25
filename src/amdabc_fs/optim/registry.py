"""The 15 optimizers compared in the paper and their settings.

Settings:
pop_size = 10, ABC family n_limits = ceil(0.5 * 50) = 25, AMDABC n_best = 3;
MPA and GTO use pop_size / 2, HHO uses ceil(pop_size / 1.5). "GTO" in the paper
is the gorilla troops optimizer (Abdollahzadeh et al. 2021), i.e. mealpy's
AGTO.OriginalAGTO, not mealpy's GTO (giant trevally). PSO, SSO, HHO and AGTO
come from the vendored mealpy 3.0.3 subset (optim/mealpy_core).
MPA is the authors' implementation (follows the published MATLAB code).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from .mealpy_core.swarm_based import AGTO, HHO, PSO, SSO

from . import paper_abc as pa
from . import paper_mpa

POP = 10
N_LIMITS = 25


@dataclass(frozen=True)
class Spec:
    cls: Callable
    kwargs: dict = field(default_factory=dict)
    uses_fe_progress: bool = False   # schedule runs on FE progress (own classes)
    group: str = ""


def _abc(cls, group, fe=False, **kw):
    return Spec(cls, dict(pop_size=POP, n_limits=N_LIMITS, **kw), fe, group)


OPTIMIZERS: dict[str, Spec] = {
    # proposed + ablation
    "AMDABC": _abc(pa.AMDABC, "proposed", fe=True, n_best=3),
    "AMDABC1": _abc(pa.AMDABC1, "ablation", fe=True, n_best=3),
    "AMDABC2": _abc(pa.AMDABC2, "ablation", n_best=3),
    "AMDABC3": _abc(pa.AMDABC3, "ablation", n_best=3),
    # ABC variants
    "ABC": _abc(pa.ABC, "abc"),
    "GBABC": _abc(pa.GBABC, "abc"),
    "GBC": _abc(pa.GBC, "abc"),
    "ABCX": Spec(pa.ABCX, dict(pop_size=POP, n_limits=None), False, "abc"),
    "DABC": _abc(pa.DABC, "abc"),
    "ABCVSS": _abc(pa.ABCVSS, "abc"),
    # other metaheuristics (stock mealpy)
    "PSO": Spec(PSO.OriginalPSO, dict(pop_size=POP), False, "meta"),
    "SSO": Spec(SSO.OriginalSSO, dict(pop_size=POP), False, "meta"),
    "HHO": Spec(HHO.OriginalHHO, dict(pop_size=math.ceil(POP / 1.5)), False, "meta"),
    "GTO": Spec(AGTO.OriginalAGTO, dict(pop_size=POP // 2), False, "meta"),
    "MPA": Spec(paper_mpa.MPA, dict(pop_size=POP // 2), False, "meta"),   # authors' MPA
}

assert len(OPTIMIZERS) == 15
