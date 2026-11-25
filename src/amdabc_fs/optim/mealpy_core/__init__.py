"""Subset of mealpy 3.0.3 (MIT licence, Nguyen Van Thieu), vendored by
tools/vendor_mealpy.py. Public names mirror mealpy's."""
from .optimizer import Optimizer
from .utils.agent import Agent
from .utils.problem import Problem
from .utils.space import BinaryVar, FloatVar, IntegerVar
from .utils.termination import Termination
from .swarm_based import AGTO, HHO, PSO, SSO

__version__ = "3.0.3"
