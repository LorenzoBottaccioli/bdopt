"""Multi-objective optimization (NSGA-II / NSGA-III via pymoo) on a surrogate or on EnergyPlus.

The search runs in the problem's unit cube and discrete coordinates (categorical / integer
parameters) are snapped to valid values before every evaluation, so mixed problems work out of the
box (including the glazing-type parameter of the platform's Test 6, which broke the Platypus code).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2 as _PymooNSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3 as _PymooNSGA3
from pymoo.algorithms.soo.nonconvex.ga import GA as _PymooGA
from pymoo.core.problem import Problem as _PymooProblem
from pymoo.core.repair import Repair
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from .problem import Problem

_BIG = 1e30


class UnitModel(Protocol):
    """Anything that maps unit-cube points to objective values: a surrogate or an evaluator."""

    def evaluate_unit(self, U: np.ndarray) -> np.ndarray:
        """Objective values ``(n, n_obj)`` for unit-cube points ``(n, n_var)``."""
        ...


@dataclass
class ParetoFront:
    """Non-dominated designs found: parameters + objectives in real units."""

    df: pd.DataFrame
    problem: Problem

    def __len__(self) -> int:
        return len(self.df)

    @property
    def parameters(self) -> pd.DataFrame:
        """The parameter columns of the front."""
        return self.df[self.problem.parameter_names]

    @property
    def objectives(self) -> pd.DataFrame:
        """The objective columns of the front."""
        return self.df[self.problem.objective_names]

    def closest_to_ideal(
        self, weights: list[float] | None = None, normalize: bool = True
    ) -> pd.Series:
        """The design closest to the ideal point (best value of every objective).

        With ``normalize=True`` (default) objectives are min-max scaled over the front first, so that an
        objective with large numbers (e.g. cooling energy in J) does not dominate the distance;
        ``normalize=False`` reproduces the raw Euclidean distance of the original notebooks.
        """
        F = self.objectives.to_numpy(float) * self.problem.signs  # minimise
        if normalize:
            lo, hi = F.min(axis=0), F.max(axis=0)
            F = (F - lo) / np.where(hi > lo, hi - lo, 1.0)
        else:
            F = F - F.min(axis=0)
        w = np.ones(F.shape[1]) if weights is None else np.asarray(weights, float)
        dist = np.sqrt(((F * w) ** 2).sum(axis=1))
        return self.df.iloc[int(np.argmin(dist))]

    def to_df(self) -> pd.DataFrame:
        """A copy of the front as a table."""
        return self.df.copy()


class _SnapRepair(Repair):
    def __init__(self, problem: Problem):
        super().__init__()
        self._p = problem

    def _do(self, problem, X, **kwargs):
        return self._p.snap(X)


class _PymooAdapter(_PymooProblem):
    def __init__(self, problem: Problem, model: UnitModel):
        super().__init__(
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            n_ieq_constr=len(problem.constraints),
            xl=np.zeros(problem.n_var),
            xu=np.ones(problem.n_var),
        )
        self.bd_problem, self.model = problem, model

    def _evaluate(self, X, out, *args, **kwargs):
        pr = self.bd_problem
        Xs = pr.snap(X)
        F = np.asarray(self.model.evaluate_unit(Xs), dtype=float).reshape(len(Xs), pr.n_obj)
        F = np.where(np.isfinite(F), F * pr.signs, _BIG)
        out["F"] = F
        if pr.constraints:
            out["G"] = pr.violation(pr.decode(Xs))


class Optimizer(Protocol):
    """Minimal optimizer interface: other back-ends can be added without touching the rest."""

    def run(self, problem: Problem, model: UnitModel) -> ParetoFront:
        """Return the non-dominated designs of ``problem`` evaluated with ``model``."""
        ...


class _PymooOptimizer:
    def __init__(self, pop_size: int = 100, n_gen: int = 100, seed: int = 0, verbose: bool = False):
        self.pop_size, self.n_gen, self.seed, self.verbose = pop_size, n_gen, seed, verbose

    def _algorithm(self, problem: Problem):
        raise NotImplementedError

    def run(self, problem: Problem, model: UnitModel) -> ParetoFront:
        res = minimize(
            _PymooAdapter(problem, model),
            self._algorithm(problem),
            ("n_gen", self.n_gen),
            seed=self.seed,
            verbose=self.verbose,
        )
        if res.X is None:
            raise RuntimeError("the optimizer found no feasible design; check the constraints")
        X = problem.snap(np.atleast_2d(res.X))
        F = np.atleast_2d(res.F) * problem.signs
        df = problem.decode(X)
        for j, name in enumerate(problem.objective_names):
            df[name] = F[:, j]
        df = df.drop_duplicates(subset=problem.parameter_names).reset_index(drop=True)
        return ParetoFront(df, problem)


class NSGA2(_PymooOptimizer):
    """NSGA-II (single objective problems use a plain genetic algorithm)."""

    def _algorithm(self, problem: Problem):
        repair = _SnapRepair(problem)
        if problem.n_obj == 1:
            return _PymooGA(pop_size=self.pop_size, repair=repair, eliminate_duplicates=True)
        return _PymooNSGA2(pop_size=self.pop_size, repair=repair, eliminate_duplicates=True)


class NSGA3(_PymooOptimizer):
    """NSGA-III (many objectives); ``pop_size`` is raised to the number of reference directions."""

    def __init__(self, n_partitions: int = 12, **kw):
        super().__init__(**kw)
        self.n_partitions = n_partitions

    def _algorithm(self, problem: Problem):
        if problem.n_obj < 2:
            raise ValueError("NSGA-III needs at least two objectives")
        ref = get_reference_directions("das-dennis", problem.n_obj, n_partitions=self.n_partitions)
        return _PymooNSGA3(
            ref_dirs=ref,
            pop_size=max(self.pop_size, len(ref)),
            repair=_SnapRepair(problem),
            eliminate_duplicates=True,
        )


def nsga2(problem: Problem, model: UnitModel, pop_size: int = 100, n_gen: int = 100, seed: int = 0,
          verbose: bool = False) -> ParetoFront:
    """Pareto front of ``problem`` using ``model`` (a surrogate or an evaluator) -- NSGA-II."""
    return NSGA2(pop_size=pop_size, n_gen=n_gen, seed=seed, verbose=verbose).run(problem, model)


def nsga3(problem: Problem, model: UnitModel, pop_size: int = 100, n_gen: int = 100, seed: int = 0,
          n_partitions: int = 12, verbose: bool = False) -> ParetoFront:
    """Pareto front of ``problem`` using ``model`` -- NSGA-III."""
    return NSGA3(n_partitions=n_partitions, pop_size=pop_size, n_gen=n_gen, seed=seed,
                 verbose=verbose).run(problem, model)
