"""Space-filling designs in the unit cube, decoded to parameter tables."""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .problem import Problem


def _rng(seed) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def _unit_points(
    problem: Problem, n: int, make: Callable[[int], np.ndarray], max_tries: int = 50
) -> np.ndarray:
    """``n`` unit-cube points satisfying the problem's constraints (rejection sampling)."""
    if not problem.constraints:
        return make(n)
    kept: list[np.ndarray] = []
    total = 0
    for _ in range(max_tries):
        U = make(max(n, 16))
        ok = problem.feasible(problem.decode(U))
        kept.append(U[ok])
        total += int(ok.sum())
        if total >= n:
            return np.vstack(kept)[:n]
    raise RuntimeError(
        f"could not draw {n} feasible points in {max_tries} tries: constraints too restrictive?"
    )


def lhs_unit(problem: Problem, n: int, seed=None, optimization: str | None = "random-cd") -> np.ndarray:
    """Latin hypercube, as unit-cube points ``(n, d)``."""
    rng = _rng(seed)
    return _unit_points(
        problem,
        n,
        lambda m: qmc.LatinHypercube(d=problem.n_var, rng=rng, optimization=optimization).random(m)
        if _supports_rng()
        else qmc.LatinHypercube(
            d=problem.n_var, seed=rng, optimization=optimization
        ).random(m),
    )


def _supports_rng() -> bool:
    import inspect

    return "rng" in inspect.signature(qmc.LatinHypercube.__init__).parameters


def lhs(problem: Problem, n: int, seed=None) -> pd.DataFrame:
    """``n`` Latin-hypercube samples of the problem, as a parameter table."""
    return problem.decode(lhs_unit(problem, n, seed))


def sobol(problem: Problem, n: int, seed=None, scramble: bool = True) -> pd.DataFrame:
    """``n`` (scrambled) Sobol' points. ``n`` should be a power of two for best balance."""
    rng = _rng(seed)
    kw = {"rng": rng} if _supports_rng_sobol() else {"seed": rng}
    sampler = qmc.Sobol(d=problem.n_var, scramble=scramble, **kw)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "n is not a power of 2"
        U = _unit_points(problem, n, lambda m: sampler.random(m))
    return problem.decode(U)


def _supports_rng_sobol() -> bool:
    import inspect

    return "rng" in inspect.signature(qmc.Sobol.__init__).parameters


def random(problem: Problem, n: int, seed=None) -> pd.DataFrame:
    """``n`` independent uniform random samples."""
    rng = _rng(seed)
    return problem.decode(_unit_points(problem, n, lambda m: rng.random((m, problem.n_var))))


def factorial(problem: Problem, levels: int | list[int] = 3) -> pd.DataFrame:
    """Full factorial grid; categorical/integer parameters use all their values (capped by ``levels``)."""
    lv = [levels] * problem.n_var if isinstance(levels, int) else list(levels)
    axes = []
    for p, k in zip(problem.parameters, lv, strict=True):
        vd = p.value_descriptor
        if p.kind == "categorical":
            m = len(vd.options)
            axes.append((np.arange(m) + 0.5) / m)
        elif p.kind == "integer":
            m = int(vd.max - vd.min) + 1
            m = min(m, k) if k else m
            axes.append((np.round(np.linspace(0, int(vd.max - vd.min), m)) + 0.5) / (vd.max - vd.min + 1))
        else:
            axes.append(np.linspace(0.0, 1.0, k))
    grid = np.array(list(itertools.product(*axes)))
    df = problem.decode(grid)
    return df[problem.feasible(df)].reset_index(drop=True)
