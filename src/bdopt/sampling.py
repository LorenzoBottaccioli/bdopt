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


def halton(problem: Problem, n: int, seed=None, scramble: bool = True) -> pd.DataFrame:
    """``n`` (scrambled) Halton points: a low-discrepancy sequence that can be extended later."""
    rng = _rng(seed)
    sampler = qmc.Halton(d=problem.n_var, scramble=scramble, seed=int(rng.integers(2**31)))
    return problem.decode(_unit_points(problem, n, sampler.random))


def hammersley(problem: Problem, n: int) -> pd.DataFrame:
    """``n`` Hammersley points: ``i / n`` in the first coordinate, the Halton sequence (without its first
    dimension) in the others. Better spread than Halton for a *known* ``n``, but not extendable."""
    d = problem.n_var
    if d == 1:
        return problem.decode((np.arange(n)[:, None] + 0.5) / n)
    rest = qmc.Halton(d=d - 1, scramble=False).random(n + 1)[1:] if n else np.empty((0, d - 1))
    U = np.column_stack([(np.arange(n) + 0.5) / n, rest])
    return problem.decode(U)


def maximin_lhs(problem: Problem, n: int, seed=None, trials: int = 200) -> pd.DataFrame:
    """Best of ``trials`` random Latin hypercubes by the maximin criterion (largest smallest pairwise distance)."""
    rng = _rng(seed)
    d = problem.n_var
    best, best_score = None, -1.0
    for _ in range(trials):
        perms = np.column_stack([rng.permutation(n) for _ in range(d)])
        U = (perms + rng.random((n, d))) / n
        D = np.sqrt(((U[:, None, :] - U[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(D, np.inf)
        if D.min() > best_score:
            best, best_score = U, float(D.min())
    return problem.decode(best)


def fps(problem: Problem, n: int, seed=None, pool: int = 20000) -> pd.DataFrame:
    """Farthest-point sampling: start from a random point, repeatedly add the pool point farthest from all chosen."""
    rng = _rng(seed)
    d = problem.n_var
    C = rng.random((pool, d))
    chosen = [int(rng.integers(pool))]
    dmin = np.sqrt(((C - C[chosen[0]]) ** 2).sum(axis=1))
    for _ in range(n - 1):
        j = int(np.argmax(dmin))
        chosen.append(j)
        dmin = np.minimum(dmin, np.sqrt(((C - C[j]) ** 2).sum(axis=1)))
    return problem.decode(problem.snap(C[chosen]))


def cvt(problem: Problem, n: int, seed=None, pool: int = 20000) -> pd.DataFrame:
    """Centroidal Voronoi tessellation: the ``n`` centroids of a k-means clustering of uniform points (each point
    sits at the centre of mass of its own Voronoi cell)."""
    from sklearn.cluster import KMeans

    rng = _rng(seed)
    C = rng.random((pool, problem.n_var))
    km = KMeans(n_clusters=n, n_init=3, random_state=int(rng.integers(2**31))).fit(C)
    return problem.decode(problem.snap(km.cluster_centers_))
