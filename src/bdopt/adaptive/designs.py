"""Infill criteria for adaptive sampling, in the unit cube (numpy / scipy only).

Every criterion answers one question: given the points already simulated (and, for some, a surrogate
fitted on them), where should the next simulation go?

================  =====================================================================================
``flola_voronoi``  fuzzy LOLA-Voronoi: exploration (Voronoi cell volume) + exploitation (non-linearity
                   of the response around each sample, neighbours weighted by a fuzzy inference system)
``lola_voronoi``   LOLA-Voronoi: same hybrid, crisp neighbourhoods chosen by the cross-polytope criterion
``cv_voronoi``     cross-validation Voronoi: refine the cell of the sample the surrogate predicts worst
                   when it is left out (k-fold)
``voronoi``        pure exploration: farthest point of the largest cell
``probabilistic``  point of largest predictive standard deviation (global optimisation by differential
                   evolution)
``gp_variance``    same criterion on a random candidate set (cheap)
``nipv``           negative integrated posterior variance: the point that reduces the average predictive
                   variance over the whole domain the most (closed form for a Gaussian process)
``latin_hypercube``  maximin-augmented Latin hypercube (non-adaptive space filling)
``random``         uniform random baseline
================  =====================================================================================

References
----------
* K. Crombecq, D. Gorissen, D. Deschrijver, T. Dhaene (2011), *A novel hybrid sequential design strategy
  for global surrogate modeling of computer experiments*, SIAM J. Sci. Comput. 33(4).
* J. van der Herten, I. Couckuyt, D. Deschrijver, T. Dhaene (2015), *A fuzzy hybrid sequential design
  strategy for global surrogate modeling of high-dimensional computer experiments*, SIAM J. Sci. Comput.
  37(2).
* S. Xu, H. Liu, X. Wang, X. Jiang (2014), *A robust error-pursuing sequential sampling approach for
  global metamodeling based on Voronoi diagram and cross validation*, J. Mech. Des. 136(7).
* H. Liu, S. Xu, X. Wang, J. Meng, S. Yang (2016), *A multi-response adaptive sampling approach for
  global metamodeling*, Proc. IMechE Part C.
* A. L. Kaminsky et al. (2021), *An efficient batch k-fold cross-validation Voronoi adaptive sampling
  technique for global surrogate modelling*, J. Mech. Des. 143.
* X. Chai (2019), *Probabilistic system identification and reliability updating for hydraulic structures*
  (largest-predictive-uncertainty infill).
* Seo et al. (2000) / Gramacy & Lee (2009): integrated (posterior) variance reduction designs.
"""

from __future__ import annotations

import itertools
from math import comb

import numpy as np
from scipy.optimize import differential_evolution

from ..problem import Problem

# ---------------------------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------------------------


def voronoi_estimate(
    X: np.ndarray, rng: np.random.Generator, n_mc: int | None = None, chunk: int = 4000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Monte-Carlo Voronoi tessellation of the unit cube around ``X``.

    Returns ``(relative volumes (n,), MC points (M, d), nearest sample index (M,), distance to it (M,))``.
    """
    n, d = X.shape
    m = int(n_mc or np.clip(200 * n, 5000, 80000))
    P = rng.random((m, d))
    idx = np.empty(m, dtype=int)
    dist = np.empty(m)
    x2 = (X**2).sum(axis=1)
    for s in range(0, m, chunk):
        B = P[s : s + chunk]
        d2 = (B**2).sum(axis=1)[:, None] - 2 * B @ X.T + x2[None, :]
        j = np.argmin(d2, axis=1)
        idx[s : s + chunk] = j
        dist[s : s + chunk] = np.sqrt(np.maximum(d2[np.arange(len(B)), j], 0.0))
    vol = np.bincount(idx, minlength=n) / m
    return vol, P, idx, dist


def _pairwise(X: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0.0))


def _is_new(p: np.ndarray, others: np.ndarray, tol: float = 1e-9) -> bool:
    return others.size == 0 or bool((np.abs(others - p).max(axis=1) > tol).all())


def _stack(*arrays) -> np.ndarray:
    parts = [a for a in arrays if a is not None and len(a)]
    return np.vstack(parts) if parts else np.empty((0, 0))


def _fallback_point(
    problem: Problem | None, rng: np.random.Generator, d: int, taken: np.ndarray
) -> np.ndarray:
    for _ in range(1000):
        p = rng.random(d)
        if problem is not None:
            p = problem.snap(p)[0]
        if _is_new(p, taken):
            return p
    return p  # the design space is (nearly) exhausted: allow a repeat


def _snap(problem: Problem | None, U: np.ndarray) -> np.ndarray:
    return problem.snap(U) if problem is not None else U


def _fill(chosen, batch, problem, rng, d, taken):
    while len(chosen) < batch:
        cur = np.vstack([taken, *chosen]) if chosen else taken
        chosen.append(_fallback_point(problem, rng, d, cur))
    return np.array(chosen)


# ---------------------------------------------------------------------------------------------
# exploitation scores (non-linearity of the response around each sample)
# ---------------------------------------------------------------------------------------------


def _local_nonlinearity(
    dx: np.ndarray, dy: np.ndarray, w: np.ndarray | None = None
) -> float:
    """Sum of absolute deviations of the neighbours from the (weighted) hyperplane through the reference
    sample (Crombecq et al. Eq. 4.8-4.9; the reference sample sits in the origin)."""
    if w is None:
        G = np.linalg.lstsq(dx, dy, rcond=None)[0]
        resid = np.abs(dy - dx @ G)
    else:
        s = np.sqrt(w)[:, None]
        G = np.linalg.lstsq(dx * s, dy * s, rcond=None)[0]
        resid = w[:, None] * np.abs(dy - dx @ G)
    return float(resid.sum())


def _neighbourhood_scores(Dm: np.ndarray, d_ref: np.ndarray, combos: np.ndarray, X_rel: np.ndarray) -> np.ndarray:
    """Cross-polytope based quality of candidate neighbourhoods (Crombecq et al. Eq. 4.3-4.7).

    ``Dm`` distances among the pool, ``d_ref`` pool distances to the reference sample, ``combos`` (q, k)
    index sets into the pool, ``X_rel`` pool coordinates relative to the reference sample.
    """
    k = combos.shape[1]
    cohesion = d_ref[combos].mean(axis=1)
    if X_rel.shape[1] == 1:
        p1, p2 = X_rel[combos[:, 0], 0], X_rel[combos[:, 1], 0]
        ratio = 1 - np.abs(p1 + p2) / (np.abs(p1) + np.abs(p2) + np.abs(p1 - p2) + 1e-12)
    else:
        sub = Dm[combos[:, :, None], combos[:, None, :]].copy()
        sub[:, np.arange(k), np.arange(k)] = np.inf
        adhesion = sub.min(axis=2).mean(axis=1)
        ratio = adhesion / (np.sqrt(2) * cohesion + 1e-12)
    return ratio / (cohesion + 1e-12)


def lola_nonlinearity(X: np.ndarray, Y: np.ndarray, max_combinations: int = 3000) -> np.ndarray:
    """LOLA (Crombecq et al. 2011): for every sample, pick the best neighbourhood of ``2 d`` samples by the
    cross-polytope criterion and measure how far the response there is from a hyperplane.

    The paper enumerates every subset of ``2 d`` samples; that is ``C(n, 2d)`` subsets (10^12 for 60
    samples in 6 dimensions), so the search is restricted to the nearest samples (the "not too far"
    rule of the paper), as many as keep the enumeration below ``max_combinations`` subsets.
    """
    n, d = X.shape
    Y = Y.reshape(n, -1)
    k = min(2 * d, n - 1)
    D = _pairwise(X)
    out = np.zeros(n)
    for i in range(n):
        order = np.argsort(D[i])
        order = order[order != i]
        m = k
        while m < n - 1 and comb(m + 1, k) <= max_combinations:
            m += 1
        pool = order[:m]
        if m == k:
            best = pool
        else:
            combos = np.array(list(itertools.combinations(range(m), k)))
            Xr = X[pool] - X[i]
            sc = _neighbourhood_scores(D[np.ix_(pool, pool)], D[i, pool], combos, Xr)
            best = pool[combos[int(np.argmax(sc))]]
        out[i] = _local_nonlinearity(X[best] - X[i], Y[best] - Y[i])
    return out


# fuzzy inference system of the fuzzy LOLA (van der Herten et al. 2015, section 4) -----------------

_U = np.linspace(0.0, 1.0, 101)


def _trimf(u: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    left = np.where(b > a, (u - a) / max(b - a, 1e-12), (u >= b).astype(float))
    right = np.where(c > b, (c - u) / max(c - b, 1e-12), (u <= b).astype(float))
    return np.clip(np.minimum(left, right), 0.0, 1.0)


_W_LOW = _trimf(_U, 0.0, 0.0, 0.13)
_W_AVG = _trimf(_U, 0.16, 0.33, 0.5)
_W_HIGH = _trimf(_U, 0.6, 1.01, 1.01)


def fuzzy_weights(cohesion: np.ndarray, adhesion: np.ndarray) -> np.ndarray:
    """Weight of every neighbour from a Mamdani fuzzy inference system.

    Inputs are normalised to ``[0, 1]`` (distance to the reference sample / its largest value; distance of a
    neighbour to its own nearest sample / the largest such distance). Membership functions: *close* is a
    sigmoid of the cohesion distance, *isolated* a Gaussian peaking at the largest adhesion, *clustered* a
    Gaussian peaking at zero. Rules (AND = min, aggregation = max, centroid defuzzification):

    ============================  ==========
    close and isolated            high
    close and clustered           average
    far and isolated              average
    far and clustered             low
    ============================  ==========
    """
    c = cohesion / (cohesion.max() + 1e-12)
    a = adhesion / (adhesion.max() + 1e-12)
    close = 1.0 / (1.0 + np.exp(10.0 * (c - 0.5)))
    far = 1.0 - close
    isolated = np.exp(-((a - 1.0) ** 2) / (2 * 0.27**2))
    clustered = np.exp(-(a**2) / (2 * 0.3**2))
    r_high = np.minimum(close, isolated)
    r_avg = np.maximum(np.minimum(close, clustered), np.minimum(far, isolated))
    r_low = np.minimum(far, clustered)
    agg = np.maximum.reduce(
        [
            np.minimum(r_high[:, None], _W_HIGH[None, :]),
            np.minimum(r_avg[:, None], _W_AVG[None, :]),
            np.minimum(r_low[:, None], _W_LOW[None, :]),
        ]
    )
    area = agg.sum(axis=1)
    w = np.where(area > 1e-12, (agg * _U[None, :]).sum(axis=1) / np.where(area > 1e-12, area, 1.0), 0.33)
    return np.maximum(w, 1e-3)


def flola_nonlinearity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Fuzzy LOLA (van der Herten et al. 2015): neighbourhood of every sample = samples closer than
    ``alpha = (2 / K) * sum of the K nearest distances`` (``K = 4 d``), weights from
    :func:`fuzzy_weights`, weighted least-squares hyperplane, weighted deviation as the score."""
    n, d = X.shape
    Y = Y.reshape(n, -1)
    K = min(4 * d, n - 1)
    D = _pairwise(X)
    Doff = D + np.diag(np.full(n, np.inf))
    nearest_other = Doff.min(axis=1)  # adhesion of every sample
    out = np.zeros(n)
    for i in range(n):
        row = Doff[i]
        alpha = 2.0 / K * np.sort(row)[:K].sum()
        nb = np.where(row < alpha)[0]
        if len(nb) < min(d, n - 1):
            nb = np.argsort(row)[: min(d, n - 1)]
        w = fuzzy_weights(D[i, nb], nearest_other[nb])
        out[i] = _local_nonlinearity(X[nb] - X[i], Y[nb] - Y[i], w)
    return out


def nonlinearity(X: np.ndarray, Y: np.ndarray, fuzzy: bool = True) -> np.ndarray:
    """Exploitation score of every sample (fuzzy LOLA if ``fuzzy`` else LOLA)."""
    return flola_nonlinearity(X, Y) if fuzzy else lola_nonlinearity(X, Y)


# ---------------------------------------------------------------------------------------------
# Voronoi based proposals
# ---------------------------------------------------------------------------------------------


def _points_in_cells(order, batch, P, idx, dist, taken, problem):
    """New point of each of the best cells: the Monte-Carlo point farthest from the cell's sample."""
    chosen: list[np.ndarray] = []
    for ref in order:
        if len(chosen) == batch:
            break
        cell = np.where(idx == ref)[0]
        for c in cell[np.argsort(-dist[cell])][:50]:
            p = _snap(problem, P[c][None])[0]
            cur = np.vstack([taken, *chosen]) if chosen else taken
            if _is_new(p, cur):
                chosen.append(p)
                break
    return chosen


def propose_voronoi_based(
    X: np.ndarray,
    Y: np.ndarray | None,
    batch: int,
    rng: np.random.Generator,
    problem: Problem | None = None,
    avoid: np.ndarray | None = None,
    exploit: bool = True,
    fuzzy: bool = True,
    n_mc: int | None = None,
) -> np.ndarray:
    """``batch`` new points: hybrid LOLA / fuzzy LOLA-Voronoi (``exploit``) or pure Voronoi."""
    n, d = X.shape
    vol, P, idx, dist = voronoi_estimate(X, rng, n_mc)
    score = vol.copy()
    if exploit and Y is not None and n > 2:
        e = nonlinearity(X, Y, fuzzy)
        if e.sum() > 0:
            score = score + e / e.sum()  # Crombecq et al. Eq. 5.1
    taken = np.vstack([X, avoid]) if avoid is not None and len(avoid) else X
    chosen = _points_in_cells(np.argsort(-score), batch, P, idx, dist, taken, problem)
    return _fill(chosen, batch, problem, rng, d, taken)


def propose_cv_voronoi(
    X: np.ndarray,
    Y: np.ndarray,
    model,
    batch: int,
    rng: np.random.Generator,
    problem: Problem | None = None,
    avoid: np.ndarray | None = None,
    k: int = 5,
) -> np.ndarray:
    """Cross-validation Voronoi (Xu et al. 2014, Liu et al. 2016, Kaminsky et al. 2021).

    A ``k``-fold cross-validation leaves every sample out once; the error of a sample is its prediction
    error from the model that did not see it. Errors of the objectives are combined with weights
    proportional to each objective's cross-validated NRMSE (multi-response rule of Liu et al.), the cell of
    the worst sample is refined at its farthest point.
    """
    from sklearn.model_selection import KFold

    n, d = X.shape
    Y = Y.reshape(n, -1)
    err = np.zeros_like(Y)
    for tr, te in KFold(min(k, n), shuffle=True, random_state=int(rng.integers(2**31))).split(X):
        m = model.clone().fit_unit(X[tr], Y[tr], problem)
        err[te] = np.abs(m.predict_unit(X[te]) - Y[te])
    scale = Y.std(axis=0)
    err = err / np.where(scale > 0, scale, 1.0)
    nrmse = np.sqrt((err**2).mean(axis=0))
    w = nrmse / nrmse.sum() if nrmse.sum() > 0 else np.full(Y.shape[1], 1 / Y.shape[1])
    cell_score = (err * w).sum(axis=1)
    _, P, idx, dist = voronoi_estimate(X, rng)
    taken = np.vstack([X, avoid]) if avoid is not None and len(avoid) else X
    chosen = _points_in_cells(np.argsort(-cell_score), batch, P, idx, dist, taken, problem)
    return _fill(chosen, batch, problem, rng, d, taken)


# ---------------------------------------------------------------------------------------------
# uncertainty based proposals
# ---------------------------------------------------------------------------------------------


def _need_std(surrogate, method: str) -> None:
    if not getattr(surrogate, "has_uncertainty", False):
        raise ValueError(
            f"the {method!r} method needs a surrogate with uncertainty "
            "(GPSurrogate, MultiTaskGPSurrogate, DeepKernelGPSurrogate, BayesianMLP, TorchMLP with n_ensemble > 1, ...)"
        )


def propose_gp_variance(
    X: np.ndarray,
    Y: np.ndarray,
    surrogate,
    batch: int,
    rng: np.random.Generator,
    problem: Problem | None = None,
    avoid: np.ndarray | None = None,
    n_candidates: int = 4000,
) -> np.ndarray:
    """Points with the largest predictive uncertainty among random candidates."""
    n, d = X.shape
    _need_std(surrogate, "gp_variance")
    C = _snap(problem, rng.random((n_candidates, d)))
    std = surrogate.predict_unit_std(C)
    scale = Y.reshape(n, -1).std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    score = (std / scale).mean(axis=1)
    taken = np.vstack([X, avoid]) if avoid is not None and len(avoid) else X
    chosen: list[np.ndarray] = []
    sep = 0.1 * np.sqrt(d)
    for _ in range(batch):
        for c in np.argsort(-score):
            p = C[c]
            cur = np.vstack([taken, *chosen]) if chosen else taken
            if _is_new(p, cur):
                chosen.append(p)
                score = score * np.minimum(1.0, np.linalg.norm(C - p, axis=1) / sep)
                break
    return _fill(chosen, batch, problem, rng, d, taken)


def propose_probabilistic(
    X: np.ndarray,
    Y: np.ndarray,
    surrogate,
    batch: int,
    rng: np.random.Generator,
    problem: Problem | None = None,
    avoid: np.ndarray | None = None,
    maxiter: int = 60,
) -> np.ndarray:
    """Maximum predictive standard deviation, found by differential evolution over the whole cube
    (Chai 2019). Batches are built one point at a time with a repulsion around the points already chosen."""
    n, d = X.shape
    _need_std(surrogate, "probabilistic")
    scale = Y.reshape(n, -1).std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    taken = np.vstack([X, avoid]) if avoid is not None and len(avoid) else X
    chosen: list[np.ndarray] = []
    sep = 0.1 * np.sqrt(d)

    def neg_std(u):
        p = _snap(problem, np.asarray(u)[None])
        s = float((surrogate.predict_unit_std(p) / scale).mean())
        for c in chosen:
            s *= min(1.0, float(np.linalg.norm(p[0] - c)) / sep)
        return -s

    for _ in range(batch):
        res = differential_evolution(
            neg_std, [(0.0, 1.0)] * d, maxiter=maxiter, popsize=10, tol=1e-6, polish=False,
            seed=int(rng.integers(2**31)), updating="immediate",
        )
        p = _snap(problem, res.x[None])[0]
        cur = np.vstack([taken, *chosen]) if chosen else taken
        if _is_new(p, cur):
            chosen.append(p)
    return _fill(chosen, batch, problem, rng, d, taken)


def _gp_blocks(model):
    """(signal kernel, noise variance, training inputs) of every output of a fitted ``GPSurrogate``."""
    blocks = []
    for gp in model.models_:
        k = gp.kernel_
        blocks.append((k.k1, float(k.k2.noise_level) + float(gp.alpha), gp.X_train_))
    return blocks


def propose_nipv(
    X: np.ndarray,
    Y: np.ndarray,
    surrogate,
    batch: int,
    rng: np.random.Generator,
    problem: Problem | None = None,
    avoid: np.ndarray | None = None,
    n_mc: int = 256,
    n_candidates: int = 3000,
) -> np.ndarray:
    """Negative integrated posterior variance (the qNIPV criterion).

    For a Gaussian process the posterior variance does not depend on the observed values, so the average
    variance over the domain left after adding a candidate ``c`` is known in closed form: it is lowered by
    ``mean_m cov(m, c)**2 / (var(c) + noise)`` over Monte-Carlo points ``m``. The candidate with the largest
    reduction (averaged over the objectives) is taken; batches add the fantasised point and repeat.
    Needs a :class:`~bdopt.surrogate.GPSurrogate`.
    """
    if not hasattr(surrogate, "models_") or not hasattr(surrogate.models_[0], "kernel_"):
        raise ValueError("the 'nipv' method needs a GPSurrogate")
    n, d = X.shape
    M = rng.random((n_mc, d))
    C = _snap(problem, rng.random((n_candidates, d)))
    blocks = _gp_blocks(surrogate)
    train = [b[2].copy() for b in blocks]
    taken = np.vstack([X, avoid]) if avoid is not None and len(avoid) else X
    chosen: list[np.ndarray] = []
    for _ in range(batch):
        red = np.zeros(len(C))
        for (kern, noise, _x), Xt in zip(blocks, train, strict=True):
            A = np.linalg.inv(kern(Xt) + noise * np.eye(len(Xt)))
            Kc, Km = kern(C, Xt), kern(M, Xt)
            cov = kern(M, C) - Km @ A @ Kc.T
            var = np.diag(kern(C)) - np.einsum("ij,jk,ik->i", Kc, A, Kc)
            red += (cov**2).mean(axis=0) / (np.maximum(var, 0) + noise)
        red /= len(blocks)
        for c in np.argsort(-red):
            cur = np.vstack([taken, *chosen]) if chosen else taken
            if _is_new(C[c], cur):
                chosen.append(C[c])
                train = [np.vstack([t, C[c]]) for t in train]
                break
    return _fill(chosen, batch, problem, rng, d, taken)


# ---------------------------------------------------------------------------------------------
# space filling / baseline
# ---------------------------------------------------------------------------------------------


def _latin_hypercube(d: int, n: int, rng: np.random.Generator) -> np.ndarray:
    # independent permutations of the n strata, jittered inside each stratum
    perms = np.column_stack([rng.permutation(n) for _ in range(d)])
    return (perms + rng.random((n, d))) / n


def propose_latin_hypercube(
    X: np.ndarray,
    batch: int,
    rng: np.random.Generator,
    problem: Problem | None = None,
    avoid: np.ndarray | None = None,
    n_trials: int = 50,
) -> np.ndarray:
    """``batch`` points from the best of ``n_trials`` random Latin hypercubes, judged by the maximin
    distance to the points already taken (so the design stays space filling as it grows)."""
    n, d = X.shape
    taken = np.vstack([X, avoid]) if avoid is not None and len(avoid) else X
    best, best_score = None, -1.0
    for _ in range(n_trials):
        U = _latin_hypercube(d, batch, rng)
        U = _snap(problem, U)
        cur = np.vstack([taken, U])
        D = _pairwise(cur)
        D[np.diag_indices(len(cur))] = np.inf
        score = float(D[len(taken):].min())
        if score > best_score:
            best, best_score = U, score
    chosen = [p for p in best if _is_new(p, taken)]
    return _fill(chosen, batch, problem, rng, d, taken)


def propose_random(
    X: np.ndarray, batch: int, rng: np.random.Generator, problem: Problem | None = None,
    avoid: np.ndarray | None = None,
) -> np.ndarray:
    n, d = X.shape
    taken = np.vstack([X, avoid]) if avoid is not None and len(avoid) else X
    return _fill([], batch, problem, rng, d, taken)
