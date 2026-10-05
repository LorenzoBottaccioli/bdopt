"""More infill criteria for adaptive sampling (committee, error, optimisation and sequence based).

All functions work in the unit cube, return ``batch`` new points that are not in ``X``/``avoid`` and are snapped to
valid discrete values. ``Y`` is the table of objectives in real units (the criteria standardise it themselves).

=====================  ====================================================================================
``qbc``                query by committee: bootstrap committee of surrogates, largest disagreement
``loocv``              leave-one-out error of the samples, interpolated over the domain, times the distance
``lipschitz``          local Lipschitz constant of the nearest samples times the distance to them
``idw``                inverse-distance-weighted variance of the data around the surrogate + exploration term
``greedy``/``fps``     farthest point from every sample (maximin)
``cvt``                centroids of the least covered region (k-means on the poorest covered Monte-Carlo points)
``fisher_information`` D-optimal for the hyper-parameters of a Gaussian process
``expected_improvement``, ``probability_improvement``, ``lcb``, ``ucb``, ``sample_min``
                       optimisation oriented acquisition functions on a randomly weighted sum of the objectives
``soas``               low predicted value and high committee variance (Bamdad et al. 2020)
``sobol_seq``, ``halton``  continuation of a scrambled low-discrepancy sequence
``mixed``              round robin of random, Latin hypercube and Sobol' sequence (Legashev et al. 2024)
``error_window``       uniform in a window around the worst predicted point (Legashev et al. 2024)
=====================  ====================================================================================

References: Mehrotra & Yi (2023) J. Build. Eng. 72, 106614 (query by committee, leave-one-out, Lipschitz);
Bamdad, Cholette & Bell (2020) J. Build. Perform. Simul. 13(6) (committee with low predicted energy);
Lahmar et al. (2024) J. Build. Eng. 82, 108341 (error, sample-min, expected improvement); Nguyen & Nghiem (2026)
arXiv:2606.25301 (greedy, inverse distance weighting, Fisher information, UCB, PI); Legashev et al. (2024)
Modelling 5(4) (mixed sampling, error window); Bemporad (2021) inverse-distance-weighting exploration.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, qmc

from .designs import _fill, _is_new, _pairwise, _snap, propose_random

# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _taken(X, avoid):
    return np.vstack([X, avoid]) if avoid is not None and len(avoid) else X


def _pool(rng, d, n, problem):
    return _snap(problem, rng.random((n, d)))


def _pick(C, score, batch, taken, problem, rng, repulsion: bool = True):
    """Best-scoring candidates; after each pick the scores close to it are damped so batches spread out."""
    d = C.shape[1]
    s = np.asarray(score, dtype=float).copy()
    s[~np.isfinite(s)] = np.nanmin(s[np.isfinite(s)]) if np.isfinite(s).any() else 0.0
    lo, hi = s.min(), s.max()
    s = (s - lo) / (hi - lo) if hi > lo else np.ones_like(s)
    sep = 0.1 * np.sqrt(d)
    chosen: list[np.ndarray] = []
    for _ in range(batch):
        for c in np.argsort(-s):
            p = C[c]
            cur = np.vstack([taken, *chosen]) if chosen else taken
            if _is_new(p, cur):
                chosen.append(p)
                if repulsion:
                    s = s * np.minimum(1.0, np.linalg.norm(C - p, axis=1) / sep)
                break
    return _fill(chosen, batch, problem, rng, d, taken)


def _scale(Y):
    Y = np.asarray(Y, dtype=float).reshape(len(Y), -1)
    sd = Y.std(axis=0)
    return Y.mean(axis=0), np.where(sd > 0, sd, 1.0)


def _committee(model, X, Y, n_members, rng, problem):
    """``n_members`` clones of ``model`` fitted on bootstrap resamples of the data."""
    members = []
    n = len(X)
    for _ in range(n_members):
        for _try in range(10):
            idx = rng.integers(0, n, n)
            if len(np.unique(idx)) >= max(4, n // 2):
                break
        members.append(model.clone().fit_unit(X[idx], Y[idx], problem))
    return members


def _committee_stats(members, C):
    P = np.stack([m.predict_unit(C) for m in members])  # (members, candidates, objectives)
    return P.mean(axis=0), P.std(axis=0)


def _scalarise(mean, std, signs, weights, ym, ys):
    """Randomly weighted sum of the standardised objectives (all to be minimised) and its std."""
    sgn = np.asarray(signs, dtype=float)
    z = (mean - ym) / ys * sgn
    mu = z @ weights
    sd = None if std is None else np.sqrt((((std / ys) ** 2) * weights**2).sum(axis=1))
    return mu, sd


def _signs(problem, m):
    return problem.signs if problem is not None and len(problem.signs) == m else np.ones(m)


def _need_uncertainty(model, name):
    if not getattr(model, "has_uncertainty", False):
        raise ValueError(f"the {name!r} method needs a surrogate with uncertainty (e.g. GPSurrogate)")


# ---------------------------------------------------------------------------------------------
# error / disagreement based
# ---------------------------------------------------------------------------------------------


def propose_qbc(X, Y, model, batch, rng, problem=None, avoid=None, n_members=5, n_pool=3000):
    """Query by committee: the pool point on which a bootstrap committee of ``model`` disagrees most."""
    d = X.shape[1]
    members = _committee(model, X, Y, n_members, rng, problem)
    C = _pool(rng, d, n_pool, problem)
    _, sd = _committee_stats(members, C)
    _, ys = _scale(Y)
    return _pick(C, (sd / ys).mean(axis=1), batch, _taken(X, avoid), problem, rng)


def propose_loocv(X, Y, model, batch, rng, problem=None, avoid=None, n_pool=3000, max_folds=20):
    """Leave-one-out cross-validation error of every sample, interpolated (inverse distance weighting) over a pool
    of candidates and multiplied by the distance to the nearest sample (error-weighted maximin)."""
    from sklearn.model_selection import KFold

    n, d = X.shape
    Y = np.asarray(Y, dtype=float).reshape(n, -1)
    _, ys = _scale(Y)
    err = np.zeros(n)
    for tr, te in KFold(min(n, max_folds), shuffle=True, random_state=int(rng.integers(2**31))).split(X):
        m = model.clone().fit_unit(X[tr], Y[tr], problem)
        err[te] = (np.abs(m.predict_unit(X[te]) - Y[te]) / ys).mean(axis=1)
    C = _pool(rng, d, n_pool, problem)
    D = np.sqrt(((C[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    w = 1.0 / np.maximum(D, 1e-9) ** 2
    interp = (w * err[None, :]).sum(axis=1) / w.sum(axis=1)
    return _pick(C, interp * D.min(axis=1), batch, _taken(X, avoid), problem, rng)


def propose_lipschitz(X, Y, batch, rng, problem=None, avoid=None, n_pool=3000, k=5):
    """Lipschitz criterion: for the sample nearest to a candidate, the largest slope towards its ``k`` nearest
    samples times the distance to the candidate is an upper bound of how much the response may change there."""
    n, d = X.shape
    Y = np.asarray(Y, dtype=float).reshape(n, -1)
    _, ys = _scale(Y)
    Z = Y / ys
    D = _pairwise(X)
    np.fill_diagonal(D, np.inf)
    slope = np.zeros(n)
    for i in range(n):
        nb = np.argsort(D[i])[: min(k, n - 1)]
        slope[i] = (np.abs(Z[nb] - Z[i]).sum(axis=1) / D[i, nb]).max()
    C = _pool(rng, d, n_pool, problem)
    dist = np.sqrt(((C[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    nearest = dist.argmin(axis=1)
    return _pick(C, slope[nearest] * dist.min(axis=1), batch, _taken(X, avoid), problem, rng)


def propose_idw(X, Y, model, batch, rng, problem=None, avoid=None, n_pool=3000, delta=1.0):
    """Inverse-distance-weighted exploration (Bemporad 2021; Nguyen & Nghiem 2026): the IDW standard deviation of the
    data around the surrogate prediction plus ``delta`` times an exploration term that vanishes near samples."""
    n, d = X.shape
    Y = np.asarray(Y, dtype=float).reshape(n, -1)
    ym, ys = _scale(Y)
    C = _pool(rng, d, n_pool, problem)
    D2 = ((C[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    w = 1.0 / np.maximum(D2, 1e-12)
    wn = w / w.sum(axis=1, keepdims=True)
    pred = (model.predict_unit(C) - ym) / ys
    Z = (Y - ym) / ys
    s = np.sqrt((wn[:, :, None] * (Z[None, :, :] - pred[:, None, :]) ** 2).sum(axis=1)).mean(axis=1)
    z = (2 / np.pi) * np.arctan(1.0 / w.sum(axis=1))
    return _pick(C, s + delta * z, batch, _taken(X, avoid), problem, rng)


# ---------------------------------------------------------------------------------------------
# space filling
# ---------------------------------------------------------------------------------------------


def propose_greedy(X, batch, rng, problem=None, avoid=None, n_pool=5000):
    """Greedy farthest-point (maximin) sampling: the pool point farthest from every sample."""
    n, d = X.shape
    C = _pool(rng, d, n_pool, problem)
    taken = _taken(X, avoid)
    dmin = np.sqrt(((C[:, None, :] - taken[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return _pick(C, dmin, batch, taken, problem, rng)


def propose_cvt(X, batch, rng, problem=None, avoid=None, n_pool=8000, share=0.25):
    """CVT-inspired: k-means (``batch`` clusters) on the ``share`` of Monte-Carlo points that are farthest from every
    sample; the centroids sit in the least covered region (existing samples cannot move, so this is not a full CVT)."""
    from sklearn.cluster import KMeans

    n, d = X.shape
    C = rng.random((n_pool, d))
    taken = _taken(X, avoid)
    dmin = np.sqrt(((C[:, None, :] - taken[None, :, :]) ** 2).sum(-1)).min(axis=1)
    far = C[np.argsort(-dmin)[: max(batch * 5, int(share * n_pool))]]
    km = KMeans(n_clusters=batch, n_init=3, random_state=int(rng.integers(2**31))).fit(far)
    chosen = [p for p in _snap(problem, km.cluster_centers_) if _is_new(p, taken)]
    return _fill(chosen, batch, problem, rng, d, taken)


def propose_sequence(kind, X, batch, rng, problem=None, avoid=None, seed=0):
    """Continue a scrambled ``'sobol'`` or ``'halton'`` sequence after the points already taken."""
    n, d = X.shape
    taken = _taken(X, avoid)
    engine = (qmc.Sobol if kind == "sobol" else qmc.Halton)(d=d, scramble=True, seed=int(seed))
    skip = len(taken)
    if kind == "sobol":
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            engine.fast_forward(skip)
            cand = engine.random(batch * 20)
    else:
        engine.fast_forward(skip)
        cand = engine.random(batch * 20)
    chosen: list[np.ndarray] = []
    for p in _snap(problem, cand):
        cur = np.vstack([taken, *chosen]) if chosen else taken
        if _is_new(p, cur):
            chosen.append(p)
        if len(chosen) == batch:
            break
    return _fill(chosen, batch, problem, rng, d, taken)


def propose_mixed(X, batch, rng, problem=None, avoid=None, iteration=1, seed=0):
    """Round robin of random, Latin hypercube and Sobol' sequence by iteration (Legashev et al. 2024; their full
    factorial is left out because it piles points in the corners)."""
    from .designs import propose_latin_hypercube

    k = (iteration - 1) % 3
    if k == 0:
        return propose_random(X, batch, rng, problem, avoid)
    if k == 1:
        return propose_latin_hypercube(X, batch, rng, problem, avoid)
    return propose_sequence("sobol", X, batch, rng, problem, avoid, seed)


def propose_error_window(X, anchor, batch, rng, problem=None, avoid=None, window=0.3):
    """Uniform points in a box of relative width ``window`` centred on ``anchor`` (the worst predicted point)."""
    n, d = X.shape
    taken = _taken(X, avoid)
    lo = np.clip(anchor - window / 2, 0, 1)
    hi = np.clip(anchor + window / 2, 0, 1)
    chosen: list[np.ndarray] = []
    for _ in range(200):
        p = _snap(problem, (lo + rng.random(d) * (hi - lo))[None])[0]
        cur = np.vstack([taken, *chosen]) if chosen else taken
        if _is_new(p, cur):
            chosen.append(p)
        if len(chosen) == batch:
            break
    return _fill(chosen, batch, problem, rng, d, taken)


# ---------------------------------------------------------------------------------------------
# optimisation oriented
# ---------------------------------------------------------------------------------------------


def _acquisition(kind, X, Y, model, batch, rng, problem, avoid, n_pool=3000, kappa=2.0, xi=0.0):
    n, d = X.shape
    Y = np.asarray(Y, dtype=float).reshape(n, -1)
    ym, ys = _scale(Y)
    signs = _signs(problem, Y.shape[1])
    w = rng.dirichlet(np.ones(Y.shape[1]))
    C = _pool(rng, d, n_pool, problem)
    mean = model.predict_unit(C)
    std = model.predict_unit_std(C) if getattr(model, "has_uncertainty", False) else None
    mu, sd = _scalarise(mean, std, signs, w, ym, ys)
    best = ((Y - ym) / ys * signs @ w).min()
    if kind == "sample_min":
        score = -mu
    elif kind == "lcb":
        score = -(mu - kappa * sd)
    elif kind == "ucb":
        score = mu + kappa * sd
    else:
        sd = np.maximum(sd, 1e-9)
        z = (best - mu - xi) / sd
        score = sd * (z * norm.cdf(z) + norm.pdf(z)) if kind == "expected_improvement" else norm.cdf(z)
    return _pick(C, score, batch, _taken(X, avoid), problem, rng)


def propose_acquisition(kind, X, Y, model, batch, rng, problem=None, avoid=None):
    """``'expected_improvement'``, ``'probability_improvement'``, ``'lcb'`` (lower confidence bound), ``'ucb'``
    (upper confidence bound, an uncertainty-seeking criterion) or ``'sample_min'`` (the surrogate's minimum) on a randomly
    weighted sum of the standardised objectives (maximised objectives enter with the opposite sign; the weights are
    redrawn every iteration, as in ParEGO, so the Pareto front is covered)."""
    if kind != "sample_min":
        _need_uncertainty(model, kind)
    return _acquisition(kind, X, Y, model, batch, rng, problem, avoid)


def propose_soas(X, Y, model, batch, rng, problem=None, avoid=None, iteration=1, n_members=5, n_pool=3000):
    """Bamdad et al. (2020): among the pool points whose committee-mean predicted (weighted) value is below the
    ``p``-th percentile, those with the largest committee variance. ``p`` starts at 50 % and falls by 5 points per
    iteration to 1 %."""
    n, d = X.shape
    Y = np.asarray(Y, dtype=float).reshape(n, -1)
    ym, ys = _scale(Y)
    members = _committee(model, X, Y, n_members, rng, problem)
    C = _pool(rng, d, n_pool, problem)
    mean, sd = _committee_stats(members, C)
    w = rng.dirichlet(np.ones(Y.shape[1]))
    mu, _ = _scalarise(mean, None, _signs(problem, Y.shape[1]), w, ym, ys)
    p = max(1.0, 50.0 - 5.0 * (iteration - 1))
    keep = mu <= np.percentile(mu, p)
    var = ((sd / ys) ** 2).mean(axis=1)
    score = np.where(keep, var, -1.0)
    return _pick(C, score, batch, _taken(X, avoid), problem, rng)


# ---------------------------------------------------------------------------------------------
# Fisher information (Gaussian process)
# ---------------------------------------------------------------------------------------------


def propose_fisher_information(X, Y, model, batch, rng, problem=None, avoid=None, n_pool=200):
    """D-optimal design for the kernel hyper-parameters of a fitted :class:`~bdopt.surrogate.GPSurrogate`: the candidate
    that raises ``log det`` of the Fisher information matrix ``I_ij = 1/2 tr(K^-1 dK_i K^-1 dK_j)`` the most."""
    if not hasattr(model, "models_") or not hasattr(model.models_[0], "kernel_"):
        raise ValueError("the 'fisher_information' method needs a GPSurrogate")
    n, d = X.shape
    C = _pool(rng, d, n_pool, problem)
    taken = _taken(X, avoid)
    kernels = [gp.kernel_ for gp in model.models_]

    def logdet_fisher(kern, Xa):
        K, dK = kern(Xa, eval_gradient=True)
        K = K + 1e-8 * np.eye(len(Xa))
        Kinv = np.linalg.inv(K)
        M = [Kinv @ dK[:, :, i] for i in range(dK.shape[2])]
        F = 0.5 * np.array([[np.trace(a @ b) for b in M] for a in M])
        sign, ld = np.linalg.slogdet(F + 1e-9 * np.eye(len(F)))
        return ld if sign > 0 else -np.inf

    score = np.zeros(len(C))
    for kern in kernels:
        for i, c in enumerate(C):
            score[i] += logdet_fisher(kern, np.vstack([X, c]))
    return _pick(C, score / len(kernels), batch, taken, problem, rng)


__all__ = [
    "propose_acquisition",
    "propose_cvt",
    "propose_error_window",
    "propose_fisher_information",
    "propose_greedy",
    "propose_idw",
    "propose_lipschitz",
    "propose_loocv",
    "propose_mixed",
    "propose_qbc",
    "propose_sequence",
    "propose_soas",
]
