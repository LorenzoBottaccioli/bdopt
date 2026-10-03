"""Infill criteria for adaptive sampling, in the unit cube (numpy only, no TensorFlow).

``flola_voronoi`` is the hybrid exploration/exploitation strategy of

* K. Crombecq, D. Gorissen, D. Deschrijver, T. Dhaene (2011), *A novel hybrid sequential design
  strategy for global surrogate modeling of computer experiments*, SIAM J. Sci. Comput. 33(4).
* J. van der Herten, I. Couckuyt, D. Deschrijver, T. Dhaene (2015), *A fuzzy hybrid sequential design
  strategy for global surrogate modeling of high-dimensional computer experiments*, SIAM J. Sci.
  Comput. 37(2).

Exploration = Monte-Carlo estimate of every sample's Voronoi cell volume (big cell = poorly covered
region). Exploitation = how far the response around each sample is from a local linear model
(nonlinear region = needs more samples). The next point goes into the cell with the highest
``volume + nonlinearity`` score, at the spot of the cell farthest from its sample. The fuzzy variant
weights neighbours by closeness (cohesion) and isolation (adhesion); here the fuzzy inference system
of the paper is replaced by an equivalent smooth weighting, which avoids the scikit-fuzzy dependency.
"""

from __future__ import annotations

import numpy as np

from ..problem import Problem


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


def nonlinearity(X: np.ndarray, Y: np.ndarray, fuzzy: bool = True) -> np.ndarray:
    """Per-sample deviation of the response from a locally fitted linear model (summed over outputs)."""
    n, d = X.shape
    Y = Y.reshape(n, -1)
    k = min(n - 1, (4 if fuzzy else 2) * d)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    out = np.zeros(n)
    for i in range(n):
        nb = np.argsort(D[i])[1 : k + 1]
        dx = X[nb] - X[i]
        dy = Y[nb] - Y[i]
        if fuzzy:
            dist = D[i, nb]
            cohesion = np.exp(-((dist / (np.median(dist) + 1e-12)) ** 2))
            sub = D[np.ix_(nb, nb)] + np.eye(len(nb)) * 1e9
            adhesion = sub.min(axis=1)
            isolation = adhesion / (adhesion.max() + 1e-12)
            w = cohesion * (0.5 + 0.5 * isolation)
            w = w / w.max()
        else:
            w = np.ones(len(nb))
        A = dx.T @ (w[:, None] * dx)
        A += np.eye(d) * 1e-8 * max(np.trace(A), 1e-12)
        G = np.linalg.solve(A, dx.T @ (w[:, None] * dy))
        resid = dy - dx @ G
        out[i] = float((w[:, None] * np.abs(resid)).sum())
    return out


def _is_new(p: np.ndarray, others: np.ndarray, tol: float = 1e-9) -> bool:
    return others.size == 0 or bool((np.abs(others - p).max(axis=1) > tol).all())


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
    """``batch`` new unit-cube points: hybrid LOLA/FLOLA-Voronoi (``exploit``) or pure Voronoi."""
    n, d = X.shape
    vol, P, idx, dist = voronoi_estimate(X, rng, n_mc)
    score = vol.copy()
    if exploit and Y is not None and n > 2:
        e = nonlinearity(X, Y, fuzzy)
        if e.sum() > 0:
            score = score + e / e.sum()
    taken = np.vstack([X, avoid]) if avoid is not None and len(avoid) else X
    chosen: list[np.ndarray] = []
    for ref in np.argsort(-score):
        if len(chosen) == batch:
            break
        cell = np.where(idx == ref)[0]
        for c in cell[np.argsort(-dist[cell])][:50]:  # farthest from the sample first
            p = P[c]
            if problem is not None:
                p = problem.snap(p)[0]
            cur = np.vstack([taken, *chosen]) if chosen else taken
            if _is_new(p, cur):
                chosen.append(p)
                break
    while len(chosen) < batch:
        cur = np.vstack([taken, *chosen]) if chosen else taken
        chosen.append(_fallback_point(problem, rng, d, cur))
    return np.array(chosen)


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
    """Points with the largest predictive uncertainty (needs a surrogate with ``predict_unit_std``)."""
    n, d = X.shape
    C = rng.random((n_candidates, d))
    if problem is not None:
        C = problem.snap(C)
    std = surrogate.predict_unit_std(C)
    if std is None:
        raise ValueError("the 'gp_variance' method needs a surrogate with uncertainty (GPSurrogate, ensemble TorchMLP)")
    scale = Y.reshape(n, -1).std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    score = (std / scale).mean(axis=1)
    taken = np.vstack([X, avoid]) if avoid is not None and len(avoid) else X
    chosen: list[np.ndarray] = []
    sep = 0.1 * np.sqrt(d)
    for _ in range(batch):
        order = np.argsort(-score)
        for c in order:
            p = C[c]
            cur = np.vstack([taken, *chosen]) if chosen else taken
            if _is_new(p, cur):
                chosen.append(p)
                score = score * np.minimum(1.0, np.linalg.norm(C - p, axis=1) / sep)
                break
        else:
            cur = np.vstack([taken, *chosen]) if chosen else taken
            chosen.append(_fallback_point(problem, rng, d, cur))
    return np.array(chosen)
