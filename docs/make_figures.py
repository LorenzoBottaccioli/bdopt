"""Regenerate the figures used by the documentation from real runs (no EnergyPlus needed).

    python docs/make_figures.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bdopt import FunctionEvaluator, GenericSelector, Parameter, Problem, RangeParameter, sampling, viz
from bdopt.adaptive import AdaptiveSampler
from bdopt.surrogate import GPSurrogate

OUT = Path(__file__).parent / "_static" / "img"
OUT.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")


def bump_problem():
    params = [
        Parameter(GenericSelector(lambda i, v: None, n), RangeParameter(0, 1), name=n) for n in ("x1", "x2")
    ]
    problem = Problem(params, ["f"])

    def f(r):
        return [0.3 * r["x1"] + np.exp(-(((r["x1"] - 0.7) ** 2 + (r["x2"] - 0.3) ** 2) / 0.012))]

    return problem, FunctionEvaluator(problem, f)


def main() -> None:
    problem, ev = bump_problem()
    gx, gy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    truth = 0.3 * gx + np.exp(-(((gx - 0.7) ** 2 + (gy - 0.3) ** 2) / 0.012))

    result = AdaptiveSampler(problem, ev, method="flola_voronoi", seed=3).run(
        n_initial=12, n_test=10, max_iter=38, tol=0, verbose=False
    )
    n = len(result.data)
    lhs = sampling.lhs(problem, n, seed=3)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    for ax, pts, title in (
        (axes[0], lhs, f"Latin hypercube ({n} points)"),
        (axes[1], result.data, f"Adaptive FLOLA-Voronoi ({n} points)"),
    ):
        ax.contourf(gx, gy, truth, levels=20, cmap="Greys", alpha=0.55)
        if ax is axes[1]:
            order = np.arange(n)
            sc = ax.scatter(pts["x1"], pts["x2"], c=order, cmap="viridis", s=26, edgecolor="k", linewidth=0.4)
        else:
            ax.scatter(pts["x1"], pts["x2"], c="tab:blue", s=26, edgecolor="k", linewidth=0.4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x1")
    axes[0].set_ylabel("x2")
    cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("order in which the point was simulated")
    fig.savefig(OUT / "adaptive_vs_lhs.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig = viz.convergence(result.history)
    fig.savefig(OUT / "convergence.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    test = result.test
    pred = GPSurrogate().fit(result.data, problem).predict(test)
    fig = viz.parity(test[["f"]], pred)
    fig.savefig(OUT / "parity.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("figures written to", OUT, "| final scaled RMSE", round(float(result.history["scaled_rmse"].iloc[-1]), 4))


if __name__ == "__main__":
    main()
