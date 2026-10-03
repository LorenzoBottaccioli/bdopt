"""Plots (matplotlib only; ``pip install 'bdopt[viz]'``). Every function returns the figure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .problem import Problem


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("plots need matplotlib: pip install 'bdopt[viz]'") from e
    return plt


def _numeric(df: pd.DataFrame, problem: Problem) -> pd.DataFrame:
    """Columns as numbers (categorical parameters -> option index)."""
    out = pd.DataFrame(index=df.index)
    for p in problem.parameters:
        col = df[p.name]
        if p.kind == "categorical":
            lookup = {str(o): i for i, o in enumerate(p.value_descriptor.options)}
            out[p.name] = col.map(lambda v, lk=lookup: lk[str(v)])
        else:
            out[p.name] = col.astype(float)
    for n in problem.objective_names:
        if n in df:
            out[n] = df[n].astype(float)
    return out


def pairplot(data: pd.DataFrame, problem: Problem, size: float = 2.6):
    """Scatter of every objective (rows) against every parameter (columns)."""
    plt = _plt()
    num = _numeric(data, problem)
    ps, os_ = problem.parameter_names, problem.objective_names
    fig, axes = plt.subplots(len(os_), len(ps), figsize=(size * len(ps), size * len(os_)), squeeze=False)
    for i, o in enumerate(os_):
        for j, p in enumerate(ps):
            ax = axes[i][j]
            ax.scatter(num[p], num[o], s=10, alpha=0.7)
            if i == len(os_) - 1:
                ax.set_xlabel(p, fontsize=8)
            if j == 0:
                ax.set_ylabel(o, fontsize=8)
            ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def correlation_heatmap(data: pd.DataFrame, problem: Problem, annotate: bool = True):
    """Correlation of every parameter with every objective."""
    plt = _plt()
    num = _numeric(data, problem)
    corr = num.corr().loc[problem.parameter_names, problem.objective_names]
    fig, ax = plt.subplots(figsize=(1.2 * len(corr.columns) + 3, 0.5 * len(corr) + 2))
    im = ax.imshow(corr.to_numpy(float), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)), corr.index, fontsize=8)
    if annotate:
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="correlation")
    fig.tight_layout()
    return fig


def parallel_coordinates(data: pd.DataFrame, problem: Problem, color: str | None = None):
    """Parallel coordinates of parameters and objectives (each axis scaled to [0, 1])."""
    plt = _plt()
    num = _numeric(data, problem)
    cols = problem.parameter_names + [n for n in problem.objective_names if n in num]
    lo, hi = num[cols].min(), num[cols].max()
    norm = (num[cols] - lo) / (hi - lo).replace(0, 1)
    fig, ax = plt.subplots(figsize=(1.4 * len(cols) + 2, 4))
    c = norm[color] if color else norm[problem.objective_names[0]]
    cmap = plt.get_cmap("viridis")
    for k in range(len(norm)):
        ax.plot(range(len(cols)), norm.iloc[k].to_numpy(), color=cmap(c.iloc[k]), alpha=0.4, lw=1)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("normalised value")
    fig.tight_layout()
    return fig


def convergence(history: pd.DataFrame, tol: float | None = None):
    """Test error of the surrogate against the number of training points."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.semilogy(history["n_train"], history["scaled_rmse"], "o-")
    if tol is not None:
        ax.axhline(tol, color="r", ls="--", label=f"tol = {tol}")
        ax.legend()
    ax.set_xlabel("training points")
    ax.set_ylabel("scaled RMSE (test)")
    fig.tight_layout()
    return fig


def parity(y_true: pd.DataFrame, y_pred: pd.DataFrame):
    """Predicted vs true values, one panel per objective."""
    plt = _plt()
    cols = list(y_true.columns)
    fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 3.4), squeeze=False)
    for ax, c in zip(axes[0], cols, strict=True):
        ax.scatter(y_true[c], y_pred[c], s=12)
        lo, hi = float(np.nanmin(y_true[c])), float(np.nanmax(y_true[c]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(c, fontsize=9)
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
    fig.tight_layout()
    return fig


def pareto(front, background: pd.DataFrame | None = None, highlight: pd.Series | None = None):
    """Pareto front (2 objectives: scatter; 3+: 3-D scatter of the first three)."""
    plt = _plt()
    objs = front.problem.objective_names
    k = min(len(objs), 3)
    fig = plt.figure(figsize=(6, 4.6))
    ax = fig.add_subplot(111, projection="3d" if k == 3 else None)

    def pts(df):
        return [df[o] for o in objs[:k]]

    if background is not None:
        ax.scatter(*pts(background), marker="x", alpha=0.3, label="samples")
    ax.scatter(*pts(front.df), c="r", s=25, label="Pareto front")
    if highlight is not None:
        ax.scatter(*[[highlight[o]] for o in objs[:k]], c="k", s=90, marker="*", label="selected")
    ax.set_xlabel(objs[0])
    ax.set_ylabel(objs[1] if k > 1 else "")
    if k == 3:
        ax.set_zlabel(objs[2])
    ax.legend()
    fig.tight_layout()
    return fig
