"""How many simulations were enough? Stopping points read off the history of an adaptive run."""

from __future__ import annotations

import pandas as pd


def suggest_budget(
    history: pd.DataFrame,
    tols: tuple[float, ...] = (0.1, 0.05, 0.02, 0.01),
    patience: int = 3,
    min_improvement: float = 0.02,
    max_std: tuple[float, ...] = (0.5, 0.3, 0.2),
) -> pd.DataFrame:
    """Number of simulations at which each stopping rule would have fired, from ``AdaptiveResult.history``.

    One row per rule: the iteration, the training points (``n_train``) and the total simulations
    (``n_evaluations``, which include the held-out test set); ``NaN`` where the rule never fired. Rules: the test
    error falling to each of ``tols`` (``scaled_rmse``), a plateau (no improvement of the best error by more than
    ``min_improvement`` over ``patience`` iterations) and, when the history has ``max_std``, the predictive
    uncertainty falling to each value of ``max_std``. Run the adaptive loop once with a generous budget and read the
    knee of the curve off this table instead of guessing the size of a design.
    """
    h = history.reset_index(drop=True)
    rows = []

    def add(rule: str, i):
        if i is None:
            rows.append({"rule": rule, "iteration": float("nan"), "n_train": float("nan"), "n_evaluations": float("nan")})
        else:
            rows.append({"rule": rule, "iteration": int(h.loc[i, "iteration"]), "n_train": int(h.loc[i, "n_train"]),
                         "n_evaluations": int(h.loc[i, "n_evaluations"])})

    for t in tols:
        hit = h.index[h["scaled_rmse"] <= t]
        add(f"scaled_rmse <= {t}", None if len(hit) == 0 else hit[0])
    plateau = None
    for i in range(patience, len(h)):
        before = h["scaled_rmse"].iloc[: i - patience + 1].min()
        recent = h["scaled_rmse"].iloc[i - patience + 1 : i + 1].min()
        if recent > before * (1 - min_improvement):
            plateau = i
            break
    add(f"plateau (patience {patience})", plateau)
    if "max_std" in h:
        for e in max_std:
            hit = h.index[h["max_std"] <= e]
            add(f"max_std <= {e}", None if len(hit) == 0 else hit[0])
    return pd.DataFrame(rows)
