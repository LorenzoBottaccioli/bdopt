"""Regression metrics per objective (columns)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> pd.DataFrame:
    """One row per objective: ``rmse``, ``rmse_pct`` (RMSE / mean of the data, as in the platform's
    notebooks), ``nrmse`` (RMSE / range), ``cv_rmse`` and ``nmbe`` in % (ASHRAE Guideline 14), ``mae``, ``r2``."""
    y_true = np.asarray(y_true, dtype=float).reshape(len(y_true), -1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(len(y_pred), -1)
    err = y_pred - y_true
    n = len(y_true)
    mean = y_true.mean(axis=0)
    rmse = np.sqrt((err**2).mean(axis=0))
    rng = y_true.max(axis=0) - y_true.min(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ss_res = (err**2).sum(axis=0)
        ss_tot = ((y_true - mean) ** 2).sum(axis=0)
        out = pd.DataFrame(
            {
                "rmse": rmse,
                "rmse_pct": rmse / np.abs(mean) * 100,
                "nrmse": rmse / rng,
                "cv_rmse": rmse / np.abs(mean) * 100,
                "nmbe": err.sum(axis=0) / ((n - 1) * mean) * 100 if n > 1 else np.nan,
                "mae": np.abs(err).mean(axis=0),
                "r2": 1 - ss_res / ss_tot,
            },
            index=names,
        )
    return out
