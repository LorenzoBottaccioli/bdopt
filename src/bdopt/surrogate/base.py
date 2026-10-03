"""Surrogate model interface.

A surrogate is trained on a table of evaluated designs (parameter columns + objective columns) and
predicts the objectives for new parameter tables. Inputs live in the problem's unit cube; outputs are
standardised internally, so nothing has to be scaled by hand and nothing leaks between train and test.
"""

from __future__ import annotations

import pickle
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from ..problem import Problem
from .metrics import regression_metrics


class Surrogate(ABC):
    """Base class. Subclasses implement :meth:`_fit` and :meth:`_predict` on standardised outputs."""

    problem: Problem | None = None
    #: whether :meth:`predict_unit_std` returns an uncertainty estimate
    has_uncertainty = False

    # -- to implement ----------------------------------------------------------------------------
    @abstractmethod
    def _fit(self, U: np.ndarray, Z: np.ndarray) -> None:
        """Fit on unit-cube inputs ``U (n, d)`` and standardised outputs ``Z (n, m)``."""

    @abstractmethod
    def _predict(self, U: np.ndarray) -> np.ndarray:
        """Standardised predictions ``(n, m)``."""

    def _predict_std(self, U: np.ndarray) -> np.ndarray | None:
        return None

    # -- public API ------------------------------------------------------------------------------
    def fit(self, data: pd.DataFrame, problem: Problem) -> Surrogate:
        """Train on ``data`` (parameter + objective columns). Rows with NaN are dropped."""
        U = problem.encode(data)
        Y = data[problem.objective_names].to_numpy(dtype=float)
        return self.fit_unit(U, Y, problem)

    def fit_unit(self, U: np.ndarray, Y: np.ndarray, problem: Problem | None = None) -> Surrogate:
        U, Y = np.asarray(U, dtype=float), np.asarray(Y, dtype=float).reshape(len(U), -1)
        ok = np.isfinite(U).all(axis=1) & np.isfinite(Y).all(axis=1)
        if not ok.all():
            warnings.warn(f"dropping {int((~ok).sum())} training rows with NaN/inf", stacklevel=2)
            U, Y = U[ok], Y[ok]
        if len(U) < 3:
            raise ValueError(f"need at least 3 training points, got {len(U)}")
        self.problem = problem if problem is not None else self.problem
        self._y_mean = Y.mean(axis=0)
        sd = Y.std(axis=0)
        self._y_std = np.where(sd > 0, sd, 1.0)
        self.n_train_ = len(U)
        self._fit(U, (Y - self._y_mean) / self._y_std)
        return self

    def predict_unit(self, U: np.ndarray) -> np.ndarray:
        """Predictions ``(n, n_obj)`` in real units for unit-cube points."""
        U = np.atleast_2d(np.asarray(U, dtype=float))
        return self._predict(U) * self._y_std + self._y_mean

    def predict_unit_std(self, U: np.ndarray) -> np.ndarray | None:
        """Predictive standard deviation ``(n, n_obj)`` in real units (``None`` if unavailable)."""
        s = self._predict_std(np.atleast_2d(np.asarray(U, dtype=float)))
        return None if s is None else s * self._y_std

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict objectives for a parameter table."""
        self._need_problem()
        out = self.predict_unit(self.problem.encode(X))
        return pd.DataFrame(out, columns=self.problem.objective_names, index=X.index)

    # an optimizer can use a surrogate anywhere it expects an evaluator
    evaluate_unit = predict_unit

    def score(self, data: pd.DataFrame) -> pd.DataFrame:
        """Metrics per objective on held-out ``data``."""
        self._need_problem()
        pred = self.predict(data)
        return regression_metrics(
            data[self.problem.objective_names].to_numpy(float), pred.to_numpy(float),
            self.problem.objective_names,
        )

    def cross_validate(self, data: pd.DataFrame, problem: Problem, k: int = 5, seed: int = 0) -> pd.DataFrame:
        """K-fold cross-validated metrics (a fresh copy of the model is trained per fold)."""
        from sklearn.model_selection import KFold

        U = problem.encode(data)
        Y = data[problem.objective_names].to_numpy(float)
        pred = np.full_like(Y, np.nan)
        for tr, te in KFold(k, shuffle=True, random_state=seed).split(U):
            m = self.clone().fit_unit(U[tr], Y[tr], problem)
            pred[te] = m.predict_unit(U[te])
        return regression_metrics(Y, pred, problem.objective_names)

    def clone(self) -> Surrogate:
        """An unfitted copy with the same hyper-parameters."""
        new = object.__new__(type(self))
        new.__dict__.update(
            {k: v for k, v in self.__dict__.items() if not k.endswith("_") and not k.startswith("_")}
        )
        return new

    def _need_problem(self) -> None:
        if self.problem is None:
            raise RuntimeError("surrogate not fitted with a problem; call fit(data, problem)")

    # -- persistence -----------------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Pickle the fitted model. If the problem cannot be pickled (e.g. it holds lambdas) it is left
        out and must be given to :meth:`load`."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            blob = pickle.dumps(self)
        except (pickle.PicklingError, AttributeError, TypeError):
            keep, self.problem = self.problem, None
            try:
                blob = pickle.dumps(self)
            finally:
                self.problem = keep
        p.write_bytes(blob)
        return p

    @staticmethod
    def load(path: str | Path, problem: Problem | None = None) -> Surrogate:
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        if problem is not None:
            model.problem = problem
        return model
