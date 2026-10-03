"""scikit-learn surrogates: Gaussian process and MLP."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

from .base import Surrogate

# Inputs live in the unit cube, where typical sample spacing is ~n**(-1/d) (0.3-0.5 for tens of
# points). Length scales far below that only describe "noise" and are a degenerate optimum of the
# marginal likelihood (observed: prediction collapses to the mean), so they are excluded.
LENGTH_BOUNDS = (0.05, 100.0)

_KERNELS = {
    "matern52": lambda d: Matern(length_scale=np.full(d, 0.5), length_scale_bounds=LENGTH_BOUNDS, nu=2.5),
    "matern32": lambda d: Matern(length_scale=np.full(d, 0.5), length_scale_bounds=LENGTH_BOUNDS, nu=1.5),
    "rbf": lambda d: RBF(length_scale=np.full(d, 0.5), length_scale_bounds=LENGTH_BOUNDS),
    "rq": lambda d: RationalQuadratic(length_scale=0.5, alpha=0.5),
}


class GPSurrogate(Surrogate):
    """One Gaussian process per objective, with automatic relevance determination (a length scale per
    parameter) and a noise term. The kernel is chosen per objective by marginal likelihood among
    ``kernels``. Provides predictive uncertainty (``predict_unit_std``).

    Marginal-likelihood optimisation is multimodal, so it is restarted ``n_restarts`` times from random
    points; a fit whose length scales all sit at the lower bound (the degenerate "white noise" optimum)
    triggers up to ``max_extra_restarts`` more attempts."""

    has_uncertainty = True

    def __init__(
        self,
        kernels: tuple[str, ...] = ("matern52",),
        n_restarts: int = 8,
        random_state: int = 0,
        max_extra_restarts: int = 24,
    ):
        for k in kernels:
            if k not in _KERNELS:
                raise ValueError(f"unknown kernel {k!r}; choose from {sorted(_KERNELS)}")
        self.kernels, self.n_restarts, self.random_state = tuple(kernels), n_restarts, random_state
        self.max_extra_restarts = max_extra_restarts

    def _fit(self, U, Z):
        d = U.shape[1]
        self.models_, self.kernel_names_ = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            for j in range(Z.shape[1]):
                best, best_lml, best_name = None, -np.inf, ""
                for name in self.kernels:
                    kern = ConstantKernel(1.0, (1e-2, 1e2)) * _KERNELS[name](d) + WhiteKernel(
                        1e-4, (1e-9, 1e-1)
                    )
                    gp = self._fit_gp(kern, U, Z[:, j], self.n_restarts)
                    if self._collapsed(gp) and self.max_extra_restarts:
                        gp = self._fit_gp(kern, U, Z[:, j], self.n_restarts + self.max_extra_restarts)
                    lml = gp.log_marginal_likelihood_value_
                    if lml > best_lml:
                        best, best_lml, best_name = gp, lml, name
                self.models_.append(best)
                self.kernel_names_.append(best_name)

    def _fit_gp(self, kernel, U, z, n_restarts: int) -> GaussianProcessRegressor:
        return GaussianProcessRegressor(
            kernel, normalize_y=False, n_restarts_optimizer=n_restarts, random_state=self.random_state
        ).fit(U, z)

    @staticmethod
    def _collapsed(gp: GaussianProcessRegressor) -> bool:
        """True if every length scale is at (or next to) its lower bound: the degenerate optimum."""
        ls = [
            v
            for k, v in gp.kernel_.get_params().items()
            if k.endswith("length_scale") and not k.endswith("length_scale_bounds")
        ]
        if not ls:
            return False
        return bool(np.all(np.concatenate([np.atleast_1d(v) for v in ls]) < LENGTH_BOUNDS[0] * 1.5))

    def _predict(self, U):
        return np.column_stack([m.predict(U) for m in self.models_])

    def _predict_std(self, U):
        return np.column_stack([m.predict(U, return_std=True)[1] for m in self.models_])


class MLPSurrogate(Surrogate):
    """scikit-learn ``MLPRegressor`` with a small grid search over size and regularisation (the same
    search the platform's notebooks did by hand)."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[tuple[int, ...], ...] | None = None,
        alphas: tuple[float, ...] = (1e-3, 1e-1, 1.0),
        cv: int = 3,
        max_iter: int = 2000,
        random_state: int = 0,
    ):
        self.hidden_layer_sizes, self.alphas = hidden_layer_sizes, alphas
        self.cv, self.max_iter, self.random_state = cv, max_iter, random_state

    def _fit(self, U, Z):
        d = U.shape[1]
        sizes = self.hidden_layer_sizes or ((16 * d,), (16 * d, 16 * d))
        cv = max(2, min(self.cv, len(U)))
        grid = GridSearchCV(
            MLPRegressor(max_iter=self.max_iter, random_state=self.random_state),
            {"hidden_layer_sizes": list(sizes), "alpha": list(self.alphas)},
            cv=cv,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            grid.fit(U, Z)
        self.model_ = grid.best_estimator_
        self.best_params_ = grid.best_params_

    def _predict(self, U):
        out = self.model_.predict(U)
        return out.reshape(len(U), -1)
