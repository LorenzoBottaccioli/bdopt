"""scikit-learn surrogates: Gaussian process and MLP."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV, LassoLarsCV, RidgeCV
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .base import Surrogate

# Inputs live in the unit cube, where typical sample spacing is ~n**(-1/d) (0.3-0.5 for tens of
# points). Length scales far below that only describe "noise" and are a degenerate optimum of the
# marginal likelihood (observed: prediction collapses to the mean), so they are excluded.
LENGTH_BOUNDS = (0.05, 100.0)

def _matern52(d):
    return Matern(length_scale=np.full(d, 0.5), length_scale_bounds=LENGTH_BOUNDS, nu=2.5)


def _matern32(d):
    return Matern(length_scale=np.full(d, 0.5), length_scale_bounds=LENGTH_BOUNDS, nu=1.5)


def _rbf(d):
    return RBF(length_scale=np.full(d, 0.5), length_scale_bounds=LENGTH_BOUNDS)


def _rq(d):
    return RationalQuadratic(length_scale=0.5, alpha=0.5)


_KERNELS = {"matern52": _matern52, "matern32": _matern32, "rbf": _rbf, "rq": _rq}


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


# ---------------------------------------------------------------------------------------------
# tree ensembles
# ---------------------------------------------------------------------------------------------


class _ForestSurrogate(Surrogate):
    """Shared code of the random-forest-like models: the spread of the individual trees is the
    uncertainty estimate."""

    has_uncertainty = True
    _estimator = None

    def __init__(self, n_estimators: int = 300, min_samples_leaf: int = 1, max_features: float | str = 1.0,
                 random_state: int = 0):
        self.n_estimators, self.min_samples_leaf = n_estimators, min_samples_leaf
        self.max_features, self.random_state = max_features, random_state

    def _fit(self, U, Z):
        self.model_ = self._estimator(
            n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features, random_state=self.random_state, n_jobs=1,
        ).fit(U, Z)

    def _predict(self, U):
        return self.model_.predict(U).reshape(len(U), -1)

    def _predict_std(self, U):
        trees = np.stack([t.predict(U).reshape(len(U), -1) for t in self.model_.estimators_])
        return trees.std(axis=0)


class RandomForestSurrogate(_ForestSurrogate):
    """Random forest (bagged regression trees). Piecewise constant, so coarse on small data sets, but
    robust to mixed and discrete parameters; uncertainty from the spread of the trees."""

    _estimator = RandomForestRegressor


class ExtraTreesSurrogate(_ForestSurrogate):
    """Extremely randomised trees: smoother than a random forest on smooth responses."""

    _estimator = ExtraTreesRegressor


class GradientBoostingSurrogate(Surrogate):
    """Gradient boosted trees, one model per objective."""

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05, max_depth: int = 3,
                 subsample: float = 0.8, random_state: int = 0):
        self.n_estimators, self.learning_rate, self.max_depth = n_estimators, learning_rate, max_depth
        self.subsample, self.random_state = subsample, random_state

    def _fit(self, U, Z):
        base = GradientBoostingRegressor(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth,
            subsample=self.subsample, random_state=self.random_state,
        )
        self.model_ = MultiOutputRegressor(base).fit(U, Z)

    def _predict(self, U):
        return self.model_.predict(U).reshape(len(U), -1)


# ---------------------------------------------------------------------------------------------
# kernel and distance based models
# ---------------------------------------------------------------------------------------------


def _cv(n: int, k: int) -> int:
    return max(2, min(k, n))


class SVRSurrogate(Surrogate):
    """Support vector regression (RBF), one model per objective, ``C`` and ``gamma`` by cross-validation."""

    def __init__(self, Cs: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0),
                 gammas: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0), epsilon: float = 0.01, cv: int = 3):
        self.Cs, self.gammas, self.epsilon, self.cv = tuple(Cs), tuple(gammas), epsilon, cv

    def _fit(self, U, Z):
        d = U.shape[1]
        self.models_ = []
        for j in range(Z.shape[1]):
            grid = GridSearchCV(
                SVR(epsilon=self.epsilon),
                {"C": list(self.Cs), "gamma": [g / d * 3 for g in self.gammas]},
                cv=_cv(len(U), self.cv),
            )
            self.models_.append(grid.fit(U, Z[:, j]).best_estimator_)

    def _predict(self, U):
        return np.column_stack([m.predict(U) for m in self.models_])


class KernelRidgeSurrogate(Surrogate):
    """Kernel ridge regression (RBF or Laplacian kernel) with ``alpha`` and ``gamma`` by cross-validation."""

    def __init__(self, kernels: tuple[str, ...] = ("rbf", "laplacian"),
                 alphas: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1e-1, 1.0),
                 gammas: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0), cv: int = 3):
        self.kernels, self.alphas, self.gammas, self.cv = tuple(kernels), tuple(alphas), tuple(gammas), cv

    def _fit(self, U, Z):
        grid = GridSearchCV(
            KernelRidge(),
            {"kernel": list(self.kernels), "alpha": list(self.alphas), "gamma": list(self.gammas)},
            cv=_cv(len(U), self.cv),
        )
        self.model_ = grid.fit(U, Z).best_estimator_
        self.best_params_ = grid.best_params_

    def _predict(self, U):
        return self.model_.predict(U).reshape(len(U), -1)


class KNNSurrogate(Surrogate):
    """Distance-weighted k nearest neighbours (a baseline: it cannot extrapolate or interpolate smoothly)."""

    def __init__(self, n_neighbors: tuple[int, ...] = (2, 3, 5, 8), cv: int = 3):
        self.n_neighbors, self.cv = tuple(n_neighbors), cv

    def _fit(self, U, Z):
        ks = [k for k in self.n_neighbors if k <= len(U)] or [1]
        grid = GridSearchCV(
            KNeighborsRegressor(weights="distance"), {"n_neighbors": ks}, cv=_cv(len(U), self.cv)
        )
        self.model_ = grid.fit(U, Z).best_estimator_

    def _predict(self, U):
        return self.model_.predict(U).reshape(len(U), -1)


class PolynomialSurrogate(Surrogate):
    """Polynomial response surface with ridge regularisation; the degree (up to ``max_degree``) and the
    regularisation are chosen by cross-validation. The classical metamodel of building optimisation."""

    def __init__(self, max_degree: int = 3, alphas: tuple[float, ...] = (1e-6, 1e-3, 1e-1, 1.0, 10.0), cv: int = 5):
        self.max_degree, self.alphas, self.cv = max_degree, tuple(alphas), cv

    def _fit(self, U, Z):
        best, best_score = None, -np.inf
        for deg in range(1, self.max_degree + 1):
            pipe = make_pipeline(PolynomialFeatures(deg), RidgeCV(alphas=self.alphas))
            n_terms = PolynomialFeatures(deg).fit(U[:1]).n_output_features_
            if n_terms > 3 * len(U) and deg > 1:
                continue  # far more coefficients than samples
            score = cross_val_score(pipe, U, Z, cv=_cv(len(U), self.cv), scoring="neg_mean_squared_error").mean()
            if score > best_score:
                best, best_score, self.degree_ = pipe, score, deg
        self.model_ = best.fit(U, Z)

    def _predict(self, U):
        return self.model_.predict(U).reshape(len(U), -1)


class RBFSurrogate(Surrogate):
    """Radial basis function interpolation (scipy ``RBFInterpolator``). Kernel and smoothing are chosen by
    cross-validation. Exact (or almost exact) at the training points, no uncertainty estimate."""

    def __init__(self, kernels: tuple[str, ...] = ("thin_plate_spline", "cubic", "multiquadric"),
                 smoothings: tuple[float, ...] = (0.0, 1e-6, 1e-4, 1e-2), cv: int = 5, random_state: int = 0):
        self.kernels, self.smoothings, self.cv, self.random_state = tuple(kernels), tuple(smoothings), cv, random_state

    def _make(self, U, Z, kernel, smoothing):
        eps = 1.0 if kernel in ("multiquadric", "gaussian", "inverse_multiquadric") else None
        return RBFInterpolator(U, Z, kernel=kernel, smoothing=smoothing, epsilon=eps)

    def _fit(self, U, Z):
        from sklearn.model_selection import KFold

        best, best_err = (self.kernels[0], self.smoothings[0]), np.inf
        folds = list(KFold(_cv(len(U), self.cv), shuffle=True, random_state=self.random_state).split(U))
        for kernel in self.kernels:
            for s in self.smoothings:
                err = 0.0
                try:
                    for tr, te in folds:
                        err += float(((self._make(U[tr], Z[tr], kernel, s)(U[te]) - Z[te]) ** 2).mean())
                except np.linalg.LinAlgError:
                    continue
                if err < best_err:
                    best, best_err = (kernel, s), err
        self.best_params_ = {"kernel": best[0], "smoothing": best[1]}
        self.model_ = self._make(U, Z, *best)

    def _predict(self, U):
        return np.asarray(self.model_(U)).reshape(len(U), -1)


# ---------------------------------------------------------------------------------------------
# model selection
# ---------------------------------------------------------------------------------------------


class AutoSurrogate(Surrogate):
    """Try several surrogates, keep the one with the lowest cross-validated error.

    ``candidates`` are unfitted surrogates (default: Gaussian process, extra trees, kernel ridge, SVR,
    polynomial, RBF interpolation, MLP). ``selected_`` and ``cv_scores_`` (RMSE in standard deviations of
    each objective, mean over objectives) are available after fitting; the chosen model is refitted on all
    the data. ``has_uncertainty`` follows the chosen model.
    """

    def __init__(self, candidates: tuple[Surrogate, ...] | None = None, cv: int = 5, random_state: int = 0):
        self.candidates, self.cv, self.random_state = candidates, cv, random_state

    def _default_candidates(self) -> list[Surrogate]:
        return [GPSurrogate(), ExtraTreesSurrogate(), KernelRidgeSurrogate(), SVRSurrogate(),
                PolynomialSurrogate(), RBFSurrogate(), MLPSurrogate()]

    def _fit(self, U, Z):
        from sklearn.model_selection import KFold

        cands = list(self.candidates) if self.candidates else self._default_candidates()
        scores = {}
        folds = list(KFold(_cv(len(U), self.cv), shuffle=True, random_state=self.random_state).split(U))
        for c in cands:
            name = type(c).__name__
            pred = np.full_like(Z, np.nan)
            try:
                for tr, te in folds:
                    pred[te] = c.clone().fit_unit(U[tr], Z[tr], self.problem).predict_unit(U[te])
            except Exception:  # a candidate that cannot be fitted on this data is simply not eligible
                continue
            scores[name] = float(np.sqrt(((pred - Z) ** 2).mean()))
        if not scores:
            raise RuntimeError("no candidate surrogate could be fitted")
        best = min(scores, key=scores.get)
        self.cv_scores_ = pd.Series(scores).sort_values()
        self.selected_ = best
        chosen = next(c for c in cands if type(c).__name__ == best)
        self.model_ = chosen.clone().fit_unit(U, Z, self.problem)
        self.has_uncertainty = self.model_.has_uncertainty

    def _predict(self, U):
        return (self.model_.predict_unit(U) - self.model_._y_mean) / self.model_._y_std

    def _predict_std(self, U):
        if not self.model_.has_uncertainty:
            return None
        return self.model_._predict_std(U)


# ---------------------------------------------------------------------------------------------
# more tree models
# ---------------------------------------------------------------------------------------------


class DecisionTreeSurrogate(Surrogate):
    """A single regression tree with depth and leaf size chosen by cross-validation (the weakest common baseline:
    piecewise constant)."""

    def __init__(self, max_depths: tuple[int, ...] = (3, 5, 8, 12), min_samples_leaf: tuple[int, ...] = (1, 2, 4),
                 cv: int = 3, random_state: int = 0):
        self.max_depths, self.min_samples_leaf, self.cv = tuple(max_depths), tuple(min_samples_leaf), cv
        self.random_state = random_state

    def _fit(self, U, Z):
        grid = GridSearchCV(
            DecisionTreeRegressor(random_state=self.random_state),
            {"max_depth": list(self.max_depths), "min_samples_leaf": list(self.min_samples_leaf)},
            cv=_cv(len(U), self.cv),
        )
        self.model_ = grid.fit(U, Z).best_estimator_

    def _predict(self, U):
        return self.model_.predict(U).reshape(len(U), -1)


class HistGradientBoostingSurrogate(Surrogate):
    """scikit-learn histogram gradient boosting, one model per objective (fast, handles many points)."""

    def __init__(self, max_iter: int = 300, learning_rate: float = 0.05, max_depth: int | None = 4,
                 min_samples_leaf: int = 3, random_state: int = 0):
        self.max_iter, self.learning_rate, self.max_depth = max_iter, learning_rate, max_depth
        self.min_samples_leaf, self.random_state = min_samples_leaf, random_state

    def _fit(self, U, Z):
        self.models_ = [
            HistGradientBoostingRegressor(
                max_iter=self.max_iter, learning_rate=self.learning_rate, max_depth=self.max_depth,
                min_samples_leaf=min(self.min_samples_leaf, max(1, len(U) // 3)), random_state=self.random_state,
            ).fit(U, Z[:, j])
            for j in range(Z.shape[1])
        ]

    def _predict(self, U):
        return np.column_stack([m.predict(U) for m in self.models_])


class AdaBoostSurrogate(Surrogate):
    """AdaBoost.R2 over depth-3 trees, one model per objective."""

    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.5, max_depth: int = 3, random_state: int = 0):
        self.n_estimators, self.learning_rate, self.max_depth = n_estimators, learning_rate, max_depth
        self.random_state = random_state

    def _fit(self, U, Z):
        base = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=self.max_depth), n_estimators=self.n_estimators,
            learning_rate=self.learning_rate, random_state=self.random_state,
        )
        self.model_ = MultiOutputRegressor(base).fit(U, Z)

    def _predict(self, U):
        return self.model_.predict(U).reshape(len(U), -1)


class LightGBMSurrogate(Surrogate):
    """LightGBM gradient boosting, one model per objective. Needs ``pip install 'bdopt[ml]'``."""

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05, num_leaves: int = 8,
                 min_child_samples: int = 3, random_state: int = 0):
        self.n_estimators, self.learning_rate, self.num_leaves = n_estimators, learning_rate, num_leaves
        self.min_child_samples, self.random_state = min_child_samples, random_state

    def _fit(self, U, Z):
        try:
            from lightgbm import LGBMRegressor
        except ImportError as e:  # pragma: no cover
            raise ImportError("LightGBMSurrogate needs lightgbm: pip install 'bdopt[ml]'") from e
        self.models_ = [
            LGBMRegressor(
                n_estimators=self.n_estimators, learning_rate=self.learning_rate, num_leaves=self.num_leaves,
                min_child_samples=min(self.min_child_samples, max(2, len(U) // 3)), random_state=self.random_state,
                verbose=-1, n_jobs=1,
            ).fit(U, Z[:, j])
            for j in range(Z.shape[1])
        ]

    def _predict(self, U):
        return np.column_stack([m.predict(U) for m in self.models_])


class XGBoostSurrogate(Surrogate):
    """XGBoost gradient boosting, one model per objective. Needs ``pip install 'bdopt[ml]'``."""

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05, max_depth: int = 3,
                 subsample: float = 0.8, random_state: int = 0):
        self.n_estimators, self.learning_rate, self.max_depth = n_estimators, learning_rate, max_depth
        self.subsample, self.random_state = subsample, random_state

    def _fit(self, U, Z):
        try:
            from xgboost import XGBRegressor
        except ImportError as e:  # pragma: no cover
            raise ImportError("XGBoostSurrogate needs xgboost: pip install 'bdopt[ml]'") from e
        self.models_ = [
            XGBRegressor(
                n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth,
                subsample=self.subsample, random_state=self.random_state, n_jobs=1,
            ).fit(U, Z[:, j])
            for j in range(Z.shape[1])
        ]

    def _predict(self, U):
        return np.column_stack([m.predict(U) for m in self.models_])


# ---------------------------------------------------------------------------------------------
# regularised polynomial families
# ---------------------------------------------------------------------------------------------


class LassoPolynomialSurrogate(Surrogate):
    """Polynomial (degree up to ``max_degree``, interactions of at most ``max_interaction`` parameters) with an
    elastic-net penalty chosen by cross-validation: a sparse response surface that stays usable with many parameters."""

    def __init__(self, max_degree: int = 3, max_interaction: int = 2, l1_ratio: float = 0.9, cv: int = 5):
        self.max_degree, self.max_interaction, self.l1_ratio, self.cv = max_degree, max_interaction, l1_ratio, cv

    def _terms(self, d: int):
        return _poly_terms(d, self.max_degree, self.max_interaction)

    def _features(self, U):
        return _monomials(U, self.terms_)

    def _fit(self, U, Z):
        self.terms_ = self._terms(U.shape[1])
        F = self._features(U)
        self.scaler_ = StandardScaler().fit(F)
        Fs = self.scaler_.transform(F)
        self.models_ = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j in range(Z.shape[1]):
                self.models_.append(
                    ElasticNetCV(l1_ratio=self.l1_ratio, cv=_cv(len(U), self.cv), max_iter=20000, alphas=60).fit(Fs, Z[:, j])
                )

    def _predict(self, U):
        Fs = self.scaler_.transform(self._features(U))
        return np.column_stack([m.predict(Fs) for m in self.models_])


def _poly_terms(d: int, max_degree: int, max_interaction: int):
    """Exponent vectors of all monomials with total degree <= ``max_degree`` involving <= ``max_interaction`` variables
    (the constant term is left out)."""
    import itertools

    terms = []
    for k in range(1, max_interaction + 1):
        for idx in itertools.combinations(range(d), k):
            for degs in itertools.product(range(1, max_degree + 1), repeat=k):
                if sum(degs) <= max_degree:
                    terms.append(tuple(zip(idx, degs, strict=True)))
    return terms


def _monomials(U, terms):
    F = np.empty((len(U), len(terms)))
    for j, t in enumerate(terms):
        v = np.ones(len(U))
        for i, p in t:
            v = v * U[:, i] ** p
        F[:, j] = v
    return F


class PCESurrogate(Surrogate):
    """Sparse polynomial chaos expansion: orthonormal Legendre polynomials on the unit cube (total degree up to
    ``max_degree``, interactions of at most ``max_interaction`` parameters), coefficients by least-angle regression with
    the number of terms chosen by cross-validation. The expansion also gives Sobol' sensitivity indices through
    ``first_order_indices_``."""

    def __init__(self, max_degree: int = 3, max_interaction: int = 2, cv: int = 5):
        self.max_degree, self.max_interaction, self.cv = max_degree, max_interaction, cv

    @staticmethod
    def _legendre(x, p):
        from numpy.polynomial import legendre

        c = np.zeros(p + 1)
        c[p] = 1.0
        return legendre.legval(2 * x - 1, c) * np.sqrt(2 * p + 1)

    def _basis(self, U):
        F = np.empty((len(U), len(self.terms_)))
        for j, t in enumerate(self.terms_):
            v = np.ones(len(U))
            for i, p in t:
                v = v * self._legendre(U[:, i], p)
            F[:, j] = v
        return F

    def _fit(self, U, Z):
        self.terms_ = _poly_terms(U.shape[1], self.max_degree, self.max_interaction)
        F = self._basis(U)
        self.models_ = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j in range(Z.shape[1]):
                self.models_.append(LassoLarsCV(cv=_cv(len(U), self.cv), max_n_alphas=200).fit(F, Z[:, j]))
        d = U.shape[1]
        first = np.zeros((Z.shape[1], d))
        for j, m in enumerate(self.models_):
            var = (m.coef_**2).sum()
            for c, t in zip(m.coef_, self.terms_, strict=True):
                if len(t) == 1 and var > 0:
                    first[j, t[0][0]] += c**2 / var
        self.first_order_indices_ = first

    def _predict(self, U):
        F = self._basis(U)
        return np.column_stack([m.predict(F) for m in self.models_])


# ---------------------------------------------------------------------------------------------
# extreme learning machine, universal kriging, ensembles
# ---------------------------------------------------------------------------------------------


class ELMSurrogate(Surrogate):
    """Extreme learning machine: a random, fixed tanh hidden layer and ridge-regressed output weights (no gradient
    training). ``n_ensemble`` independent machines give a mean and a spread."""

    def __init__(self, n_hidden: int = 200, alphas: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0),
                 n_ensemble: int = 10, random_state: int = 0):
        self.n_hidden, self.alphas, self.n_ensemble, self.random_state = n_hidden, tuple(alphas), n_ensemble, random_state
        self.has_uncertainty = n_ensemble > 1

    def _fit(self, U, Z):
        rng = np.random.default_rng(self.random_state)
        d = U.shape[1]
        self.members_ = []
        for _ in range(self.n_ensemble):
            W = rng.normal(0, 2.0, (d, self.n_hidden))
            b = rng.uniform(-2, 2, self.n_hidden)
            H = np.tanh(U @ W + b)
            ridge = RidgeCV(alphas=self.alphas).fit(H, Z)
            self.members_.append((W, b, ridge))

    def _members(self, U):
        return np.stack([r.predict(np.tanh(U @ W + b)).reshape(len(U), -1) for W, b, r in self.members_])

    def _predict(self, U):
        return self._members(U).mean(axis=0)

    def _predict_std(self, U):
        return self._members(U).std(axis=0) if self.n_ensemble > 1 else None


class UniversalKrigingSurrogate(Surrogate):
    """Universal kriging: a linear trend in all parameters plus a Gaussian process on the residuals. Keeps the
    Gaussian process' uncertainty and extrapolates better than a zero-mean GP when the response has a global trend."""

    has_uncertainty = True

    def __init__(self, n_restarts: int = 8, random_state: int = 0):
        self.n_restarts, self.random_state = n_restarts, random_state

    def _fit(self, U, Z):
        self.trend_ = RidgeCV(alphas=(1e-6, 1e-3, 1e-1, 1.0)).fit(U, Z)
        resid = Z - self.trend_.predict(U).reshape(len(U), -1)
        self.gp_ = GPSurrogate(n_restarts=self.n_restarts, random_state=self.random_state).fit_unit(U, resid)

    def _predict(self, U):
        return self.trend_.predict(U).reshape(len(U), -1) + self.gp_.predict_unit(U)

    def _predict_std(self, U):
        return self.gp_.predict_unit_std(U)


class BaggingSurrogate(Surrogate):
    """Bootstrap ensemble of any surrogate: ``n_members`` copies of ``base`` fitted on resampled data. The mean is the
    prediction and the spread between members an uncertainty estimate, so models without one (kernel ridge, RBF,
    polynomial, ...) can drive uncertainty-based sampling and query by committee."""

    has_uncertainty = True

    def __init__(self, base: Surrogate | None = None, n_members: int = 10, random_state: int = 0):
        self.base, self.n_members, self.random_state = base, n_members, random_state

    def _fit(self, U, Z):
        base = self.base if self.base is not None else KernelRidgeSurrogate()
        rng = np.random.default_rng(self.random_state)
        n = len(U)
        self.members_ = []
        for _ in range(self.n_members):
            for _try in range(10):
                idx = rng.integers(0, n, n)
                if len(np.unique(idx)) >= max(4, n // 2):
                    break
            self.members_.append(base.clone().fit_unit(U[idx], Z[idx], self.problem))

    def _stack(self, U):
        return np.stack([m.predict_unit(U) for m in self.members_])

    def _predict(self, U):
        return self._stack(U).mean(axis=0)

    def _predict_std(self, U):
        return self._stack(U).std(axis=0)


class StackingSurrogate(Surrogate):
    """Stacked generalisation: out-of-fold predictions of several base surrogates are combined with non-negative
    least-squares weights (per objective, summing to about one); the bases are then refitted on all the data."""

    def __init__(self, bases: tuple[Surrogate, ...] | None = None, cv: int = 5, random_state: int = 0):
        self.bases, self.cv, self.random_state = bases, cv, random_state

    def _fit(self, U, Z):
        from scipy.optimize import nnls
        from sklearn.model_selection import KFold

        bases = list(self.bases) if self.bases else [GPSurrogate(), KernelRidgeSurrogate(), RBFSurrogate(), SVRSurrogate()]
        oof = np.full((len(bases), *Z.shape), np.nan)
        for tr, te in KFold(_cv(len(U), self.cv), shuffle=True, random_state=self.random_state).split(U):
            for b, base in enumerate(bases):
                oof[b, te] = base.clone().fit_unit(U[tr], Z[tr], self.problem).predict_unit(U[te])
        self.weights_ = np.zeros((Z.shape[1], len(bases)))
        for j in range(Z.shape[1]):
            w, _ = nnls(oof[:, :, j].T, Z[:, j])
            self.weights_[j] = w / w.sum() if w.sum() > 0 else np.full(len(bases), 1 / len(bases))
        self.members_ = [base.clone().fit_unit(U, Z, self.problem) for base in bases]

    def _predict(self, U):
        P = np.stack([m.predict_unit(U) for m in self.members_])  # (bases, n, objectives)
        return np.einsum("jb,bnj->nj", self.weights_, P)
