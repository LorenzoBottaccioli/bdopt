"""Optional bridge to `harlow <https://github.com/TNO/harlow>`_ (TNO) adaptive samplers.

bdopt has its own adaptive samplers (:class:`bdopt.adaptive.AdaptiveSampler`) that need neither
TensorFlow nor harlow. Use this bridge when you specifically want harlow's implementations
(``FuzzyLolaVoronoi``, ``LolaVoronoi``, ``ProbabilisticSampler``, ``CVVoronoi`` ...) and its
Gaussian-process surrogates.

harlow 0.1.0 imports TensorFlow and TensorFlow-Probability at module level, so installing it pulls in
TensorFlow (``pip install harlow tf_keras networkx``; set ``TF_USE_LEGACY_KERAS=1``, done here
automatically). The bridge removes the manual glue the platform's notebooks needed:

* parameter bounds and ordering come from the :class:`~bdopt.Problem` (harlow works in the unit cube);
* the ``target_function`` that scales/unscales and calls EnergyPlus is built for you;
* categorical / integer parameters are snapped to valid values before evaluation;
* the ``surrogate.update`` shape patch is applied;
* inputs and outputs are scaled with fixed scalers, so no train/test leakage.
"""

from __future__ import annotations

import os
import tempfile
import time

import numpy as np
import pandas as pd

from ..evaluator import Evaluator
from ..problem import Problem
from ..sampling import lhs_unit
from ..surrogate import GPSurrogate
from .sampler import AdaptiveResult


def _import_harlow():
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        import harlow.sampling as samplers
        import harlow.surrogating.surrogate_model as surrogates
    except ImportError as e:
        raise ImportError(
            "HarlowAdaptiveSampler needs harlow, which itself requires TensorFlow: "
            "`pip install harlow tf_keras networkx`. bdopt.adaptive.AdaptiveSampler provides the same "
            f"kind of adaptive sampling without it. (original error: {e})"
        ) from e
    try:
        from loguru import logger

        logger.disable("harlow")
    except ImportError:  # pragma: no cover
        pass
    return samplers, surrogates


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


class _UpdateRecorder:
    """Replaces ``surrogate.update`` to (a) restore the 2-D shape of ``y`` that harlow flattens and
    (b) record the test error after every update. A module-level class (not a closure) because harlow
    pickles the surrogate, patched attribute included."""

    def __init__(self, model, y_test_scaled, x_test, history, verbose, t0):
        self.model, self.original = model, model.update
        self.x_test, self.y_test = x_test, y_test_scaled
        self.history, self.verbose, self.t0 = history, verbose, t0
        self.n_train = None

    def __call__(self, X, y, **kwargs):
        X = np.asarray(X)
        out = self.original(X, np.asarray(y, dtype=float).reshape(len(X), -1), **kwargs)
        self.n_train = (self.n_train if self.n_train is not None else self.history[0]["n_train"]) + len(X)
        score = _rmse(self.y_test, self.model.predict(self.x_test, as_array=True))
        self.history.append(
            {"iteration": len(self.history), "n_train": self.n_train, "scaled_rmse": score,
             "elapsed_s": round(time.perf_counter() - self.t0, 2)}
        )
        if self.verbose:
            print(f"[bdopt/harlow] iter {len(self.history) - 1:3d}  train={self.n_train:4d}  scaled_rmse={score:.4f}")
        return out


class HarlowAdaptiveSampler:
    """Run a harlow sampler on a :class:`~bdopt.Problem`.

    ``method`` is the name of a class in ``harlow.sampling`` and ``surrogate`` the name of a class in
    ``harlow.surrogating.surrogate_model`` (it must be multi-output for several objectives, e.g.
    ``BatchIndependentGaussianProcess``).
    """

    def __init__(
        self,
        problem: Problem,
        evaluator: Evaluator,
        method: str = "FuzzyLolaVoronoi",
        surrogate: str = "BatchIndependentGaussianProcess",
        seed: int = 0,
    ):
        self.problem, self.evaluator, self.method, self.surrogate_name, self.seed = (
            problem, evaluator, method, surrogate, seed,
        )

    def run(
        self,
        n_initial: int | None = None,
        n_test: int | None = None,
        max_iter: int = 40,
        batch: int = 1,
        tol: float = 0.01,
        initial_data: pd.DataFrame | None = None,
        test_data: pd.DataFrame | None = None,
        verbose: bool = False,
    ) -> AdaptiveResult:
        """Run harlow's sampler; same arguments and result type as :meth:`AdaptiveSampler.run`."""
        samplers, surrogates = _import_harlow()
        pr = self.problem
        d = pr.n_var
        n_initial = n_initial if n_initial is not None else 5 * d
        n_test = n_test if n_test is not None else max(5, d)
        t0 = time.perf_counter()

        def evaluate(U: np.ndarray) -> np.ndarray:
            Y = self.evaluator.evaluate_unit(pr.snap(U))
            if not np.isfinite(Y).all():
                raise RuntimeError(
                    "an evaluation failed (NaN); harlow cannot skip points. See evaluator.failures"
                )
            return Y

        if initial_data is not None:
            U0, Y0 = pr.encode(initial_data), initial_data[pr.objective_names].to_numpy(float)
        else:
            U0 = pr.snap(lhs_unit(pr, n_initial, seed=[self.seed, 0]))
            Y0 = evaluate(U0)
        if test_data is not None:
            Ut, Yt = pr.encode(test_data), test_data[pr.objective_names].to_numpy(float)
        else:
            Ut = pr.snap(lhs_unit(pr, n_test, seed=[self.seed, 2]))
            Yt = evaluate(Ut)

        mean, sd = Y0.mean(axis=0), Y0.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        scale = lambda Y: (Y - mean) / sd  # noqa: E731

        def target(X):
            return scale(evaluate(np.atleast_2d(np.asarray(X, dtype=float))))

        history: list[dict] = [{"iteration": 0, "n_train": len(U0), "scaled_rmse": np.nan,
                                "elapsed_s": 0.0}]
        sampler_cls = getattr(samplers, self.method)
        sampler = sampler_cls(
            target_function=target,
            surrogate_model=getattr(surrogates, self.surrogate_name),
            domain_lower_bound=np.zeros(d),
            domain_upper_bound=np.ones(d),
            fit_points_x=U0,
            fit_points_y=scale(Y0),
            test_points_x=Ut,
            test_points_y=scale(Yt),
            evaluation_metric=_rmse,
            save_dir=tempfile.mkdtemp(prefix="bdopt_harlow_"),
            run_name="bdopt",
        )
        sampler.surrogate_model.update = _UpdateRecorder(
            sampler.surrogate_model, scale(Yt), Ut, history, verbose, t0
        )
        X, Y = sampler.sample(
            n_initial_points=len(U0),
            n_new_points_per_iteration=batch,
            max_n_iterations=max_iter,
            stopping_criterium=[tol],
        )
        Xf, Yf = np.asarray(X, float), np.asarray(Y, float) * sd + mean

        def table(U, Yv):
            df = pr.decode(U)
            for j, n in enumerate(pr.objective_names):
                df[n] = Yv[:, j]
            return df

        data = table(Xf, Yf)
        final = history[-1]["scaled_rmse"] if len(history) > 1 else float("nan")
        history[0]["scaled_rmse"] = _rmse(scale(Yt), sampler.surrogate_model.predict(Ut, as_array=True)) \
            if len(history) == 1 else np.nan
        hist = pd.DataFrame(history)
        return AdaptiveResult(
            data=data,
            test=table(Ut, Yt),
            history=hist,
            surrogate=GPSurrogate().fit(data, pr),
            converged=bool(final <= tol),
            stop_reason="tolerance" if final <= tol else "max_iter",
            n_evaluations=len(Xf) + len(Ut),
            extra={"harlow_sampler": sampler},
        )
