"""Adaptive sampling loop: spend expensive EnergyPlus runs where the surrogate needs them."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluator import Evaluator
from ..problem import Problem
from ..sampling import lhs_unit
from ..surrogate import GPSurrogate, Surrogate
from . import designs

METHODS = ("flola_voronoi", "lola_voronoi", "voronoi", "gp_variance", "random")


@dataclass
class AdaptiveResult:
    """Outcome of :meth:`AdaptiveSampler.run`.

    ``data``/``test`` are tables of parameters + objectives in real units, ``history`` has one row per
    iteration (``scaled_rmse`` is the stopping metric: test RMSE in units of the std of each
    objective over the initial samples, as in the harlow notebooks), ``surrogate`` is fitted on
    ``data``.
    """

    data: pd.DataFrame
    test: pd.DataFrame
    history: pd.DataFrame
    surrogate: Surrogate
    converged: bool
    stop_reason: str
    n_evaluations: int
    failed: pd.DataFrame = field(default_factory=pd.DataFrame)
    extra: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        last = self.history["scaled_rmse"].iloc[-1]
        return (
            f"AdaptiveResult({len(self.data)} train + {len(self.test)} test points, "
            f"scaled_rmse={last:.4f}, stop='{self.stop_reason}')"
        )


class AdaptiveSampler:
    """Adaptive design of experiments for global surrogate modelling.

    Parameters
    ----------
    problem, evaluator:
        the problem and what evaluates it (usually an :class:`~bdopt.EnergyPlusEvaluator`).
    method:
        ``'flola_voronoi'`` (default; fuzzy hybrid LOLA-Voronoi), ``'lola_voronoi'`` (crisp
        neighbourhoods), ``'voronoi'`` (pure exploration), ``'gp_variance'`` (largest predictive
        uncertainty) or ``'random'`` (baseline).
    surrogate:
        model refitted at every iteration to measure convergence (default :class:`GPSurrogate`).
    seed:
        everything is deterministic given the seed, so with the evaluator's cache an interrupted run is
        resumed by simply running it again: already simulated points are not repeated.
    """

    def __init__(
        self,
        problem: Problem,
        evaluator: Evaluator,
        method: str = "flola_voronoi",
        surrogate: Surrogate | None = None,
        seed: int = 0,
    ):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.problem, self.evaluator, self.method = problem, evaluator, method
        self.surrogate = surrogate or GPSurrogate()
        self.seed = seed

    # -- helpers ---------------------------------------------------------------------------------
    def _evaluate(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate unit points; return ``(U_ok, Y_ok, U_failed)``."""
        U = self.problem.snap(U)
        Y = self.evaluator.evaluate_unit(U)
        ok = np.isfinite(Y).all(axis=1)
        return U[ok], Y[ok], U[~ok]

    def _propose(self, U, Y, batch: int, it: int, avoid, model: Surrogate) -> np.ndarray:
        rng = np.random.default_rng([self.seed, 1, it])
        if self.method == "random":
            out = []
            taken = np.vstack([U, avoid]) if len(avoid) else U
            for _ in range(batch):
                cur = np.vstack([taken, *out]) if out else taken
                out.append(designs._fallback_point(self.problem, rng, U.shape[1], cur))
            return np.array(out)
        if self.method == "gp_variance":
            return designs.propose_gp_variance(U, Y, model, batch, rng, self.problem, avoid)
        return designs.propose_voronoi_based(
            U, self._scale(Y), batch, rng, self.problem, avoid,
            exploit=self.method != "voronoi", fuzzy=self.method == "flola_voronoi",
        )

    def _scale(self, Y: np.ndarray) -> np.ndarray:
        return (Y - self._y_mean) / self._y_std

    def _table(self, U: np.ndarray, Y: np.ndarray) -> pd.DataFrame:
        df = self.problem.decode(U)
        for j, n in enumerate(self.problem.objective_names):
            df[n] = Y[:, j]
        return df

    def _metric(self, model: Surrogate, U, Y, Ut, Yt) -> tuple[float, pd.Series]:
        if len(Ut):
            pred, truth = model.predict_unit(Ut), Yt
        else:  # no test set: 5-fold cross-validation on the training data
            from sklearn.model_selection import KFold

            pred = np.full_like(Y, np.nan)
            for tr, te in KFold(min(5, len(U)), shuffle=True, random_state=self.seed).split(U):
                pred[te] = model.clone().fit_unit(U[tr], Y[tr], self.problem).predict_unit(U[te])
            truth = Y
        err = (pred - truth) / self._y_std
        per_obj = pd.Series(np.sqrt((err**2).mean(axis=0)), index=self.problem.objective_names)
        ss_res = ((pred - truth) ** 2).sum(axis=0)
        ss_tot = ((truth - truth.mean(axis=0)) ** 2).sum(axis=0)
        r2 = pd.Series(1 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan), index=per_obj.index)
        self._last_r2 = r2
        return float(np.sqrt((err**2).mean())), per_obj

    # -- main loop -------------------------------------------------------------------------------
    def run(
        self,
        n_initial: int | None = None,
        n_test: int | None = None,
        max_iter: int = 40,
        batch: int = 1,
        tol: float = 0.05,
        max_evaluations: int | None = None,
        initial_data: pd.DataFrame | None = None,
        test_data: pd.DataFrame | None = None,
        checkpoint_dir: str | Path | None = None,
        verbose: bool = True,
        callback: Callable[[int, pd.Series], None] | None = None,
    ) -> AdaptiveResult:
        """Run the adaptive loop.

        ``n_initial`` Latin-hypercube points (default ``5 * n_parameters``) and ``n_test`` held-out points
        (default ``max(5, n_parameters)``; ``0`` = use 5-fold cross-validation instead) are simulated
        first, then ``batch`` points per iteration are added until the test ``scaled_rmse`` is
        ``<= tol``, ``max_iter`` iterations or ``max_evaluations`` simulations are reached.
        ``initial_data`` / ``test_data`` reuse tables of already evaluated points.
        """
        pr = self.problem
        d = pr.n_var
        n_initial = n_initial if n_initial is not None else 5 * d
        n_test = n_test if n_test is not None else max(5, d)
        t0 = time.perf_counter()

        # ---- initial and test designs ---------------------------------------------------------
        failed: list[np.ndarray] = []
        if initial_data is not None:
            U = pr.encode(initial_data)
            Y = initial_data[pr.objective_names].to_numpy(float)
        else:
            U, Y, bad = self._evaluate(lhs_unit(pr, n_initial, seed=[self.seed, 0]))
            failed.append(bad)
        if test_data is not None:
            Ut = pr.encode(test_data)
            Yt = test_data[pr.objective_names].to_numpy(float)
        elif n_test > 0:
            Ut, Yt, bad = self._evaluate(lhs_unit(pr, n_test, seed=[self.seed, 2]))
            failed.append(bad)
        else:
            Ut, Yt = np.empty((0, d)), np.empty((0, pr.n_obj))
        if len(U) < 3:
            raise RuntimeError(f"only {len(U)} initial simulations succeeded; check evaluator.failures")
        n_evals = (len(U) if initial_data is None else 0) + (len(Ut) if test_data is None else 0)
        n_evals += sum(len(f) for f in failed)

        self._y_mean = Y.mean(axis=0)
        sd = Y.std(axis=0)
        self._y_std = np.where(sd > 0, sd, 1.0)  # fixed from the initial samples, as in harlow

        model = self.surrogate.clone()
        rows: list[dict] = []

        def fit_and_score(it: int) -> float:
            model.fit_unit(U, Y, pr)
            score, per_obj = self._metric(model, U, Y, Ut, Yt)
            row = {"iteration": it, "n_train": len(U), "n_evaluations": n_evals, "scaled_rmse": score,
                   "elapsed_s": round(time.perf_counter() - t0, 2)}
            row.update({f"rmse_scaled:{k}": v for k, v in per_obj.items()})
            row.update({f"r2:{k}": v for k, v in self._last_r2.items()})
            rows.append(row)
            if verbose:
                print(f"[bdopt] iter {it:3d}  train={len(U):4d}  evals={n_evals:4d}  scaled_rmse={score:.4f}")
            if checkpoint_dir is not None:
                self._checkpoint(checkpoint_dir, U, Y, Ut, Yt, rows)
            if callback:
                callback(it, pd.Series(row))
            return score

        score = fit_and_score(0)
        stop = None
        it = 0
        while score > tol and it < max_iter:
            if max_evaluations is not None and n_evals >= max_evaluations:
                stop = "max_evaluations"
                break
            it += 1
            b = batch if max_evaluations is None else min(batch, max_evaluations - n_evals)
            avoid = np.vstack(failed) if failed and sum(len(f) for f in failed) else np.empty((0, d))
            Unew = self._propose(U, Y, b, it, np.vstack([avoid, Ut]) if len(Ut) else avoid, model)
            Uok, Yok, bad = self._evaluate(Unew)
            n_evals += len(Unew)
            failed.append(bad)
            if len(Uok):
                U, Y = np.vstack([U, Uok]), np.vstack([Y, Yok])
            score = fit_and_score(it)
        if stop is None:
            stop = "tolerance" if score <= tol else "max_iter"
        failed_all = np.vstack(failed) if failed and sum(len(f) for f in failed) else np.empty((0, d))
        return AdaptiveResult(
            data=self._table(U, Y),
            test=self._table(Ut, Yt) if len(Ut) else pd.DataFrame(columns=pr.names()),
            history=pd.DataFrame(rows),
            surrogate=model,
            converged=score <= tol,
            stop_reason=stop,
            n_evaluations=n_evals,
            failed=pr.decode(failed_all) if len(failed_all) else pd.DataFrame(columns=pr.parameter_names),
        )

    def _checkpoint(self, directory, U, Y, Ut, Yt, rows) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self._table(U, Y).to_csv(d / "train.csv", index=False)
        if len(Ut):
            self._table(Ut, Yt).to_csv(d / "test.csv", index=False)
        pd.DataFrame(rows).to_csv(d / "history.csv", index=False)
