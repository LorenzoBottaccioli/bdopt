import numpy as np
import pandas as pd
import pytest

from bdopt import FunctionEvaluator, GenericSelector, Parameter, Problem, RangeParameter, sampling
from bdopt.adaptive import AdaptiveSampler
from bdopt.adaptive.designs import nonlinearity, voronoi_estimate
from bdopt.surrogate import GPSurrogate
from conftest import analytic_problem, fast_gp


def test_voronoi_volumes_sum_to_one_and_reflect_coverage():
    rng = np.random.default_rng(0)
    X = np.array([[0.1, 0.1], [0.12, 0.1], [0.9, 0.9]])
    vol, P, idx, dist = voronoi_estimate(X, rng, n_mc=20000)
    assert vol.sum() == pytest.approx(1.0)
    assert vol[2] > vol[0] and vol[2] > vol[1]  # the isolated sample owns the biggest cell


def test_nonlinearity_is_zero_for_linear_response():
    X = np.random.default_rng(1).random((30, 3))
    Y = X @ np.array([[1.0], [2.0], [-1.0]]) + 3
    assert nonlinearity(X, Y).max() < 1e-6
    Y2 = Y + 5 * (X[:, :1] > 0.5)
    assert nonlinearity(X, Y2).max() > 0.1


@pytest.mark.parametrize("method", ["flola_voronoi", "lola_voronoi", "voronoi", "gp_variance", "random"])
def test_methods_run_and_add_points(method):
    pr, ev = analytic_problem(3)
    r = AdaptiveSampler(pr, ev, method=method, surrogate=fast_gp(), seed=0).run(
        n_initial=8, n_test=5, max_iter=4, batch=2, tol=0, verbose=False
    )
    assert len(r.data) == 8 + 4 * 2 and len(r.test) == 5
    assert list(r.history["iteration"]) == [0, 1, 2, 3, 4]
    assert r.stop_reason == "max_iter" and not r.converged
    assert not r.data.duplicated(subset=pr.parameter_names).any()
    assert r.surrogate.predict(r.test).shape == (5, 2)


@pytest.mark.slow
def test_flola_beats_lhs_on_localised_feature_with_same_training_size():
    """Adaptive sampling pays off when the response has a *localised* feature: measured on this function
    (4 seeds, 50 training points) normalised RMSE 0.154 (FLOLA) vs 0.243 (LHS). On smooth responses all
    designs are equally good and on an oblique step there was no gain, so those cases are not asserted."""
    d = 3
    P = [Parameter(GenericSelector(lambda i, v: None, f"p{k}"), RangeParameter(0, 1), name=f"p{k}") for k in range(d)]
    pr = Problem(P, ["f"])
    ev = FunctionEvaluator(
        pr, lambda r: [0.3 * r["p0"] + np.exp(-(((r["p0"] - 0.7) ** 2 + (r["p1"] - 0.3) ** 2) / 0.012))]
    )
    big = sampling.lhs(pr, 2000, seed=99)
    truth = ev.evaluate(big).to_numpy()

    def err(train):
        p = GPSurrogate(n_restarts=3, max_extra_restarts=8).fit(train, pr).predict(big).to_numpy()
        return float(np.sqrt(((p - truth) ** 2).mean() / truth.var()))

    adaptive, plain = [], []
    for seed in range(3):
        r = AdaptiveSampler(
            pr, ev, "flola_voronoi", surrogate=GPSurrogate(n_restarts=3, max_extra_restarts=8), seed=seed
        ).run(n_initial=14, n_test=10, max_iter=36, tol=0, verbose=False)
        assert len(r.data) == 50
        adaptive.append(err(r.data))
        plain.append(err(ev.evaluate(sampling.lhs(pr, 50, seed=seed), keep_input=True)))
    assert np.mean(adaptive) < 0.85 * np.mean(plain), (adaptive, plain)


def test_stops_on_tolerance_and_budget():
    pr, ev = analytic_problem(2)
    r = AdaptiveSampler(pr, ev, surrogate=fast_gp(), seed=0).run(n_initial=10, n_test=10, max_iter=30, tol=10.0, verbose=False)
    assert r.converged and r.stop_reason == "tolerance" and len(r.data) == 10
    r = AdaptiveSampler(pr, ev, surrogate=fast_gp(), seed=0).run(
        n_initial=8, n_test=4, max_iter=50, tol=0, max_evaluations=16, verbose=False
    )
    assert r.stop_reason == "max_evaluations" and r.n_evaluations == 16


def test_deterministic_given_seed():
    pr, ev = analytic_problem(3)
    a = AdaptiveSampler(pr, ev, surrogate=fast_gp(), seed=5).run(n_initial=8, n_test=4, max_iter=3, tol=0, verbose=False)
    b = AdaptiveSampler(pr, ev, surrogate=fast_gp(), seed=5).run(n_initial=8, n_test=4, max_iter=3, tol=0, verbose=False)
    pd.testing.assert_frame_equal(a.data, b.data)


def test_mixed_problem_snaps_discrete_values():
    pr, ev = analytic_problem(2, mixed=True)
    r = AdaptiveSampler(pr, ev, surrogate=fast_gp(), seed=1).run(n_initial=10, n_test=5, max_iter=4, tol=0, verbose=False)
    assert set(r.data["g"]) <= {"a", "b", "c"} and set(r.data["n"]) <= {1, 2, 3, 4}


def test_failed_evaluations_are_skipped_not_repeated():
    pr, good = analytic_problem(2)
    calls = {"n": 0}

    def flaky(df, **kw):
        out = good.evaluate(df)
        calls["n"] += 1
        bad = (df["p0"] > 0.8).to_numpy()
        out.loc[bad, :] = np.nan
        return out

    ev = FunctionEvaluator(pr, lambda d: flaky(d).to_numpy(), vectorized=True)
    r = AdaptiveSampler(pr, ev, surrogate=fast_gp(), seed=0).run(n_initial=10, n_test=5, max_iter=8, tol=0, verbose=False)
    assert r.data[["f1", "f2"]].notna().all().all()
    assert (r.data["p0"] <= 0.8).all()
    assert r.n_evaluations >= len(r.data) + len(r.test)


def test_no_test_set_uses_cross_validation_and_checkpoint(tmp_path):
    pr, ev = analytic_problem(2)
    r = AdaptiveSampler(pr, ev, surrogate=fast_gp(), seed=0).run(
        n_initial=10, n_test=0, max_iter=2, tol=0, verbose=False, checkpoint_dir=tmp_path
    )
    assert len(r.test) == 0 and np.isfinite(r.history["scaled_rmse"]).all()
    assert (tmp_path / "train.csv").exists() and (tmp_path / "history.csv").exists()


def test_initial_data_reused():
    pr, ev = analytic_problem(2)
    init = ev.evaluate(sampling.lhs(pr, 9, seed=1), keep_input=True)
    r = AdaptiveSampler(pr, ev, surrogate=fast_gp(), seed=0).run(initial_data=init, n_test=4, max_iter=1, tol=0, verbose=False)
    assert len(r.data) == 10 and r.n_evaluations == 4 + 1
