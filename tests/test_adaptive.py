import numpy as np
import pandas as pd
import pytest

from bdopt import FunctionEvaluator, GenericSelector, Parameter, Problem, RangeParameter, sampling
from bdopt.adaptive import METHODS, AdaptiveSampler
from bdopt.adaptive.designs import (
    flola_nonlinearity,
    fuzzy_weights,
    lola_nonlinearity,
    nonlinearity,
    propose_nipv,
    voronoi_estimate,
)
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


@pytest.mark.parametrize("fn", [lola_nonlinearity, flola_nonlinearity])
def test_exploitation_scores_zero_for_linear_and_large_at_a_kink(fn):
    X = np.random.default_rng(3).random((40, 2))
    Y = X @ np.array([[1.0], [-2.0]]) + 1
    assert fn(X, Y).max() < 1e-6
    kinked = Y + 3 * np.abs(X[:, :1] - 0.5)
    score = fn(X, kinked)
    near = np.abs(X[:, 0] - 0.5) < 0.1
    assert score[near].mean() > 2 * score[~near].mean()


def test_fuzzy_weights_rank_close_isolated_neighbours_highest():
    cohesion = np.array([0.05, 0.05, 0.9, 0.9])      # close, close, far, far
    adhesion = np.array([0.5, 0.01, 0.5, 0.01])      # isolated, clustered, isolated, clustered
    w = fuzzy_weights(cohesion, adhesion)
    assert w[0] > w[1] and w[0] > w[2] > w[3] and np.all(w > 0) and np.all(w <= 1)


def test_nipv_prefers_the_unexplored_region():
    pr, ev = analytic_problem(2)
    X = np.random.default_rng(0).random((15, 2)) * 0.4   # every sample in the lower-left corner
    Y = ev.evaluate_unit(X)
    model = fast_gp().fit_unit(X, Y, pr)
    new = propose_nipv(X, Y, model, 1, np.random.default_rng(1), pr)
    assert new.max() > 0.5


@pytest.mark.parametrize("method", METHODS)
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


# ---- criteria added from the literature ---------------------------------------------------------


def _gp_state(n=15, d=2, seed=0, corner=0.4):
    pr, ev = analytic_problem(d)
    X = np.random.default_rng(seed).random((n, d)) * corner  # every sample in one corner
    Y = ev.evaluate_unit(X)
    return pr, X, Y, fast_gp().fit_unit(X, Y, pr)


@pytest.mark.parametrize("name", ["greedy", "cvt", "lipschitz", "qbc", "loocv", "idw", "soas", "sample_min", "ucb"])
def test_new_criteria_look_outside_the_sampled_corner(name):
    from bdopt.adaptive import criteria

    pr, X, Y, model = _gp_state()
    rng = np.random.default_rng(1)
    fn = {
        "greedy": lambda: criteria.propose_greedy(X, 1, rng, pr),
        "cvt": lambda: criteria.propose_cvt(X, 1, rng, pr),
        "lipschitz": lambda: criteria.propose_lipschitz(X, Y, 1, rng, pr),
        "qbc": lambda: criteria.propose_qbc(X, Y, model, 1, rng, pr),
        "loocv": lambda: criteria.propose_loocv(X, Y, model, 1, rng, pr),
        "idw": lambda: criteria.propose_idw(X, Y, model, 1, rng, pr),
        "soas": lambda: criteria.propose_soas(X, Y, model, 1, rng, pr),
        "sample_min": lambda: criteria.propose_acquisition("sample_min", X, Y, model, 1, rng, pr),
        "ucb": lambda: criteria.propose_acquisition("ucb", X, Y, model, 1, rng, pr),
    }[name]
    new = fn()
    assert new.shape == (1, 2) and np.isfinite(new).all()
    if name not in ("sample_min", "soas"):  # these seek a low value, not a gap
        assert new.max() > 0.5, new


def test_greedy_increases_the_smallest_distance_and_batches_do_not_collide():
    from bdopt.adaptive import criteria

    pr, X, _, _ = _gp_state()
    new = criteria.propose_greedy(X, 4, np.random.default_rng(0), pr)
    assert len(np.unique(new.round(9), axis=0)) == 4
    d_old = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    np.fill_diagonal(d_old, np.inf)
    d_new = np.sqrt(((new[:, None] - X[None]) ** 2).sum(-1)).min(axis=1)
    assert d_new.min() > d_old.min()


def test_sequence_continuation_is_deterministic_and_extends_without_repeats():
    from bdopt.adaptive import criteria

    pr, X, _, _ = _gp_state()
    a = criteria.propose_sequence("sobol", X, 4, np.random.default_rng(0), pr, seed=3)
    b = criteria.propose_sequence("sobol", X, 4, np.random.default_rng(9), pr, seed=3)
    assert np.allclose(a, b)
    both = np.vstack([X, a])
    c = criteria.propose_sequence("halton", both, 4, np.random.default_rng(0), pr, seed=3)
    assert len(np.unique(np.vstack([both, c]).round(9), axis=0)) == len(both) + 4


def test_error_window_stays_in_its_box():
    from bdopt.adaptive import criteria

    pr, X, _, _ = _gp_state()
    anchor = np.array([0.8, 0.2])
    new = criteria.propose_error_window(X, anchor, 6, np.random.default_rng(0), pr, window=0.2)
    assert (np.abs(new - anchor) <= 0.1 + 1e-9).all()


def test_acquisition_criteria_require_uncertainty_and_fisher_requires_a_gp():
    from bdopt.adaptive import criteria
    from bdopt.surrogate import PolynomialSurrogate

    pr, X, Y, gp = _gp_state()
    poly = PolynomialSurrogate().fit_unit(X, Y, pr)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="uncertainty"):
        criteria.propose_acquisition("lcb", X, Y, poly, 1, rng, pr)
    with pytest.raises(ValueError, match="GPSurrogate"):
        criteria.propose_fisher_information(X, Y, poly, 1, rng, pr)
    assert criteria.propose_fisher_information(X, Y, gp, 1, rng, pr, n_pool=30).shape == (1, 2)


def test_expected_improvement_prefers_promising_regions():
    """With a response that is lowest at x = 0 the improvement criteria should not choose the worst corner."""
    from bdopt.adaptive import criteria

    pr, X, Y, gp = _gp_state(n=25, corner=1.0)
    new = criteria.propose_acquisition("expected_improvement", X, Y, gp, 1, np.random.default_rng(0), pr)
    mean = gp.predict_unit(new)[0]
    assert np.all(mean < gp.predict_unit(X).max(axis=0))


@pytest.mark.parametrize("design", ["halton", "hammersley", "maximin_lhs", "fps", "cvt"])
def test_new_fixed_designs(design):
    pr, _ = analytic_problem(4)
    df = sampling.hammersley(pr, 16) if design == "hammersley" else getattr(sampling, design)(pr, 16, seed=0)
    assert df.shape == (16, 4) and not df.duplicated().any()
    U = pr.encode(df)
    assert U.min() >= 0 and U.max() <= 1


# ---- stopping rules ---------------------------------------------------------------------------


def test_plateau_stops_a_run_whose_error_stops_improving():
    pr, ev = analytic_problem(2)
    r = AdaptiveSampler(pr, ev, "latin_hypercube", surrogate=fast_gp(), seed=0).run(
        n_initial=12, n_test=10, max_iter=40, tol=0, patience=3, min_improvement=0.5, verbose=False
    )
    assert r.stop_reason == "plateau" and len(r.history) < 41 and not r.converged


def test_uncertainty_rule_needs_a_surrogate_with_uncertainty_and_records_max_std():
    from bdopt.surrogate import PolynomialSurrogate

    pr, ev = analytic_problem(2)
    with pytest.raises(ValueError, match="uncertainty"):
        AdaptiveSampler(pr, ev, "random", surrogate=PolynomialSurrogate(), seed=0).run(
            n_initial=8, n_test=4, max_iter=2, max_std=0.1, verbose=False
        )
    r = AdaptiveSampler(pr, ev, "random", surrogate=fast_gp(), seed=0).run(
        n_initial=10, n_test=5, max_iter=6, tol=0, max_std=100.0, verbose=False
    )
    assert r.stop_reason == "uncertainty" and r.converged and "max_std" in r.history


def test_suggest_budget_reads_stopping_points_off_the_history():
    from bdopt.adaptive import suggest_budget

    pr, ev = analytic_problem(2)
    r = AdaptiveSampler(pr, ev, "latin_hypercube", surrogate=fast_gp(), seed=0).run(
        n_initial=10, n_test=8, max_iter=8, tol=0, verbose=False
    )
    table = suggest_budget(r.history, tols=(1e9, 1e-9), patience=2)
    assert list(table["rule"][:2]) == ["scaled_rmse <= 1000000000.0", "scaled_rmse <= 1e-09"]
    assert table.loc[0, "iteration"] == 0 and np.isnan(table.loc[1, "iteration"])
    assert "max_std <= 0.5" in set(table["rule"])
