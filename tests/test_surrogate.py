import numpy as np
import pytest

from bdopt import sampling
from bdopt.surrogate import GPSurrogate, MLPSurrogate, Surrogate, regression_metrics
from conftest import analytic_problem


@pytest.fixture(scope="module")
def data():
    pr, ev = analytic_problem(4)
    train = ev.evaluate(sampling.lhs(pr, 60, seed=0), keep_input=True)
    test = ev.evaluate(sampling.lhs(pr, 300, seed=7), keep_input=True)
    return pr, train, test


def test_gp_accuracy_and_uncertainty(data):
    pr, train, test = data
    m = GPSurrogate().fit(train, pr)
    s = m.score(test)
    assert s.loc["f1", "r2"] > 0.95
    std = m.predict_unit_std(pr.encode(test))
    assert std.shape == (300, 2) and (std > 0).all()
    assert list(m.predict(test).columns) == ["f1", "f2"]


@pytest.mark.parametrize("seed", range(6))
def test_gp_not_degenerate_across_lhs_designs(seed):
    """Regression: with LHS 'random-cd' designs a 2-restart GP collapsed to predicting the mean
    (all length scales at the lower bound, error ~1.0). The fit must be reliable for any design."""
    pr, ev = analytic_problem(4)
    train = ev.evaluate(sampling.lhs(pr, 60, seed=seed), keep_input=True)
    test = ev.evaluate(sampling.lhs(pr, 400, seed=99), keep_input=True)
    nrmse = GPSurrogate().fit(train, pr).score(test)["rmse"] / test[["f1", "f2"]].std()
    assert (nrmse < 0.5).all(), nrmse


def test_mlp_and_cross_validation(data):
    pr, train, test = data
    m = MLPSurrogate(alphas=(1e-3,), hidden_layer_sizes=((32,),)).fit(train, pr)
    assert m.score(test).loc["f1", "r2"] > 0.7
    cv = GPSurrogate().cross_validate(train, pr, k=3)
    assert list(cv.index) == ["f1", "f2"] and (cv["r2"] > 0.8).all()


def test_torch_mlp(data):
    pytest.importorskip("torch")
    from bdopt.surrogate import TorchMLP

    pr, train, test = data
    m = TorchMLP(hidden=(32, 32), epochs=1500, n_ensemble=2).fit(train, pr)
    assert m.score(test).loc["f1", "r2"] > 0.8
    assert m.predict_unit_std(pr.encode(test)).shape == (300, 2)
    p1 = m.predict(test).to_numpy()
    p2 = TorchMLP(hidden=(32, 32), epochs=1500, n_ensemble=2).fit(train, pr).predict(test).to_numpy()
    np.testing.assert_allclose(p1, p2, rtol=1e-4, atol=1e-5)  # seeded: reproducible


def test_nan_rows_dropped_and_too_few(data):
    pr, train, _ = data
    bad = train.copy()
    bad.loc[0, "f1"] = np.nan
    with pytest.warns(UserWarning, match="dropping 1"):
        GPSurrogate(n_restarts=0).fit(bad, pr)
    with pytest.raises(ValueError, match="at least 3"):
        GPSurrogate().fit(train.head(2), pr)


def test_save_load_roundtrip(tmp_path, data):
    pr, train, test = data
    m = GPSurrogate(n_restarts=1).fit(train, pr)
    path = m.save(tmp_path / "m.pkl")  # problem holds lambdas -> saved without it
    m2 = Surrogate.load(path, problem=pr)
    np.testing.assert_allclose(m.predict(test).to_numpy(), m2.predict(test).to_numpy())


def test_metrics_definitions():
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    m = regression_metrics(y, y + 0.5, ["a"])
    assert m.loc["a", "rmse"] == pytest.approx(0.5)
    assert m.loc["a", "rmse_pct"] == pytest.approx(0.5 / 2.5 * 100)
    assert m.loc["a", "nrmse"] == pytest.approx(0.5 / 3)
    assert m.loc["a", "r2"] == pytest.approx(1 - 1.0 / 5.0)


def test_predict_keeps_the_index(data):
    pr, train, test = data
    t = test.head(3).copy()
    t.index = [10, 20, 30]
    m = GPSurrogate(n_restarts=1).fit(train, pr)
    assert list(m.predict(t).index) == [10, 20, 30]


ACCURACY = {  # minimum R^2 on the smooth analytic test problem with 60 training points
    "gp": 0.95, "random_forest": 0.8, "extra_trees": 0.85, "gradient_boosting": 0.85, "svr": 0.8,
    "kernel_ridge": 0.85, "knn": 0.6, "polynomial": 0.7, "rbf": 0.85,
}


@pytest.mark.parametrize("kind", sorted(ACCURACY))
def test_every_sklearn_surrogate_learns_the_response(data, kind):
    from bdopt.surrogate import make_surrogate

    pr, train, test = data
    m = make_surrogate(kind).fit(train, pr)
    assert m.score(test)["r2"].min() > ACCURACY[kind], m.score(test)
    assert m.predict(test).shape == (300, 2)
    assert (m.predict_unit_std(pr.encode(test)) is not None) == m.has_uncertainty


def test_auto_surrogate_picks_a_model_and_reports_scores(data):
    from bdopt.surrogate import AutoSurrogate, GPSurrogate, PolynomialSurrogate, make_surrogate

    pr, train, test = data
    m = AutoSurrogate(candidates=[GPSurrogate(n_restarts=2), PolynomialSurrogate()], cv=3).fit(train, pr)
    assert m.selected_ in ("GPSurrogate", "PolynomialSurrogate")
    assert len(m.cv_scores_) == 2 and m.score(test)["r2"].min() > 0.9
    assert isinstance(make_surrogate("gp"), GPSurrogate)
    with pytest.raises(ValueError, match="unknown surrogate"):
        make_surrogate("nope")


@pytest.mark.parametrize("kind", ["multitask_gp", "deep_kernel_gp", "mc_dropout_mlp"])
def test_torch_surrogates_with_uncertainty(data, kind):
    pytest.importorskip("torch")
    from bdopt.surrogate import make_surrogate

    pr, train, test = data
    m = make_surrogate(kind).fit(train, pr)
    assert m.score(test)["r2"].min() > 0.5, m.score(test)
    std = m.predict_unit_std(pr.encode(test))
    assert std.shape == (300, 2) and (std > 0).all()
    m2 = m.clone().fit(train, pr)                      # reproducible for a fixed seed
    assert np.allclose(m.predict(test).to_numpy(), m2.predict(test).to_numpy(), atol=1e-6)
