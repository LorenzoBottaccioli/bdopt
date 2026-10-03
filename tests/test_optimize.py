import numpy as np

from bdopt import (
    CategoryParameter,
    FunctionEvaluator,
    GenericSelector,
    IntegerParameter,
    MeterReader,
    Parameter,
    Problem,
    RangeParameter,
)
from bdopt.optimize import NSGA2, nsga2, nsga3
from bdopt.surrogate import GPSurrogate
from conftest import analytic_problem


def _problem():
    P = [
        Parameter(GenericSelector(lambda i, v: None, "a"), RangeParameter(0, 1), name="a"),
        Parameter(GenericSelector(lambda i, v: None, "b"), RangeParameter(0, 1), name="b"),
        Parameter(GenericSelector(lambda i, v: None, "g"), CategoryParameter(["single", "double", "triple"]), name="glazing"),
        Parameter(GenericSelector(lambda i, v: None, "n"), IntegerParameter(1, 4), name="n"),
    ]
    loss = {"single": (1.0, 0.5), "double": (0.6, 1.0), "triple": (0.3, 2.0)}  # heat loss vs cost: a real trade-off

    def f(r):
        h, c = loss[r["glazing"]]
        return [h * (1 - 0.5 * r["a"]) + 0.2 * r["n"] * (1 - r["b"]), c * (0.5 + r["a"]) + r["b"], r["a"] * r["b"]]

    pr = Problem(P, ["heat", "cost", MeterReader("gain", direction="max")],
                 constraints=[lambda r: r["a"] + r["b"] - 1.8])
    return pr, FunctionEvaluator(pr, f)


def _dominated(F):
    return any(np.all(F[j] <= F[i]) and np.any(F[j] < F[i]) for i in range(len(F)) for j in range(len(F)) if i != j)


def test_nsga2_mixed_constrained_maximise():
    pr, ev = _problem()
    front = nsga2(pr, ev, pop_size=60, n_gen=40, seed=1)
    assert len(front) > 10
    assert (front.df["a"] + front.df["b"] <= 1.8 + 1e-9).all()  # constraint
    assert set(front.df["glazing"]) == {"single", "double", "triple"}  # real trade-off found
    assert set(front.df["n"]) <= {1, 2, 3, 4}
    assert not _dominated(front.objectives.to_numpy() * pr.signs)  # 'gain' is maximised
    best = front.closest_to_ideal()
    assert best["heat"] <= front.df["heat"].max() and best["gain"] >= front.df["gain"].min()


def test_nsga3_and_reproducible():
    pr, ev = _problem()
    f = nsga3(pr, ev, pop_size=40, n_gen=20, n_partitions=6)
    assert len(f) >= 5
    a = NSGA2(pop_size=30, n_gen=10, seed=3).run(pr, ev).df
    b = NSGA2(pop_size=30, n_gen=10, seed=3).run(pr, ev).df
    assert a.equals(b)


def test_optimizes_on_surrogate():
    pr, ev = analytic_problem(3)
    from bdopt import sampling

    data = ev.evaluate(sampling.lhs(pr, 50, seed=0), keep_input=True)
    sur = GPSurrogate().fit(data, pr)
    front = nsga2(pr, sur, pop_size=40, n_gen=20)
    # what the optimizer reports for each design is exactly what the surrogate predicts for it
    np.testing.assert_allclose(front.objectives.to_numpy(), sur.predict(front.parameters).to_numpy(), rtol=1e-6)
    assert len(front) > 5


def test_closest_to_ideal_normalisation_matters():
    import pandas as pd

    from bdopt.optimize import ParetoFront

    pr, _ = _problem()
    df = pd.DataFrame({"a": [0, 0], "b": [0, 0], "glazing": ["single"] * 2, "n": [1, 1],
                       "heat": [1.0, 2.0], "cost": [1e9, 1.0], "gain": [0.0, 0.0]})
    front = ParetoFront(df, pr)
    assert front.closest_to_ideal(normalize=False)["cost"] == 1.0  # raw distance dominated by 'cost'
    assert front.closest_to_ideal(normalize=True) is not None
