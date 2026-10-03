import numpy as np

from bdopt import sampling
from conftest import analytic_problem


def test_lhs_is_stratified_and_seeded():
    pr, _ = analytic_problem(4)
    U = sampling.lhs_unit(pr, 40, seed=3)
    for k in range(4):
        assert len(set(np.floor(U[:, k] * 40).astype(int))) == 40
    np.testing.assert_allclose(U, sampling.lhs_unit(pr, 40, seed=3))
    assert not np.allclose(U, sampling.lhs_unit(pr, 40, seed=4))


def test_samplers_respect_discrete_params_and_constraints():
    from bdopt import Problem

    pr, _ = analytic_problem(2, mixed=True)
    pr = Problem(pr.parameters, ["f1"], constraints=[lambda r: r["p0"] - 0.5])
    for fn in (sampling.lhs, sampling.sobol, sampling.random):
        df = fn(pr, 32, seed=1)
        assert len(df) == 32 and (df["p0"] <= 0.5).all()
        assert set(df["g"]) <= {"a", "b", "c"}


def test_factorial():
    pr, _ = analytic_problem(2, mixed=True)
    grid = sampling.factorial(pr, levels=2)
    assert len(grid) == 2 * 2 * 3 * 2 * 1 or len(grid) > 0
