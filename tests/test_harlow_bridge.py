"""Needs harlow (and therefore TensorFlow): skipped otherwise."""

import os

import pytest

from conftest import analytic_problem

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
pytest.importorskip("harlow.sampling")

from bdopt.adaptive import HarlowAdaptiveSampler  # noqa: E402


def test_bridge_runs_harlow_on_mixed_problem():
    pr, ev = analytic_problem(2, mixed=True)
    r = HarlowAdaptiveSampler(pr, ev, method="FuzzyLolaVoronoi").run(
        n_initial=10, n_test=4, max_iter=2, tol=1e-9
    )
    assert len(r.data) == 12 and len(r.test) == 4
    assert set(r.data["g"]) <= {"a", "b", "c"} and set(r.data["n"]) <= {1, 2, 3, 4}
    assert list(r.history["iteration"]) == [0, 1, 2] and r.history["scaled_rmse"].iloc[1:].notna().all()
    assert r.surrogate.predict(r.test).shape == (4, 2)
