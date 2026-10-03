from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from bdopt import (
    CategoryParameter,
    FunctionEvaluator,
    GenericSelector,
    IntegerParameter,
    Parameter,
    Problem,
    RangeParameter,
)
from bdopt.eplus import find_installations

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "examples" / "models"
DATA = Path(__file__).parent / "data"


def installs():
    return find_installations()


def wanted_versions() -> list[str]:
    """Versions to test: BDOPT_TEST_EPLUS_VERSIONS (comma separated) or every installed one."""
    env = os.environ.get("BDOPT_TEST_EPLUS_VERSIONS")
    have = {e.version.short for e in installs()}
    if env:
        return [v.strip() for v in env.split(",") if v.strip()]
    return sorted(have, key=lambda s: tuple(map(int, s.split("."))))


def pytest_generate_tests(metafunc):
    if "eplus_version" in metafunc.fixturenames:
        metafunc.parametrize("eplus_version", wanted_versions() or ["none"])


@pytest.fixture
def eplus(eplus_version):
    from bdopt.eplus import EnergyPlus, EnergyPlusNotFound

    if eplus_version == "none":
        pytest.skip("no EnergyPlus installation found (set BDOPT_EPLUS_PATH)")
    try:
        return EnergyPlus.find(eplus_version)
    except EnergyPlusNotFound:
        pytest.skip(f"EnergyPlus {eplus_version} not installed")


def need_eplus(version: str | None = None):
    """Skip decorator/marker helper for tests that need an installation."""
    have = {e.version.short for e in installs()}
    ok = bool(have) if version is None else version in have
    return pytest.mark.skipif(not ok, reason=f"EnergyPlus {version or ''} not installed")


def analytic_problem(d: int = 4, mixed: bool = False):
    """A cheap multi-output analytic problem with a sharp feature, for sampler/surrogate tests."""
    params = [
        Parameter(GenericSelector(lambda i, v: None, f"p{k}"), RangeParameter(0, 1), name=f"p{k}")
        for k in range(d)
    ]
    if mixed:
        params.append(
            Parameter(GenericSelector(lambda i, v: None, "g"), CategoryParameter(["a", "b", "c"]), name="g")
        )
        params.append(Parameter(GenericSelector(lambda i, v: None, "n"), IntegerParameter(1, 4), name="n"))
    pr = Problem(params, ["f1", "f2"])

    def f(r):
        x = np.array([r[f"p{k}"] for k in range(d)])
        a, b, c, e = (x[min(i, d - 1)] for i in range(4))
        base = [
            np.tanh(12 * (a - 0.3)) + 0.3 * b + 0.2 * c * e,
            0.5 * np.sin(6 * b) * (a > 0.2) + c**2,
        ]
        if mixed:
            k = {"a": 0.0, "b": 0.5, "c": 1.0}[r["g"]]
            base = [base[0] + k + 0.1 * r["n"], base[1] - k]
        return base

    return pr, FunctionEvaluator(pr, f)


def fast_gp():
    """GP with few restarts: enough for loop/plumbing tests (the robust default is tested separately)."""
    from bdopt.surrogate import GPSurrogate

    return GPSurrogate(n_restarts=2, max_extra_restarts=6)
