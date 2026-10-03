"""BESOS-compatible API: code written for BESOS notebooks runs unchanged."""

import sys

import numpy as np
import pandas as pd
import pytest

import bdopt.compat
from conftest import installs


@pytest.fixture
def besos():
    saved = {k: v for k, v in sys.modules.items() if k == "besos" or k.startswith("besos.")}
    bdopt.compat.use_as_besos()
    yield
    for k in [k for k in sys.modules if k == "besos" or k.startswith("besos.")]:
        del sys.modules[k]
    sys.modules.update(saved)


def test_imports_and_problem_api(besos):
    from besos import eppy_funcs as ef  # noqa: F401
    from besos import sampling
    from besos.evaluator import EvaluatorEP, EvaluatorGeneric  # noqa: F401
    from besos.parameters import (
        CategoryParameter,
        FieldSelector,
        Parameter,
        RangeParameter,
        expand_plist,
    )
    from besos.problem import EPProblem

    sel = FieldSelector(class_name="Lights", object_name="*", field_name="Watts per Zone Floor Area")
    # `value_descriptors=` (sic) is used in the platform's notebooks
    p = Parameter(selector=sel, value_descriptors=RangeParameter(min_val=5, max_val=20), name="Lights")
    q = Parameter(selector=sel, value_descriptor=CategoryParameter(["a", "b"]), name="Cat")
    problem = EPProblem([p], ["Electricity:Facility"])
    assert problem.names() == ["Lights", "Electricity:Facility"]
    assert p.value_descriptor.min == 5 and p.value_descriptor.max == 20  # used by the Platypus cell
    df = sampling.dist_sampler(sampling.lhs, problem, num_samples=12)
    assert df.shape == (12, 1) and df["Lights"].between(5, 20).all()
    assert q.kind == "categorical"
    assert len(expand_plist({"Lights": {"Watts per Zone Floor Area": (5, 20)}})) == 1


def test_evaluator_generic_and_nsgaii(besos):
    from besos import sampling
    from besos.evaluator import EvaluatorGeneric
    from besos.optimizer import NSGAII
    from besos.parameters import GenericSelector, Parameter, RangeParameter
    from besos.problem import EPProblem

    P = [Parameter(GenericSelector(lambda i, v: None), RangeParameter(0, 1), name=n) for n in ("x", "y")]
    pr = EPProblem(P, ["f1", "f2"])
    ev = EvaluatorGeneric(lambda ind: [ind[0] ** 2 + ind[1], (1 - ind[0]) ** 2 + ind[1]], pr)
    X = sampling.dist_sampler(sampling.lhs, pr, 5)
    out = ev.df_apply(X, keep_input=True)
    assert list(out.columns) == ["x", "y", "f1", "f2"]
    res = NSGAII(ev, evaluations=400, population_size=20)
    assert res["pareto-optimal"].all() and (res["y"] < 0.05).mean() > 0.5  # y -> 0 is optimal


def test_unsupported_features_explain_themselves(besos):
    from besos.parameters import wwr

    with pytest.raises(NotImplementedError, match="GenericSelector"):
        wwr()


@pytest.mark.eplus
@pytest.mark.skipif(not installs(), reason="no EnergyPlus")
def test_notebook_cells_run_unchanged(besos, tmp_path):
    """The BESOS cells of the platform's Test 1 notebook (get_building, selectors, EPProblem, dist_sampler,
    EvaluatorEP, df_apply) with the original arguments."""
    from besos import eppy_funcs as ef
    from besos import sampling
    from besos.evaluator import EvaluatorEP
    from besos.parameters import FieldSelector, Parameter, RangeParameter
    from besos.problem import EPProblem

    from conftest import MODELS

    building = ef.get_building(str(MODELS / "Office_IDF.idf"))
    epw_file = str(MODELS / "Larnaca_Larnax_Airp_-hour.epw")
    wall_ins = FieldSelector(class_name="Material", object_name="_InsulationEX_.01", field_name="Thickness")
    params = [Parameter(selector=wall_ins, value_descriptor=RangeParameter(min_val=0.1, max_val=0.5), name="Wall Insulation")]
    objectives = ["Electricity:Facility", "DistrictHeating:Facility", "DistrictCooling:Facility"]
    problem = EPProblem(params, objectives)
    samples_df = sampling.dist_sampler(sampling.lhs, problem, num_samples=2)
    evaluator = EvaluatorEP(problem, building, out_dir=str(tmp_path), err_dir=str(tmp_path),
                            epw_file=epw_file, progress_bar=False)
    sim = evaluator.df_apply(samples_df, keep_input=True, processes=2)
    assert list(sim.columns) == ["Wall Insulation", *objectives]
    assert np.isfinite(sim[objectives].to_numpy()).all()
    assert isinstance(sim, pd.DataFrame)
