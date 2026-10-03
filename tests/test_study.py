import json

import pandas as pd
import pytest

from bdopt.cli import main
from bdopt.parameters import CategoryParameter, IntegerParameter, ParameterError, RangeParameter
from bdopt.study import DesignStudy, objective_from_config, parameter_from_config
from conftest import MODELS, installs


def test_parameter_config_forms():
    p = parameter_from_config({"name": "a", "class": "Lights", "field": "Lighting Level", "min": 1, "max": 2})
    assert isinstance(p.value_descriptor, RangeParameter) and p.name == "a"
    p = parameter_from_config({"class": "Zone", "field": "Multiplier", "min": 1, "max": 4, "type": "integer"})
    assert isinstance(p.value_descriptor, IntegerParameter)
    p = parameter_from_config({"name": "g", "class": "FenestrationSurface:Detailed", "field": "Construction Name",
                               "where": {"Surface Type": "Window"}, "options": ["a", "b"]})
    assert isinstance(p.value_descriptor, CategoryParameter) and p.selector.where == {"Surface Type": "Window"}
    with pytest.raises(ParameterError):
        parameter_from_config({"class": "Lights", "field": "x", "type": "weird"})


def test_objective_config_forms():
    assert objective_from_config("Electricity:Facility") == "Electricity:Facility"
    o = objective_from_config({"meter": "Electricity:Facility", "name": "kWh", "unit": "kWh", "direction": "max"})
    assert o.name == "kWh" and o.unit == "kWh" and o.direction == "max"
    v = objective_from_config({"variable": "Zone Mean Air Temperature", "aggregate": "mean"})
    assert v.variable == "Zone Mean Air Temperature" and v.aggregate == "mean"
    with pytest.raises(ParameterError):
        objective_from_config({"name": "nothing"})


def _config(tmp_path):
    return {
        "output_dir": str(tmp_path / "out"),
        "model": {"idf": str(MODELS.parent.parent / "NOT_USED.idf"), "epw": "x.epw"},
        "parameters": [{"name": "w", "class": "Material", "object": "_InsulationEX_.01", "field": "Thickness", "min": 0.1, "max": 0.5}],
        "objectives": ["Electricity:Facility"],
    }


def test_problem_built_from_config(tmp_path):
    study = DesignStudy(_config(tmp_path))
    assert study.problem.parameter_names == ["w"] and study.problem.objective_names == ["Electricity:Facility"]


@pytest.mark.eplus
@pytest.mark.skipif(not any(e.version.matches("9.6") for e in installs()), reason="needs EnergyPlus 9.6")
def test_study_end_to_end_and_cli(tmp_path, capsys):
    cfg = {
        "output_dir": str(tmp_path / "out"),
        "model": {"idf": str(MODELS / "Office_IDF.idf"), "epw": str(MODELS / "Larnaca_Larnax_Airp_-hour.epw")},
        "parameters": [
            {"name": "Wall", "class": "Material", "object": "_InsulationEX_.01", "field": "Thickness", "min": 0.1, "max": 0.5},
            {"name": "Lights", "class": "Lights", "field": "Watts per Zone Floor Area", "min": 5, "max": 20},
        ],
        "objectives": ["Electricity:Facility", "DistrictCooling:Facility"],
        "evaluation": {"processes": 4},
        "sampling": {"method": "adaptive", "n_initial": 4, "n_test": 2, "max_iter": 2, "tol": 0.0, "verbose": False},
        "surrogate": {"type": "gp", "params": {"n_restarts": 1}},
        "optimization": {"algorithm": "nsga2", "pop_size": 16, "n_gen": 5, "verify": 2},
    }
    path = tmp_path / "study.json"
    path.write_text(json.dumps(cfg))
    assert main(["run", str(path)]) == 0
    out = tmp_path / "out"
    for f in ("train.csv", "test.csv", "history.csv", "metrics.csv", "pareto.csv", "best.csv", "verification.csv",
              "surrogate.pkl", "stats.json"):
        assert (out / f).exists(), f
    assert len(pd.read_csv(out / "train.csv")) == 4 + 2
    ver = pd.read_csv(out / "verification.csv")  # up to `verify` designs (fewer if the front is smaller)
    assert 1 <= len(ver) <= 2 and "Electricity:Facility (surrogate)" in ver.columns
    assert "selected design" in capsys.readouterr().out
    # run again: everything comes from the cache
    stats = json.loads((out / "stats.json").read_text())
    assert stats["failed"] == 0 and stats["simulations"] >= 8


def test_cli_eplus_list(capsys):
    rc = main(["eplus", "list"])
    out = capsys.readouterr()
    assert rc == (0 if installs() else 1)
    if installs():
        assert str(installs()[0].version) in out.out
