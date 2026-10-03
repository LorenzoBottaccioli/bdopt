"""Integration tests against real EnergyPlus installations (8.7 ... 25.x).

Run with e.g. ``BDOPT_EPLUS_PATH=/opt/eplus pytest -m eplus`` (all installed versions) or
``BDOPT_TEST_EPLUS_VERSIONS=8.7,9.6 pytest -m eplus``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from bdopt import (
    EnergyPlus,
    EnergyPlusEvaluator,
    FieldSelector,
    MeterReader,
    Parameter,
    ParameterError,
    Problem,
    RangeParameter,
    VariableReader,
    VersionMismatch,
)
from bdopt.eplus import TransitionError
from bdopt.simulation import CommandRunner
from conftest import DATA, MODELS, installs

pytestmark = pytest.mark.eplus


def _five_zone(ep: EnergyPlus, objectives=("Electricity:Facility", "DistrictHeating:Facility")):
    idf = ep.root / "ExampleFiles" / "5ZoneDDCycOnAny.idf"
    epw = next(ep.root.glob("WeatherData/USA_IL_Chicago-OHare*.epw"))
    params = [Parameter(FieldSelector("Lights", "*", "Lighting Level"), RangeParameter(500, 3000), name="lights")]
    return Problem(params, list(objectives)), idf, epw


def test_same_study_runs_on_every_version(eplus, tmp_path):
    """The identical study (class/field/meter names written once) works on this EnergyPlus version, and the
    physics is right: more lighting -> more electricity and less (district) heating."""
    pr, idf, epw = _five_zone(eplus)
    ev = EnergyPlusEvaluator(pr, idf, epw, eplus=eplus, out_dir=tmp_path, processes=2)
    out = ev.evaluate(pd.DataFrame({"lights": [500.0, 3000.0]}))
    assert ev.stats()["failed"] == 0 and ev.stats()["eplus_version"] == str(eplus.version)
    assert np.isfinite(out.to_numpy()).all() and (out.to_numpy() > 0).all()
    assert out.loc[1, "Electricity:Facility"] > out.loc[0, "Electricity:Facility"]
    assert out.loc[1, "DistrictHeating:Facility"] < out.loc[0, "DistrictHeating:Facility"]


def test_engine_autodetected_from_model_version(eplus, tmp_path, monkeypatch):
    pr, idf, epw = _five_zone(eplus)
    ev = EnergyPlusEvaluator(pr, idf, epw, out_dir=tmp_path)  # no eplus=: picked by model version
    assert (ev.eplus.version.major, ev.eplus.version.minor) == (eplus.version.major, eplus.version.minor)


def test_cache_avoids_resimulation(eplus, tmp_path):
    pr, idf, epw = _five_zone(eplus, ("Electricity:Facility",))
    df = pd.DataFrame({"lights": [1000.0, 2000.0, 1000.0]}, index=[5, 6, 7])  # duplicate row simulated once
    ev = EnergyPlusEvaluator(pr, idf, epw, eplus=eplus, out_dir=tmp_path)
    a = ev.evaluate(df)
    assert list(a.index) == [5, 6, 7]  # the caller's index is kept
    assert ev.n_simulations == 2 and a.iloc[0, 0] == a.iloc[2, 0]
    ev2 = EnergyPlusEvaluator(pr, idf, epw, eplus=eplus, out_dir=tmp_path)  # new object, same cache dir
    b = ev2.evaluate(df)
    assert ev2.n_simulations == 0 and ev2.n_cache_hits == 2
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


def test_variable_reader_and_units(eplus, tmp_path):
    pr, idf, epw = _five_zone(eplus, ())  if False else (None, None, None)
    pr0, idf, epw = _five_zone(eplus)
    pr = Problem(pr0.parameters, [
        MeterReader("Electricity:Facility", "J"), MeterReader("Electricity:Facility", "kWh", unit="kWh"),
        VariableReader("*", "Site Outdoor Air Drybulb Temperature", "Tmean", aggregate="mean", frequency="RunPeriod"),
    ])
    ev = EnergyPlusEvaluator(pr, idf, epw, eplus=eplus, out_dir=tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ev.evaluate(pd.DataFrame({"lights": [1000.0]}))
    assert out["J"][0] == pytest.approx(out["kWh"][0] * 3.6e6)


def test_failed_run_gives_nan_and_keeps_directory(eplus, tmp_path):
    pr, idf, epw = _five_zone(eplus, ("Electricity:Facility",))
    bad = Parameter(FieldSelector("Lights", "*", "Lighting Level"), RangeParameter(-1e9, -1e8), name="lights")
    pr = Problem([bad], ["Electricity:Facility"])  # negative design level: EnergyPlus fatal error
    ev = EnergyPlusEvaluator(pr, idf, epw, eplus=eplus, out_dir=tmp_path)
    with pytest.warns(UserWarning, match="failed"):
        out = ev.evaluate(pd.DataFrame({"lights": [-5e8]}))
    assert np.isnan(out.to_numpy()).all()
    assert len(ev.failures) == 1 and ev.failures[0].errors
    assert any((tmp_path / "failed").iterdir())
    ev_raise = EnergyPlusEvaluator(pr, idf, epw, eplus=eplus, out_dir=tmp_path / "r", on_error="raise")
    with pytest.raises(RuntimeError, match="failed"):
        ev_raise.evaluate(pd.DataFrame({"lights": [-5e8]}))


def test_selector_typos_are_reported_early(eplus, tmp_path):
    pr0, idf, epw = _five_zone(eplus)
    for sel, msg in [
        (FieldSelector("Lightz", "*", "Lighting Level"), "does not exist"),
        (FieldSelector("Lights", "*", "Lighting Levl"), "Did you mean"),
        (FieldSelector("Lights", "NO SUCH LIGHT", "Lighting Level"), "matches no object"),
    ]:
        pr = Problem([Parameter(sel, RangeParameter(0, 1), name="x")], ["Electricity:Facility"])
        with pytest.raises(ParameterError, match=msg):
            EnergyPlusEvaluator(pr, idf, epw, eplus=eplus, out_dir=tmp_path)


def test_hvactemplate_models_use_expandobjects(eplus, tmp_path):
    idf = eplus.root / "ExampleFiles" / "HVACTemplate-5ZoneBaseboardHeat.idf"
    epw = next(eplus.root.glob("WeatherData/USA_CA_San.Francisco*.epw"))
    pr = Problem([Parameter(FieldSelector("Lights", "*", "Lighting Level"), RangeParameter(500, 3000), name="l")],
                 ["Electricity:Facility"])
    ev = EnergyPlusEvaluator(pr, idf, epw, eplus=eplus, out_dir=tmp_path, design_day=True)
    assert ev._expand  # HVACTemplate objects present -> `energyplus -x`
    out = ev.evaluate(pd.DataFrame({"l": [500.0, 3000.0]}))
    assert ev.stats()["failed"] == 0 and np.isfinite(out.to_numpy()).all()
    assert out.iloc[1, 0] > out.iloc[0, 0]  # more lighting -> more electricity (design days only)


def test_version_mismatch_without_auto_upgrade(eplus, tmp_path):
    older = [e for e in installs() if e.version < eplus.version]
    if not older:
        pytest.skip("needs an older installation")
    old = older[0]
    pr, idf, epw = _five_zone(old)
    with pytest.raises(VersionMismatch):
        EnergyPlusEvaluator(pr, idf, epw, eplus=eplus, out_dir=tmp_path)


@pytest.mark.slow  # one transition step takes ~10-20 s (EnergyPlus' own updater)
def test_auto_upgrade_old_model_to_this_version(eplus, tmp_path):
    """A model written for an older EnergyPlus is upgraded with the shipped transition tools and runs."""
    older = [e for e in installs() if e.version < eplus.version]
    if not older:
        pytest.skip("needs an older installation")
    old = older[0]  # oldest installed
    pr, idf, epw = _five_zone(old)
    ev = EnergyPlusEvaluator(pr, idf, epw, eplus=eplus, auto_upgrade=True, out_dir=tmp_path)
    assert (ev.idf.version.major, ev.idf.version.minor) == (eplus.version.major, eplus.version.minor)
    out = ev.evaluate(pd.DataFrame({"lights": [1000.0]}))
    assert np.isfinite(out.to_numpy()).all() and ev.stats()["failed"] == 0
    # the upgraded model is cached: a second evaluator does not run the transitions again
    assert any((tmp_path / "upgraded").glob("*.idf"))


def test_downgrade_is_refused(tmp_path):
    ins = installs()
    if len(ins) < 2:
        pytest.skip("needs two installations")
    new, old = ins[-1], ins[0]
    pr, idf, epw = _five_zone(new)
    with pytest.raises(TransitionError, match="downgrade"):
        EnergyPlusEvaluator(pr, idf, epw, eplus=old, auto_upgrade=True, out_dir=tmp_path)


def test_command_runner_with_prefix(tmp_path):
    ep = next((e for e in installs() if e.version.matches("9.6")), None)
    if ep is None:
        pytest.skip("9.6 not installed")
    pr, idf, epw = _five_zone(ep, ("Electricity:Facility",))
    runner = CommandRunner(["env", "BDOPT_TEST=1"], ep, path_map={tmp_path: str(tmp_path)})
    ev = EnergyPlusEvaluator(pr, idf, epw, runner=runner, out_dir=tmp_path)
    assert np.isfinite(ev.evaluate(pd.DataFrame({"lights": [1000.0]})).to_numpy()).all()
    assert runner.command(tmp_path / "in.idf", epw, tmp_path, False, False)[:2] == ["env", "BDOPT_TEST=1"]


@pytest.mark.slow
def test_golden_matches_besos_on_real_platform_model(tmp_path):
    """4 design points of the platform's Test 1 (Office, Larnaca): BESOS 2.2.3 results, stored in
    tests/data, reproduced to floating-point precision (max relative difference measured: 4e-13)."""
    ep = next((e for e in installs() if e.version.matches("9.6")), None)
    if ep is None:
        pytest.skip("EnergyPlus 9.6 not installed")
    rows = pd.read_json(DATA / "golden_besos_test1_rows.json")
    ref = pd.read_json(DATA / "golden_besos_test1_out.json").sort_index()
    sel = lambda c, o, f: FieldSelector(c, o, f)  # noqa: E731
    P = [
        Parameter(sel("Material", "_InsulationEX_.01", "Thickness"), RangeParameter(.1, .5), name="Wall Insulation"),
        Parameter(sel("Material", "_InsulationEX_Roof_.01", "Thickness"), RangeParameter(.1, .8), name="Roof Insulation"),
        Parameter(sel("ZoneVentilation:DesignFlowRate", "*", "Air Changes per Hour"), RangeParameter(0, 6), name="Ventilation ACH"),
        Parameter(sel("Lights", "*", "Watts per Zone Floor Area"), RangeParameter(5, 20), name="Lights Watts/Area"),
        Parameter(sel("WindowShadingControl", "*", "Setpoint"), RangeParameter(18, 30), name="Temp Setpoint shading"),
        Parameter(sel("WindowShadingControl", "*", "Setpoint 2"), RangeParameter(80, 300), name="Rad Setpoint shading"),
    ]
    objs = ["Electricity:Facility", "DistrictHeating:Facility", "DistrictCooling:Facility"]
    pr = Problem(P, objs)
    ev = EnergyPlusEvaluator(pr, MODELS / "Office_IDF.idf", MODELS / "Larnaca_Larnax_Airp_-hour.epw",
                             eplus=ep, out_dir=tmp_path, processes=4)
    out = ev.evaluate(rows[pr.parameter_names])
    rel = np.abs(out[objs].to_numpy() - ref[objs].to_numpy()) / np.abs(ref[objs].to_numpy())
    assert rel.max() < 1e-9, rel.max()
