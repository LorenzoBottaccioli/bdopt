"""Integration tests against real EnergyPlus installations (8.7 ... 25.x).

Run with e.g. ``BDOPT_EPLUS_PATH=/opt/eplus pytest -m eplus`` (all installed versions) or
``BDOPT_TEST_EPLUS_VERSIONS=8.7,9.6 pytest -m eplus``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

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
from bdopt.simulation import CommandRunner, LocalRunner
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


def _invalid_meter_warnings(run_dir) -> list[str]:
    err = (Path(run_dir) / "eplusout.err").read_text(errors="replace")
    return sorted(line.strip() for line in err.splitlines() if "Output:Meter: invalid Key Name" in line)


def test_bdopt_adds_no_energyplus_warnings(eplus, tmp_path):
    """EnergyPlus must have nothing to say about what bdopt adds. The same design run directly and through the
    evaluator gives the same warnings (same count, same invalid-meter messages) and no severe error. A meter
    that does not exist in a version makes EnergyPlus warn, so the requests are chosen per version."""
    from bdopt import IDF
    from bdopt.simulation import LocalRunner

    pr, idf_path, epw = _five_zone(eplus)
    direct = IDF.from_file(idf_path)
    pr.apply(direct, {"lights": 1000.0}, eplus.idd)
    direct.add("Output:SQLite", "SimpleAndTabular")  # the runner needs the SQL file; bdopt adds the same
    baseline = LocalRunner(eplus).run(direct.to_text(), epw, tmp_path / "direct")
    assert baseline.ok

    ev = EnergyPlusEvaluator(pr, idf_path, epw, eplus=eplus, out_dir=tmp_path / "bdopt", keep_dirs=True)
    ev.evaluate(pd.DataFrame({"lights": [1000.0]}))
    rec = ev.history[-1]
    assert _invalid_meter_warnings(rec.run_dir) == _invalid_meter_warnings(tmp_path / "direct")
    assert rec.eplus_warnings == baseline.warnings, (rec.eplus_warnings, baseline.warnings)
    assert rec.eplus_severe == 0 and baseline.severe == 0
    stats = ev.stats()
    assert stats["energyplus_severe"] == 0 and stats["energyplus_warnings"] == baseline.warnings


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
def test_reproduces_stored_reference_results(tmp_path):
    """4 design points of an office model (Larnaca weather, 6 parameters, 3 objectives, EnergyPlus 9.6): the
    objectives reproduce the stored reference values to floating-point precision (measured: 4e-13)."""
    ep = next((e for e in installs() if e.version.matches("9.6")), None)
    if ep is None:
        pytest.skip("EnergyPlus 9.6 not installed")
    rows = pd.read_json(DATA / "reference_office_rows.json")
    ref = pd.read_json(DATA / "reference_office_out.json").sort_index()
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


class _SevereRunner(LocalRunner):
    """A real EnergyPlus run whose outcome is reported with severe errors (to test the policies)."""

    def run(self, *a, **k):
        out = super().run(*a, **k)
        out.severe = 2
        out.severe_messages = ["** Severe  ** CheckWarmupConvergence: Zone=X did not converge after 25 warmup days.",
                               "** Severe  ** second one"]
        return out


@pytest.mark.parametrize("policy", ["warn", "nan", "ignore"])
def test_severe_errors_policy(eplus, tmp_path, policy):
    """A run that finishes with severe errors: ``warn`` keeps the value and warns, ``nan`` treats it as failed,
    ``ignore`` is silent. The same applies when the result later comes from the cache."""
    pr, idf, epw = _five_zone(eplus, ("Electricity:Facility",))
    ev = EnergyPlusEvaluator(pr, idf, epw, runner=_SevereRunner(eplus), out_dir=tmp_path, on_severe=policy)
    df = pd.DataFrame({"lights": [1000.0]})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = ev.evaluate(df)
    texts = " ".join(str(x.message) for x in w)
    assert ev.stats()["energyplus_severe"] == 2
    if policy == "nan":
        assert np.isnan(out.iloc[0, 0]) and "severe" in texts and any((tmp_path / "failed").iterdir())
    else:
        assert np.isfinite(out.iloc[0, 0])
        assert ("severe" in texts) == (policy == "warn")
    # second call: from the cache; the policy still applies
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        again = ev.evaluate(df)
    texts2 = " ".join(str(x.message) for x in w2)
    if policy == "nan":
        assert np.isnan(again.iloc[0, 0])  # failed runs are not cached: simulated (and rejected) again
    else:
        assert ev.n_cache_hits == 1 and again.iloc[0, 0] == out.iloc[0, 0]
        assert ("severe" in texts2) == (policy == "warn")
    assert ev.history[-1].parameters == {"lights": 1000.0}


def test_on_severe_must_be_valid(tmp_path):
    from bdopt import EnergyPlusEvaluator as E
    from conftest import analytic_problem

    pr, _ = analytic_problem(2)
    with pytest.raises(ValueError, match="on_severe"):
        E(pr, "x.idf", None, on_severe="maybe")


@pytest.mark.slow
def test_real_case_warmup_convergence_severe_error(tmp_path):
    """One design of the office model (EnergyPlus 9.6) finishes with two *severe* errors, ``CheckWarmupConvergence:
    Loads Initialization ... did not converge``. It was found in the tutorial study; raising the maximum number of
    warm-up days (25, 60, 200) does not remove it and does not change the objectives, so bdopt reports it instead of
    hiding it and lets the user choose what to do with such designs (``on_severe``)."""
    ep = next((e for e in installs() if e.version.matches("9.6")), None)
    if ep is None:
        pytest.skip("EnergyPlus 9.6 not installed")
    P = [
        Parameter(FieldSelector("Material", "_InsulationEX_.01", "Thickness"), RangeParameter(0.1, 0.5), name="wall"),
        Parameter(FieldSelector("Material", "_InsulationEX_Roof_.01", "Thickness"), RangeParameter(0.1, 0.8), name="roof"),
        Parameter(FieldSelector("Lights", "*", "Watts per Zone Floor Area"), RangeParameter(5, 20), name="lights"),
        Parameter(FieldSelector("ZoneVentilation:DesignFlowRate", "*", "Air Changes per Hour"), RangeParameter(0, 6), name="ach"),
    ]
    pr = Problem(P, ["Electricity:Facility"])
    design = pd.DataFrame({"wall": [0.1159], "roof": [0.4066], "lights": [18.0846], "ach": [3.8009]})
    kwargs = dict(eplus=ep, out_dir=tmp_path)
    with pytest.warns(UserWarning, match="did not converge after 25 warmup days"):
        ev = EnergyPlusEvaluator(pr, MODELS / "Office_IDF.idf", MODELS / "Larnaca_Larnax_Airp_-hour.epw", **kwargs)
        value = ev.evaluate(design).iloc[0, 0]
    assert np.isfinite(value) and ev.history[-1].eplus_severe == 2
    strict = EnergyPlusEvaluator(pr, MODELS / "Office_IDF.idf", MODELS / "Larnaca_Larnax_Airp_-hour.epw",
                                 eplus=ep, out_dir=tmp_path / "strict", on_severe="nan")
    with pytest.warns(UserWarning, match="failed"):
        assert np.isnan(strict.evaluate(design).iloc[0, 0])
