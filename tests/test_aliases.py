import pytest

from bdopt.eplus import Version, meter_candidates


@pytest.mark.parametrize("version", ["8.7", "9.0.1", "9.6", "22.1", "22.2", "23.1"])
def test_district_heating_plain_up_to_23_1(version):
    assert meter_candidates("DistrictHeating:Facility", Version.parse(version)) == ["DistrictHeating:Facility"]


@pytest.mark.parametrize("version", ["23.2", "24.1", "24.2", "25.2"])
def test_district_heating_water_from_23_2(version):
    v = Version.parse(version)
    assert meter_candidates("DistrictHeating:Facility", v, has_steam=False) == ["DistrictHeatingWater:Facility"]
    assert meter_candidates("DistrictHeating:Facility", v, has_steam=True) == [
        "DistrictHeatingWater:Facility", "DistrictHeatingSteam:Facility"]


def test_explicit_names_follow_the_version_and_other_meters_are_untouched():
    assert meter_candidates("DistrictHeatingWater:Facility", Version.parse("9.6")) == ["DistrictHeating:Facility"]
    assert meter_candidates("DistrictHeatingSteam:Facility", Version.parse("25.2")) == ["DistrictHeatingSteam:Facility"]
    assert meter_candidates("Heating:DistrictHeating", Version.parse("25.2"), False) == ["Heating:DistrictHeatingWater"]
    assert meter_candidates("Electricity:Facility", Version.parse("25.2")) == ["Electricity:Facility"]
    assert meter_candidates("DistrictCooling:Facility", Version.parse("8.7")) == ["DistrictCooling:Facility"]


def test_reading_considers_every_variant():
    assert meter_candidates("DistrictHeating:Facility") == [
        "DistrictHeating:Facility", "DistrictHeatingWater:Facility", "DistrictHeatingSteam:Facility"]


def test_err_file_counts_come_from_the_final_summary(tmp_path):
    """eplusout.err has warm-up, sizing and final summaries; the final one counts the whole run."""
    from bdopt.simulation.runner import parse_err_file

    err = tmp_path / "eplusout.err"
    err.write_text(
        "   ** Severe  ** something bad\\n"
        "   ************* EnergyPlus Warmup Error Summary. During Warmup: 0 Warning; 0 Severe Errors.\\n"
        "   ************* EnergyPlus Sizing Error Summary. During Sizing: 7 Warning; 0 Severe Errors.\\n"
        "   ************* EnergyPlus Completed Successfully-- 38 Warning; 1 Severe Errors; Elapsed Time=00hr 00min 28.71sec\\n"
    )
    msgs, warn, sev, completed = parse_err_file(err)
    assert (warn, sev, completed) == (38, 1, True) and len(msgs) == 1
