"""One design variable writing in several places: object-name lists, MultiSelector, YAML ``targets``."""

import pickle

import pytest

from bdopt import (
    CategoryParameter,
    FieldSelector,
    FunctionReader,
    MeterReader,
    MultiSelector,
    Parameter,
    ParameterError,
    Problem,
    RangeParameter,
    VariableReader,
)
from bdopt.eplus.idd import IDD
from bdopt.idf import IDF
from bdopt.objectives import function_fingerprint
from bdopt.parameters import selector_from_spec
from bdopt.study import parameter_from_config

IDD_TEXT = r"""!IDD_Version 9.6.0
Schedule:Compact,
  A1 , \field Name
  A2 , \field Schedule Type Limits Name
  A3 , \field Field 1
  A4 , \field Field 2
  A5 , \field Field 3
  A6 , \field Field 4
FenestrationSurface:Detailed,
  A1 , \field Name
  A2 , \field Surface Type
  A3 , \field Construction Name
Lights,
  A1 , \field Name
  A2 , \field Zone or ZoneList Name
  N1 , \field Lighting Level
  N2 , \field Watts per Zone Floor Area
"""

IDF_TEXT = """
Schedule:Compact, Heat Z1, Temp, Through: 12/31, For: AllDays, Until: 24:00, 20;
Schedule:Compact, Heat Z2, Temp, Through: 12/31, For: AllDays, Until: 24:00, 20;
Schedule:Compact, Cool Z1, Temp, Through: 12/31, For: AllDays, Until: 24:00, 26;
FenestrationSurface:Detailed, Win A, Window, single;
FenestrationSurface:Detailed, Win B, Window, single;
FenestrationSurface:Detailed, Win C, Window, single;
FenestrationSurface:Detailed, Door D, Door, single;
Lights, L1, Z1, 100, 8;
Lights, L2, Z2, 100, 8;
"""


@pytest.fixture
def idd():
    return IDD.from_text(IDD_TEXT)


@pytest.fixture
def idf():
    return IDF.from_text(IDF_TEXT)


def constructions(idf):
    return {o.name: o.fields[2] for o in idf.by_class("FenestrationSurface:Detailed")}


# -- IDF.find / FieldSelector with a list of names ------------------------------------------------
def test_find_accepts_a_collection_of_names(idf):
    names = lambda objs: [o.name for o in objs]  # noqa: E731
    assert names(idf.find("FenestrationSurface:Detailed", ["win a", "WIN C"])) == ["Win A", "Win C"]
    assert names(idf.find("FenestrationSurface:Detailed", ("Win B",))) == ["Win B"]
    assert names(idf.find("FenestrationSurface:Detailed", {"Win A", "nope"})) == ["Win A"]
    assert idf.find("FenestrationSurface:Detailed", []) == []
    assert len(idf.find("FenestrationSurface:Detailed")) == 4  # unchanged: '*' and a single name
    assert len(idf.find("FenestrationSurface:Detailed", "*")) == 4
    assert names(idf.find("FenestrationSurface:Detailed", "Win B")) == ["Win B"]


def test_field_selector_with_a_list_writes_only_those(idf, idd):
    sel = FieldSelector("FenestrationSurface:Detailed", ["Win A", "Win C"], "Construction Name")
    assert sel.validate(idf, idd) == 2
    sel.apply(idf, "double", idd)
    assert constructions(idf) == {"Win A": "double", "Win B": "single", "Win C": "double", "Door D": "single"}


def test_field_selector_list_reports_every_missing_name(idf, idd):
    sel = FieldSelector("FenestrationSurface:Detailed", ["Win A", "Win Z", "Win Y"], "Construction Name")
    with pytest.raises(ParameterError, match=r"2 of 3 listed .* 'Win Z', 'Win Y'"):
        sel.validate(idf, idd)  # a typo must not silently shrink the selection


def test_field_selector_list_names_and_errors():
    one = FieldSelector("Lights", ["L1"], "Lighting Level")
    many = FieldSelector("Lights", ["L1", "L2"], "Lighting Level")
    assert one.default_name == "Lights:L1:Lighting Level"
    assert many.default_name == "Lights:2 objects:Lighting Level"
    assert "2 objects" in repr(many)
    with pytest.raises(ParameterError, match="empty list"):
        FieldSelector("Lights", [], "Lighting Level")
    assert FieldSelector("Lights", ("L1", "L2"), "Lighting Level").object_name == ["L1", "L2"]


# -- MultiSelector -------------------------------------------------------------------------------
def heating_selector():
    return MultiSelector([
        FieldSelector("Schedule:Compact", "Heat Z1", field_index=5),
        FieldSelector("Schedule:Compact", "Heat Z2", field_index=5),
        FieldSelector("Lights", "*", "Watts per Zone Floor Area"),  # another class and field
    ])


def test_multi_selector_writes_every_place(idf, idd):
    sel = heating_selector()
    assert sel.validate(idf, idd) == 4
    sel.apply(idf, 21.5, idd)
    sched = {o.name: o.fields[5] for o in idf.by_class("Schedule:Compact")}
    assert sched == {"Heat Z1": "21.5", "Heat Z2": "21.5", "Cool Z1": "26"}
    assert [o.fields[3] for o in idf.by_class("Lights")] == ["21.5", "21.5"]


def test_multi_selector_error_names_the_failing_target(idf, idd):
    sel = MultiSelector([
        FieldSelector("Lights", "*", "Watts per Zone Floor Area"),
        FieldSelector("Schedule:Compact", "Heat Z9", field_index=5),
    ])
    with pytest.raises(ParameterError, match="target 2 of 2: .*matches no object"):
        sel.validate(idf, idd)
    bad_field = MultiSelector([FieldSelector("Lights", "*", "Lightning Level")])
    with pytest.raises(KeyError, match="target 1 of 1"):
        bad_field.validate(idf, idd)


def test_multi_selector_construction_rules():
    with pytest.raises(ParameterError, match="at least one"):
        MultiSelector([])
    with pytest.raises(ParameterError, match="not a selector"):
        MultiSelector([object()])
    one = MultiSelector([FieldSelector("Lights", "L1", "Lighting Level")])
    assert one.default_name == "Lights:L1:Lighting Level"
    assert heating_selector().default_name.endswith("(+2 more)")
    assert MultiSelector([FieldSelector("Lights", "L1", "Lighting Level")], name="Light").default_name == "Light"


def test_multi_selector_targets_are_unique(idf, idd):
    sel = MultiSelector([
        FieldSelector("Lights", "*", "Lighting Level"),
        FieldSelector("Lights", ["L1"], "Watts per Zone Floor Area"),  # L1 is reached twice
    ])
    assert sorted(o.name for o in sel.targets(idf, idd)) == ["L1", "L2"]


# -- from_targets / selector_from_spec -----------------------------------------------------------
def test_from_targets_groups_by_class_and_field():
    sel = MultiSelector.from_targets(
        ["Win A", "Win B", "Win C"], class_name="FenestrationSurface:Detailed", field_name="Construction Name"
    )
    assert len(sel.selectors) == 1
    assert sel.selectors[0].object_name == ["Win A", "Win B", "Win C"]

    mixed = MultiSelector.from_targets([
        {"class": "Schedule:Compact", "object": "Heat Z1", "field_index": 5},
        {"class": "Schedule:Compact", "object": "Heat Z2", "field_index": 5},
        {"class": "Lights", "object": "*", "field": "Lighting Level"},
        {"class": "Schedule:Compact", "object": "Cool Z1", "field_index": 5},  # not consecutive: own group
    ])
    assert [type(s.object_name).__name__ for s in mixed.selectors] == ["list", "str", "list"]
    assert mixed.selectors[0].object_name == ["Heat Z1", "Heat Z2"]


def test_from_targets_inherits_and_overrides_defaults(idf, idd):
    sel = MultiSelector.from_targets(
        ["Win A", {"object": "Door D", "field": "Surface Type"}],
        class_name="FenestrationSurface:Detailed", field_name="Construction Name",
    )
    sel.apply(idf, "X", idd)
    assert constructions(idf)["Win A"] == "X"
    assert [o.fields[1] for o in idf.by_class("FenestrationSurface:Detailed") if o.name == "Door D"] == ["X"]


def test_from_targets_needs_class_and_field():
    with pytest.raises(ParameterError, match="needs a class and a field"):
        MultiSelector.from_targets(["Win A"])
    with pytest.raises(ParameterError, match="needs a class and a field"):
        MultiSelector.from_targets([{"object": "Win A"}], class_name="Lights")


def test_selector_from_spec_forms():
    assert isinstance(selector_from_spec({"class": "Lights", "field": "Lighting Level"}), FieldSelector)
    listed = selector_from_spec({"class": "Lights", "object": ["L1", "L2"], "field": "Lighting Level"})
    assert listed.object_name == ["L1", "L2"]
    multi = selector_from_spec({"class": "Lights", "field": "Lighting Level", "targets": ["L1", "L2"]})
    assert isinstance(multi, MultiSelector)
    with pytest.raises(ParameterError, match="needs 'class'"):
        selector_from_spec({"field": "x"})


# -- the promise: one parameter, one column -------------------------------------------------------
def test_one_parameter_is_one_column_and_writes_all_targets(idf, idd):
    glazing = Parameter(
        MultiSelector.from_targets(
            ["Win A", "Win B", "Win C"], class_name="FenestrationSurface:Detailed", field_name="Construction Name"
        ),
        CategoryParameter(["single", "double", "triple"]),
        name="Glazing",
    )
    heating = Parameter(heating_selector(), RangeParameter(18, 22), name="Heating set point")
    pr = Problem([glazing, heating], ["Electricity:Facility"])
    assert pr.n_var == 2 and pr.parameter_names == ["Glazing", "Heating set point"]
    pr.validate(idf, idd)
    pr.apply(idf, {"Glazing": "triple", "Heating set point": 19.0}, idd)
    assert constructions(idf) == {"Win A": "triple", "Win B": "triple", "Win C": "triple", "Door D": "single"}
    assert {o.fields[5] for o in idf.by_class("Schedule:Compact") if o.name.startswith("Heat")} == {"19.0"}


def test_problem_validate_names_the_parameter(idf, idd):
    bad = Parameter(
        MultiSelector.from_targets(["Win A", "Win Q"], class_name="FenestrationSurface:Detailed",
                                   field_name="Construction Name"),
        CategoryParameter(["a", "b"]), name="Glazing",
    )
    with pytest.raises(ParameterError, match=r"parameter 'Glazing'.*1 of 2 listed.*'Win Q'"):
        Problem([bad], ["Electricity:Facility"]).validate(idf, idd)


def test_multi_selector_parameters_pickle():
    p = Parameter(heating_selector(), RangeParameter(18, 22), name="H")
    q = pickle.loads(pickle.dumps(p))
    assert q.name == "H" and len(q.selector.selectors) == 3


# -- YAML / JSON ----------------------------------------------------------------------------------
def test_config_with_targets_and_object_lists(idf, idd):
    p = parameter_from_config({
        "name": "Glazing", "options": ["single", "double"],
        "class": "FenestrationSurface:Detailed", "field": "Construction Name",
        "targets": ["Win A", "Win B"],
    })
    assert isinstance(p.selector, MultiSelector) and p.kind == "categorical"
    p.selector.apply(idf, "double", idd)
    assert constructions(idf)["Win B"] == "double" and constructions(idf)["Win C"] == "single"

    q = parameter_from_config({"name": "L", "min": 1, "max": 2, "class": "Lights", "object": ["L1", "L2"],
                               "field": "Lighting Level"})
    assert isinstance(q.selector, FieldSelector) and q.selector.object_name == ["L1", "L2"]

    r = parameter_from_config({"name": "H", "min": 18, "max": 22, "targets": [
        {"class": "Schedule:Compact", "object": "Heat Z1", "field_index": 5},
        {"class": "Lights", "field": "Watts per Zone Floor Area"},
    ]})
    r.selector.apply(idf, 20.5, idd)
    assert idf.by_class("Lights")[0].fields[3] == "20.5"


# -- cache fingerprint of custom objectives -------------------------------------------------------
def _fn_a(run):
    return 1.0


def _fn_b(run):
    return 2.0


def test_function_reader_cache_token_follows_the_code():
    assert FunctionReader(_fn_a, "x").cache_token() == FunctionReader(_fn_a, "y").cache_token()
    assert FunctionReader(_fn_a, "x").cache_token() != FunctionReader(_fn_b, "x").cache_token()
    assert len(FunctionReader(_fn_a, "x").cache_token()) == 16


def test_function_fingerprint_without_source_and_with_defaults():
    ns1, ns2 = {}, {}
    exec("def f(run, k=1):\n    return k * 2", ns1)  # no source file: falls back to bytecode
    exec("def f(run, k=1):\n    return k * 3", ns2)
    assert function_fingerprint(ns1["f"]) != function_fingerprint(ns2["f"])
    ns3 = {}
    exec("def f(run, k=2):\n    return k * 2", ns3)
    assert function_fingerprint(ns1["f"]) != function_fingerprint(ns3["f"])  # default values count
    assert function_fingerprint(ns1["f"]) == function_fingerprint(ns1["f"])


def test_cache_token_of_a_partial_includes_its_arguments():
    from functools import partial

    def g(run, k):
        return k

    assert function_fingerprint(partial(g, k=1)) != function_fingerprint(partial(g, k=2))


def test_meters_and_variables_do_not_change_the_cache_key():
    assert MeterReader("Electricity:Facility").cache_token() == ""
    assert VariableReader("*", "Zone Mean Air Temperature").cache_token() == ""
