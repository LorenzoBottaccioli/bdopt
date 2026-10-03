import numpy as np
import pandas as pd
import pytest

from bdopt import (
    CategoryParameter,
    GenericSelector,
    IntegerParameter,
    Parameter,
    ParameterError,
    Problem,
    RangeParameter,
)
from bdopt.eplus.idd import IDD
from conftest import analytic_problem

MINI_IDD = """!IDD_Version 9.6.0
Lights,
  A1 , \\field Name
  A2 , \\field Zone or ZoneList Name
  N1 , \\field Lighting Level
  N2 , \\field Watts per Zone Floor Area
"""


def test_decode_encode_roundtrip_mixed():
    pr, _ = analytic_problem(3, mixed=True)
    U = np.random.default_rng(0).random((200, pr.n_var))
    df = pr.decode(U)
    assert set(df["g"]) == {"a", "b", "c"} and set(df["n"]) == {1, 2, 3, 4}
    U2 = pr.encode(df)
    pd.testing.assert_frame_equal(pr.decode(U2), df)  # decode(encode(x)) == x
    np.testing.assert_allclose(pr.snap(U), U2)


def test_range_descriptor():
    r = RangeParameter(2, 6)
    assert r.min == 2 and r.max == 6 and r.min_val == 2
    np.testing.assert_allclose(r.decode(np.array([0, 0.5, 1])), [2, 4, 6])
    with pytest.raises(ParameterError):
        RangeParameter(1, 1)


def test_integer_and_category_bins():
    i = IntegerParameter(1, 4)
    assert list(i.decode(np.array([0, 0.24, 0.26, 0.99, 1.0]))) == [1, 1, 2, 4, 4]
    c = CategoryParameter(["x", "y"])
    assert list(c.decode(np.array([0.1, 0.9]))) == ["x", "y"]
    with pytest.raises(ParameterError):
        c.encode(["z"])
    with pytest.raises(ParameterError):
        CategoryParameter(["only"])


def test_duplicate_names_rejected():
    p = Parameter(GenericSelector(lambda i, v: None), RangeParameter(0, 1), name="a")
    with pytest.raises(ParameterError, match="duplicate"):
        Problem([p, p], ["f"])


def test_value_descriptors_typo_accepted_and_shortcuts():
    p = Parameter(GenericSelector(lambda i, v: None), value_descriptors=RangeParameter(0, 1), name="a")
    assert isinstance(p.value_descriptor, RangeParameter)
    assert isinstance(Parameter(GenericSelector(lambda i, v: None), (0, 2), "b").value_descriptor, RangeParameter)
    assert isinstance(Parameter(GenericSelector(lambda i, v: None), ["u", "v"], "c").value_descriptor, CategoryParameter)


def test_constraints():
    pr, _ = analytic_problem(2)
    pr2 = Problem(pr.parameters, ["f1"], constraints=[lambda r: r["p0"] + r["p1"] - 1.0])
    df = pr2.decode(np.array([[0.2, 0.3], [0.9, 0.9]]))
    assert list(pr2.feasible(df)) == [True, False]


def test_selector_resolution_with_aliases():
    from bdopt import FieldSelector
    from bdopt.idf import IDF

    idd = IDD.from_text(MINI_IDD)
    idf = IDF.from_text("Lights, L1, Z1, 100, ;")
    FieldSelector("Lights", "*", "Watts per Floor Area").apply(idf, 7.5, idd)  # new name -> old IDD
    assert idf.by_class("Lights")[0].fields[3] == "7.5"
    with pytest.raises(KeyError, match="Did you mean"):
        FieldSelector("Lights", "*", "Lightning Level").apply(idf, 1, idd)
    with pytest.raises(ParameterError, match="matches no object"):
        FieldSelector("Lights", "nope", "Lighting Level").validate(idf, idd)


def test_function_evaluator_keeps_the_index():
    pr, ev = analytic_problem(2)
    df = pr.decode(np.random.default_rng(0).random((3, 2)))
    df.index = [7, 8, 9]
    assert list(ev.evaluate(df).index) == [7, 8, 9]
    assert list(ev.evaluate(df, keep_input=True).index) == [7, 8, 9]
