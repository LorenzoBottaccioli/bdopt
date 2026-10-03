"""The seven surrogate-modelling tests of the Smart Building Educational Platform, expressed with bdopt.

Each test has the same six design parameters (wall and roof insulation thickness, lighting power, ventilation
air changes, and the two shading set points) and the same three objectives; tests differ in building,
weather file and ranges, and test 6 adds the glazing type of the windows as a categorical parameter.

    python examples/platform_tests.py ROOT --test 1 --check      # compare with the platform's stored results
    python examples/platform_tests.py ROOT --test all --surrogates

``ROOT`` is the platform's folder ``Surrogate modeling examples``. Needs an EnergyPlus 9.6 installation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bdopt import (
    CategoryParameter,
    EnergyPlusEvaluator,
    FieldSelector,
    Parameter,
    Problem,
    RangeParameter,
)

OBJECTIVES = ["Electricity:Facility", "DistrictHeating:Facility", "DistrictCooling:Facility"]

#: test -> (idf, epw, wall material, roof material, wall range, roof range, ventilation range, lights range,
#:          radiation set point range, glazing types)
TESTS = {
    1: ("Office_IDF.idf", "Larnaca_Larnax_Airp_-hour.epw", "_InsulationEX_.01", "_InsulationEX_Roof_.01",
        (0.1, 0.5), (0.1, 0.8), (0, 6), (5, 20), (80, 300), None),
    2: ("Office_IDF_2_win.idf", "PARIS_FR-hour.epw", "_InsulationEX_.01", "_InsulationEX_Roof_.01",
        (0.1, 0.5), (0.1, 0.8), (0, 6), (5, 20), (80, 300), None),
    3: ("Res_flat1.idf", "Torino_IT-hour.epw", "_Insulation_.0001", "MW Glass Wool (rolls)_.0001",
        (0.1, 0.8), (0.1, 1.0), (0, 8), (8, 12), (80, 300), None),
    4: ("Res-Flat2_IDF.idf", "Larnaca_Larnax_Airp_-hour.epw", "_Insulation_.0001", "MW Glass Wool (rolls)_.0001",
        (0.1, 1.0), (0.1, 1.0), (0, 6), (8, 20), (80, 300), None),
    5: ("res-multifloor_IDF.idf", "PARIS_FR-hour.epw", "_Insulation_.0001", "MW Glass Wool (rolls)_.0001",
        (0.1, 0.5), (0.1, 0.8), (0, 8), (5, 20), (100, 300), None),
    6: ("lab_building.idf", "Torino_IT-hour.epw", "EPS Expanded Polystyrene (Heavyweight)_.1",
        "MW Glass Wool (rolls)_.1445", (0.1, 0.5), (0.1, 0.8), (0, 6), (5, 20), (30, 300),
        ["single_glazing", "double_glazing", "triple_glazing"]),
    7: ("office.idf", "Larnaca_Larnax_Airp_-hour.epw", "_InsulationEX_.01", "_InsulationEX_Roof_.01",
        (0.1, 0.5), (0.1, 0.8), (0, 6), (5, 20), (80, 300), None),
}


def build(test: int, root: str | Path) -> tuple[Problem, Path, Path]:
    """Problem, model and weather file of platform test ``test`` (folder ``ROOT/Test <n>``)."""
    idf, epw, wall, roof, wr, rr, vr, lr, sr, glazing = TESTS[test]
    folder = Path(root) / f"Test {test}"
    parameters = [
        Parameter(FieldSelector("Material", wall, "Thickness"), RangeParameter(*wr), name="Wall Insulation"),
        Parameter(FieldSelector("Material", roof, "Thickness"), RangeParameter(*rr), name="Roof Insulation"),
        Parameter(FieldSelector("Lights", "*", "Watts per Zone Floor Area"), RangeParameter(*lr),
                  name="Lights Watts/Area"),
        Parameter(FieldSelector("ZoneVentilation:DesignFlowRate", "*", "Air Changes per Hour"),
                  RangeParameter(*vr), name="Ventilation ACH"),
        Parameter(FieldSelector("WindowShadingControl", "*", "Setpoint"), RangeParameter(18, 30),
                  name="Temp Setpoint shading"),
        Parameter(FieldSelector("WindowShadingControl", "*", "Setpoint 2"), RangeParameter(*sr),
                  name="Rad Setpoint shading"),
    ]
    if glazing:
        windows = FieldSelector("FenestrationSurface:Detailed", "*", "Construction Name",
                                where={"Surface Type": "Window"})
        parameters.insert(2, Parameter(windows, CategoryParameter(glazing), name="Windows types"))
    return Problem(parameters, OBJECTIVES), folder / idf, folder / epw


def stored_tables(test: int, root: str | Path) -> dict[str, pd.DataFrame]:
    """The platform's own EnergyPlus results (made with BESOS) for this test, by file."""
    out = {}
    for f in sorted((Path(root) / f"Test {test}").glob("simulation_*_points_*.pkl")) + sorted(
        (Path(root) / f"Test {test}").glob("simulation_samples_*.pkl")
    ):
        out[f.name] = pd.read_pickle(f)
    return out


def stored_to_bdopt(df: pd.DataFrame, problem: Problem) -> pd.DataFrame:
    """Platform tables use 1, 2, 3 for the glazing type (test 6); bdopt uses the construction names."""
    df = df.copy()
    if "Windows types" in df:
        names = problem.parameters[2].value_descriptor.options if hasattr(
            problem.parameters[2].value_descriptor, "options") else None
        if names and pd.api.types.is_numeric_dtype(df["Windows types"]):
            df["Windows types"] = [names[int(round(v)) - 1] for v in df["Windows types"]]
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--test", default="all")
    ap.add_argument("--check", action="store_true", help="re-simulate the platform's test points and compare")
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--out", default="platform_runs")
    args = ap.parse_args()
    tests = list(TESTS) if args.test == "all" else [int(args.test)]
    for t in tests:
        problem, idf, epw = build(t, args.root)
        print(f"Test {t}: {idf.name} / {epw.name}, {problem.n_var} parameters")
        if args.check:
            stored = [v for k, v in stored_tables(t, args.root).items() if "test_points" in k][0]
            stored = stored_to_bdopt(stored, problem)
            ev = EnergyPlusEvaluator(problem, idf, epw, out_dir=Path(args.out) / f"test{t}", processes=args.processes)
            ours = ev.evaluate(stored[problem.parameter_names])
            rel = ((ours.to_numpy() - stored[OBJECTIVES].to_numpy()) / stored[OBJECTIVES].to_numpy()).__abs__()
            print(f"  max relative difference to the platform's results: {rel.max():.2e}")


if __name__ == "__main__":
    main()
