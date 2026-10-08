"""A 15-parameter EnergyPlus problem on the platform's Office model (envelope, glazing, internal gains, shading).

    python examples/office_15d.py --test          # check that every selector matches the model
    python examples/office_15d.py --run 100       # adaptive sampling with 100 training simulations

The model and weather file are the ones in ``examples/models`` (EnergyPlus 9.6). The objectives are the three annual
facility meters of the platform's notebooks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bdopt import EnergyPlusEvaluator, FieldSelector, Parameter, Problem, RangeParameter

MODELS = Path(__file__).parent / "models"
OBJECTIVES = ["Electricity:Facility", "DistrictHeating:Facility", "DistrictCooling:Facility"]


def build() -> tuple[Problem, Path, Path]:
    """The problem, the IDF and the weather file."""
    wall, roof = "_InsulationEX_.01", "_InsulationEX_Roof_.01"
    parameters = [
        Parameter(FieldSelector("Material", wall, "Thickness"), RangeParameter(0.05, 0.5), name="Wall insulation [m]"),
        Parameter(FieldSelector("Material", roof, "Thickness"), RangeParameter(0.05, 0.8), name="Roof insulation [m]"),
        Parameter(FieldSelector("Material", wall, "Conductivity"), RangeParameter(0.02, 0.06), name="Wall insulation k"),
        Parameter(FieldSelector("Material", roof, "Conductivity"), RangeParameter(0.02, 0.06), name="Roof insulation k"),
        Parameter(FieldSelector("Material", "Brickwork Outer_.1", "Solar Absorptance"), RangeParameter(0.3, 0.9),
                  name="Wall absorptance"),
        Parameter(FieldSelector("Material", "Asphalt 1_.01", "Solar Absorptance"), RangeParameter(0.3, 0.95),
                  name="Roof absorptance"),
        Parameter(FieldSelector("Lights", "*", "Watts per Zone Floor Area"), RangeParameter(5, 20), name="Lights [W/m2]"),
        Parameter(FieldSelector("ZoneVentilation:DesignFlowRate", "*", "Air Changes per Hour"), RangeParameter(0, 6),
                  name="Ventilation ACH"),
        Parameter(FieldSelector("WindowMaterial:Gas", "1002", "Thickness"), RangeParameter(0.006, 0.02),
                  name="Gas gap [m]"),
        Parameter(FieldSelector("WindowMaterial:Glazing", "3", "Conductivity"), RangeParameter(0.5, 1.5),
                  name="Glass conductivity"),
        Parameter(FieldSelector("WindowShadingControl", "*", "Setpoint"), RangeParameter(18, 30),
                  name="Shading temperature set point"),
    ]
    for name, label in (("11001", "SW"), ("21001", "SE"), ("31001", "NW"), ("41001", "NE")):
        parameters.append(
            Parameter(FieldSelector("WindowShadingControl", name, "Setpoint 2"), RangeParameter(80, 300),
                      name=f"Shading radiation set point {label}")
        )
    return Problem(parameters, OBJECTIVES), MODELS / "Office_IDF.idf", MODELS / "Larnaca_Larnax_Airp_-hour.epw"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--method", default="flola_voronoi")
    ap.add_argument("--out", default="office_15d_runs")
    args = ap.parse_args()
    problem, idf, epw = build()
    print(problem)
    if args.test:
        from bdopt import load_idf
        from bdopt.eplus import EnergyPlus

        model = load_idf(idf)
        ep = EnergyPlus.find("9.6")
        problem.validate(model, ep.idd)
        print(f"all {problem.n_var} selectors match the model")
    if args.run:
        from bdopt.adaptive import AdaptiveSampler

        ev = EnergyPlusEvaluator(problem, idf, epw, out_dir=args.out, processes=4)
        n0 = 3 * problem.n_var
        r = AdaptiveSampler(problem, ev, args.method, seed=0).run(
            n_initial=n0, n_test=10, max_iter=(args.run - n0) // 5, batch=5, tol=0)
        print(r)


if __name__ == "__main__":
    main()
