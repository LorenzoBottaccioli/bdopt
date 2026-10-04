"""bdopt quick start: the whole flow of the platform's surrogate notebook, without glue code.

    python examples/quickstart.py

Needs an EnergyPlus 9.6 installation (python tools/install_eplus.py 9.6).
"""

from pathlib import Path

from bdopt import (
    EnergyPlusEvaluator,
    FieldSelector,
    Parameter,
    Problem,
    RangeParameter,
)
from bdopt.adaptive import AdaptiveSampler
from bdopt.optimize import nsga2
from bdopt.surrogate import GPSurrogate

models = Path(__file__).parent / "models"

parameters = [
    Parameter(FieldSelector("Material", "_InsulationEX_.01", "Thickness"), RangeParameter(0.1, 0.5), name="Wall Insulation"),
    Parameter(FieldSelector("Material", "_InsulationEX_Roof_.01", "Thickness"), RangeParameter(0.1, 0.8), name="Roof Insulation"),
    Parameter(FieldSelector("Lights", "*", "Watts per Zone Floor Area"), RangeParameter(5, 20), name="Lights Watts/Area"),
]
problem = Problem(parameters, ["Electricity:Facility", "DistrictHeating:Facility", "DistrictCooling:Facility"])

# The engine is found automatically (matching the model's version); no Docker, no fixed path.
evaluator = EnergyPlusEvaluator(
    problem, models / "Office_IDF.idf", models / "Larnaca_Larnax_Airp_-hour.epw",
    out_dir="runs", processes=4,
)

# Adaptive sampling: bounds, scaling, categorical handling and the target function are built for you.
result = AdaptiveSampler(problem, evaluator, method="flola_voronoi").run(
    n_initial=12, n_test=4, max_iter=12, tol=0.05
)
print(result)

surrogate = GPSurrogate().fit(result.data, problem)
print(surrogate.score(result.test))                      # RMSE, RMSE %, CV(RMSE), NMBE, R2 per objective

front = nsga2(problem, surrogate, pop_size=100, n_gen=100)
best = front.closest_to_ideal()
print(best)

# check the selected design with a real simulation
print(evaluator.evaluate(best[problem.parameter_names].to_frame().T))
