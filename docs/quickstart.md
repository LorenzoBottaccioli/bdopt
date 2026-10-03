# Quick start

## Try it in a second (no EnergyPlus)

An {class}`~bdopt.FunctionEvaluator` stands in for the simulator, so the whole flow runs anywhere.

<!-- runnable -->
```python
import numpy as np
from bdopt import FunctionEvaluator, NullSelector, Parameter, Problem, RangeParameter
from bdopt.adaptive import AdaptiveSampler
from bdopt.surrogate import GPSurrogate
from bdopt.optimize import nsga2

# two parameters in [0, 1]; the selector is irrelevant here because no model file is edited
params = [Parameter(NullSelector(n), RangeParameter(0, 1), name=n) for n in ("x1", "x2")]
problem = Problem(params, ["cost", "heat_loss"])

def simulator(row):                      # a stand-in for an EnergyPlus run
    x1, x2 = row["x1"], row["x2"]
    return [x1 ** 2 + x2, (1 - x1) ** 2 + 0.5 * x2]

evaluator = FunctionEvaluator(problem, simulator)

result = AdaptiveSampler(problem, evaluator, seed=0).run(
    n_initial=8, n_test=5, max_iter=10, tol=0.02, verbose=False
)
print(result)                            # AdaptiveResult(.. train + 5 test points, scaled_rmse=..)

surrogate = GPSurrogate().fit(result.data, problem)
front = nsga2(problem, surrogate, pop_size=40, n_gen=30)
print(front.closest_to_ideal())          # parameters and objectives of the selected design
```

## With EnergyPlus

Same flow; only the evaluator and the parameters change. Needs an EnergyPlus installation matching the model
(see {doc}`installation`). The model and weather file used below are in `examples/models/` of the repository.

```python
from bdopt import (EnergyPlusEvaluator, FieldSelector, Parameter, Problem, RangeParameter)
from bdopt.adaptive import AdaptiveSampler
from bdopt.surrogate import GPSurrogate
from bdopt.optimize import nsga2

parameters = [
    Parameter(FieldSelector("Material", "_InsulationEX_.01", "Thickness"), RangeParameter(0.1, 0.5), name="Wall insulation"),
    Parameter(FieldSelector("Material", "_InsulationEX_Roof_.01", "Thickness"), RangeParameter(0.1, 0.8), name="Roof insulation"),
    Parameter(FieldSelector("Lights", "*", "Watts per Zone Floor Area"), RangeParameter(5, 20), name="Lighting W/m2"),
]
problem = Problem(parameters, ["Electricity:Facility", "DistrictHeating:Facility", "DistrictCooling:Facility"])

# The EnergyPlus version is picked to match the model; results are cached under ./bdopt_runs/cache
evaluator = EnergyPlusEvaluator(problem, "Office_IDF.idf", "Larnaca_Larnax_Airp_-hour.epw", processes=4)

result = AdaptiveSampler(problem, evaluator).run(n_initial=12, n_test=4, max_iter=12, tol=0.05)
surrogate = GPSurrogate().fit(result.data, problem)
print(surrogate.score(result.test))                 # RMSE, RMSE %, CV(RMSE), NMBE, R2 per objective

front = nsga2(problem, surrogate, pop_size=100, n_gen=100)
best = front.closest_to_ideal()
print(evaluator.evaluate(best[problem.parameter_names].to_frame().T))   # check with a real simulation
```

The same study as a configuration file: {doc}`guide/studies`. A complete walk-through with figures and the
numbers it produced: {doc}`tutorials/02_office_building`.

## Where next

* {doc}`concepts`: how the pieces fit together;
* {doc}`guide/index`: one page per step;
* {doc}`tutorials/index`: executed notebooks.
