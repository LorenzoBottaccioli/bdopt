"""Build and execute the tutorial notebooks, storing their outputs.

    python docs/tutorials/build_notebooks.py            # all
    python docs/tutorials/build_notebooks.py 01 03      # only these (they need no EnergyPlus)

Tutorial 02 runs real EnergyPlus simulations (version 9.6, about 8 minutes on 4 cores): set
BDOPT_EPLUS_PATH if EnergyPlus is not in a standard location. The documentation build does not execute
notebooks, it displays the stored outputs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell as code
from nbformat.v4 import new_markdown_cell as md
from nbformat.v4 import new_notebook

HERE = Path(__file__).parent

SETUP = '''import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.precision", 4)
pd.set_option("display.width", 120)'''

# ----------------------------------------------------------------------------------------------------
NB01 = [
    md('''# 1. Adaptive sampling: the basics

This tutorial needs no EnergyPlus: a Python function plays the simulator, so every step runs in seconds and you
can see what adaptive sampling does. The response has a **narrow feature** (a bump) on top of a smooth trend, the
situation where adaptive sampling pays off.

We will

1. describe a problem with two parameters;
2. build a surrogate from a fixed Latin-hypercube design (the baseline);
3. build one with the adaptive sampler, using the same number of training simulations;
4. compare them.'''),
    code(SETUP),
    md('''## The problem

A {class}`~bdopt.Parameter` couples *where* a value goes in the model (a selector) with *which values* are
allowed. Here there is no model file, so the selector does nothing (`GenericSelector`), and a
{class}`~bdopt.FunctionEvaluator` computes the response.'''),
    code('''from bdopt import FunctionEvaluator, GenericSelector, Parameter, Problem, RangeParameter

params = [
    Parameter(GenericSelector(lambda idf, value: None), RangeParameter(0, 1), name=name)
    for name in ("x1", "x2")
]
problem = Problem(params, ["f"])

def simulator(row):
    """A smooth trend plus a narrow bump around (0.7, 0.3)."""
    bump = np.exp(-(((row["x1"] - 0.7) ** 2 + (row["x2"] - 0.3) ** 2) / 0.012))
    return [0.3 * row["x1"] + bump]

evaluator = FunctionEvaluator(problem, simulator)
problem'''),
    md('''## Baseline: a fixed Latin-hypercube design

{func}`bdopt.sampling.lhs` returns a table of designs; `evaluate` returns the objectives.'''),
    code('''from bdopt import sampling
from bdopt.surrogate import GPSurrogate

N = 50
lhs_designs = sampling.lhs(problem, N, seed=3)
lhs_data = evaluator.evaluate(lhs_designs, keep_input=True)       # parameters + objective
lhs_data.head()'''),
    md('''To compare fairly we need independent points the surrogates have never seen:'''),
    code('''reference = sampling.lhs(problem, 2000, seed=99)
truth = evaluator.evaluate(reference)["f"].to_numpy()

def error(surrogate):
    """RMSE on the reference points, divided by the standard deviation of the response."""
    prediction = surrogate.predict(reference)["f"].to_numpy()
    return float(np.sqrt(np.mean((prediction - truth) ** 2)) / truth.std())

lhs_surrogate = GPSurrogate().fit(lhs_data, problem)
print(f"Latin hypercube, {N} points: relative error {error(lhs_surrogate):.3f}")'''),
    md('''## Adaptive sampling

{class}`~bdopt.adaptive.AdaptiveSampler` starts from a small Latin hypercube (`n_initial`), holds out `n_test`
simulations to measure the surrogate's error, and then adds simulations where the surrogate needs them. Here we
ask for exactly the same number of training points as the baseline: 12 initial + 38 adaptive = 50
(`tol=0` disables early stopping).'''),
    code('''from bdopt.adaptive import AdaptiveSampler

sampler = AdaptiveSampler(problem, evaluator, method="flola_voronoi", seed=3)
result = sampler.run(n_initial=12, n_test=10, max_iter=38, tol=0, verbose=False)
print(result)
result.history[["iteration", "n_train", "n_evaluations", "scaled_rmse"]].tail()'''),
    code('''adaptive_surrogate = GPSurrogate().fit(result.data, problem)
print(f"Adaptive, {len(result.data)} points:      relative error {error(adaptive_surrogate):.3f}")'''),
    md('''## Where did the simulations go?

Grey contours show the true response. The Latin hypercube spreads points evenly; the adaptive sampler starts the
same way (dark points) and then places new simulations (green, yellow) on and around the bump, where the
response is nonlinear.'''),
    code('''gx, gy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
surface = 0.3 * gx + np.exp(-(((gx - 0.7) ** 2 + (gy - 0.3) ** 2) / 0.012))

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
for ax, points, title in ((axes[0], lhs_data, "Latin hypercube"), (axes[1], result.data, "Adaptive sampling")):
    ax.contourf(gx, gy, surface, levels=20, cmap="Greys", alpha=0.55)
    colour = np.arange(len(points)) if ax is axes[1] else "tab:blue"
    sc = ax.scatter(points["x1"], points["x2"], c=colour, cmap="viridis", s=26, edgecolor="k", linewidth=0.4)
    ax.set_title(f"{title} ({len(points)} points)", fontsize=10)
    ax.set_xlabel("x1")
axes[0].set_ylabel("x2")
fig.colorbar(sc, ax=axes, shrink=0.85, label="order of simulation")
plt.show()'''),
    md('''## Convergence

`history` has one row per iteration. `scaled_rmse` is the surrogate's error on the held-out simulations, in units
of the standard deviation of the response. With a real budget you would set `tol` (for example `0.05`) and let
the loop stop by itself, learning how many simulations were enough.'''),
    code('''from bdopt import viz

fig = viz.convergence(result.history)
plt.show()'''),
    md('''## What to take away

* Both surrogates used 50 training simulations; the numbers above are what each achieved on 2000 unseen points.
* The gain depends on the response. It is large for a localised feature like this one; for a smooth response a
  Latin hypercube is as good, and the adaptive loop is then mainly a way to *know* when you have simulated
  enough. The measured comparison is in {doc}`../guide/adaptive-sampling`.
* In a real study replace the `FunctionEvaluator` with an {class}`~bdopt.EnergyPlusEvaluator` and the
  `GenericSelector` with `FieldSelector` objects: tutorial 2 does that.'''),
]

# ----------------------------------------------------------------------------------------------------
NB02 = [
    md('''# 2. An office building with EnergyPlus

A complete study on a real model: an office building (EnergyPlus 9.6) in Larnaca. Four design parameters,
three energy objectives, adaptive sampling, three surrogate models compared on held-out simulations,
multi-objective optimization, and a check of the selected design with a real simulation.

Everything below was executed against EnergyPlus 9.6 with a deliberately small budget (32 simulations, about 8
minutes on 4 cores) so that the tutorial can be rebuilt; a real study would use more. The model and weather file
are in `examples/models/` of the repository.

:::{note}
You need an EnergyPlus **9.6** installation to run it yourself (`python tools/install_eplus.py 9.6`), or
`eplus=...` pointing at any version together with `auto_upgrade=True`.
:::'''),
    code(SETUP.replace('warnings.filterwarnings("ignore")', 'warnings.filterwarnings("ignore", module="sklearn")  # bdopt\'s own warnings stay visible') + '''
from pathlib import Path

from bdopt import (CategoryParameter, EnergyPlusEvaluator, FieldSelector, Parameter, Problem,
                   RangeParameter, load_idf, viz)

MODELS = Path("../../examples/models")        # relative to this notebook'''),
    md('''## The model

{func}`~bdopt.load_idf` reads the file without needing EnergyPlus; `version` is what the file says it was written
for.'''),
    code('''building = load_idf(MODELS / "Office_IDF.idf")
epw = MODELS / "Larnaca_Larnax_Airp_-hour.epw"
print(building, "|", len(building.by_class("Lights")), "lighting objects,",
      len(building.by_class("Zone")), "zones")'''),
    md('''## The design space

Each {class}`~bdopt.Parameter` has a selector (where in the IDF) and a range. `object_name="*"` writes the value
into every object of the class.'''),
    code('''parameters = [
    Parameter(FieldSelector("Material", "_InsulationEX_.01", "Thickness"),
              RangeParameter(0.1, 0.5), name="Wall insulation [m]"),
    Parameter(FieldSelector("Material", "_InsulationEX_Roof_.01", "Thickness"),
              RangeParameter(0.1, 0.8), name="Roof insulation [m]"),
    Parameter(FieldSelector("Lights", "*", "Watts per Zone Floor Area"),
              RangeParameter(5, 20), name="Lighting [W/m2]"),
    Parameter(FieldSelector("ZoneVentilation:DesignFlowRate", "*", "Air Changes per Hour"),
              RangeParameter(0, 6), name="Ventilation [ACH]"),
]
objectives = ["Electricity:Facility", "DistrictHeating:Facility", "DistrictCooling:Facility"]
problem = Problem(parameters, objectives)
problem'''),
    md('''## The evaluator

The EnergyPlus version is chosen to match the model; results are cached under `runs/cache`, so re-running a cell
or the whole notebook does not repeat finished simulations.'''),
    code('''evaluator = EnergyPlusEvaluator(problem, building, epw, out_dir="runs", processes=4)
print(evaluator.eplus)'''),
    md('''## Adaptive sampling

12 initial simulations, 4 held out to measure the error, then 8 iterations of 2 new simulations each.'''),
    code('''from bdopt.adaptive import AdaptiveSampler

sampler = AdaptiveSampler(problem, evaluator, method="flola_voronoi", seed=0)
result = sampler.run(n_initial=12, n_test=4, max_iter=8, batch=2, tol=0.0)
print(result)
print(evaluator.stats())'''),
    code('''# runs that EnergyPlus finished with severe errors (bdopt also raised a Python warning for each, above)
flagged = [r for r in evaluator.history if r.eplus_severe]
for r in flagged:
    print({name: round(float(value), 3) for name, value in r.parameters.items()})
    print("  ", r.warnings[0][:260])
print(len(flagged), "of", len(evaluator.history), "runs flagged")'''),
    md('''`stats()` also counts the messages EnergyPlus wrote to `eplusout.err` in every run
(`energyplus_warnings`, `energyplus_severe`). The warnings come from the model itself (an office model of this size
has a few dozen: unused constructions, monthly reports naming meters the model does not use ...); bdopt adds none.
A run that ends with **severe** errors is different: the cell above lists them. The result is kept by default
(`on_severe="warn"`); `on_severe="nan"` would treat such a design as failed, so adaptive sampling would skip it and
never propose it again. EnergyPlus' message here is about the warm-up of two zones (see the troubleshooting page).
See {doc}`../guide/evaluation`.'''),
    code('''result.history[["iteration", "n_train", "n_evaluations", "scaled_rmse", "elapsed_s"]]'''),
    md('''`scaled_rmse` is the error on the 4 held-out simulations, in units of the standard deviation of each
objective. It need not fall monotonically: with so few points a new simulation can temporarily make the model
worse before it makes it better. A real study would set `tol` and a larger budget.'''),
    code('''viz.convergence(result.history)
plt.show()'''),
    md('''## The simulated designs

Objectives are in joules (the unit EnergyPlus reports); use `MeterReader(..., unit="kWh")` for others.'''),
    code('''data = result.data
data.describe().T[["min", "mean", "max"]]'''),
    code('''viz.pairplot(data, problem, size=2.2)
plt.show()'''),
    code('''viz.correlation_heatmap(data, problem)
plt.show()'''),
    md('''## Surrogate models

Three models trained on the same simulations and scored on the same held-out simulations. Inputs and outputs
are scaled inside the models, using the training data only.'''),
    code('''from bdopt.surrogate import GPSurrogate, MLPSurrogate, TorchMLP

models = {
    "Gaussian process": GPSurrogate(),
    "MLP (scikit-learn)": MLPSurrogate(),
    "MLP (PyTorch, 3 nets)": TorchMLP(hidden=(32, 64, 32), n_ensemble=3),
}
scores = {}
for name, model in models.items():
    model.fit(data, problem)
    scores[name] = model.score(result.test)

summary = pd.DataFrame({name: s["rmse_pct"] for name, s in scores.items()})
summary.round(2)          # RMSE as % of the mean of the held-out values'''),
    md('''With only 12 training points the models are rough; the point here is the workflow. The table says which
model to trust on this data.'''),
    code('''best_name = summary.mean().idxmin()
surrogate = models[best_name]
print("selected:", best_name)
viz.parity(result.test[problem.objective_names], surrogate.predict(result.test))
plt.show()'''),
    md('''## Optimization

NSGA-II searches the surrogate (thousands of evaluations in seconds) for the designs that cannot be improved in
one objective without worsening another.'''),
    code('''from bdopt.optimize import nsga2

front = nsga2(problem, surrogate, pop_size=100, n_gen=100, seed=0)
print(len(front), "non-dominated designs")
front.df.head()'''),
    code('''best = front.closest_to_ideal()      # min-max scaled distance to the ideal point
best.to_frame("selected design")'''),
    code('''viz.pareto(front, background=data, highlight=best)
plt.show()
viz.parallel_coordinates(front.df, problem)
plt.show()'''),
    md('''## Check the selected design

The surrogate is approximate, and optimizers like to find the places where it is wrong. Simulate the selected
design with EnergyPlus and compare.'''),
    code('''x = best[problem.parameter_names].to_frame().T.reset_index(drop=True)
comparison = pd.DataFrame({"EnergyPlus": evaluator.evaluate(x).iloc[0], "surrogate": surrogate.predict(x).iloc[0]})
comparison["difference [%]"] = 100 * (comparison["surrogate"] - comparison["EnergyPlus"]) / comparison["EnergyPlus"]
comparison'''),
    md('''If the difference is larger than you can accept, add this design to the training data
(`initial_data=` of {meth}`~bdopt.adaptive.AdaptiveSampler.run`) or simply spend more simulations and optimize
again. The same study written as a configuration file is in {doc}`../guide/studies`.'''),
]

# ----------------------------------------------------------------------------------------------------
NB03 = [
    md('''# 3. Mixed variables, constraints and choosing a design

Building choices are often **categorical** (a glazing type) or **integer** (a number of shading levels), and
often **constrained** (a budget). This tutorial shows how bdopt handles them, with a Python function standing in
for the simulator so it runs in seconds. Everything shown works the same with an
{class}`~bdopt.EnergyPlusEvaluator`.

The toy problem: choose a wall insulation thickness, a glazing type and a number of shading levels to minimise
heat loss and cost and maximise daylight, with a cost cap.'''),
    code(SETUP),
    md('''## Parameters of three kinds'''),
    code('''from bdopt import (CategoryParameter, FunctionEvaluator, GenericSelector, IntegerParameter, Objective,
                   Parameter, Problem, RangeParameter)

noop = GenericSelector(lambda idf, value: None)        # no model file in this tutorial

parameters = [
    Parameter(noop, RangeParameter(0.05, 0.40), name="insulation [m]"),
    Parameter(noop, CategoryParameter(["single", "double", "triple"]), name="glazing"),
    Parameter(noop, IntegerParameter(0, 4), name="shading levels"),
]

# glazing: (U-value, daylight transmittance, cost per m2)
GLAZING = {"single": (5.7, 0.90, 40.0), "double": (2.8, 0.78, 90.0), "triple": (1.2, 0.68, 160.0)}

def simulator(row):
    u, daylight, glazing_cost = GLAZING[row["glazing"]]
    heat_loss = 100 / (1 + 12 * row["insulation [m]"]) + 9 * u - 1.5 * row["shading levels"]
    cost = 400 * row["insulation [m]"] + glazing_cost + 6 * row["shading levels"]
    daylight_score = 100 * daylight * (1 - 0.12 * row["shading levels"])
    return [heat_loss, cost, daylight_score]

objectives = [
    "heat loss",
    "cost",
    Objective("daylight", direction="max"),            # an objective to maximise
]
problem = Problem(
    parameters, objectives,
    constraints=[lambda row: GLAZING[row["glazing"]][2] + 400 * row["insulation [m]"] - 190.0],   # cost cap
)
evaluator = FunctionEvaluator(problem, simulator)
problem.names()'''),
    md('''A plain string is an objective to minimise (with an EnergyPlus evaluator it would be a meter name);
`Objective("daylight", direction="max")` names an objective to **maximise**. The constraint is a function of a
design that returns a number, feasible when it is `<= 0`.

## Sampling respects discrete values and constraints'''),
    code('''from bdopt import sampling

designs = sampling.lhs(problem, 200, seed=0)
print(designs["glazing"].value_counts().to_dict(), sorted(designs["shading levels"].unique()))
print("all feasible:", bool(problem.feasible(designs).all()))
designs.head()'''),
    md('''## A surrogate over mixed variables

Categories and integers are encoded in the unit cube, so the same surrogate classes work unchanged.'''),
    code('''from bdopt.surrogate import GPSurrogate

train = evaluator.evaluate(sampling.lhs(problem, 80, seed=1), keep_input=True)
test = evaluator.evaluate(sampling.lhs(problem, 200, seed=2), keep_input=True)
surrogate = GPSurrogate().fit(train, problem)
surrogate.score(test).round(3)'''),
    md('''## NSGA-II on the surrogate

Discrete coordinates are snapped to valid values during the search, so every design on the front has a real
glazing type and an integer number of shading levels, and the front respects the cost cap.'''),
    code('''from bdopt.optimize import nsga2

front = nsga2(problem, surrogate, pop_size=120, n_gen=80, seed=0)
print(len(front), "designs;", "glazings:", sorted(front.df["glazing"].unique()),
      "| shading levels:", sorted(front.df["shading levels"].unique()))
print("cost cap respected:", bool(problem.feasible(front.parameters).all()))
front.df.sort_values("cost").head(8)'''),
    md('''## Choosing one design

{meth}`~bdopt.optimize.ParetoFront.closest_to_ideal` picks the design nearest to the ideal point, after scaling
each objective to $[0, 1]$ over the front. `weights` say which objective matters more.'''),
    code('''equal = front.closest_to_ideal()
daylight_first = front.closest_to_ideal(weights=[1, 1, 3])
pd.DataFrame({"equal weights": equal, "daylight x3": daylight_first})'''),
    code('''from bdopt import viz

viz.pareto(front, highlight=equal)
plt.show()
viz.parallel_coordinates(front.df, problem)
plt.show()'''),
    md('''## Check against the "simulator"

The front holds the surrogate's predictions; evaluate the same designs with the simulator to see the
surrogate's error on the designs that matter.'''),
    code('''actual = evaluator.evaluate(front.parameters)
error = (front.objectives.to_numpy() - actual.to_numpy()) / actual.to_numpy().std(axis=0)
pd.Series(np.sqrt((error ** 2).mean(axis=0)), index=problem.objective_names, name="RMSE / std")'''),
    md('''## Next

* Real model: {doc}`02_office_building`.
* Whole pipeline from a file: {doc}`../guide/studies`.'''),
]

BOOKS = {"01": ("01_adaptive_sampling_basics.ipynb", NB01), "02": ("02_office_building.ipynb", NB02),
         "03": ("03_mixed_variables_and_constraints.ipynb", NB03)}


def build(key: str) -> None:
    name, cells = BOOKS[key]
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    t0 = time.time()
    NotebookClient(nb, timeout=3000, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    nbformat.write(nb, HERE / name)
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    print(f"{name}: executed in {time.time() - t0:.0f} s, {len(errors)} errors")


if __name__ == "__main__":
    for k in sys.argv[1:] or list(BOOKS):
        build(k)
