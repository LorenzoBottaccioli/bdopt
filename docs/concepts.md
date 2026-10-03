# Concepts

A design study answers one question: *which combination of design choices gives the best energy
performance?* Running EnergyPlus for every combination is too slow, so bdopt works in stages.

```{image} _static/img/workflow.svg
:alt: bdopt workflow: model and weather, problem, evaluator, sampler, surrogate, optimizer, Pareto front
:width: 100%
```

## The pieces

**Model and weather.** An EnergyPlus input file (`.idf`) and a weather file (`.epw`). The model is read with
{func}`bdopt.load_idf` into an {class}`~bdopt.IDF`, a plain description of the objects in the file. No
EnergyPlus installation is needed to read or edit it.

**Problem.** A {class}`~bdopt.Problem` lists the *parameters* (what you may change) and the *objectives*
(what you want to be small or large).

* A {class}`~bdopt.Parameter` couples a **selector**, which says where in the model a value is written
  (`FieldSelector("Material", "Insulation", "Thickness")`), with a **value descriptor**, which says which values
  are allowed ({class}`~bdopt.RangeParameter`, {class}`~bdopt.IntegerParameter`,
  {class}`~bdopt.CategoryParameter`).
* An objective is usually a meter such as `"Electricity:Facility"`; see {doc}`guide/objectives` for units,
  aggregation, variables and custom functions.

**Evaluator.** Turns a table of designs into a table of objective values.
{class}`~bdopt.EnergyPlusEvaluator` edits the model for each row, runs EnergyPlus (in parallel), and reads the
results; finished simulations are cached on disk. {class}`~bdopt.FunctionEvaluator` does the same with a Python
function, which is how a surrogate or a test function plugs in.

**Sampler.** Chooses which designs to simulate. A fixed design ({mod}`bdopt.sampling`) picks all points up
front; an {class}`~bdopt.adaptive.AdaptiveSampler` picks a few, builds a surrogate, measures its error on held-out
simulations and adds new points where the surrogate needs them, until the error is small enough.

**Surrogate.** A fast statistical model of the simulator trained on the simulated designs
({class}`~bdopt.surrogate.GPSurrogate`, {class}`~bdopt.surrogate.MLPSurrogate`,
{class}`~bdopt.surrogate.TorchMLP`). Predicting a design takes microseconds instead of minutes.

**Optimizer.** NSGA-II or NSGA-III ({func}`~bdopt.optimize.nsga2`, {func}`~bdopt.optimize.nsga3`) searches the
surrogate for the non-dominated designs (the Pareto front); {meth}`~bdopt.optimize.ParetoFront.closest_to_ideal`
selects one. The selected design should then be checked with a real simulation.

## Everything lives in the unit cube

Internally every design is a point in $[0, 1]^d$, one coordinate per parameter.

* {meth}`Problem.decode <bdopt.Problem.decode>` turns unit-cube points into a table of real values (a range
  parameter is scaled, an integer or category picks the bin the coordinate falls in);
* {meth}`Problem.encode <bdopt.Problem.encode>` does the reverse;
* {meth}`Problem.snap <bdopt.Problem.snap>` moves discrete coordinates to the middle of their bin.

Samplers, surrogates and optimizers all work on these points, so **bounds, parameter order and scaling are never
typed by hand**, and categorical parameters (a glazing type, a construction set) need no special code anywhere.

## Tables in, tables out

Designs are {class}`pandas.DataFrame` objects with one column per parameter, named as in the problem; results
have one column per objective. Everything is in real units; a categorical column holds the option itself
(`"double_glazing"`).

## Reproducibility and failures

* Everything that involves randomness takes a `seed`; with the evaluator's cache an interrupted study can simply
  be run again and repeats no simulation.
* A simulation that fails gives `NaN` for its design and a warning (or an exception with
  `on_error="raise"`); its run directory is kept in `failed/` with EnergyPlus' `.err` file. Adaptive sampling skips
  failed points and never proposes them again.
* A selector that matches nothing, or a misspelt class or field, is reported when the evaluator is created,
  with suggestions, not halfway through a study.
