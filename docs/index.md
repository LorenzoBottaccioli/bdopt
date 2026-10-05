# bdopt

**Building design optimization with EnergyPlus.**
Parametric studies, adaptive sampling, surrogate models and multi-objective optimization,
as a plain Python library (no Docker needed).

```{image} _static/img/adaptive_vs_lhs.png
:alt: Adaptive sampling concentrates the simulations where the response is nonlinear
:width: 100%
```

*Same number of simulations (50). Left: a Latin hypercube. Right: adaptive sampling, which puts the
later simulations (green, yellow) on the narrow feature of the response. Produced by
`docs/make_figures.py`.*

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: doc
Install the package and an EnergyPlus version.
:::

:::{grid-item-card} Quick start
:link: quickstart
:link-type: doc
A few lines from a model to a Pareto front.
:::

:::{grid-item-card} Concepts
:link: concepts
:link-type: doc
How a design study is put together, and why everything lives in the unit cube.
:::

:::{grid-item-card} User guide
:link: guide/index
:link-type: doc
Parameters, objectives, evaluation, sampling, surrogates, optimization, studies from YAML.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
Executed notebooks: adaptive sampling, an office building on EnergyPlus, mixed variables.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc
Every public class and function.
:::
::::

## What it does

```{list-table}
:header-rows: 1
:widths: 30 70

* - Step
  - What bdopt provides
* - Describe the design space
  - {class}`~bdopt.Parameter` with a {class}`~bdopt.FieldSelector` (where in the IDF) and a
    {class}`~bdopt.RangeParameter`, {class}`~bdopt.IntegerParameter` or {class}`~bdopt.CategoryParameter`.
* - Run EnergyPlus
  - {class}`~bdopt.EnergyPlusEvaluator`: parallel runs, on-disk cache, failure handling, the EnergyPlus version of your model.
* - Choose the simulations
  - {mod}`bdopt.sampling` (LHS, Sobol, random, factorial) and {class}`~bdopt.adaptive.AdaptiveSampler`.
* - Build a cheap model
  - fifteen models, from {class}`~bdopt.surrogate.GPSurrogate` and
    {class}`~bdopt.surrogate.PolynomialSurrogate` to {class}`~bdopt.surrogate.DeepKernelGPSurrogate` and
    {class}`~bdopt.surrogate.TorchMLP`, or let {class}`~bdopt.surrogate.AutoSurrogate` choose.
* - Optimize
  - {func}`~bdopt.optimize.nsga2`, {func}`~bdopt.optimize.nsga3`, with categorical variables and constraints.
* - Automate
  - {class}`~bdopt.study.DesignStudy` and the `bdopt run study.yaml` command.
```

```{toctree}
:hidden:
:maxdepth: 2

installation
quickstart
concepts
guide/index
tutorials/index
api/index
benchmarks
literature
faq
contributing
changelog
```
