# Tutorials

Executed notebooks: the outputs you see are the real ones. The notebooks are in `docs/tutorials/` of the
repository; open them with Jupyter to run them yourself.

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 1. Adaptive sampling: the basics
:link: 01_adaptive_sampling_basics
:link-type: doc
No EnergyPlus needed. Latin hypercube against adaptive sampling on a response with a narrow feature, with the
same number of simulations.
:::

:::{grid-item-card} 2. An office building with EnergyPlus
:link: 02_office_building
:link-type: doc
A complete study on a real model: adaptive sampling, three surrogate models compared, NSGA-II, and a check of
the selected design with a real simulation.
:::

:::{grid-item-card} 3. Mixed variables and constraints
:link: 03_mixed_variables_and_constraints
:link-type: doc
No EnergyPlus needed. Categorical and integer parameters, a constraint, an objective to maximise, choosing a
design from the Pareto front.
:::

:::{grid-item-card} 4. Choosing a strategy and a surrogate
:link: 04_strategies_and_surrogates
:link-type: doc
No EnergyPlus needed. All nine adaptive sampling strategies and all the surrogate models compared on the same
response, and automatic model selection.
:::
::::

```{toctree}
:hidden:

01_adaptive_sampling_basics
02_office_building
03_mixed_variables_and_constraints
04_strategies_and_surrogates
```
