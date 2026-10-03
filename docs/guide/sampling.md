# Fixed designs

{mod}`bdopt.sampling` draws points in the unit cube and decodes them to a table of designs. Every function
returns a {class}`pandas.DataFrame` ready for {meth}`~bdopt.EnergyPlusEvaluator.evaluate`, respects discrete
parameters (an integer or category column only contains valid values) and constraints (infeasible points are
rejected and redrawn).

```python
from bdopt import sampling

designs = sampling.lhs(problem, 60, seed=0)          # Latin hypercube
designs = sampling.sobol(problem, 64, seed=0)        # scrambled Sobol' sequence (a power of 2 is best)
designs = sampling.random(problem, 60, seed=0)       # independent uniform
grid = sampling.factorial(problem, levels=3)         # full grid; categories use all their options
```

| Function | Use it when |
|---|---|
| `lhs` | default for a one-shot design; stratified in every parameter. Uses scipy's `random-cd` optimisation |
| `sobol` | you may want to add points later (a Sobol' sequence keeps its balance when extended) |
| `random` | a baseline |
| `factorial` | few parameters and you want every combination (cost grows as `levels ** n_parameters`) |

`lhs_unit(problem, n, seed)` returns the points in the unit cube instead of a table.

```python
designs = sampling.lhs(problem, 60, seed=0)
results = evaluator.evaluate(designs, keep_input=True)        # parameters + objectives
results.to_pickle("simulated_designs.pkl")
```

A fixed design needs you to guess how many simulations are enough. {doc}`adaptive-sampling` decides that for
you.
