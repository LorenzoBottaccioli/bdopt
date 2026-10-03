# Optimization

{func}`~bdopt.optimize.nsga2` and {func}`~bdopt.optimize.nsga3` search for the non-dominated designs (the Pareto
front) with [pymoo](https://pymoo.org/).

```python
from bdopt.optimize import nsga2

front = nsga2(problem, surrogate, pop_size=100, n_gen=100, seed=0)
front.df                      # parameters + objectives of every non-dominated design
best = front.closest_to_ideal()
```

`model` can be any object with `evaluate_unit(U)`: a surrogate (the usual case: thousands of evaluations take
seconds) or an {class}`~bdopt.EnergyPlusEvaluator` (each generation then runs `pop_size` simulations, in
parallel and cached). A problem with one objective uses a plain genetic algorithm.

| Function | When |
|---|---|
| `nsga2(problem, model, pop_size=100, n_gen=100, seed=0)` | two or three objectives |
| `nsga3(problem, model, pop_size=100, n_gen=100, seed=0, n_partitions=12)` | many objectives; `pop_size` is raised to the number of reference directions |

## Mixed variables, directions, constraints

* **Categorical and integer parameters** need no special handling: the search runs in the unit cube and discrete
  coordinates are snapped to valid values before every evaluation and in the offspring, so the front only
  contains valid designs (a glazing type is a glazing type).
* **Direction:** an objective with `direction="max"` is maximised and reported with its true sign.
* **Constraints** given to the {class}`~bdopt.Problem` are passed to pymoo; if no feasible design is found a
  `RuntimeError` is raised.
* A `NaN` objective (a failed simulation when optimizing on EnergyPlus) is penalised.
* Results are reproducible for a fixed `seed`.

## Choosing one design

{class}`~bdopt.optimize.ParetoFront` is the set of designs: `.df`, `.parameters`, `.objectives`.

```python
best = front.closest_to_ideal()                          # min-max scaled distance to the ideal point
best = front.closest_to_ideal(weights=[2, 1, 1])         # care twice as much about the first objective
best = front.closest_to_ideal(normalize=False)           # raw distance: dominated by the largest numbers
```

The ideal point holds the best value of every objective over the front. With `normalize=True` (default) each
objective is first scaled to $[0, 1]$ over the front, so an objective in joules does not drown one in degrees.
This is one reasonable rule, not the only one: look at the front ({func}`bdopt.viz.pareto`,
{func}`bdopt.viz.parallel_coordinates`) and choose with the trade-offs in view.

## Check the answer

The surrogate is approximate, and optimizers like to find the places where it is wrong. Always simulate the
selected design:

```python
x = best[problem.parameter_names].to_frame().T.reset_index(drop=True)
check = evaluator.evaluate(x)                 # EnergyPlus
guess = surrogate.predict(x)                  # what the surrogate said
```

If they disagree, add the design to the training data (`initial_data` of the adaptive sampler, or refit) and
optimize again. In a configuration file, `optimization: {verify: 5}` does this check for five Pareto designs.

## Your own algorithm

Anything with `run(problem, model) -> ParetoFront` satisfies the {class}`~bdopt.optimize.Optimizer` protocol;
the rest of the library does not care which optimizer produced a front.
