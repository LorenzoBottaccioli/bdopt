# Migrating from BESOS

Fastest path: keep the notebook and add one line before the BESOS imports.

```python
import bdopt.compat
bdopt.compat.use_as_besos()        # `from besos import ...` now resolves to bdopt
```

This covers what the platform's notebooks use: `eppy_funcs.get_building`, `FieldSelector`, `RangeParameter`,
`CategoryParameter`, `Parameter` (including the `value_descriptors=` spelling found in the notebooks),
`EPProblem`, `sampling.dist_sampler(sampling.lhs, ...)`, `EvaluatorEP(...).df_apply(df, keep_input=True,
processes=...)`, `EvaluatorGeneric`, `optimizer.NSGAII`, `expand_plist`. Other BESOS features raise
`NotImplementedError` with a pointer (`wwr`, PyEHub, geometry editing ...). The BESOS cells of the platform's
Test 1 notebook run unchanged (`tests/test_compat.py`).

For new code use the native API:

| BESOS | bdopt |
|---|---|
| `ef.get_building("x.idf")` | `load_idf("x.idf")` (any version, no IDD needed) |
| `EPProblem(params, objectives)` | `Problem(params, objectives)` |
| `sampling.dist_sampler(sampling.lhs, problem, 60)` | `sampling.lhs(problem, 60, seed=0)` |
| `EvaluatorEP(problem, building, epw_file=...)` | `EnergyPlusEvaluator(problem, building, epw, processes=4)` |
| `evaluator.df_apply(df, keep_input=True)` | `evaluator.evaluate(df, keep_input=True)` (`df_apply` also works) |
| scalers + `domain_lower_bound` + `target_f` + harlow | `AdaptiveSampler(problem, evaluator).run(...)` |
| `GridSearchCV(GaussianProcessRegressor...)` | `GPSurrogate().fit(result.data, problem)` |
| Keras `Sequential` | `TorchMLP(hidden=(32, 64, 32))` |
| Platypus `NSGAII` built by hand from `value_descriptor.min/.max` | `nsga2(problem, surrogate)` |
| `min(dist)` on raw objectives | `front.closest_to_ideal()` (min-max scaled; `normalize=False` for the old behaviour) |

## Differences that matter

* **Objectives are in Joules**, like BESOS (`MeterReader(..., unit="kWh")` converts). Meters that EnergyPlus does
  not report (no consumption, e.g. no district cooling) give 0 with a warning; `missing="nan"` or `"error"` change that.
* **Duplicate parameter names are an error.** The Test 6 notebook created one `Windows types` variable per window,
  nine variables with the same name. In bdopt use one parameter applied to all windows:
  `FieldSelector("FenestrationSurface:Detailed", "*", "Construction Name", where={"Surface Type": "Window"})`.
* **Categorical parameters work everywhere**; no `if glazing_value < 1.5` code, no `replace(win, i+1)`.
* **No data leakage:** the notebooks called `scaler.fit_transform` on the test inputs; bdopt standardises
  outputs inside the surrogate using the training data only.
* Failed simulations give NaN rows and a kept run directory (`outputdir/failed/`), not an aborted study.
* Class and field names are resolved against the IDD of the EnergyPlus version in use, with typo hints
  (`Did you mean: Watts per Floor Area?`) at construction time instead of an error halfway through a study.

## Notebooks

`tools/besos_to_bdopt_notebook.py` converts a BESOS/harlow surrogate notebook: it keeps the design-space cells
(model, selectors, ranges, objectives) and regenerates the rest with the bdopt flow. The seven platform examples
were converted this way (`Surrogate modeling examples/Test N/bdopt_Example_N.ipynb` in the platform repository).
