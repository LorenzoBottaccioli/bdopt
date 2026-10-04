# Studies from a configuration file

A {class}`~bdopt.study.DesignStudy` runs the whole pipeline (sampling, surrogate, optimization) from plain data:
a YAML or JSON file, or a Python dictionary. That makes a study reproducible, reviewable, and easy to generate
from a web form or a notebook.

## From the command line

```bash
bdopt run study.yaml                 # relative paths in the file are relative to the file
bdopt run study.yaml --out results   # override output_dir
bdopt run study.yaml --no-optimize   # stop after sampling and surrogate training
bdopt eplus list                     # EnergyPlus installations bdopt can see
```

A real run (EnergyPlus 9.6, two parameters, two objectives, a five-zone example model):

```yaml
output_dir: out
model: {idf: 5ZoneDDCycOnAny.idf, epw: weather.epw}
parameters:
  - {name: Lighting level, class: Lights, field: Lighting Level, min: 500, max: 3000}
  - {name: Equipment level, class: ElectricEquipment, field: Design Level, min: 200, max: 2000}
objectives:
  - Electricity:Facility
  - DistrictHeating:Facility
evaluation: {processes: 2}
sampling: {method: adaptive, n_initial: 6, n_test: 3, max_iter: 4, tol: 0.02, seed: 0}
surrogate: {type: gp}
optimization: {algorithm: nsga2, pop_size: 30, n_gen: 20, verify: 3}
```

```text
$ bdopt run study.yaml
[bdopt] iter   0  train=   6  evals=   9  scaled_rmse=0.0214
[bdopt] iter   1  train=   7  evals=  10  scaled_rmse=0.0202
[bdopt] iter   2  train=   8  evals=  11  scaled_rmse=0.0199
results written to /tmp/cli_demo/out
selected design:
Lighting level              6.383675e+02
Equipment level             1.935305e+03
Electricity:Facility        2.616648e+09
DistrictHeating:Facility    3.575972e+09
```

The loop stopped after two adaptive iterations because the error (0.0199) fell below `tol: 0.02`. The output
folder contains:

| File | Content |
|---|---|
| `train.csv`, `test.csv` | simulated designs: parameters and objectives |
| `history.csv` | per iteration: training size, simulations so far, `scaled_rmse`, per-objective errors |
| `metrics.csv` | surrogate error metrics on the test set (or cross-validation if there is none) |
| `pareto.csv`, `best.csv` | the non-dominated designs and the selected one |
| `verification.csv` | `verify` Pareto designs re-simulated with EnergyPlus, next to the surrogate's prediction |
| `surrogate.pkl` | the fitted surrogate ({meth}`Surrogate.load <bdopt.surrogate.Surrogate.load>`) |
| `stats.json` | simulations run, cache hits, failures, simulation time, EnergyPlus version |
| `checkpoint/` | `train.csv`, `test.csv`, `history.csv` written after every adaptive iteration |
| `runs/` | the evaluator's working directory: `cache/` and, if any simulation failed, `failed/` |

## From Python

```python
from bdopt.study import DesignStudy

study = DesignStudy.from_config("study.yaml")      # or a dict
results = study.run()                              # StudyResults
results.surrogate.predict(new_designs)
results.pareto.df
results.save("out/")
```

`study.problem` and `study.evaluator` give you the objects the file describes, in case you want to continue by
hand.

## Reference

Every section except `model`, `parameters` and `objectives` is optional.

### `output_dir`

Folder for the results (default `bdopt_study`).

### `eplus`

| Key | Meaning |
|---|---|
| `version` | EnergyPlus version to use, e.g. `"9.6"` (default: the one matching the model) |
| `path` | an installation folder |
| `search_paths` | extra folders to search for installations |
| `auto_upgrade` | upgrade an older model to the newest installed version (default `false`) |

### `model`

`idf` and `epw`: the model and the weather file.

### `parameters`

A list; each entry:

| Key | Meaning |
|---|---|
| `name` | unique name (default: derived from the selector) |
| `type` | `range` (default if `min`/`max`), `integer`, `category` (default if `options`) |
| `class`, `object`, `field` | what to change; `object` defaults to `"*"` (all objects); `field_index` can replace `field` |
| `where` | `{field name: value}` filter on the objects |
| `min`, `max` | bounds of a range or integer parameter |
| `options` | list of values of a category parameter |

```yaml
parameters:
  - {name: Wall insulation, class: Material, object: _InsulationEX_.01, field: Thickness, min: 0.1, max: 0.5}
  - name: Glazing
    class: FenestrationSurface:Detailed
    field: Construction Name
    where: {Surface Type: Window}
    options: [single_glazing, double_glazing, triple_glazing]
```

### `objectives`

A list; each entry is a meter name or a mapping:

```yaml
objectives:
  - Electricity:Facility
  - {meter: Electricity:Facility, name: Electricity kWh, unit: kWh, aggregate: sum, missing: zero, direction: min}
  - {variable: Zone Mean Air Temperature, key: "*", name: Tmean, aggregate: mean, frequency: Hourly, direction: min}
```

See {doc}`objectives` for the meaning of each key.

### `evaluation`

| Key | Default |
|---|---|
| `processes` | `1` |
| `out_dir` | `<output_dir>/runs` |
| `cache` | `true` (or a folder, or `false`) |
| `keep_dirs` | `false` |
| `timeout` | none (seconds) |
| `design_day` | `false` |
| `progress` | `false` |
| `on_error` | `nan` (or `raise`) |
| `on_severe` | `warn` (or `nan`, `ignore`): runs that finish with severe EnergyPlus errors, see {doc}`evaluation` |

### `runner`

```yaml
runner:
  type: command                       # default: local
  prefix: [docker, exec, energyplus]
  executable: /EnergyPlus/energyplus  # as seen by the command
  path_map: {/data/runs: /runs}       # host path -> path seen by the command
```

See {class}`~bdopt.simulation.CommandRunner`.

### `sampling`

| Key | Meaning | Default |
|---|---|---|
| `method` | `adaptive`, or a fixed design: `lhs`, `sobol`, `random` | `adaptive` |
| `seed` | seed of everything random | `0` |
| `adaptive_method` | `flola_voronoi`, `lola_voronoi`, `cv_voronoi`, `voronoi`, `probabilistic`, `gp_variance`, `nipv`, `latin_hypercube`, `random` | `flola_voronoi` |
| `n_initial`, `n_test`, `max_iter`, `batch`, `tol`, `max_evaluations` | as in {doc}`adaptive-sampling` | |
| `n_samples` | size of a fixed design | `5 * n_parameters` |
| `n_test` (fixed designs) | extra held-out simulations | `0` |
| `verbose` | print one line per iteration | `true` |

### `surrogate`

```yaml
surrogate: {type: gp}                                     # any name of bdopt.surrogate.SURROGATES, or torch_mlp, mc_dropout_mlp, multitask_gp, deep_kernel_gp
surrogate: {type: torch_mlp, params: {hidden: [32, 64, 32], n_ensemble: 3}}
```

`params` are passed to the class ({class}`~bdopt.surrogate.GPSurrogate`, {class}`~bdopt.surrogate.MLPSurrogate`,
{class}`~bdopt.surrogate.TorchMLP`). The same surrogate type is used during adaptive sampling and for the final
model.

### `optimization`

| Key | Meaning | Default |
|---|---|---|
| `algorithm` | `nsga2` or `nsga3` | `nsga2` |
| `pop_size`, `n_gen`, `seed`, `n_partitions` | as in {doc}`optimization` | |
| `use` | `surrogate` or `energyplus` (optimize directly on simulations) | `surrogate` |
| `weights`, `normalize` | how the design is selected, see {meth}`~bdopt.optimize.ParetoFront.closest_to_ideal` | `normalize: true` |
| `verify` | re-simulate this many Pareto designs and compare with the surrogate | `0` |

A complete example with every option commented is `examples/study.yaml` in the repository.
