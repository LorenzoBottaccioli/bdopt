# Running EnergyPlus

{class}`~bdopt.EnergyPlusEvaluator` turns a table of designs into a table of results.

```python
from bdopt import EnergyPlusEvaluator

evaluator = EnergyPlusEvaluator(problem, "Office_IDF.idf", "Larnaca_Larnax_Airp_-hour.epw",
                                out_dir="runs", processes=4)
results = evaluator.evaluate(designs)                    # objectives, same index as `designs`
both = evaluator.evaluate(designs, keep_input=True)      # parameters + objectives
```

`designs` is a {class}`pandas.DataFrame` with one column per parameter (for example from
{func}`bdopt.sampling.lhs`). For each row the evaluator copies the model, writes the parameter values, adds the
`Output:SQLite` and `Output:Meter`/`Output:Variable` objects the objectives need, runs EnergyPlus, reads the
results and deletes the run directory.

## Arguments

| Argument | Meaning | Default |
|---|---|---|
| `problem`, `building`, `epw` | the problem, an {class}`~bdopt.IDF` or a path, the weather file | |
| `eplus` | which EnergyPlus: an {class}`~bdopt.EnergyPlus`, a version string (`"9.6"`), a path, or `None` to match the model | `None` |
| `auto_upgrade` | upgrade a model older than the engine instead of failing ({doc}`energyplus-versions`) | `False` |
| `processes` | EnergyPlus runs in parallel | `1` |
| `out_dir` | working directory of the runs | `bdopt_runs` |
| `err_dir` | where failed runs are kept | `out_dir/failed` |
| `cache` | `True` (`out_dir/cache`), a folder, or `False` | `True` |
| `keep_dirs` | keep every run directory, not only failed ones | `False` |
| `timeout` | seconds per run | none |
| `design_day` | run the design days only (`-D`) | `False` |
| `on_error` | what to do with a run that fails: `"nan"` or `"raise"` | `"nan"` |
| `on_severe` | what to do with a run that *finishes* but reports severe errors: `"warn"`, `"nan"` (treat it as failed) or `"ignore"` | `"warn"` |
| `runner` | a custom {class}`~bdopt.simulation.Runner`, see below | local |
| `progress` | progress bar (needs `tqdm`) | `False` |

## Caching and resuming

The cache stores the objective values of every finished simulation under a key made from the edited model text,
the weather file's content, the EnergyPlus version and the requested outputs. Consequences:

* the same design is never simulated twice, even in a different session or by another evaluator object that
  shares the cache folder;
* identical rows inside one call are simulated once;
* after a crash, run the study again: finished simulations come from the cache, so it picks up where it stopped
  (adaptive sampling is deterministic given its `seed`, so it asks for the same points);
* changing anything that affects the result (a parameter, the weather file, the EnergyPlus version, the unit of
  an objective) changes the key, so stale results are never reused.

```python
evaluator.stats()
# {'simulations': 4, 'cache_hits': 0, 'failed': 0, 'simulation_seconds': 121.87, 'eplus_version': '9.6.0'}
```

## EnergyPlus messages

EnergyPlus writes warnings and errors to `eplusout.err`. bdopt reads the final summary of every run and keeps the
counts in the run record, so a model that finishes with problems is not silent:

```python
evaluator.stats()
# {'simulations': 32, 'cache_hits': 0, 'failed': 0, 'simulation_seconds': 978.32, 'eplus_version': '9.6.0',
#  'energyplus_warnings': 1152, 'energyplus_severe': 2}      # tutorial 2: 32 runs, one with 2 severe errors
evaluator.history[-1].eplus_warnings      # warnings of the last run
evaluator.history[-1].parameters          # the design that was simulated
```

* a run that finishes **with severe errors** (for example a zone whose warm-up did not converge) is handled by
  `on_severe`: `"warn"` (default) keeps the result and raises a `UserWarning` quoting the first severe messages;
  `"nan"` treats the design as failed (NaN, run directory kept in `failed/`, never cached, so adaptive sampling skips
  it and does not propose it again); `"ignore"` is silent. The policy also applies to results taken from the cache,
  which remember how many severe errors their run had;
* ordinary warnings are only counted (most models have a few dozen, e.g. unused constructions or monthly tables
  that name meters the model does not have): to read them keep the run directory (`keep_dirs=True`) and open its
  `eplusout.err`;
* **bdopt itself adds none**: the objects it appends (`Output:SQLite`, the meters and variables you ask for) are
  chosen so that EnergyPlus has nothing to say about them. A test compares, on every installed EnergyPlus
  version, the warnings of a model run directly and through the evaluator, and they are identical. For example the
  office model of the tutorials gives 36 warnings and no severe error, with or without bdopt.

## Failures

A run counts as failed if EnergyPlus exits with an error, does not complete, times out, or produces no SQL file.

* with `on_error="nan"` (default) the row of the result is `NaN`, a `UserWarning` names the run directory and
  the first EnergyPlus errors, and the directory (with `eplusout.err`) is moved to `err_dir`;
* with `on_error="raise"` a `RuntimeError` is raised after the batch;
* `evaluator.failures` lists the failed runs (`RunRecord` with `errors`, `run_dir`);
* adaptive sampling skips failed points, and samplers and optimizers never see NaN (the optimizers penalise it).

## Running EnergyPlus somewhere else

The default {class}`~bdopt.simulation.LocalRunner` starts the `energyplus` executable of the selected
installation. To run it through a command prefix (a container, `ssh`, a job wrapper) use
{class}`~bdopt.simulation.CommandRunner`:

```python
from bdopt import EnergyPlus, EnergyPlusEvaluator
from bdopt.simulation import CommandRunner

local_copy = EnergyPlus.find(path="/local/copy/of/EnergyPlus-9-6-0")   # at least Energy+.idd, to resolve field names
runner = CommandRunner(
    prefix=["docker", "exec", "-u", "root", "energyplus"],
    eplus=local_copy,
    executable="/EnergyPlus/energyplus",                  # as seen inside the container
    path_map={"/data/runs": "/runs", "/data/weather": "/weather"},   # host path -> path in the container
)
evaluator = EnergyPlusEvaluator(problem, "model.idf", "/data/weather/city.epw", runner=runner, out_dir="/data/runs")
```

The library itself does not depend on Docker; the prefix can be anything that runs a command.

## Evaluating with something else

{class}`~bdopt.FunctionEvaluator` wraps a Python function with the same interface, so a surrogate, an analytic
test function or another simulator can be used wherever an evaluator is expected:

```python
from bdopt import FunctionEvaluator

ev = FunctionEvaluator(problem, lambda row: [row["Wall"] ** 2, 1 - row["Wall"]])            # one row at a time
ev = FunctionEvaluator(problem, lambda df: df[["Wall"]].to_numpy() * [1, 2], vectorized=True)  # whole table
```

Every evaluator also has {meth}`~bdopt.Evaluator.evaluate_unit`, which takes unit-cube points: that is what the
samplers and optimizers call.
