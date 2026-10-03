# Troubleshooting and FAQ

## EnergyPlus is not found

```text
EnergyPlusNotFound: no EnergyPlus installation found. Install one (e.g. `python tools/install_eplus.py 9.6`),
or set ENERGYPLUS_DIR / BDOPT_EPLUS_PATH, or pass EnergyPlus(path=...).
```

Run `bdopt eplus list`. If your installation is in an unusual place, set `BDOPT_EPLUS_PATH` to its folder (or to
a folder that contains several installations), or pass `eplus="/path/to/EnergyPlus-9-6-0"` to the evaluator. See
{doc}`installation`.

## "the model is for EnergyPlus X but no such version is installed"

The model's `Version` object says it was written for another EnergyPlus than the ones installed. Options:

1. install that version (`python tools/install_eplus.py 9.6`);
2. upgrade the model to the newest installed version with `auto_upgrade=True`. It is slow the first time (about
   10 to 30 seconds per version step; a large model from 9.6 to 25.2 took about 4 minutes) and cached afterwards.
   Check the upgraded model: results between engine versions differ;
3. open and save the model in the EnergyPlus version you have (IDF Version Updater).

A model *newer* than the engine cannot be downgraded.

## Names: "does not exist" and "matches no object"

```text
KeyError: field 'Lighting Levl' does not exist in class 'Lights' of EnergyPlus 9.6.0. Did you mean: Lighting Level?
ParameterError: parameter 'x': selector 'Lights:NO SUCH LIGHT:Lighting Level' matches no object in the model ...
```

These are raised when the evaluator is created. The first is a misspelt class or field (the message lists the
closest names; `Energy+.idd` of your installation is the reference); the second means the object name or
`where` filter selects nothing in *this* model. Names are case-insensitive, whitespace is normalised.

## Every simulation fails

Look in the folder of a failed run, `out_dir/failed/run_<id>/` (the path is in the warning):

* `eplusout.err` has EnergyPlus' own messages (`** Severe **`, `**  Fatal  **`);
* `stdout.txt` is what the program printed;
* `in.idf` is exactly the model that was run.

Frequent causes: a weather file path that does not exist, a parameter value outside what the model accepts
(a negative thickness, a fraction above 1), an `HVACTemplate` model run with a version too old for it, or an
EnergyPlus that is not the model's version.

## A warning says a meter was "not reported"

```text
UserWarning: meter 'DistrictCooling:Facility' not reported by EnergyPlus (no consumption, or unknown name)
```

EnergyPlus only creates a meter if something uses that resource. If the model has no cooling, `0` is the right
answer; if you expected a value, check the spelling. Set `missing="error"` on the {class}`~bdopt.MeterReader` to
turn this into an exception.

## `NaN` in the results

A simulation failed (see above). Adaptive sampling skips those points; for a fixed design drop them with
`results.dropna()` before training. Surrogates drop `NaN` rows with a warning.

## The adaptive sampler raises "only n initial simulations succeeded"

Fewer than three of the initial simulations worked, so there is nothing to build a surrogate from. Fix the cause
of the failures first (`evaluator.failures[0].errors`).

## It is slow

* **Simulations dominate.** Use `processes` up to the number of cores (each EnergyPlus run uses one core), a
  shorter `RunPeriod` while developing, `design_day=True` for a quick test of the plumbing.
* **Surrogate fitting dominates** (many points, many objectives): `GPSurrogate(n_restarts=2)` is much faster; or use
  {class}`~bdopt.surrogate.TorchMLP` for thousands of points.
* **The model upgrade is slow** the first time: it is cached under `out_dir/upgraded/`.

## `TorchMLP` raises ImportError

Install PyTorch: `pip install "bdopt[torch]"`. The default PyPI wheel includes CUDA libraries (several GB); for a
CPU-only install use `pip install torch --index-url https://download.pytorch.org/whl/cpu` first.

## Results differ between EnergyPlus versions

They do: the engine changes between releases. A model upgraded from 9.6 to 25.2 gave results that differed by a
few percent on annual totals. Compare results only within one version, or check how much they move before you
mix them.

## Can I use my own simulator, or a surrogate, as the evaluator?

Yes: {class}`~bdopt.FunctionEvaluator` wraps a Python function, and every sampler and optimizer only needs
`evaluate_unit`. See {doc}`guide/evaluation`.

## A parameter is not a single field in the IDF

Use a {class}`~bdopt.GenericSelector` with a function that edits the model; it can change any number of objects.

## Does it run on Windows and macOS?

The code paths are written for them (executable names, install folders, the installer helper) but bdopt has only
been tested on Linux so far. Reports are welcome.

## How do I run it where EnergyPlus is in a container?

Use a {class}`~bdopt.simulation.CommandRunner` with a command prefix such as
`["docker", "exec", "energyplus"]` and a `path_map`; see {doc}`guide/evaluation`.

## How much disk do runs use?

A run directory is deleted after a successful run (`keep_dirs=True` keeps them). The cache holds a few hundred
bytes per simulation. Failed runs are kept in `failed/` until you delete them.

## How do I cite bdopt?

See `CITATION.cff` in the repository.
