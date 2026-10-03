# bdopt

Building design optimization with **EnergyPlus 8.7 and newer**: parametric studies, **adaptive sampling**,
surrogate models (scikit-learn and PyTorch) and multi-objective optimization (NSGA-II / NSGA-III).
Built for the Smart Building Educational Platform, with adaptive sampling as a first-class feature.

* **Independent library.** `pip install` + any EnergyPlus install, on any OS. No Docker, no fixed paths
  (a Dockerfile is provided as a convenience, nothing depends on it).
* **Every EnergyPlus version from 8.7** (tested on 8.7, 9.0, 9.6, 23.1, 24.2, 25.2): the installation matching
  the model is found automatically; names that changed between versions are aliased; older models can be
  upgraded with EnergyPlus' own transition tools.
* **Adaptive sampling without TensorFlow.** `harlow` imports TensorFlow at module level; bdopt implements the
  same family of algorithms (FLOLA-Voronoi, LOLA-Voronoi, Voronoi, GP variance) in numpy/scipy, and offers
  an optional bridge if you do want harlow itself.
* **PyTorch, not TensorFlow/Keras**, for neural surrogates.
* **Categorical parameters are first-class** (glazing type, construction sets ...), in sampling, surrogates
  and optimization.

```python
from bdopt import *
from bdopt.adaptive import AdaptiveSampler
from bdopt.surrogate import GPSurrogate
from bdopt.optimize import nsga2

parameters = [
    Parameter(FieldSelector("Material", "_InsulationEX_.01", "Thickness"), RangeParameter(0.1, 0.5), name="Wall"),
    Parameter(FieldSelector("Lights", "*", "Watts per Zone Floor Area"), RangeParameter(5, 20), name="Lights"),
    Parameter(FieldSelector("FenestrationSurface:Detailed", "*", "Construction Name",
                            where={"Surface Type": "Window"}),
              CategoryParameter(["single_glazing", "double_glazing", "triple_glazing"]), name="Glazing"),
]
problem = Problem(parameters, ["Electricity:Facility", "DistrictHeating:Facility", "DistrictCooling:Facility"])
evaluator = EnergyPlusEvaluator(problem, "model.idf", "weather.epw", processes=4)   # engine auto-detected, results cached

result = AdaptiveSampler(problem, evaluator).run(n_initial=20, n_test=5, max_iter=40, tol=0.05)
surrogate = GPSurrogate().fit(result.data, problem)         # or TorchMLP(), MLPSurrogate()
print(surrogate.score(result.test))                          # RMSE, RMSE %, CV(RMSE), NMBE, R2
front = nsga2(problem, surrogate, pop_size=100, n_gen=100)   # Pareto front, categorical variables included
best = front.closest_to_ideal()
```

## Install

```bash
pip install "bdopt[torch,viz] @ git+https://github.com/lorenzobottaccioli/buidlingdesginoptimization.git"
python tools/install_eplus.py 9.6          # optional helper: downloads an official EnergyPlus release
bdopt eplus list                           # what bdopt can see
```

`pip install bdopt` alone gives sampling, scikit-learn surrogates, adaptive sampling and NSGA-II; extras:
`torch`, `viz` (matplotlib), `yaml`, `harlow`. Python >= 3.10.
EnergyPlus is found through, in order, `EnergyPlus(path=...)`, `BDOPT_EPLUS_PATH`, `ENERGYPLUS_DIR`,
`EPLUS_PATH`, `PATH`, and the usual install folders (`/usr/local`, `/opt`, `/Applications`, `C:\`,
`~/.bdopt/eplus`). `tools/install_eplus.py` installs into `~/.bdopt/eplus`, which is searched automatically.

## What is in it

| | |
|---|---|
| `bdopt.Problem`, `Parameter`, `FieldSelector`, `RangeParameter`, `IntegerParameter`, `CategoryParameter`, `GenericSelector` | the design space; everything numeric happens in the unit cube, so bounds, order and scaling are never typed by hand |
| `MeterReader`, `VariableReader`, `FunctionReader` | objectives (units, aggregation, direction min/max, custom post-processing of `eplusout.sql`) |
| `EnergyPlusEvaluator` | parallel runs, on-disk cache (resume after a crash for free), failure handling (NaN + kept run directory), `auto_upgrade` |
| `bdopt.sampling` | `lhs`, `sobol`, `random`, `factorial` with constraints |
| `bdopt.adaptive.AdaptiveSampler` | `flola_voronoi` (default), `lola_voronoi`, `voronoi`, `gp_variance`, `random`; stops on test error, iterations or simulation budget |
| `bdopt.adaptive.HarlowAdaptiveSampler` | optional bridge to harlow's samplers (needs harlow + TensorFlow) |
| `bdopt.surrogate` | `GPSurrogate` (ARD, uncertainty), `MLPSurrogate`, `TorchMLP` (optional ensemble), metrics, cross-validation |
| `bdopt.optimize` | `nsga2`, `nsga3` (pymoo), mixed variables, constraints, `ParetoFront.closest_to_ideal()` |
| `bdopt.DesignStudy`, `bdopt run study.yaml` | the whole pipeline from a YAML/JSON file (`examples/study.yaml`) |
| `bdopt.viz` | pairplot, correlation heatmap, parallel coordinates, convergence, parity, Pareto (matplotlib) |

More: [docs/VERSIONS.md](docs/VERSIONS.md) (EnergyPlus versions),
[docs/ADAPTIVE.md](docs/ADAPTIVE.md) (algorithms and when they help).

## Measured

All measured in the development environment (4 cores, EnergyPlus from the official releases).

* **Reproducible results.** 4 design points of an office model (Larnaca weather, 6 parameters, 3 objectives,
  EnergyPlus 9.6) reproduce stored reference values to a maximum relative difference of 4e-13
  (`tests/test_eplus_versions.py::test_reproduces_stored_reference_results`). A study that is interrupted and
  re-run takes its finished simulations from the cache (4 cached results: 0.4 s instead of 33 s).
* **One study, six EnergyPlus versions.** The same code (class, field and meter names written once) runs on
  8.7, 9.0, 9.6, 23.1, 24.2 and 25.2, including `HVACTemplate` models (`-x` expansion), and the cache,
  failure-handling and selector-error tests pass on each.
* **Adaptive vs plain Latin hypercube, same number of training points** (50, Gaussian-process surrogate,
  4 random seeds, normalised RMSE on 3000 independent points; analytic functions, 3 parameters):

  | response | FLOLA-Voronoi | LOLA-Voronoi | GP variance | random | LHS |
  |---|---|---|---|---|---|
  | smooth | 0.001 | 0.002 | 0.001 | 0.002 | 0.002 |
  | narrow local bump | **0.154** ± 0.036 | 0.166 ± 0.028 | 0.342 | 0.353 | 0.243 ± 0.029 |
  | step across an oblique surface | 0.224 | 0.186 | 0.197 | 0.257 | 0.203 |

  Adaptive sampling is clearly better only when the response has a **localised** feature (37 % lower error
  than LHS here). For smooth responses every design is equally good, and for the oblique step it did not help.
  If your responses are smooth (I would expect that for insulation, lighting or ventilation parameters, but I
  have not measured it on the platform's models), the practical value of the adaptive loop is the convergence
  monitoring and the automatic stopping, not a more accurate model for the same budget.
* **Upgrading a model.** The platform's Office model (EnergyPlus 9.6) upgraded to 25.2 (8 transition steps)
  and simulated; the upgrade took about 230 s once (cached afterwards), and the three objectives differ from
  the 9.6 results by -2.3 %, +4.9 %, -5.8 %. I did not isolate how much of that is the engine and how much the
  transition, so check an upgraded model before relying on it.

## Limitations

* EnergyPlus Windows and macOS code paths and the CI workflow are written but **not executed**: development
  and tests ran on Linux only. The Dockerfiles were not built (no Docker daemon available).
* Results are read from `eplusout.sql` at run-period level (meters and variables, aggregated with sum/max/mean)
  or through your own `FunctionReader`. Geometry edits are not provided: use a `GenericSelector` with your
  own function.
* `lola_voronoi` uses nearest-neighbour neighbourhoods rather than the cross-polytope selection of the paper, and
  the fuzzy weighting of `flola_voronoi` is a smooth closed form of the paper's fuzzy inference system, so
  neither is numerically identical to harlow's. The harlow bridge runs harlow's own code.
* A model must match the engine version, or be upgraded (`auto_upgrade=True`); downgrading is not possible.

## Development

```bash
pip install -e ".[dev]"
pytest -m "not eplus and not slow"                                   # no EnergyPlus needed
BDOPT_EPLUS_PATH=/opt/eplus pytest -m eplus                          # every installed version
BDOPT_TEST_EPLUS_VERSIONS=8.7,9.6 pytest -m eplus -k "not golden"   # chosen versions
python tools/idd_diff.py OLD_ROOT NEW_ROOT                           # what changed between two versions
```

MIT licensed. References for the algorithms: Crombecq et al. (2011), SIAM J. Sci. Comput. 33(4);
van der Herten et al. (2015), SIAM J. Sci. Comput. 37(2); pymoo (Blank & Deb, 2020).
