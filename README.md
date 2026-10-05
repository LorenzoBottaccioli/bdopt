# bdopt

Building design optimization with **EnergyPlus**: parametric studies, **adaptive sampling**, a wide choice of
surrogate models (Gaussian processes, tree ensembles, kernel methods, scikit-learn and PyTorch networks) and
multi-objective optimization (NSGA-II / NSGA-III). Built for the Smart Building Educational Platform.

* **EnergyPlus is the core.** bdopt edits the IDF, runs EnergyPlus in parallel (with an on-disk cache and
  failure handling) and reads `eplusout.sql`. The installation matching the model is found automatically
  and names that changed between releases are aliased (see [EnergyPlus versions](docs/guide/energyplus-versions.md)).
* **Independent library.** `pip install` + any EnergyPlus install, on any OS. No Docker, no fixed paths
  (a Dockerfile is provided as a convenience, nothing depends on it).
* **Twenty-six adaptive sampling strategies** in numpy/scipy: fuzzy LOLA-Voronoi, LOLA-Voronoi, cross-validation
  Voronoi, Voronoi, probabilistic, GP variance, negative integrated posterior variance, query by committee,
  leave-one-out, Lipschitz, inverse-distance weighting, greedy, CVT, Fisher information, expected improvement,
  probability of improvement, lower/upper confidence bound, sample-min, SOAS, Sobol'/Halton sequences, mixed,
  error window, maximin Latin hypercube and random.
* **Twenty-eight surrogate models**, from polynomials, PCE and RBF interpolation to boosting libraries, multi-task,
  deep-kernel and sparse Gaussian processes, ensembles and automatic selection by cross-validation.
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
`torch`, `viz` (matplotlib), `yaml`. Python >= 3.10.
EnergyPlus is found through, in order, `EnergyPlus(path=...)`, `BDOPT_EPLUS_PATH`, `ENERGYPLUS_DIR`,
`EPLUS_PATH`, `PATH`, and the usual install folders (`/usr/local`, `/opt`, `/Applications`, `C:\`,
`~/.bdopt/eplus`). `tools/install_eplus.py` installs into `~/.bdopt/eplus`, which is searched automatically.

## What is in it

| | |
|---|---|
| `bdopt.Problem`, `Parameter`, `FieldSelector`, `RangeParameter`, `IntegerParameter`, `CategoryParameter`, `GenericSelector` | the design space; everything numeric happens in the unit cube, so bounds, order and scaling are never typed by hand |
| `MeterReader`, `VariableReader`, `FunctionReader` | objectives (units, aggregation, direction min/max, custom post-processing of `eplusout.sql`) |
| `EnergyPlusEvaluator` | parallel runs, on-disk cache (resume after a crash for free), failure handling (NaN + kept run directory), EnergyPlus warning/severe-error counts and an `on_severe` policy, `auto_upgrade` |
| `bdopt.sampling` | `lhs`, `sobol`, `halton`, `hammersley`, `maximin_lhs`, `fps`, `cvt`, `random`, `factorial` with constraints |
| `bdopt.adaptive.AdaptiveSampler` | 26 methods (`bdopt.adaptive.METHODS`, default `flola_voronoi`; see the [algorithms guide](docs/guide/adaptive-algorithms.md)); stops on test error, iterations or simulation budget |
| `bdopt.surrogate` | 28 models (`bdopt.surrogate.SURROGATES`; see the [models guide](docs/guide/surrogate-models.md)) incl. `GPSurrogate`, `SparseGPSurrogate`, `UniversalKrigingSurrogate`, `PCESurrogate`, boosting, `BaggingSurrogate`, `AutoSurrogate`; metrics, cross-validation |
| `bdopt.optimize` | `nsga2`, `nsga3` (pymoo), mixed variables, constraints, `ParetoFront.closest_to_ideal()` |
| `bdopt.DesignStudy`, `bdopt run study.yaml` | the whole pipeline from a YAML/JSON file (`examples/study.yaml`) |
| `bdopt.viz` | pairplot, correlation heatmap, parallel coordinates, convergence, parity, Pareto (matplotlib) |

## Documentation

Benchmark results (all tables, protocol, how to reproduce): [docs/benchmarks.md](docs/benchmarks.md). What the literature says: [docs/literature.md](docs/literature.md).

The documentation is a Sphinx site in `docs/` (build it with `pip install -e ".[docs]"` then
`sphinx-build -W -b html docs docs/_build/html`):

* [Installation](docs/installation.md), [quick start](docs/quickstart.md), [concepts](docs/concepts.md);
* [user guide](docs/guide/index.md): model and parameters, objectives, running EnergyPlus, fixed designs,
  [adaptive sampling](docs/guide/adaptive-sampling.md), surrogates, optimization,
  [studies from a YAML file](docs/guide/studies.md), [EnergyPlus versions](docs/guide/energyplus-versions.md);
* [tutorials](docs/tutorials/index.md): executed notebooks (adaptive sampling basics, an office building on
  EnergyPlus, mixed variables and constraints);
* API reference (generated from the docstrings), [troubleshooting](docs/faq.md),
  [contributing](docs/contributing.md), [changelog](docs/changelog.md).

## Measured

All measured in the development environment (4 cores, EnergyPlus from the official releases).

* **Reproducible results.** 4 design points of an office model (Larnaca weather, 6 parameters, 3 objectives,
  EnergyPlus 9.6) reproduce stored reference values to a maximum relative difference of 4e-13
  (`tests/test_eplus_versions.py::test_reproduces_stored_reference_results`). A study that is interrupted and
  re-run takes its finished simulations from the cache (4 cached results: 0.4 s instead of 33 s).
* **The seven platform tests.** `examples/platform_tests.py` expresses the platform's seven surrogate-modelling
  tests (offices, two flats, a multi-floor residence, a laboratory with a categorical glazing type) in bdopt.
  Re-simulating the platform's stored test points with EnergyPlus 9.6 gives a maximum relative difference to
  its BESOS results of 7.7e-8, 2.4e-8, 1.6e-4, 4.7e-5, 9.3e-5, 3.5e-4 and 2.8e-7 for tests 1 to 7 (I did not
  investigate the 1e-4 differences of the residential and laboratory models).
* **Strategies and surrogates on the seven tests**, real EnergyPlus runs (`docs/benchmarks/platform_benchmark.py`,
  raw results in `docs/benchmarks/platform_*.csv`). Per test: 15 initial + 5 test simulations, then 20 adaptive
  ones (35 training points); the plain Latin hypercube has 35 points; error = RMSE / range averaged over the
  three objectives (`nrmse`) of a Gaussian process on 25 independent Latin-hypercube simulations; one seed
  per test.

  Adaptive strategies (same Gaussian process):

  | method | test 1 | test 2 | test 3 | test 4 | test 5 | test 6 | test 7 | mean |
  |---|---|---|---|---|---|---|---|---|
  | plain_lhs | 0.032 | 0.015 | 0.016 | 0.029 | 0.008 | 0.027 | 0.014 | 0.020 |
  | nipv | 0.032 | 0.018 | 0.015 | 0.026 | 0.007 | 0.035 | 0.016 | 0.021 |
  | voronoi | 0.036 | 0.020 | 0.012 | 0.024 | 0.007 | 0.033 | 0.019 | 0.021 |
  | latin_hypercube | 0.030 | 0.019 | 0.015 | 0.026 | 0.008 | 0.033 | 0.024 | 0.022 |
  | flola_voronoi | 0.039 | 0.021 | 0.012 | 0.022 | 0.007 | 0.038 | 0.017 | 0.022 |
  | cv_voronoi | 0.037 | 0.019 | 0.015 | 0.020 | 0.009 | 0.044 | 0.015 | 0.023 |
  | gp_variance | 0.037 | 0.024 | 0.016 | 0.026 | 0.008 | 0.040 | 0.018 | 0.024 |
  | random | 0.035 | 0.026 | 0.016 | 0.030 | 0.009 | 0.031 | 0.019 | 0.024 |
  | lola_voronoi | 0.040 | 0.019 | 0.019 | 0.033 | 0.008 | 0.057 | 0.017 | 0.027 |
  | probabilistic | 0.056 | 0.025 | 0.019 | 0.063 | 0.012 | 0.057 | 0.024 | 0.036 |

  **On these models no adaptive strategy beat the plain Latin hypercube** (mean 0.020 against 0.021 to 0.036);
  `nipv`, `flola_voronoi`, `cv_voronoi` and `latin_hypercube` are within the noise of one seed per test, and
  `probabilistic` is clearly worse. This agrees with the analytic study below: the platform's responses are
  smooth, so adaptivity has nothing to find. Surrogates trained on the 35-point Latin hypercube:

  |  | test 1 | test 2 | test 3 | test 4 | test 5 | test 6 | test 7 | mean |
  |---|---|---|---|---|---|---|---|---|
  | auto | 0.032 | 0.015 | 0.016 | 0.029 | 0.008 | 0.027 | 0.014 | 0.020 |
  | gp | 0.032 | 0.015 | 0.016 | 0.029 | 0.008 | 0.027 | 0.014 | 0.020 |
  | multitask_gp | 0.035 | 0.027 | 0.048 | 0.050 | 0.031 | 0.062 | 0.019 | 0.039 |
  | mc_dropout_mlp | 0.042 | 0.040 | 0.037 | 0.044 | 0.038 | 0.059 | 0.023 | 0.041 |
  | deep_kernel_gp | 0.047 | 0.036 | 0.044 | 0.037 | 0.038 | 0.079 | 0.025 | 0.044 |
  | rbf | 0.044 | 0.046 | 0.053 | 0.059 | 0.032 | 0.065 | 0.027 | 0.046 |
  | kernel_ridge | 0.048 | 0.059 | 0.051 | 0.052 | 0.040 | 0.054 | 0.032 | 0.048 |
  | svr | 0.041 | 0.061 | 0.058 | 0.059 | 0.035 | 0.063 | 0.041 | 0.051 |
  | mlp | 0.046 | 0.049 | 0.057 | 0.062 | 0.041 | 0.080 | 0.033 | 0.053 |
  | gradient_boosting | 0.064 | 0.067 | 0.055 | 0.055 | 0.034 | 0.080 | 0.050 | 0.058 |
  | torch_mlp | 0.048 | 0.069 | 0.058 | 0.067 | 0.063 | 0.097 | 0.033 | 0.062 |
  | polynomial | 0.053 | 0.075 | 0.083 | 0.077 | 0.061 | 0.083 | 0.048 | 0.068 |
  | extra_trees | 0.078 | 0.085 | 0.103 | 0.101 | 0.091 | 0.115 | 0.060 | 0.090 |
  | random_forest | 0.088 | 0.099 | 0.119 | 0.116 | 0.112 | 0.132 | 0.071 | 0.105 |
  | knn | 0.133 | 0.133 | 0.135 | 0.156 | 0.150 | 0.158 | 0.107 | 0.139 |

  The Gaussian process is the best model on every test; `auto` selects it every time. Trees and k-NN are the
  least accurate with this few points. Neural networks need more data than 35 points.
* **Adaptive strategies vs plain Latin hypercube, analytic responses** (`docs/benchmarks/adaptive_vs_lhs.py`):

  Same Gaussian process, 3 seeds, 50 training points, error = RMSE / standard deviation of the response on 3000
  independent points (± over seeds):

  | response | FLOLA | LOLA | CV-Voronoi | NIPV | probabilistic | GP variance | maximin LHS | random | plain LHS |
  |---|---|---|---|---|---|---|---|---|---|
  | smooth | 0.001 | 0.001 | 0.001 | 0.002 | 0.001 | 0.001 | 0.001 | 0.002 | 0.002 |
  | narrow bump | **0.134** ± 0.021 | 0.151 ± 0.042 | 0.151 ± 0.045 | 0.318 ± 0.041 | 0.701 ± 0.454 | 0.380 ± 0.075 | 0.525 ± 0.317 | 0.378 ± 0.110 | 0.228 ± 0.012 |
  | oblique step | 0.240 ± 0.023 | 0.224 ± 0.013 | 0.212 ± 0.019 | **0.184** ± 0.012 | 0.194 ± 0.020 | 0.204 ± 0.006 | 0.256 ± 0.036 | 0.244 ± 0.038 | 0.207 ± 0.033 |

  The three Voronoi/LOLA strategies are clearly better than a Latin hypercube only on the **localised** feature
  (FLOLA 41 % lower error); NIPV and the variance-based methods help a little on the step, within the spread
  between seeds; on the smooth response nothing matters. Methods that chase uncertainty alone
  (`probabilistic`, `gp_variance`) can be much worse than a plain design (they pile samples on the boundary).
* **Upgrading a model.** The platform's Office model (EnergyPlus 9.6) upgraded to 25.2 (8 transition steps)
  and simulated; the upgrade took about 230 s once (cached afterwards), and the three objectives differ from
  the 9.6 results by -2.3 %, +4.9 %, -5.8 %. I did not isolate how much of that is the engine and how much the
  transition, so check an upgraded model before relying on it.

## Limitations

* EnergyPlus Windows and macOS code paths and the CI workflow are written but **not executed**: development
  and tests ran on Linux only. The Dockerfiles were not built (no Docker daemon available).
* Results are read from `eplusout.sql` at run-period level (meters and variables, aggregated with sum/max/mean)
  or through your own `FunctionReader`. Geometry edits are not provided: use a `GenericSelector` with your
  own setter function.
* `lola_voronoi` restricts the neighbourhood search to the nearest samples (the paper enumerates every subset), and
  the fuzzy membership functions of `flola_voronoi` are my reading of the paper (see the adaptive sampling
  guide), so neither is guaranteed numerically identical to other implementations.
* The platform benchmark uses one seed per test and 35 training points; differences below about 0.005 are noise.
* A model must match the engine version, or be upgraded (`auto_upgrade=True`); downgrading is not possible.

## Development

```bash
pip install -e ".[dev]"
pytest -m "not eplus and not slow"                                   # no EnergyPlus needed
BDOPT_EPLUS_PATH=/opt/eplus pytest -m eplus                          # every installed version
BDOPT_TEST_EPLUS_VERSIONS=9.6,24.2 pytest -m eplus -k "not golden"   # chosen versions
python tools/idd_diff.py OLD_ROOT NEW_ROOT                           # what changed between two versions
sphinx-build -W -b html docs docs/_build/html                        # the documentation
```

MIT licensed. References for the algorithms: Crombecq et al. (2011), SIAM J. Sci. Comput. 33(4);
van der Herten et al. (2015), SIAM J. Sci. Comput. 37(2); pymoo (Blank & Deb, 2020).
