# Adaptive sampling

A fixed design (say 60 Latin-hypercube simulations) has to guess how many simulations are enough.
{class}`~bdopt.adaptive.AdaptiveSampler` starts from a small Latin hypercube plus a held-out test set, then adds
simulations where the surrogate needs them and stops when the test error is small enough (or a budget is reached).

```{image} ../_static/img/adaptive_vs_lhs.png
:alt: Latin hypercube against adaptive sampling on a response with a narrow feature
:width: 100%
```

```python
from bdopt.adaptive import AdaptiveSampler
from bdopt.surrogate import GPSurrogate

sampler = AdaptiveSampler(problem, evaluator, method="flola_voronoi", surrogate=GPSurrogate(), seed=0)
result = sampler.run(n_initial=20, n_test=5, max_iter=40, batch=1, tol=0.05, max_evaluations=80)
```

## `run()` arguments

| Argument | Meaning | Default |
|---|---|---|
| `n_initial` | first Latin-hypercube simulations | `5 * n_parameters` |
| `n_test` | held-out simulations used to measure the error (`0`: 5-fold cross-validation instead) | `max(5, n_parameters)` |
| `max_iter` | adaptive iterations | `40` |
| `batch` | new simulations per iteration (they can run in parallel) | `1` |
| `tol` | stop when `scaled_rmse` is at or below this | `0.05` |
| `max_evaluations` | stop after this many simulations in total (initial + test + adaptive) | none |
| `initial_data`, `test_data` | tables of already simulated designs to reuse instead of simulating new ones | none |
| `checkpoint_dir` | write `train.csv`, `test.csv`, `history.csv` after every iteration | none |
| `callback` | `callback(iteration, history_row)` after every iteration | none |

## Result

{class}`~bdopt.adaptive.AdaptiveResult` holds

* `data`, `test`: tables of parameters and objectives (real units) of the training and test simulations;
* `history`: one row per iteration with `iteration`, `n_train`, `n_evaluations`, `scaled_rmse`, `elapsed_s`, and
  per objective `rmse_scaled:<name>` and `r2:<name>`;
* `surrogate`: the model fitted on `data` (a {class}`~bdopt.surrogate.GPSurrogate` unless you passed another);
* `converged`, `stop_reason` (`"tolerance"`, `"max_iter"` or `"max_evaluations"`), `n_evaluations`;
* `failed`: designs whose simulation failed; `extra`: method-specific objects.

```{image} ../_static/img/convergence.png
:alt: scaled RMSE against number of training points
:width: 60%
```

## How it works

Each iteration:

1. fit the surrogate on all training simulations and measure its error on the held-out test simulations;
2. stop if the error is at or below `tol`, or `max_iter` / `max_evaluations` is reached;
3. propose `batch` new designs with the chosen method, simulate them, add them to the training set.

**Stopping metric.** `scaled_rmse` is the RMSE on the test points in units of the standard deviation of each
objective over the initial samples (fixed from the start, as harlow does), averaged over objectives. With
`n_test=0` the same quantity is computed from 5-fold cross-validation on the training data instead.

### Methods

| `method` | How the next point is chosen |
|---|---|
| `flola_voronoi` (default) | Hybrid of *exploration* and *exploitation* (Crombecq et al. 2011; van der Herten et al. 2015). Exploration: a Monte-Carlo estimate of the volume of each sample's Voronoi cell (a big cell is a poorly covered region). Exploitation: how far the response around each sample is from a locally fitted linear model, with neighbours weighted by closeness and isolation. The sample with the highest *volume + nonlinearity* score gets a new neighbour at the farthest point of its cell. |
| `lola_voronoi` | Same score with equal weights on the nearest neighbours. |
| `voronoi` | Exploration only: fills the largest gaps (tends towards the corners of the cube). |
| `gp_variance` | The candidate with the largest predictive standard deviation of the surrogate (needs a surrogate with uncertainty: {class}`~bdopt.surrogate.GPSurrogate` or an ensemble {class}`~bdopt.surrogate.TorchMLP`). |
| `random` | Uniform random, as a baseline. |

The Voronoi/FLOLA scores are implemented here in numpy; the fuzzy inference system of the paper is replaced by an
equivalent smooth weighting, and `lola_voronoi` uses nearest-neighbour neighbourhoods instead of the
cross-polytope selection of the paper. They are not numerically identical to harlow's (see below).

### Behaviour worth knowing

* **Discrete parameters** (integer, categorical) live in the unit cube and are snapped to valid values before
  every simulation; a design is never proposed twice.
* **Failures:** a failed simulation is skipped and never proposed again; the loop continues. If fewer than three
  initial simulations succeed, `run()` raises.
* **Determinism and resuming:** everything is deterministic given `seed`, and the evaluator caches finished
  simulations, so running an interrupted study again repeats nothing. `checkpoint_dir` also writes the tables
  after every iteration.
* **Parallelism:** with `batch > 1` the new designs of an iteration are simulated in parallel
  (`EnergyPlusEvaluator(processes=...)`); with `batch=1` and several cores, a larger `batch` keeps them busy.

## When does it help?

It depends on the shape of the response. Measured on analytic functions of three parameters (4 seeds, 50
training points, Gaussian-process surrogate; error = RMSE on 3000 independent points divided by the standard
deviation of the response; `±` is the standard deviation over seeds):

| Response | FLOLA-Voronoi | LOLA-Voronoi | GP variance | random | Latin hypercube |
|---|---|---|---|---|---|
| smooth | 0.001 | 0.002 | 0.001 | 0.002 | 0.002 |
| narrow local bump | **0.154** ± 0.036 | 0.166 ± 0.028 | 0.342 ± 0.092 | 0.353 ± 0.104 | 0.243 ± 0.029 |
| step across an oblique surface | 0.224 ± 0.020 | 0.186 ± 0.027 | 0.197 ± 0.012 | 0.257 ± 0.040 | 0.203 ± 0.030 |

* **Localised nonlinearity** (a narrow bump): FLOLA-Voronoi has 37 % lower error than a Latin hypercube of the
  same size, and about half the error of random or variance-based sampling.
* **Smooth responses:** every design is equally good; there is nothing to gain from adaptivity.
* **An oblique step:** no gain over the Latin hypercube (the differences are within the spread between seeds).

So expect the adaptive loop to help when something happens in a small region of the design space (a threshold, a
switch, a resonance). For smooth responses, which I would expect for insulation, lighting or ventilation
parameters but have not measured on real models, its main value is the **convergence monitoring and automatic
stopping**: you learn how many simulations were enough instead of guessing.

The benchmark is `docs/benchmarks/adaptive_vs_lhs.py`.

## Why a Gaussian process with restarts

During development the GP's marginal-likelihood optimisation, started from a few random points, sometimes
collapsed to a "white noise" solution (all length scales at the lower bound) that predicts the mean, giving
errors around 1.0 for some Latin-hypercube designs and 0.1 for others, which made any comparison between designs
meaningless. {class}`~bdopt.surrogate.GPSurrogate` therefore bounds the length scales to what makes sense in the
unit cube, restarts the optimisation 8 times and retries with more restarts if it detects the collapsed solution
(`tests/test_surrogate.py::test_gp_not_degenerate_across_lhs_designs`).

## Using harlow's samplers

{class}`~bdopt.adaptive.HarlowAdaptiveSampler` runs [harlow](https://github.com/TNO/harlow)'s own sampler
(`FuzzyLolaVoronoi`, `LolaVoronoi`, `ProbabilisticSampler` ...) on a bdopt problem; bounds, scaling, categorical
handling and the target function are generated for you.

```python
from bdopt.adaptive import HarlowAdaptiveSampler

result = HarlowAdaptiveSampler(problem, evaluator, method="FuzzyLolaVoronoi").run(
    n_initial=20, n_test=5, max_iter=40, tol=0.01
)
```

It returns the same {class}`~bdopt.adaptive.AdaptiveResult`. harlow 0.1.0 imports TensorFlow and
TensorFlow-Probability when it is imported, so installing it pulls in TensorFlow; with current releases you also
need `pip install tf_keras networkx`, and `TF_USE_LEGACY_KERAS=1` (set for you). The bridge is tested
(`tests/test_harlow_bridge.py`) but nothing else in bdopt needs it.
