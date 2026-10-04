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
objective over the initial samples (fixed from the start), averaged over objectives. With
`n_test=0` the same quantity is computed from 5-fold cross-validation on the training data instead.

### Methods

Each method is explained step by step in {doc}`adaptive-algorithms`; this table is the summary.

`AdaptiveSampler(method=...)` takes one of the names in `bdopt.adaptive.METHODS`. All of them are implemented in
numpy/scipy in `bdopt.adaptive.designs`.

| `method` | How the next point is chosen | Needs |
|---|---|---|
| `flola_voronoi` (default) | **Fuzzy LOLA-Voronoi** (Crombecq et al. 2011; van der Herten et al. 2015). Exploration: a Monte-Carlo estimate of the volume of each sample's Voronoi cell (a big cell is a poorly covered region). Exploitation: how far the response around each sample is from a locally fitted hyperplane. The neighbourhood of a sample is every sample closer than $\alpha = \frac{2}{K}\sum$ (the $K = 4d$ nearest distances); each neighbour is weighted by a Mamdani fuzzy inference system fed with its *cohesion* (distance to the sample) and *adhesion* (distance to its own nearest sample), see {func}`~bdopt.adaptive.designs.fuzzy_weights`. The sample with the highest *volume + nonlinearity* score gets a new neighbour at the farthest point of its cell. | |
| `lola_voronoi` | **LOLA-Voronoi**: same hybrid score, with crisp neighbourhoods. For every sample the best set of $2d$ neighbours is chosen by the cross-polytope criterion (cohesion and adhesion of the set, Crombecq et al. Eq. 4.3 to 4.7). The paper enumerates every subset, which is $\binom{n}{2d}$ (about $10^{12}$ for 60 samples in 6 dimensions); here the search is restricted to the nearest samples, as many as keep it below 3000 subsets. | |
| `cv_voronoi` | **Cross-validation Voronoi** (Xu et al. 2014; Liu et al. 2016; Kaminsky et al. 2021). A 5-fold cross-validation leaves every sample out once; the cell of the sample the surrogate predicts worst is refined at its farthest point. Objectives are combined with weights proportional to their cross-validated NRMSE. | refits the surrogate 5 times per iteration |
| `voronoi` | Exploration only: fills the largest gaps (tends towards the corners of the cube). | |
| `probabilistic` | The point of largest predictive standard deviation, searched over the whole cube by differential evolution (Chai 2019). | surrogate with uncertainty |
| `gp_variance` | The same criterion on 4000 random candidates (cheap). | surrogate with uncertainty |
| `nipv` | **Negative integrated posterior variance**: the candidate that lowers the *average* predictive variance over the domain the most. For a Gaussian process this is a closed form, $\overline{\mathrm{cov}(m, c)^2 / (\mathrm{var}(c) + \sigma^2_n)}$ over Monte-Carlo points $m$, and a batch is built by adding the fantasised point and repeating. | {class}`~bdopt.surrogate.GPSurrogate` |
| `latin_hypercube` | Maximin-augmented Latin hypercube: each batch is the best of 50 random Latin hypercubes by distance to the points already taken. Non-adaptive, space filling. | |
| `random` | Uniform random, as a baseline. | |

Surrogates with uncertainty: {class}`~bdopt.surrogate.GPSurrogate`, {class}`~bdopt.surrogate.MultiTaskGPSurrogate`,
{class}`~bdopt.surrogate.DeepKernelGPSurrogate`, {class}`~bdopt.surrogate.RandomForestSurrogate`,
{class}`~bdopt.surrogate.ExtraTreesSurrogate`, {class}`~bdopt.surrogate.MCDropoutMLP`, an ensemble
{class}`~bdopt.surrogate.TorchMLP`.

Departures from the papers, so that nobody has to guess: the fuzzy membership functions act on normalised
distances (close = sigmoid of the cohesion distance, isolated and clustered = Gaussians of the adhesion
distance) and the rules are *close and isolated = high weight*, *close and clustered = average*, *far and isolated
= average*, *far and clustered = low*; the response of several objectives is handled by summing the
non-linearity of the standardised objectives; the non-linearity uses the proper dot product of gradient and
displacement.

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

It depends on the shape of the response.

**Analytic responses** (three parameters, 3 seeds, 50 training points, Gaussian-process surrogate; error = RMSE on
3000 independent points divided by the standard deviation of the response; `±` over seeds):

| response | FLOLA | LOLA | CV-Voronoi | NIPV | probabilistic | GP variance | maximin LHS | random | plain LHS |
|---|---|---|---|---|---|---|---|---|---|
| smooth | 0.001 | 0.001 | 0.001 | 0.002 | 0.001 | 0.001 | 0.001 | 0.002 | 0.002 |
| narrow bump | **0.134** ± 0.021 | 0.151 ± 0.042 | 0.151 ± 0.045 | 0.318 ± 0.041 | 0.701 ± 0.454 | 0.380 ± 0.075 | 0.525 ± 0.317 | 0.378 ± 0.110 | 0.228 ± 0.012 |
| oblique step | 0.240 ± 0.023 | 0.224 ± 0.013 | 0.212 ± 0.019 | **0.184** ± 0.012 | 0.194 ± 0.020 | 0.204 ± 0.006 | 0.256 ± 0.036 | 0.244 ± 0.038 | 0.207 ± 0.033 |

* **Localised nonlinearity** (a narrow bump): FLOLA, LOLA and CV-Voronoi cut the error of a Latin hypercube by 34 to
  41 %. Uncertainty-only criteria (`probabilistic`, `gp_variance`) and `nipv` are worse than a plain design.
* **Smooth responses:** every design is equally good.
* **An oblique step:** differences are within the spread between seeds.

**The platform's seven tests** (real EnergyPlus, 35 training points, one seed per test, error on 25 independent
simulations; `docs/benchmarks/platform_benchmark.py`):

| strategy | mean nrmse over the 7 tests |
|---|---|
| plain Latin hypercube | 0.020 |
| `nipv` | 0.021 |
| `latin_hypercube` (maximin) | 0.022 |
| `flola_voronoi` | 0.022 |
| `cv_voronoi` | 0.023 |
| `lola_voronoi` | 0.027 |
| `probabilistic` | 0.036 |

No adaptive strategy beat the plain Latin hypercube on these models: their responses to insulation, lighting,
ventilation and shading set points are smooth, and a Gaussian process with 35 points already reaches 1 to 3 %
error. So expect the adaptive loop to help when something happens in a small region of the design space (a
threshold, a switch, a resonance). For smooth building responses its main value is the **convergence monitoring
and automatic stopping**: you learn how many simulations were enough instead of guessing.

## Why a Gaussian process with restarts

During development the GP's marginal-likelihood optimisation, started from a few random points, sometimes
collapsed to a "white noise" solution (all length scales at the lower bound) that predicts the mean, giving
errors around 1.0 for some Latin-hypercube designs and 0.1 for others, which made any comparison between designs
meaningless. {class}`~bdopt.surrogate.GPSurrogate` therefore bounds the length scales to what makes sense in the
unit cube, restarts the optimisation 8 times and retries with more restarts if it detects the collapsed solution
(`tests/test_surrogate.py::test_gp_not_degenerate_across_lhs_designs`).
