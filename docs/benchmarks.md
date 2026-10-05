# Benchmarks

Everything below was measured in this repository's development environment (4 cores, EnergyPlus 9.6, real
simulations). Raw numbers: `docs/benchmarks/platform_adaptive.csv`, `platform_surrogates.csv`. Scripts:
`docs/benchmarks/platform_benchmark.py` (platform tests), `docs/benchmarks/adaptive_vs_lhs.py` (analytic
responses), `examples/platform_tests.py` (the seven tests as bdopt problems).

## The seven platform tests

`examples/platform_tests.py ROOT --test N --check` re-simulates the five stored test points of each platform test
(`ROOT` = the platform's `Surrogate modeling examples` folder) and compares with the platform's own BESOS results.
Maximum relative difference, tests 1 to 7: 7.7e-8, 2.4e-8, 1.6e-4, 4.7e-5, 9.3e-5, 3.5e-4, 2.8e-7. The 1e-4
differences of tests 3 to 6 were not investigated.

## Protocol

Per test: 15 initial Latin-hypercube simulations + 5 held-out test simulations, then 20 adaptive simulations
(35 training points); a plain Latin hypercube with 35 points for comparison; 25 independent Latin-hypercube
simulations to judge every model. Error `nrmse` = RMSE divided by the range of the objective, averaged over the 3
objectives, of a Gaussian process (adaptive part) or of the named surrogate (surrogate part). Seeds: the tables list which seeds are included (the rows come from `platform_adaptive_full.csv`
and `platform_surrogates_full.csv`, regenerated into this page by `docs/benchmarks/make_tables.py`). With a single
seed, differences below about 0.005 are noise; with several, compare them with the ± over seeds. A 60-point budget
is supported by the script (`--budgets`) but has not been run.

## Platform tests: strategies and surrogates

<!-- BENCH:start -->
Seeds used: 0, 1 (budget 35 points; error = nrmse of a Gaussian process / of the named surrogate).

**Adaptive strategies**

| strategy | test 1 | test 2 | test 3 | test 4 | test 5 | test 6 | test 7 | mean (± std over seeds) |
|---|---|---|---|---|---|---|---|---|
| plain_lhs | 0.034 | 0.016 | 0.016 | 0.024 | 0.007 | 0.032 | 0.016 | 0.021 ± 0.001 |
| voronoi | 0.035 | 0.017 | 0.013 | 0.028 | 0.009 | 0.033 | 0.017 | 0.022 ± 0.000 |
| mixed | 0.039 | 0.017 | 0.015 | 0.023 | 0.008 | 0.033 | 0.019 | 0.022 ± 0.001 |
| nipv | 0.037 | 0.018 | 0.014 | 0.028 | 0.008 | 0.033 | 0.016 | 0.022 ± 0.001 |
| sobol_seq | 0.031 | 0.018 | 0.014 | 0.024 | 0.009 | 0.038 | 0.020 | 0.022 ± 0.001 |
| flola_voronoi | 0.037 | 0.021 | 0.012 | 0.029 | 0.007 | 0.035 | 0.016 | 0.023 ± 0.000 |
| latin_hypercube | 0.032 | 0.020 | 0.016 | 0.027 | 0.008 | 0.035 | 0.022 | 0.023 ± 0.001 |
| cv_voronoi | 0.041 | 0.019 | 0.015 | 0.021 | 0.008 | 0.042 | 0.015 | 0.023 ± 0.001 |
| random | 0.038 | 0.023 | 0.016 | 0.030 | 0.009 | 0.032 | 0.017 | 0.023 ± 0.001 |
| greedy | 0.037 | 0.020 | 0.014 | 0.024 | 0.009 | 0.045 | 0.016 | 0.024 ± 0.001 |
| gp_variance | 0.039 | 0.024 | 0.016 | 0.029 | 0.009 | 0.035 | 0.017 | 0.024 ± 0.001 |
| loocv | 0.036 | 0.019 | 0.016 | 0.032 | 0.008 | 0.043 | 0.016 | 0.024 ± 0.003 |
| qbc | 0.042 | 0.017 | 0.018 | 0.036 | 0.007 | 0.032 | 0.019 | 0.024 ± 0.001 |
| lola_voronoi | 0.038 | 0.018 | 0.017 | 0.031 | 0.009 | 0.047 | 0.017 | 0.025 ± 0.003 |
| lipschitz | 0.041 | 0.018 | 0.019 | 0.036 | 0.008 | 0.042 | 0.019 | 0.026 ± 0.001 |
| probabilistic | 0.056 | 0.025 | 0.019 | 0.063 | 0.012 | 0.053 | 0.024 | 0.043 ± 0.009 |

**Surrogates** (trained on the plain Latin hypercube)

| surrogate | test 1 | test 2 | test 3 | test 4 | test 5 | test 6 | test 7 | mean (± std over seeds) |
|---|---|---|---|---|---|---|---|---|
| sparse_gp | 0.033 | 0.015 | 0.014 | 0.022 | 0.006 | 0.032 | 0.015 | 0.019 ± 0.001 |
| auto | 0.034 | 0.016 | 0.016 | 0.024 | 0.007 | 0.032 | 0.016 | 0.021 ± 0.001 |
| gp | 0.034 | 0.016 | 0.016 | 0.024 | 0.007 | 0.032 | 0.016 | 0.021 ± 0.001 |
| stacking | 0.033 | 0.018 | 0.016 | 0.024 | 0.008 | 0.032 | 0.019 | 0.022 ± 0.001 |
| universal_kriging | 0.036 | 0.021 | 0.018 | 0.028 | 0.006 | 0.045 | 0.024 | 0.025 ± 0.001 |
| pce | 0.036 | 0.032 | 0.032 | 0.032 | 0.021 | 0.040 | 0.021 | 0.031 ± 0.000 |
| deep_ensemble | 0.042 | 0.037 | 0.037 | 0.033 | 0.032 | 0.058 | 0.026 | 0.038 ± 0.005 |
| multitask_gp | 0.035 | 0.028 | 0.043 | 0.060 | 0.030 | 0.071 | 0.022 | 0.041 ± 0.004 |
| mc_dropout_mlp | 0.044 | 0.040 | 0.041 | 0.041 | 0.041 | 0.061 | 0.027 | 0.042 ± 0.002 |
| lasso_polynomial | 0.045 | 0.052 | 0.052 | 0.054 | 0.037 | 0.064 | 0.031 | 0.048 ± 0.002 |
| deep_kernel_gp | 0.045 | 0.036 | 0.049 | 0.039 | 0.038 | 0.099 | 0.030 | 0.048 ± 0.006 |
| kernel_ridge | 0.049 | 0.057 | 0.052 | 0.052 | 0.047 | 0.050 | 0.036 | 0.049 ± 0.002 |
| rbf | 0.043 | 0.050 | 0.058 | 0.058 | 0.037 | 0.068 | 0.032 | 0.049 ± 0.004 |
| svr | 0.043 | 0.060 | 0.058 | 0.057 | 0.041 | 0.069 | 0.039 | 0.052 ± 0.002 |
| mlp | 0.048 | 0.050 | 0.060 | 0.073 | 0.039 | 0.084 | 0.035 | 0.056 ± 0.004 |
| gradient_boosting | 0.063 | 0.064 | 0.056 | 0.049 | 0.033 | 0.076 | 0.053 | 0.056 ± 0.002 |
| bagging | 0.049 | 0.061 | 0.065 | 0.067 | 0.051 | 0.071 | 0.037 | 0.057 ± 0.002 |
| xgboost | 0.067 | 0.068 | 0.054 | 0.058 | 0.038 | 0.080 | 0.062 | 0.061 ± 0.004 |
| elm | 0.050 | 0.058 | 0.078 | 0.077 | 0.056 | 0.081 | 0.038 | 0.063 ± 0.003 |
| torch_mlp | 0.049 | 0.072 | 0.064 | 0.072 | 0.061 | 0.103 | 0.037 | 0.065 ± 0.005 |
| hist_gradient_boosting | 0.075 | 0.078 | 0.061 | 0.062 | 0.043 | 0.083 | 0.062 | 0.066 ± 0.003 |
| polynomial | 0.053 | 0.074 | 0.076 | 0.077 | 0.056 | 0.084 | 0.049 | 0.067 ± 0.002 |
| lightgbm | 0.074 | 0.076 | 0.065 | 0.068 | 0.050 | 0.080 | 0.067 | 0.069 ± 0.002 |
| adaboost | 0.090 | 0.089 | 0.072 | 0.063 | 0.050 | 0.104 | 0.073 | 0.077 ± 0.003 |
| extra_trees | 0.078 | 0.083 | 0.104 | 0.103 | 0.088 | 0.116 | 0.062 | 0.090 ± 0.000 |
| random_forest | 0.088 | 0.100 | 0.119 | 0.114 | 0.110 | 0.133 | 0.071 | 0.105 ± 0.000 |
| knn | 0.134 | 0.135 | 0.157 | 0.155 | 0.152 | 0.163 | 0.117 | 0.145 ± 0.008 |
| decision_tree | 0.128 | 0.129 | 0.165 | 0.186 | 0.161 | 0.192 | 0.106 | 0.152 ± 0.001 |
<!-- BENCH:end -->

## Many parameters

The platform tests have 6 to 7 parameters. Two tests with more, because the literature says adaptive sampling helps
most there.

**Analytic responses with 15 and 20 parameters** (`docs/benchmarks/highdim_analytic.py`; one localised bump in two
parameters plus a smooth trend in three more, the rest nearly inert; Gaussian process; 120 training points; 3 seeds;
error = RMSE / standard deviation on 3000 independent points; mean ± std over seeds; raw rows in
`highdim_analytic.csv`). Strategies not listed were not run here.

| 15 parameters | error |
|---|---|
| random | 0.251 ± 0.011 |
| sobol_seq | 0.269 ± 0.015 |
| latin_hypercube | 0.271 ± 0.035 |
| plain_lhs | 0.273 ± 0.035 |
| nipv | 0.275 ± 0.042 |
| mixed | 0.292 ± 0.048 |
| cv_voronoi | 0.307 ± 0.030 |
| qbc | 0.323 ± 0.034 |
| lola_voronoi | 0.328 ± 0.070 |
| lipschitz | 0.328 ± 0.069 |
| cvt | 0.360 ± 0.011 |
| greedy | 0.402 ± 0.111 |
| flola_voronoi | 0.415 ± 0.132 |

| 20 parameters | error |
|---|---|
| nipv | 0.240 ± 0.023 |
| random | 0.256 ± 0.031 |
| mixed | 0.273 ± 0.039 |
| plain_lhs | 0.273 ± 0.033 |
| sobol_seq | 0.275 ± 0.026 |
| latin_hypercube | 0.276 ± 0.026 |
| flola_voronoi | 0.276 ± 0.013 |
| cv_voronoi | 0.280 ± 0.009 |
| qbc | 0.298 ± 0.022 |
| greedy | 0.309 ± 0.012 |
| lola_voronoi | 0.311 ± 0.043 |
| lipschitz | 0.311 ± 0.085 |
| cvt | 0.334 ± 0.022 |

No strategy beat the plain Latin hypercube clearly; all errors are high (0.24 to 0.42) because 120 points are few for a
Gaussian process in 15 to 20 dimensions with a localised bump. `nipv` was best at 20 parameters and level with the
Latin hypercube at 15; FLOLA-Voronoi was the worst at 15 parameters with a large spread across seeds.

**EnergyPlus office with 15 parameters** (`examples/office_15d.py`, `docs/benchmarks/office_15d_benchmark.py`;
insulation thickness and conductivity, wall and roof absorptance, lights, ventilation, glazing gas gap and glass
conductivity, and five shading set points; 80 training simulations; 2 seeds; Gaussian process; error = mean
`nrmse` over the three objectives on 30 independent simulations; mean ± std over the two seeds; raw rows in
`office_15d.csv`):

| strategy | nrmse |
|---|---|
| qbc | 0.0326 ± 0.0047 |
| lola_voronoi | 0.0357 ± 0.0001 |
| flola_voronoi | 0.0371 ± 0.0030 |
| nipv | 0.0400 ± 0.0111 |
| latin_hypercube | 0.0458 ± 0.0122 |
| plain_lhs | 0.0464 ± 0.0020 |

Here the committee-based strategy (`qbc`) and the two LOLA-Voronoi variants were 20 to 30 % below the plain Latin
hypercube, and the gap is larger than the spread between the two seeds for those three. With two seeds this is
suggestive, not conclusive; `nipv` and the maximin Latin hypercube were level with the baseline within the spread.

## Analytic responses

Three parameters, 3 seeds, 50 training points, Gaussian-process surrogate, error = RMSE / standard deviation on 3000
independent points.

| response | FLOLA | LOLA | CV-Voronoi | NIPV | probabilistic | GP variance | maximin LHS | random | plain LHS |
|---|---|---|---|---|---|---|---|---|---|
| smooth | 0.001 | 0.001 | 0.001 | 0.002 | 0.001 | 0.001 | 0.001 | 0.002 | 0.002 |
| narrow bump | **0.134** | 0.151 | 0.151 | 0.318 | 0.701 | 0.380 | 0.525 | 0.378 | 0.228 |
| oblique step | 0.240 | 0.224 | 0.212 | **0.184** | 0.194 | 0.204 | 0.256 | 0.244 | 0.207 |

The standard deviations over seeds are in {doc}`guide/adaptive-sampling`. Tutorial 4
({doc}`tutorials/04_strategies_and_surrogates`) runs a smaller version of this comparison with its outputs stored.

## Reproduce

```bash
python docs/benchmarks/platform_benchmark.py "ROOT" --out platform_bench      # about 2.5 hours on 4 cores
python docs/benchmarks/adaptive_vs_lhs.py                                      # about 40 minutes
```
