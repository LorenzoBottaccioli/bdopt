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
objectives, of a Gaussian process (adaptive part) or of the named surrogate (surrogate part). One seed per test:
differences below about 0.005 are noise.

## Adaptive strategies

| method | test 1 | test 2 | test 3 | test 4 | test 5 | test 6 | test 7 | mean |
|---|---|---|---|---|---|---|---|---|
| plain_lhs | 0.032 | 0.015 | 0.016 | 0.029 | 0.008 | 0.027 | 0.014 | 0.020 |
| nipv | 0.032 | 0.018 | 0.015 | 0.026 | 0.007 | 0.035 | 0.016 | 0.021 |
| latin_hypercube | 0.030 | 0.019 | 0.015 | 0.026 | 0.008 | 0.033 | 0.024 | 0.022 |
| flola_voronoi | 0.039 | 0.021 | 0.012 | 0.022 | 0.007 | 0.038 | 0.017 | 0.022 |
| cv_voronoi | 0.037 | 0.019 | 0.015 | 0.020 | 0.009 | 0.044 | 0.015 | 0.023 |
| lola_voronoi | 0.040 | 0.019 | 0.019 | 0.033 | 0.008 | 0.057 | 0.017 | 0.027 |
| probabilistic | 0.056 | 0.025 | 0.019 | 0.063 | 0.012 | 0.057 | 0.024 | 0.036 |

No adaptive strategy beat the plain Latin hypercube; `probabilistic` is clearly worse. The responses of these
models are smooth (see {doc}`guide/adaptive-sampling`).

## Surrogate models

Trained on the 35-point Latin hypercube, error on the 25 independent simulations.

| surrogate | test 1 | test 2 | test 3 | test 4 | test 5 | test 6 | test 7 | mean |
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

R² (mean over tests and objectives) is 0.989 for the Gaussian process and for `auto` (which selects it on every
test), 0.745 for k-NN. Fit times:

| surrogate | mean fit time (s) |
|---|---|
| rbf | 0.01 |
| knn | 0.02 |
| polynomial | 0.05 |
| extra_trees | 0.15 |
| random_forest | 0.23 |
| svr | 0.28 |
| kernel_ridge | 0.28 |
| gradient_boosting | 0.37 |
| torch_mlp | 0.64 |
| deep_kernel_gp | 0.69 |
| mc_dropout_mlp | 0.81 |
| gp | 0.84 |
| multitask_gp | 1.63 |
| mlp | 2.04 |
| auto | 18.81 |

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
