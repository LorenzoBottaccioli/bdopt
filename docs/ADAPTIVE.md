# Adaptive sampling

A fixed design (e.g. 60 Latin-hypercube simulations) has to guess how many simulations are enough.
`AdaptiveSampler` starts from a small Latin hypercube plus a held-out test set, then adds simulations where the
surrogate needs them and stops when the test error is small enough (or a budget is reached).

```python
sampler = AdaptiveSampler(problem, evaluator, method="flola_voronoi", surrogate=GPSurrogate(), seed=0)
result = sampler.run(n_initial=20, n_test=5, max_iter=40, batch=1, tol=0.05, max_evaluations=80)
result.data      # training points (parameters + objectives, real units)
result.history   # per iteration: n_train, scaled_rmse, per-objective error and R2
```

* **Stopping metric** `scaled_rmse`: RMSE on the test points in units of the standard deviation of each objective
  over the initial samples (fixed from the start, as in the harlow notebooks), averaged over objectives.
  `n_test=0` uses 5-fold cross-validation instead of a test set.
* **Methods.** `flola_voronoi` (default), `lola_voronoi`: Voronoi-cell volume (exploration, Monte-Carlo estimate)
  plus deviation of the response from a local linear fit (exploitation); the new point goes into the best cell,
  at the spot farthest from its sample. `voronoi`: exploration only. `gp_variance`: largest predictive
  uncertainty. `random`: baseline.
* **Discrete parameters** (categorical, integer) are handled in the unit cube and snapped to valid values; a
  point is never proposed twice.
* **Failures:** a failed simulation is skipped and never proposed again.
* **Resuming:** everything is deterministic given the seed, and the evaluator caches finished simulations, so
  re-running an interrupted study repeats nothing. `checkpoint_dir` also writes `train.csv`, `test.csv` and
  `history.csv` at every iteration.

## When does it help?

See the table in the README. In short: **localised nonlinearity** (a narrow bump) gave 37 % lower error than LHS
for the same number of training points; **smooth** responses are equally well served by any design; an
**oblique step** showed no gain. Use the convergence history to decide how many simulations you need rather than
to expect a better model from the same budget on smooth problems.

## Why a Gaussian process with restarts

During development the GP's marginal-likelihood optimisation, started from few random points, sometimes
collapsed to a "white noise" solution (all length scales at the lower bound) that predicts the mean, giving
errors around 1.0 for some Latin-hypercube designs and 0.1 for others. `GPSurrogate` therefore bounds the length
scales to what makes sense in the unit cube, restarts 8 times, and retries more if it detects the collapsed
solution (`tests/test_surrogate.py::test_gp_not_degenerate_across_lhs_designs`).
Any earlier comparison between adaptive and plain designs made with the unstable fit was meaningless.

## harlow

`HarlowAdaptiveSampler(problem, evaluator, method="FuzzyLolaVoronoi")` runs harlow's own sampler on a bdopt
problem (bounds, scaling, categorical handling and the `target_function` are generated). harlow 0.1.0 needs
TensorFlow at import; with current releases you also need `pip install tf_keras networkx` and
`TF_USE_LEGACY_KERAS=1` (set automatically). It is tested here (`tests/test_harlow_bridge.py`) but it is not required
for anything else in bdopt.
