# Surrogate models

A surrogate is a fast statistical model of the simulator, trained on simulated designs. All surrogates share
one interface ({class}`~bdopt.surrogate.Surrogate`), take a table of parameters and predict the objectives.

```python
from bdopt.surrogate import GPSurrogate

model = GPSurrogate().fit(result.data, problem)          # data: parameters + objective columns
predictions = model.predict(new_designs)                  # table with one column per objective
metrics = model.score(result.test)                        # table of error metrics per objective
```

Inputs live in the unit cube and outputs are **standardised inside the model with the training data only**, so
there is nothing to scale by hand and nothing leaks from the test set. Rows with `NaN` in the training data are
dropped with a warning; at least three points are needed.

## Which model?

How each model works, step by step: {doc}`surrogate-models`.

All of them are built from a short name too: `make_surrogate("random_forest")`, `make_surrogate("rbf")`, ...
(also the `surrogate: {type: ...}` entry of a study file).

| Name | Class | Strengths | Notes |
|---|---|---|---|
| `gp` | {class}`~bdopt.surrogate.GPSurrogate` | most accurate on small, smooth data; predictive standard deviation | one Gaussian process per objective, a length scale per parameter; cost grows as $n^3$ |
| `multitask_gp` | {class}`~bdopt.surrogate.MultiTaskGPSurrogate` | objectives share one kernel and are coupled by a learned task covariance; uncertainty | PyTorch; useful when objectives are strongly correlated |
| `deep_kernel_gp` | {class}`~bdopt.surrogate.DeepKernelGPSurrogate` | a network learns the coordinates in which a GP kernel acts; uncertainty | PyTorch; for responses that are smooth only after a non-linear change of coordinates |
| `random_forest` | {class}`~bdopt.surrogate.RandomForestSurrogate` | robust, mixed and discrete parameters, uncertainty from the trees | piecewise constant |
| `extra_trees` | {class}`~bdopt.surrogate.ExtraTreesSurrogate` | like the forest, smoother; uncertainty | |
| `gradient_boosting` | {class}`~bdopt.surrogate.GradientBoostingSurrogate` | strong on larger data sets | one model per objective |
| `svr` | {class}`~bdopt.surrogate.SVRSurrogate` | RBF support vector regression, `C` and `gamma` by cross-validation | |
| `kernel_ridge` | {class}`~bdopt.surrogate.KernelRidgeSurrogate` | RBF or Laplacian kernel, regularisation by cross-validation | |
| `knn` | {class}`~bdopt.surrogate.KNNSurrogate` | baseline | cannot interpolate smoothly |
| `polynomial` | {class}`~bdopt.surrogate.PolynomialSurrogate` | the classical response surface; degree 1 to 3 and ridge by cross-validation | needs few points; coefficients are interpretable |
| `rbf` | {class}`~bdopt.surrogate.RBFSurrogate` | radial basis function interpolation (thin-plate, cubic, multiquadric) | almost exact at the training points |
| `mlp` | {class}`~bdopt.surrogate.MLPSurrogate` | cheap neural network | scikit-learn `MLPRegressor` with a grid search over size and regularisation |
| `torch_mlp` | {class}`~bdopt.surrogate.TorchMLP` | flexible, many points, ensembles for uncertainty | PyTorch |
| `mc_dropout_mlp` | {class}`~bdopt.surrogate.MCDropoutMLP` | network with a predictive standard deviation from dropout (an approximate Bayesian network) | PyTorch |
| `auto` | {class}`~bdopt.surrogate.AutoSurrogate` | cross-validates a list of candidates and keeps the best | `selected_` and `cv_scores_` tell what it chose |

On the platform's seven tests (35 training simulations, error on 25 independent simulations, mean nrmse) the
Gaussian process was best on every test (0.020), then the multi-task GP (0.039), the MC-dropout network (0.041), the
deep-kernel GP (0.044), RBF interpolation (0.046), kernel ridge (0.048), SVR, MLP, gradient boosting, PyTorch MLP,
polynomial, extra trees, random forest and k-NN (0.139); `auto` picked the GP every time. Full table in the README.

For a few dozen to a few hundred simulations start with the Gaussian process, or let `auto` decide. Compare models
on the same held-out simulations rather than by intuition; {doc}`../tutorials/02_office_building` and the
measurements on the platform tests (see the README) do exactly that.

### Gaussian process

```python
GPSurrogate(kernels=("matern52",), n_restarts=8, random_state=0, max_extra_restarts=24)
```

* `kernels`: any of `matern52`, `matern32`, `rbf`, `rq`; with several, the best marginal likelihood wins per
  objective.
* The length scales of the Matérn and RBF kernels (one per parameter) are bounded to `[0.05, 100]` in the unit cube. Without this lower bound the likelihood
  optimisation sometimes lands on a degenerate "white noise" solution that predicts the mean; if every length
  scale still ends up at the bound, the fit is retried with up to `max_extra_restarts` more restarts.
* {meth}`~bdopt.surrogate.Surrogate.predict_unit_std` returns the predictive standard deviation.

### PyTorch MLP

```python
TorchMLP(hidden=(32, 64, 32), activation="relu", lr=1e-3, weight_decay=1e-5,
         epochs=3000, patience=200, val_fraction=0.2, n_ensemble=1, seed=0)
```

Full-batch Adam with early stopping on a validation split (used when there are at least 10 points).
`n_ensemble > 1` trains several networks with different seeds; the mean is the prediction and the spread is the
uncertainty. The default architecture is a 32-64-32 ReLU network. Training runs on one thread, which is about ten
times faster than the default for networks this small. Results are reproducible for a fixed `seed`.

### Choosing automatically

```python
from bdopt.surrogate import AutoSurrogate

model = AutoSurrogate(cv=5).fit(result.data, problem)
model.selected_, model.cv_scores_     # the winner and every candidate's cross-validated error
```

Pass `candidates=[...]` to restrict or extend the list.

## Metrics

{meth}`~bdopt.surrogate.Surrogate.score` returns one row per objective:

| Column | Definition |
|---|---|
| `rmse` | root mean squared error, in the units of the objective |
| `rmse_pct` | `rmse` as a percentage of the mean of the true values |
| `cv_rmse` | the same quantity as `rmse_pct` (no degrees-of-freedom correction) |
| `nrmse` | `rmse` divided by the range of the true values |
| `nmbe` | normalised mean bias error, % (`sum(error) / ((n - 1) * mean)`) |
| `mae` | mean absolute error |
| `r2` | coefficient of determination |

With few test points `r2` can be strongly negative for an objective that barely varies; read it together with
`rmse_pct`.

```python
model.cross_validate(data, problem, k=5)       # k-fold metrics; a fresh model is trained per fold
```

## Saving and loading

```python
model.save("surrogate.pkl")
from bdopt.surrogate import Surrogate
model = Surrogate.load("surrogate.pkl")
```

A problem that contains local functions cannot be pickled; in that case `save` leaves it out and you pass it back with
`Surrogate.load("surrogate.pkl", problem=problem)`.

## Writing your own

Subclass {class}`~bdopt.surrogate.Surrogate` and implement `_fit(U, Z)` and `_predict(U)` on standardised
outputs (and optionally `_predict_std`). Set `has_uncertainty = True` if you provide a standard deviation, which
enables the `gp_variance` sampling method. A surrogate can be passed to the optimizers directly (it has
`evaluate_unit`).
