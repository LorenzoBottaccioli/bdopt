# Surrogate models

A surrogate is a fast statistical model of the simulator, trained on simulated designs. All surrogates share
one interface ({class}`~bdopt.surrogate.Surrogate`), take a table of parameters and predict the objectives.

```python
from bdopt.surrogate import GPSurrogate, MLPSurrogate, TorchMLP

model = GPSurrogate().fit(result.data, problem)          # data: parameters + objective columns
predictions = model.predict(new_designs)                  # table with one column per objective
metrics = model.score(result.test)                        # table of error metrics per objective
```

Inputs live in the unit cube and outputs are **standardised inside the model with the training data only**, so
there is nothing to scale by hand and nothing leaks from the test set. Rows with `NaN` in the training data are
dropped with a warning; at least three points are needed.

## Which model?

| Model | Strengths | Notes |
|---|---|---|
| {class}`~bdopt.surrogate.GPSurrogate` | most accurate on small data (tens of points), smooth responses; gives a predictive standard deviation | one Gaussian process per objective with a length scale per parameter; cost grows as $n^3$, fine up to a few hundred points |
| {class}`~bdopt.surrogate.MLPSurrogate` | cheap, no extra dependency | scikit-learn `MLPRegressor` with a small grid search over size and regularisation |
| {class}`~bdopt.surrogate.TorchMLP` | flexible, many points, ensembles for uncertainty | PyTorch (`pip install "bdopt[torch]"`) |

For a few dozen to a few hundred simulations start with the Gaussian process. Compare models on the same
held-out simulations rather than by intuition; {doc}`../tutorials/02_office_building` does exactly that.

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

A problem that contains lambdas cannot be pickled; in that case `save` leaves it out and you pass it back with
`Surrogate.load("surrogate.pkl", problem=problem)`.

## Writing your own

Subclass {class}`~bdopt.surrogate.Surrogate` and implement `_fit(U, Z)` and `_predict(U)` on standardised
outputs (and optionally `_predict_std`). Set `has_uncertainty = True` if you provide a standard deviation, which
enables the `gp_variance` sampling method. A surrogate can be passed to the optimizers directly (it has
`evaluate_unit`).
