# Surrogate models, one by one

All surrogates share the interface of {class}`~bdopt.surrogate.Surrogate`: parameters in the unit cube,
objectives standardised inside the model with the training data, `fit(table, problem)`, `predict(table)`,
`score(table)`. Build one by name with `make_surrogate("rbf")`. Overview and measurements: {doc}`surrogates`.
"Uncertainty" means `predict_unit_std` returns a predictive standard deviation (needed by `probabilistic`,
`gp_variance`).

## Gaussian-process family

**`gp` {class}`~bdopt.surrogate.GPSurrogate`** (uncertainty). One Gaussian process per objective with kernel
$\sigma^2 k(x, x')$ + noise, where $k$ is Matérn-5/2 (also `matern32`, `rbf`, `rq`) with **one length scale per
parameter** (ARD: an unimportant parameter gets a long length scale). Hyper-parameters maximise the marginal
likelihood from 8 random starts; length scales are bounded below by 0.05 because the likelihood otherwise has a
degenerate optimum ("white noise", predicts the mean), and the fit is retried with more restarts if it lands
there. Prediction: posterior mean and standard deviation. Cost $O(n^3)$.

**`multitask_gp` {class}`~bdopt.surrogate.MultiTaskGPSurrogate`** (uncertainty, PyTorch). Intrinsic coregionalisation:
the covariance between objective $a$ at $x$ and objective $b$ at $x'$ is $B_{ab}\,k(x, x')$ with a shared ARD kernel
and a learned task matrix $B = LL^\top + \mathrm{diag}(\kappa)$. Objectives that move together share information.
Trained by L-BFGS on the joint marginal likelihood (dense Cholesky of the $mn \times mn$ matrix), best of 3 starts.

**`deep_kernel_gp` {class}`~bdopt.surrogate.DeepKernelGPSurrogate`** (uncertainty, PyTorch). A small network
$g: \mathbb{R}^d \to \mathbb{R}^4$ (tanh, hidden 32-16) maps the parameters to features, and the GP kernel acts on
$g(x)$. Network weights, length scales, one output scale and one noise per objective are trained together
(Adam, 300 epochs) on the marginal likelihood. Helps when the response is smooth only after a nonlinear change
of coordinates; with few points it can overfit.

## Tree ensembles

**`random_forest` {class}`~bdopt.surrogate.RandomForestSurrogate`** and **`extra_trees`
{class}`~bdopt.surrogate.ExtraTreesSurrogate`** (uncertainty). 300 regression trees on bootstrap samples (random
forest) or with random split thresholds (extra trees, smoother); prediction = mean of the trees, uncertainty =
their standard deviation. Piecewise constant: robust to discrete parameters, coarse on smooth responses with few
points.

**`gradient_boosting` {class}`~bdopt.surrogate.GradientBoostingSurrogate`.** Depth-3 trees added one at a time to
fit the residual (300 trees, learning rate 0.05, subsample 0.8), one model per objective. No uncertainty.

## Kernel and distance methods

**`svr` {class}`~bdopt.surrogate.SVRSurrogate`.** Support vector regression with an RBF kernel and an
$\epsilon$-insensitive loss, one model per objective; `C` (1 to 1000) and `gamma` chosen by 3-fold cross-validation.

**`kernel_ridge` {class}`~bdopt.surrogate.KernelRidgeSurrogate`.** Ridge regression in the feature space of an RBF
or Laplacian kernel; kernel, `alpha` (1e-6 to 1) and `gamma` chosen by 3-fold cross-validation. Like a GP posterior
mean without the uncertainty.

**`rbf` {class}`~bdopt.surrogate.RBFSurrogate`.** Radial-basis-function interpolation (`scipy.interpolate.RBFInterpolator`):
the prediction is a weighted sum of kernels centred on the samples, with a polynomial tail; it passes (almost)
through the training points. Kernel (thin-plate, cubic, multiquadric) and smoothing chosen by 5-fold
cross-validation. Cheap (milliseconds).

**`knn` {class}`~bdopt.surrogate.KNNSurrogate`.** Distance-weighted mean of the $k$ nearest samples ($k$ from
2, 3, 5, 8 by cross-validation). A baseline: it cannot extrapolate or interpolate smoothly.

## Regression models

**`polynomial` {class}`~bdopt.surrogate.PolynomialSurrogate`.** Polynomial of degree 1 to 3 in all parameters
(including interactions) with ridge regularisation; the degree and the ridge strength are chosen by 5-fold
cross-validation, and degrees with far more terms than samples are skipped. The classical response surface;
coefficients are interpretable.

## Neural networks

**`mlp` {class}`~bdopt.surrogate.MLPSurrogate`.** scikit-learn `MLPRegressor`, grid search over layer sizes
($16d$ or $16d, 16d$) and L2 strength, 3-fold cross-validation.

**`torch_mlp` {class}`~bdopt.surrogate.TorchMLP`** (PyTorch). Multi-output MLP (32-64-32, ReLU), full-batch Adam,
early stopping on a validation split. `n_ensemble > 1` trains several networks with different seeds; mean =
prediction, spread = uncertainty.

**`mc_dropout_mlp` {class}`~bdopt.surrogate.MCDropoutMLP`** (uncertainty, PyTorch). An MLP (64-64, SiLU) with dropout
0.1 that stays active at prediction time; 50 stochastic passes give a mean and a standard deviation (Monte-Carlo
dropout, an approximate Bayesian neural network).

## More models

**`decision_tree` {class}`~bdopt.surrogate.DecisionTreeSurrogate`.** One regression tree, depth (3 to 12) and leaf size
chosen by 3-fold cross-validation. The weakest baseline.

**`hist_gradient_boosting` {class}`~bdopt.surrogate.HistGradientBoostingSurrogate`.** scikit-learn histogram boosting
(300 rounds, depth 4, learning rate 0.05), one model per objective; fast on many points.

**`adaboost` {class}`~bdopt.surrogate.AdaBoostSurrogate`.** AdaBoost.R2 over depth-3 trees (200 rounds).

**`lightgbm` {class}`~bdopt.surrogate.LightGBMSurrogate`, `xgboost` {class}`~bdopt.surrogate.XGBoostSurrogate`.**
The two boosting libraries used in the literature (Legashev et al. found LightGBM best). Install with
`pip install "bdopt[ml]"`.

**`lasso_polynomial` {class}`~bdopt.surrogate.LassoPolynomialSurrogate`.** Monomials of degree up to 3 involving at most
two parameters (so the term count stays manageable with 15 parameters), elastic-net penalty by 5-fold cross-validation:
a sparse response surface.

**`pce` {class}`~bdopt.surrogate.PCESurrogate`.** Sparse polynomial chaos: orthonormal Legendre polynomials on the unit
cube (same truncation), coefficients by least-angle regression with the number of terms chosen by cross-validation.
`first_order_indices_` gives first-order Sobol' sensitivity indices straight from the coefficients.

**`elm` {class}`~bdopt.surrogate.ELMSurrogate`** (uncertainty). Extreme learning machine: a random fixed tanh layer of 200
units and ridge-regressed output weights; 10 independent machines give a mean and a spread.

**`universal_kriging` {class}`~bdopt.surrogate.UniversalKrigingSurrogate`** (uncertainty). A ridge-regressed linear trend
plus a Gaussian process on the residuals; extrapolates better than a zero-mean GP when the response has a global trend.

**`bagging` {class}`~bdopt.surrogate.BaggingSurrogate`** (uncertainty). 10 bootstrap copies of any base surrogate (kernel ridge
by default); mean and spread give uncertainty to models that have none, which also enables `probabilistic`, `gp_variance`
and the acquisition methods.

**`stacking` {class}`~bdopt.surrogate.StackingSurrogate`.** Out-of-fold predictions of GP, kernel ridge, RBF and SVR
combined with non-negative least-squares weights per objective, bases refitted on all data.

**`deep_ensemble` {class}`~bdopt.surrogate.DeepEnsembleMLP`** (uncertainty, PyTorch). Five independently initialised
64-64 networks; mean and spread.

**`sparse_gp` {class}`~bdopt.surrogate.SparseGPSurrogate`** (uncertainty, PyTorch). Variational sparse GP (Titsias): 50
inducing points (k-means centres), hyper-parameters by L-BFGS on the collapsed bound, cost $O(nm^2)$; equals the exact
GP when there are no more samples than inducing points.

## Automatic choice

**`auto` {class}`~bdopt.surrogate.AutoSurrogate`.** Cross-validates (5-fold) a list of candidates (default: GP,
extra trees, kernel ridge, SVR, polynomial, RBF, MLP), keeps the one with the lowest RMSE and refits it on all the
data. `selected_` and `cv_scores_` show the outcome. It costs one cross-validated fit of every candidate.
