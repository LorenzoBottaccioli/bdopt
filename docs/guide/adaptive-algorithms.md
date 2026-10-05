# Adaptive sampling algorithms, step by step

Every strategy of {class}`~bdopt.adaptive.AdaptiveSampler` answers one question: *given the simulations I already
have, where should the next one go?* All of them work in the unit cube (every parameter scaled to $[0, 1]$,
discrete ones snapped to valid values) and never propose a design that already exists or that failed before.
The code is in `bdopt.adaptive.designs`; the loop around it is described in {doc}`adaptive-sampling`.

Notation: $X$ are the $n$ samples simulated so far ($n \times d$), $Y$ their objectives standardised with the
standard deviation of the initial samples, $d$ the number of parameters.

## The building block: Monte-Carlo Voronoi cells

Used by `voronoi`, `lola_voronoi`, `flola_voronoi` and `cv_voronoi`.

1. Draw $M = \mathrm{clip}(200 n, 5000, 80000)$ random points in the cube.
2. Assign each to its nearest sample. The cell of sample $i$ is the set of points assigned to it.
3. The **relative volume** $V_i$ is the fraction of random points in the cell of $i$: a big cell is a region far
   from every simulation.
4. To place a new point in cell $i$, take the random point of that cell **farthest from sample $i$** (the point of
   the cell least covered by anything).

## `voronoi`: exploration only

1. Compute the cells and volumes.
2. Take the cell with the largest volume.
3. New point: the farthest point of that cell from its sample. For a batch, repeat with the next cells.

It fills gaps and tends towards the corners of the cube, because corners are far from everything. It ignores the
response entirely.

## `lola_voronoi`: exploration + local nonlinearity (Crombecq et al. 2011)

Exploitation asks: *around which sample is the response far from a plane?* A nonlinear region needs more samples.

1. **Neighbourhood.** For sample $i$ pick $2d$ neighbours (a cross-polytope has $2d$ vertices). Among candidate
   sets, the best one maximises the *cross-polytope ratio* divided by the *cohesion*:
   * cohesion $= $ mean distance from the neighbours to $i$ (small = close);
   * adhesion $= $ mean over neighbours of the distance to the nearest *other* neighbour (large = spread out);
   * ratio $= \text{adhesion} / (\sqrt{2}\cdot\text{cohesion})$, which is 1 for a perfect cross-polytope;
   * score $= \text{ratio} / \text{cohesion}$.

   The paper enumerates every subset of $2d$ samples. That is $\binom{n}{2d}$ subsets ($10^{12}$ for 60 samples in 6
   dimensions), so bdopt restricts the search to the nearest samples, as many as keep it below 3000 subsets; "not
   too far" is also what the paper recommends.
2. **Local plane.** Least-squares fit of a hyperplane through sample $i$ to its neighbours (one per objective).
3. **Nonlinearity** $E_i = \sum_{\text{neighbours, objectives}} |y_j - (y_i + g\cdot(x_j - x_i))|$: how far the
   neighbours are from the plane.
4. **Hybrid score** $H_i = V_i + E_i / \sum_k E_k$ (volume plus normalised nonlinearity).
5. New point: farthest point of the Voronoi cell of the sample with the largest $H_i$. Batches take the top
   samples.

## `flola_voronoi`: the fuzzy version (van der Herten et al. 2015), the default

Same hybrid score, but the neighbourhood is variable and neighbours are weighted by a fuzzy system.

1. **Neighbourhood.** With $K = 4d$, $\alpha = \frac{2}{K}\sum$ (the $K$ nearest distances of sample $i$). The
   neighbours are all samples closer than $\alpha$ (at least $d$, taking the nearest if fewer).
2. **Two numbers per neighbour:** *cohesion* $c$ = its distance to $i$; *adhesion* $a$ = its distance to *its own*
   nearest sample. Both are divided by their largest value in the neighbourhood.
3. **Fuzzy inference** (Mamdani, `fuzzy_weights`):

   | membership | function |
   |---|---|
   | close | $1 / (1 + e^{10 (c - 0.5)})$ |
   | far | $1 - \text{close}$ |
   | isolated | $\exp(-(a-1)^2 / (2 \cdot 0.27^2))$ |
   | clustered | $\exp(-a^2 / (2 \cdot 0.3^2))$ |

   | rule | weight |
   |---|---|
   | close **and** isolated | high |
   | close and clustered | average |
   | far and isolated | average |
   | far and clustered | low |

   AND is the minimum; the output sets are triangles over $[0, 1]$ (low $0, 0, 0.13$; average $0.16, 0.33, 0.5$;
   high $0.6, 1.01, 1.01$); the rules are aggregated with the maximum and defuzzified by the centroid. A neighbour
   that is close to $i$ and not crowded by other samples counts most.
4. **Weighted plane:** weighted least squares with those weights, then the weighted deviation $E_i$.
5. $H_i = V_i + E_i / \sum E$ and the new point as in `lola_voronoi`.

## `cv_voronoi`: refine where the surrogate is wrong (Xu 2014; Liu 2016; Kaminsky 2021)

1. 5-fold cross-validation: five times, fit the surrogate without one fifth of the samples and predict that fifth.
   Every sample gets an error $e_{ij}$ for each objective $j$ from a model that never saw it.
2. Objective weights proportional to each objective's cross-validated NRMSE (a badly predicted objective counts
   more, Liu et al.).
3. Score of a sample: $\sum_j w_j e_{ij}$.
4. New point: farthest point of the Voronoi cell of the worst sample.

It costs five surrogate fits per iteration.

## `probabilistic`: largest predictive uncertainty (Chai 2019)

Needs a surrogate with a standard deviation $s(x)$ (Gaussian process, random forest, ...).

1. Maximise the mean over objectives of $s_j(x) / \text{std}(Y_j)$ over the whole cube with differential evolution
   (population 10 per dimension, up to 60 generations).
2. New point: the maximiser. For a batch, each point after the first is penalised close to those already chosen
   (factor $\min(1, \text{dist} / (0.1\sqrt{d}))$).

The most uncertain places are usually the boundary of the cube, so on its own this tends to pile points there.

## `gp_variance`: the same, cheap

Evaluate $s(x)$ on 4000 random candidates, take the best, penalise the neighbourhood of what was taken for the
next point of the batch. Faster than differential evolution, less exact.

## `nipv`: negative integrated posterior variance

Needs a {class}`~bdopt.surrogate.GPSurrogate`. Instead of "where is the model most uncertain", it asks "which
point, once simulated, lowers the uncertainty *averaged over the whole domain* the most".

1. Draw 256 Monte-Carlo points $m$ (the domain) and 3000 candidates $c$.
2. For a Gaussian process the posterior variance does not depend on the observed values. Adding candidate $c$
   reduces the variance at $m$ by $\mathrm{cov}(m, c)^2 / (\mathrm{var}(c) + \sigma_n^2)$, with the posterior
   covariance computed from the fitted kernel.
3. Score of $c$ = mean of that reduction over $m$, averaged over objectives. Take the best candidate.
4. For a batch: add $c$ to the training inputs (its value is not needed), recompute, repeat.

Unlike `probabilistic` it does not push to the boundary, because a point there reduces little variance elsewhere.

## `latin_hypercube`: space filling, not adaptive

Each iteration: draw 50 random Latin hypercubes of `batch` points and keep the one whose smallest distance to
the already taken points (including the new ones) is largest (maximin).

## `random`

Uniform random points in the cube: the baseline against which anything else must be judged.

## Criteria from the building literature

These live in `bdopt.adaptive.criteria`. Most work on a pool of 3000 random candidates in the unit cube (snapped to
valid discrete values) and pick the best-scoring ones; for a batch, scores close to a point already chosen are damped
(factor $\min(1, \text{dist}/(0.1\sqrt{d}))$) so the points spread out.

### `qbc`: query by committee (Mehrotra & Yi 2023; Bamdad 2020; Nguyen & Nghiem 2026)

1. Fit 5 copies of the surrogate on bootstrap resamples of the samples (a point may appear several times, others not at
   all).
2. Predict the pool with every copy.
3. Score = standard deviation across the copies, divided by the standard deviation of each objective, averaged over
   objectives. Take the pool point where the committee disagrees most.

Works with any surrogate, with or without its own uncertainty. Costs 5 fits per iteration.

### `loocv`: leave-one-out error (Mehrotra & Yi 2023)

1. Leave-one-out cross-validation (a 20-fold one when there are more than 20 samples): every sample gets the error of a
   model that did not see it.
2. Interpolate those errors over the pool by inverse-distance weighting ($w = 1/d^2$).
3. Score = interpolated error times the distance to the nearest sample (an error-weighted maximin). Take the best.

### `lipschitz`: Lipschitz criterion (Mehrotra & Yi 2023)

1. For every sample, the largest slope $|y_i - y_k| / \|x_i - x_k\|$ towards its 5 nearest samples (summed over the
   standardised objectives): a local Lipschitz constant $L_i$.
2. For a pool point, take its nearest sample $i$; score = $L_i \times$ distance to that sample, the largest change the
   response could have there.

Cheap: no surrogate is needed.

### `idw`: inverse-distance-weighted exploration (Bemporad 2021; Nguyen & Nghiem 2026)

1. With weights $w_i(x) = 1/\|x - x_i\|^2$, normalised, the IDW deviation is
   $s(x) = \sqrt{\sum_i w_i (y_i - \hat y(x))^2}$, where $\hat y$ is the surrogate's prediction: large where the data around
   $x$ disagree with the model.
2. Exploration term $z(x) = \frac{2}{\pi}\arctan(1/\sum_i w_i)$, which is large far from every sample.
3. Score = $s + \delta z$ ($\delta = 1$).

### `greedy` (farthest point) and `cvt`

* `greedy`: the pool point (5000) whose distance to the nearest sample is largest (maximin). Nguyen & Nghiem found it
  never beat random sampling for a GP.
* `cvt`: keep the 25 % of 8000 Monte-Carlo points farthest from every sample and run k-means with `batch` clusters; the
  centroids are the new points. Existing samples cannot move, so this is a CVT-inspired addition, not a full centroidal
  tessellation (`bdopt.sampling.cvt` builds a real one for a fixed design).

### `fisher_information`: D-optimal for a Gaussian process (Nguyen & Nghiem 2026)

Needs a fitted `GPSurrogate`. For hyper-parameters $\theta$ of the kernel, the Fisher information of the data is
$I_{ij} = \tfrac12 \mathrm{tr}(K^{-1} \partial_i K\, K^{-1} \partial_j K)$. For each of 200 candidates, append it to the
samples, recompute $\log\det I$ with the kernel gradients of the fitted model, and take the candidate that raises it
most (averaged over objectives): the point that tells the most about the length scales and the noise.

### Optimisation-oriented: `expected_improvement`, `probability_improvement`, `lcb`, `ucb`, `sample_min`

For problems where finding good designs matters more than a global model (Lahmar 2024; Bamdad 2020; Nguyen & Nghiem
2026).

1. Draw random weights $w$ (a Dirichlet draw, redrawn every iteration as in ParEGO) and scalarise the *standardised*
   objectives, maximised ones with the opposite sign, so lower is better. The mean is $\sum_j w_j \mu_j$ and the
   deviation $\sqrt{\sum_j w_j^2 s_j^2}$.
2. Score each pool point: `sample_min` = $-\mu$ (the surrogate's minimum); `lcb` = $-(\mu - 2s)$; `ucb` = $\mu + 2s$
   (uncertainty seeking); `expected_improvement` = $s\,[z\Phi(z) + \phi(z)]$ and `probability_improvement` = $\Phi(z)$ with
   $z = (\text{best so far} - \mu)/s$.
3. Take the best; EI, PI, LCB and UCB need a surrogate with uncertainty.

### `soas`: committee with a low-value filter (Bamdad et al. 2020)

1. Build a bootstrap committee as in `qbc`; the pool's committee-mean value is scalarised as above.
2. Keep the pool points whose mean is below the $p$-th percentile; $p$ starts at 50 % and falls by 5 points per
   iteration to 1 %.
3. Among those, take the largest committee variance: uncertain *and* promising.

### Sequences and mixes: `sobol_seq`, `halton`, `mixed`, `error_window`

* `sobol_seq`, `halton`: continue a scrambled low-discrepancy sequence from position $n$ (the number of points
  already taken), so the full design is the sequence's first $n + k$ points (apart from duplicates after snapping).
* `mixed` (Legashev et al. 2024): round robin by iteration: random, Latin hypercube, Sobol' sequence. Their full
  factorial is left out because it piles points in the corners.
* `error_window` (Legashev et al. 2024): take the held-out test point (or, without a test set, the training point)
  with the largest scaled prediction error and draw uniform points in a box of relative width 0.3 around it.

### Fixed designs (`bdopt.sampling`)

Besides `lhs`, `sobol`, `random` and `factorial`: `halton`, `hammersley` (needs the final size, not extendable),
`maximin_lhs` (best of 200 Latin hypercubes by the smallest pairwise distance), `fps` (farthest-point sampling) and
`cvt` (k-means centroids of uniform points).

## Which one?

* Unknown response, want something safe: `flola_voronoi` (default) or `cv_voronoi`.
* The response has a localised feature (threshold, switch): the Voronoi/LOLA family. Measured gains: see
  {doc}`adaptive-sampling`.
* Smooth responses (insulation, lighting, ventilation, as in the platform's seven tests): a plain Latin hypercube
  was as good; use the loop for its stopping rule.
* Avoid `probabilistic`, `gp_variance` and `greedy` as the only criterion.
* Optimising a design rather than modelling the response: the acquisition functions above.
