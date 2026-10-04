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

## Which one?

* Unknown response, want something safe: `flola_voronoi` (default) or `cv_voronoi`.
* The response has a localised feature (threshold, switch): the Voronoi/LOLA family. Measured gains: see
  {doc}`adaptive-sampling`.
* Smooth responses (insulation, lighting, ventilation, as in the platform's seven tests): a plain Latin hypercube
  was as good; use the loop for its stopping rule.
* Avoid `probabilistic` and `gp_variance` as the only criterion.
