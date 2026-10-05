# Literature

What published studies say about adaptive sampling and surrogate choice for building simulation, and how it
relates to the measurements in {doc}`benchmarks`. The five works in the first part were read in full; numbers
were extracted from the PDFs and should be spot-checked against the originals before they are quoted elsewhere.
The works in the second part were **not** read (only titles and search summaries were seen) and are listed as
pointers only.

## Read in full

| Work | Simulator | Setup | What it found |
|---|---|---|---|
| Mehrotra & Yi (2023), *Effect of adaptive intelligent sampling and machine-learning emulators in surrogate energy modeling of architectural massing*, J. Build. Eng. 72, 106614 | EnergyPlus 9.6 (via Climate Studio), one medium office, Suwon weather | 19 geometry/glazing/shading inputs; 8 non-adaptive samplers (random, LHS, Sobol, Halton, Hammersley, grid, CVT, farthest point) against 4 adaptive ones (query-by-committee, leave-one-out CV, k-fold CV, Lipschitz); 6 surrogates (RF, GPR, SVR, XGBoost, decision tree, DNN); 100 to 2000 training points | Adaptive sampling gave higher mean R² at small training sizes (up to about 300 to 400 points) and only slightly higher at large sizes; at 2000 points accuracies were similar. SVR and GPR were the best surrogates for heating, cooling and total load (highest R² 0.988); decision tree worst. Adaptive training took 40 to 9000 s against 1 to 10 s for non-adaptive. Seeds or repetitions are not reported. |
| Lahmar, Maalmi & Idchabani (2024), *Investigating adaptive sampling strategies for optimal building energy performance using artificial neural networks and kriging surrogate models*, J. Build. Eng. 82, 108341 | TRNSYS (not EnergyPlus), single zone, Morocco | 2 and 6 design variables; static LHS against error-based, LOLA-Voronoi, SampleMin and expected improvement; ANN and kriging; 100 and 400 final samples | The abstract claims 25 to 68 % fewer samples than static sampling. The paper's own tables do not clearly support a general advantage: static kriging has a lower RRSE than most adaptive variants (case 1: 0.0114; case 2: 0.1882) and the lowest cost among the 400-sample surrogate runs in case 2. One run per method, no statistics. |
| Legashev et al. (2024), *Investigation into the Hyperparameters of Error-Based Adaptive Sampling Approach for Surrogate Modeling*, Modelling 5(4), 2051-2074 | EnergyPlus 24.2, the `5ZoneAirCooled` example, Samara weather | 11 inputs (8 of them weather variables with unrealistically wide ranges); random, LHS and mixed sampling; error-based adaptive loop against static; RF, gradient boosting, LightGBM | Adaptive sampling is "preferable for a small number of iterations" once its hyperparameters (window size, sampling method, model) are tuned; for many iterations both give similar accuracy. With untuned settings (LHS, 10 % window) adaptive was not better than static random. Final tuned model MSE 7.76 (R² 0.99). No seeds, one case. |
| Elwy & Hagishima (2024), *The artificial intelligence reformation of sustainable building design approach: a systematic review on building design optimization methods using surrogate models*, Energy and Buildings 323, 114769 | review of 72 studies (mostly EnergyPlus-based) | no experiments of its own | Only 11 % of the studies used adaptive sampling; LHS was used in 40 %, Sobol in 4 %. Neural networks were 47 % of surrogates, Gaussian processes 7 %. The authors report a "lack of a robust adaptive sampling technique": in one study adaptive refinement worked "exceptionally well only during the initial iterations" and more than 1000 samples did not help. They recommend uncertainty-aware surrogates and error checks in critical regions rather than global R² alone. |
| Weber, Subramaniam & Leistner (2026), *A parametric design integrated sampling and general training approach for optimal control oriented surrogate models of light-related quantities*, J. Build. Perform. Simul. 19(4), 826-849 | Radiance | sampling of incidence directions (Fibonacci), regression on spherical harmonics | Not about adaptive against static sampling and not EnergyPlus; it only suggests, untested, that active learning at a classifier decision boundary could remove outliers. Listed because it was requested; no conclusion for bdopt can be drawn from it. |

## What this means for bdopt

* **Adaptive sampling helps at small budgets and little at large ones.** Mehrotra & Yi (with EnergyPlus) and Legashev
  et al. (with EnergyPlus, after tuning) both say so. The platform benchmark (35 points, six or seven parameters,
  EnergyPlus) is a small budget, yet no adaptive strategy beat the plain Latin hypercube there. The two
  results are not in contradiction: Mehrotra & Yi's gains are measured over 19 inputs and surrogates trained on
  100 or more points, and their strongest adaptive criteria (query-by-committee, cross-validation) are not
  among those bdopt shipped when the benchmark ran (`cv_voronoi` is the closest).
* **The claimed gains are fragile.** Two of the four experimental studies report no seeds or repetitions, and
  Lahmar et al.'s own tables undercut its abstract. bdopt's benchmark also uses one seed per test; see the caveats
  in {doc}`benchmarks`.
* **Adaptive sampling costs time.** Mehrotra & Yi report adaptive training sets that took up to 9000 s to
  build against 1 to 10 s for static ones; bdopt's per-iteration surrogate refits are the same kind of cost.
* **Surrogate choice.** SVR and Gaussian-process regression were the best thermal-load surrogates in the largest
  EnergyPlus comparison read here (Mehrotra & Yi), which agrees with the platform benchmark, where the Gaussian
  process was best on all seven tests. The review notes that neural networks dominate the literature (47 %)
  but that accuracy comparisons across model types cannot be generalised for that reason.
* **Parameter ranges matter.** Legashev et al. varied weather variables far beyond realistic ranges; their
  error levels are not comparable with design-parameter studies such as the platform's.

## Pointers, not read

Titles and search summaries only; verify before citing.

* Crombecq, Gorissen, Deschrijver & Dhaene (2011), *A novel hybrid sequential design strategy for global
  surrogate modeling of computer experiments*, SIAM J. Sci. Comput. 33(4): LOLA-Voronoi.
* van der Herten, Couckuyt, Deschrijver & Dhaene (2015), *A fuzzy hybrid sequential design strategy for global
  surrogate modeling of high-dimensional computer experiments*, SIAM J. Sci. Comput. 37(2): fuzzy LOLA-Voronoi.
* Westermann & Evins (2019), *Adaptive sampling for building simulation surrogate model derivation using the
  LOLA-Voronoi algorithm*, Building Simulation 2019 (IBPSA): LOLA-Voronoi on building models with a neural
  network surrogate, 35 inputs and 12 outputs.
* *Comparison of ANN and XGBoost surrogate models trained on small numbers of building energy simulations*,
  PLOS ONE (2024): a sequential method slightly improved on maximin LHS.
* *Efficient surrogate model development: impact of sample size and underlying model dimensions* (OSTI): Sobol
  sequences gave the best estimates at small sample sizes, and the influence of the sampling method shrank as the
  sample size grew.
* *Building energy optimization using surrogate model and active sampling*, J. Build. Perform. Simul. (2020).
