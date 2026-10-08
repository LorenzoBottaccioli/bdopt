# Literature

What published studies say about adaptive sampling and surrogate choice for building simulation, and how it
relates to the measurements in {doc}`benchmarks`. The works in the first part were read in full; numbers
were extracted from the PDFs and should be spot-checked against the originals before they are quoted elsewhere.
The works in the second part were **not** read (only titles and search summaries were seen) and are listed as
pointers only.

## Read in full

| Work | Simulator | Setup | What it found |
|---|---|---|---|
| Mehrotra & Yi (2023), *Effect of adaptive intelligent sampling and machine-learning emulators in surrogate energy modeling of architectural massing*, J. Build. Eng. 72, 106614 | EnergyPlus 9.6 (via Climate Studio), one medium office, Suwon weather | 19 geometry/glazing/shading inputs; 8 non-adaptive samplers (random, LHS, Sobol, Halton, Hammersley, grid, CVT, farthest point) against 4 adaptive ones (query-by-committee, leave-one-out CV, k-fold CV, Lipschitz); 6 surrogates (RF, GPR, SVR, XGBoost, decision tree, DNN); 100 to 2000 training points | Adaptive sampling gave higher mean R² at small training sizes (up to about 300 to 400 points) and only slightly higher at large sizes; at 2000 points accuracies were similar. SVR and GPR were the best surrogates for heating, cooling and total load (highest R² 0.988); decision tree worst. Adaptive training took 40 to 9000 s against 1 to 10 s for non-adaptive. Seeds or repetitions are not reported. |
| Westermann & Evins (2019), *Adaptive sampling for building simulation surrogate model derivation using the LOLA-Voronoi algorithm*, Proc. 16th IBPSA Conference (Building Simulation 2019), Rome, 1559-1563 | not named in the text; the case is the DOE small office reference building (90.1-2004), five zones | 4 and 8 design parameters (screened by Morris), one output (annual energy), one LOLA-Voronoi variant against static LHS, neural network, 5-fold CV R² on 100 random test samples, 20 repetitions | "None of the two sampling schemes clearly outperforms the other." Adaptive sampling was more sampling-efficient with 4 parameters (higher mean R², narrower band; both exceeded R² of about 0.95) and the difference "appears to vanish" with 8. The authors say more experiments are needed to confirm. No tables of R² or sample savings; values are read off plots. |
| Bamdad, Cholette & Bell (2020), *Building energy optimization using surrogate model and active sampling*, J. Build. Perform. Simul. 13(6), 760-776 | EnergyPlus 8.1.0, three-storey office, Brisbane and Melbourne | 15 continuous design variables; surrogate-based optimisation with committee-based active sampling (SOAS, L-SOAS) against random sampling and simulation-based PSO and ACOR; ANN committee; 15 runs per method, 2000 simulations | Active sampling beat random sampling early and L-SOAS was the best surrogate variant, but the advantage was "lost" beyond about 1000 labelled samples and simulation-based ACOR was slightly better at the end. Best-of-15 energy in Brisbane: 630.60 (random), 630.18 (L-SOAS), 629.62 (ACOR) MJ/m²; the authors call the differences about 0.1 %. No pure-LHS static baseline. |
| Lahmar, Maalmi & Idchabani (2024), *Investigating adaptive sampling strategies for optimal building energy performance using artificial neural networks and kriging surrogate models*, J. Build. Eng. 82, 108341 | TRNSYS (not EnergyPlus), single zone, Morocco | 2 and 6 design variables; static LHS against error-based, LOLA-Voronoi, SampleMin and expected improvement; ANN and kriging; 100 and 400 final samples | The abstract claims 25 to 68 % fewer samples than static sampling. The paper's own tables do not clearly support a general advantage: static kriging has a lower RRSE than most adaptive variants (case 1: 0.0114; case 2: 0.1882) and the lowest cost among the 400-sample surrogate runs in case 2. One run per method, no statistics. |
| Legashev et al. (2024), *Investigation into the Hyperparameters of Error-Based Adaptive Sampling Approach for Surrogate Modeling*, Modelling 5(4), 2051-2074 | EnergyPlus 24.2, the `5ZoneAirCooled` example, Samara weather | 11 inputs (8 of them weather variables with unrealistically wide ranges); random, LHS and mixed sampling; error-based adaptive loop against static; RF, gradient boosting, LightGBM | Adaptive sampling is "preferable for a small number of iterations" once its hyperparameters (window size, sampling method, model) are tuned; for many iterations both give similar accuracy. With untuned settings (LHS, 10 % window) adaptive was not better than static random. Final tuned model MSE 7.76 (R² 0.99). No seeds, one case. |
| Elwy & Hagishima (2024), *The artificial intelligence reformation of sustainable building design approach: a systematic review on building design optimization methods using surrogate models*, Energy and Buildings 323, 114769 | review of 72 studies (mostly EnergyPlus-based) | no experiments of its own | Only 11 % of the studies used adaptive sampling; LHS was used in 40 %, Sobol in 4 %. Neural networks were 47 % of surrogates, Gaussian processes 7 %. The authors report a "lack of a robust adaptive sampling technique": in one study adaptive refinement worked "exceptionally well only during the initial iterations" and more than 1000 samples did not help. They recommend uncertainty-aware surrogates and error checks in critical regions rather than global R² alone. |
| Manmatharasan, Bitsuamlak & Grolinger (2025), *AI-driven design optimization for sustainable buildings: a systematic review*, Energy and Buildings 332, 115440 | review of 70 studies (EnergyPlus alone in 51 %) | no experiments of its own | LHS is the most common sampling method; only 5 of 70 studies used adaptive sampling; feed-forward networks are 65 % of surrogates, tree models 17 %, SVR 6 %. Adaptive sampling is described only qualitatively, with no head-to-head comparison; the authors rate it a "medium" research gap and recommend further work. |
| Nguyen & Nghiem (2026), *Active Learning for Optimal Experimental Design in Machine Learning-Based Building Energy System Identification*, arXiv:2606.25301 (preprint) | BOPTEST (Modelica), a fan-coil test case, not EnergyPlus | identifying one-step room-temperature dynamics (4 inputs, 1 output); 14 active-learning techniques (greedy and inverse-distance data-space methods, query-by-committee, MC dropout, model-change and, for Gaussian processes only, variance, integrated variance reduction, Fisher information, entropy, UCB, PI) against uniform random inputs; GP and small NN; 6 scenarios; single runs | Active learning lowered RMSE in most window and metric combinations (up to 54 % with Gaussian processes), mostly when data were scarce (2 initial points, tight ramp limits). With 10 initial points and loose limits the advantage narrowed or vanished, and random sampling had the best final 24 h RMSE for the NN. Greedy farthest-point sampling never beat random for the GP; MC dropout and entropy often did not. No repetitions; the authors list statistics over repeated trials as future work. This is system identification of dynamics, not annual-energy surrogates. |
| Weber, Subramaniam & Leistner (2026), *A parametric design integrated sampling and general training approach for optimal control oriented surrogate models of light-related quantities*, J. Build. Perform. Simul. 19(4), 826-849 | Radiance | sampling of incidence directions (Fibonacci), regression on spherical harmonics | Not about adaptive against static sampling and not EnergyPlus; it only suggests, untested, that active learning at a classifier decision boundary could remove outliers. Listed because it was requested; no conclusion for bdopt can be drawn from it. |

## What this means for bdopt

* **Adaptive sampling helps at small budgets and little at large ones, and not reliably.** Mehrotra & Yi and
  Legashev et al. (EnergyPlus) say so, Bamdad et al. find the advantage lost beyond about 1000 samples, and
  Westermann & Evins, the LOLA-Voronoi paper nearest to bdopt, conclude that neither scheme clearly wins and that
  the benefit appears to vanish going from 4 to 8 parameters. The platform benchmark (35 points, six or seven
  parameters, EnergyPlus, 3 seeds) agrees: no adaptive strategy beat the plain Latin hypercube reliably (the best ones
  win in about half of the test/seed pairs). Mehrotra & Yi's gains come with 19 inputs; bdopt now also has
  query-by-committee, leave-one-out and Lipschitz criteria, and none of them stood out at six or seven parameters.
* **Variance-reduction criteria work for Gaussian processes, farthest-point criteria do not.** In the BOPTEST study
  integrated variance reduction (the criterion behind bdopt's `nipv`) and maximum variance were the strongest GP
  criteria, while greedy farthest-point sampling never beat random sampling.
* **Many parameters.** The literature expects more benefit from adaptive sampling with many parameters. bdopt's
  own tests are mixed: on a 15-parameter EnergyPlus office (two seeds) committee-based and LOLA-Voronoi sampling had
  20 to 30 % lower error than a Latin hypercube, while on analytic 15- and 20-parameter responses nothing clearly beat it
  ({doc}`benchmarks`).
* **The claimed gains are fragile.** Two of the four experimental studies report no seeds or repetitions, and
  Lahmar et al.'s own tables undercut its abstract. bdopt's benchmark also uses one seed per test; see the caveats
  in {doc}`benchmarks`.
* **Adaptive sampling costs time.** Mehrotra & Yi report adaptive training sets that took up to 9000 s to
  build against 1 to 10 s for static ones; bdopt's per-iteration surrogate refits are the same kind of cost.
* **Surrogate choice.** SVR and Gaussian-process regression were the best thermal-load surrogates in the largest
  EnergyPlus comparison read here (Mehrotra & Yi), which agrees with the platform benchmark, where the Gaussian
  process family (GP, sparse GP, `auto`) was best on all seven tests. The review notes that neural networks dominate the literature (47 %)
  but that accuracy comparisons across model types cannot be generalised for that reason.
* **Parameter ranges matter.** Legashev et al. varied weather variables far beyond realistic ranges; their
  error levels are not comparable with design-parameter studies such as the platform's.

## Pointers, not read

Titles and search summaries only; verify before citing.

* Crombecq, Gorissen, Deschrijver & Dhaene (2011), *A novel hybrid sequential design strategy for global
  surrogate modeling of computer experiments*, SIAM J. Sci. Comput. 33(4): LOLA-Voronoi.
* van der Herten, Couckuyt, Deschrijver & Dhaene (2015), *A fuzzy hybrid sequential design strategy for global
  surrogate modeling of high-dimensional computer experiments*, SIAM J. Sci. Comput. 37(2): fuzzy LOLA-Voronoi.
* *Comparison of ANN and XGBoost surrogate models trained on small numbers of building energy simulations*,
  PLOS ONE (2024): a sequential method slightly improved on maximin LHS.
* *Efficient surrogate model development: impact of sample size and underlying model dimensions* (OSTI): Sobol
  sequences gave the best estimates at small sample sizes, and the influence of the sampling method shrank as the
  sample size grew.
