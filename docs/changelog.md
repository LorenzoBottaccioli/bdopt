# Changelog

## 0.2.0

* {class}`~bdopt.MultiSelector`: one design variable that writes the same value in several objects, classes and
  fields (one column in the design table), with `MultiSelector.from_targets`. YAML/JSON studies get the same through
  `targets:`.
* {class}`~bdopt.FieldSelector` and {meth}`IDF.find <bdopt.IDF.find>` accept a list of object names. Every listed
  name must exist in the model: a typo is an error, not a smaller selection.
* The result cache of a custom objective ({class}`~bdopt.FunctionReader`) now depends on the code of its function:
  editing the function no longer returns results cached for the old one. Meters and variables keep their
  cache keys, so existing caches stay valid.

## 0.1.0

First release.

* EnergyPlus integration: installation discovery, version-aware field/class/meter names, model upgrade through
  the transitions shipped with EnergyPlus, results read from `eplusout.sql`.
* {class}`~bdopt.EnergyPlusEvaluator`: parallel runs, on-disk cache, failure handling, command runners.
* Range, integer and categorical parameters; constraints; meters, variables and custom objectives.
* Fixed designs (Latin hypercube, Sobol', random, factorial) and nine adaptive sampling strategies
  (fuzzy LOLA-Voronoi, LOLA-Voronoi, cross-validation Voronoi, Voronoi, probabilistic, GP variance, negative
  integrated posterior variance, maximin Latin hypercube, random).
* Fifteen surrogates (Gaussian process, multi-task and deep-kernel GPs, random forest, extra trees, gradient
  boosting, SVR, kernel ridge, k-NN, polynomial, RBF interpolation, scikit-learn MLP, PyTorch MLP, MC-dropout
  MLP) and automatic selection by cross-validation.
* NSGA-II and NSGA-III with mixed variables and constraints.
* EnergyPlus messages: warning and severe-error counts of every run are recorded (`stats()`), severe errors
  surface as Python warnings (policy `on_severe`: `warn`, `nan` or `ignore`), and bdopt adds no warnings of its own on any tested version (district heating meter
  names are requested per version; conflicting `Output:Meter:MeterFileOnly` requests are replaced).
* Studies from YAML/JSON and the `bdopt` command; plots; documentation; tools to install EnergyPlus versions and
  to compare their IDDs.
