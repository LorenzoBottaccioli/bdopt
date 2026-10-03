# Changelog

## 0.1.0

First release.

* EnergyPlus 8.7 and newer: installation discovery, version-aware field/class/meter names, model upgrade through
  the transitions shipped with EnergyPlus, results read from `eplusout.sql`.
* {class}`~bdopt.EnergyPlusEvaluator`: parallel runs, on-disk cache, failure handling, command runners.
* Range, integer and categorical parameters; constraints; meters, variables and custom objectives.
* Fixed designs (Latin hypercube, Sobol', random, factorial) and adaptive sampling (FLOLA-Voronoi, LOLA-Voronoi,
  Voronoi, GP variance), with an optional bridge to harlow.
* Surrogates: Gaussian process, scikit-learn MLP, PyTorch MLP.
* NSGA-II and NSGA-III with mixed variables and constraints.
* EnergyPlus messages: warning and severe-error counts of every run are recorded (`stats()`), severe errors
  surface as Python warnings (policy `on_severe`: `warn`, `nan` or `ignore`), and bdopt adds no warnings of its own on any tested version (district heating meter
  names are requested per version; conflicting `Output:Meter:MeterFileOnly` requests are replaced).
* Studies from YAML/JSON and the `bdopt` command; plots; documentation; tools to install EnergyPlus versions and
  to compare their IDDs.
