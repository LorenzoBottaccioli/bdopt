# Sampling

```{eval-rst}
.. py:module:: bdopt.sampling

.. currentmodule:: bdopt.sampling

Fixed designs
-------------

.. autofunction:: lhs
.. autofunction:: lhs_unit
.. autofunction:: sobol
.. autofunction:: random
.. autofunction:: factorial
.. autofunction:: halton
.. autofunction:: hammersley
.. autofunction:: maximin_lhs
.. autofunction:: fps
.. autofunction:: cvt

.. currentmodule:: bdopt.adaptive

Adaptive sampling
-----------------

.. autoclass:: AdaptiveSampler
.. autoclass:: AdaptiveResult

.. py:data:: METHODS
   :module: bdopt.adaptive

   The names accepted by ``AdaptiveSampler(method=...)``.

Infill criteria
---------------

.. currentmodule:: bdopt.adaptive.designs

.. autofunction:: voronoi_estimate
.. autofunction:: lola_nonlinearity
.. autofunction:: flola_nonlinearity
.. autofunction:: fuzzy_weights
.. autofunction:: propose_voronoi_based
.. autofunction:: propose_cv_voronoi
.. autofunction:: propose_probabilistic
.. autofunction:: propose_gp_variance
.. autofunction:: propose_nipv
.. autofunction:: propose_latin_hypercube

.. currentmodule:: bdopt.adaptive.criteria

.. autofunction:: propose_qbc
.. autofunction:: propose_loocv
.. autofunction:: propose_lipschitz
.. autofunction:: propose_idw
.. autofunction:: propose_greedy
.. autofunction:: propose_cvt
.. autofunction:: propose_fisher_information
.. autofunction:: propose_acquisition
.. autofunction:: propose_soas
.. autofunction:: propose_sequence
.. autofunction:: propose_mixed
.. autofunction:: propose_error_window
```
