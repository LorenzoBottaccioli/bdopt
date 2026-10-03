"""Compatibility with BESOS (https://github.com/building-energy/besos).

Existing BESOS notebooks run unchanged after one line::

    import bdopt.compat
    bdopt.compat.use_as_besos()          # now `from besos import ...` resolves to bdopt

or change the imports to ``from bdopt.compat.besos import ...``. Only what the Smart Building
Educational Platform used is provided (get_building, FieldSelector / RangeParameter /
CategoryParameter / Parameter, EPProblem, dist_sampler / lhs, EvaluatorEP / EvaluatorGeneric,
NSGAII); everything else raises ``NotImplementedError`` with a pointer to the bdopt equivalent.
"""

from __future__ import annotations

import importlib
import sys

_SUBMODULES = ("eppy_funcs", "sampling", "evaluator", "parameters", "problem", "optimizer")


def use_as_besos() -> None:
    """Register ``bdopt.compat.besos`` as the ``besos`` package in ``sys.modules``."""
    root = importlib.import_module("bdopt.compat.besos")
    sys.modules["besos"] = root
    for name in _SUBMODULES:
        sys.modules[f"besos.{name}"] = importlib.import_module(f"bdopt.compat.besos.{name}")
