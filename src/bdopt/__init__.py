"""bdopt -- building design optimization with EnergyPlus.

Typical flow::

    from bdopt import *
    problem = Problem(parameters, ["Electricity:Facility", ...])
    ev = EnergyPlusEvaluator(problem, "model.idf", epw="weather.epw")
    result = AdaptiveSampler(problem, ev).run(n_initial=20, max_iter=40)
    surrogate = GPSurrogate().fit(result.data, problem)
    front = nsga2(problem, surrogate)
"""

from . import sampling
from .eplus import EnergyPlus, EnergyPlusNotFound, Version, find_installations
from .evaluator import EnergyPlusEvaluator, Evaluator, FunctionEvaluator, VersionMismatch
from .idf import IDF, IDFObject
from .model import load_idf
from .objectives import FunctionReader, MeterReader, Objective, VariableReader
from .parameters import (
    CategoryParameter,
    FieldSelector,
    GenericSelector,
    IntegerParameter,
    NullSelector,
    Parameter,
    ParameterError,
    RangeParameter,
)
from .problem import Problem

__version__ = "0.1.0"

__all__ = [
    "IDF",
    "CategoryParameter",
    "EnergyPlus",
    "EnergyPlusEvaluator",
    "EnergyPlusNotFound",
    "Evaluator",
    "FieldSelector",
    "FunctionEvaluator",
    "FunctionReader",
    "GenericSelector",
    "NullSelector",
    "IDFObject",
    "IntegerParameter",
    "MeterReader",
    "Objective",
    "Parameter",
    "ParameterError",
    "Problem",
    "RangeParameter",
    "VariableReader",
    "Version",
    "VersionMismatch",
    "find_installations",
    "load_idf",
    "sampling",
]
