"""Objectives: how to read a result out of an EnergyPlus run.

``Problem(parameters, ["Electricity:Facility", ...])`` accepts plain meter names; use
:class:`MeterReader` / :class:`VariableReader` / :class:`FunctionReader` for units, aggregation or
custom post-processing.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass

from .eplus.aliases import meter_candidates

#: multiply a value in Joules by this to get the unit
ENERGY_UNITS = {"J": 1.0, "kJ": 1e-3, "MJ": 1e-6, "GJ": 1e-9, "kWh": 1 / 3.6e6, "MWh": 1 / 3.6e9}


@dataclass
class Objective:
    """Base class. ``direction`` is ``'min'`` or ``'max'`` (used by the optimizers)."""

    name: str
    direction: str = "min"

    def __post_init__(self) -> None:
        if self.direction not in ("min", "max"):
            raise ValueError("direction must be 'min' or 'max'")

    @property
    def sign(self) -> float:
        """``+1`` for an objective to minimise, ``-1`` to maximise."""
        return 1.0 if self.direction == "min" else -1.0

    def idf_outputs(self, version=None, idf=None) -> list[tuple[str, ...]]:
        """``Output:*`` objects (class, fields...) to add to the IDF so the result is reported.

        ``version`` (the EnergyPlus version) and ``idf`` (the model) let an objective request only what
        exists there, so that EnergyPlus has nothing to warn about."""
        return []

    def cache_token(self) -> str:
        """Extra text that goes into the result-cache key, for objectives whose value depends on code.

        Empty for meters and variables (their name and options already identify the result)."""
        return ""


class MeterReader(Objective):
    """A meter (e.g. ``Electricity:Facility``) aggregated over the run period.

    ``aggregate`` is ``'sum'`` (energy, over all run periods), ``'max'`` or ``'mean'`` of the
    run-period values. ``unit`` converts Joules to J/kJ/MJ/GJ/kWh/MWh (default: J).
    ``missing`` controls meters EnergyPlus does not report (no consumption, or a wrong name):
    ``'zero'`` (default, with a warning), ``'nan'`` or ``'error'``.
    """

    def __init__(
        self,
        meter: str,
        name: str | None = None,
        unit: str = "J",
        aggregate: str = "sum",
        missing: str = "zero",
        direction: str = "min",
    ):
        super().__init__(name or meter, direction)
        if unit not in ENERGY_UNITS:
            raise ValueError(f"unit must be one of {sorted(ENERGY_UNITS)}")
        if aggregate not in ("sum", "max", "mean"):
            raise ValueError("aggregate must be 'sum', 'max' or 'mean'")
        if missing not in ("zero", "nan", "error"):
            raise ValueError("missing must be 'zero', 'nan' or 'error'")
        self.meter, self.unit, self.aggregate, self.missing = meter, unit, aggregate, missing

    @property
    def candidates(self) -> list[str]:
        """Meter names requested for this meter across EnergyPlus versions."""
        return meter_candidates(self.meter)

    def idf_outputs(self, version=None, idf=None) -> list[tuple[str, ...]]:
        """``Output:Meter`` objects needed to report this meter, only with names valid for ``version``."""
        # the ``Output:Meter`` key field is called ``Name`` (<9.0) / ``Key Name``: both are field 0
        has_steam = idf is None or bool(idf.by_class("DistrictHeating:Steam"))
        return [("Output:Meter", c, "RunPeriod") for c in meter_candidates(self.meter, version, has_steam)]


class VariableReader(Objective):
    """An ``Output:Variable`` (``key_value`` may be ``'*'``) aggregated over the run period."""

    def __init__(
        self,
        key_value: str,
        variable: str,
        name: str | None = None,
        aggregate: str = "sum",
        frequency: str = "RunPeriod",
        missing: str = "nan",
        direction: str = "min",
    ):
        super().__init__(name or f"{variable}", direction)
        if aggregate not in ("sum", "max", "mean"):
            raise ValueError("aggregate must be 'sum', 'max' or 'mean'")
        self.key_value, self.variable = key_value, variable
        self.aggregate, self.frequency, self.missing = aggregate, frequency, missing

    def idf_outputs(self, version=None, idf=None) -> list[tuple[str, ...]]:
        """``Output:Variable`` object needed to report this variable."""
        return [("Output:Variable", self.key_value, self.variable, self.frequency)]


class FunctionReader(Objective):
    """Custom objective: ``func(run)`` receives a :class:`bdopt.simulation.sql.RunResults`."""

    def __init__(
        self,
        func: Callable,
        name: str,
        outputs: list[tuple[str, ...]] | None = None,
        direction: str = "min",
    ):
        super().__init__(name, direction)
        self.func, self.outputs = func, list(outputs or [])

    def idf_outputs(self, version=None, idf=None) -> list[tuple[str, ...]]:
        """The ``Output:*`` objects declared for the custom function."""
        return list(self.outputs)

    def cache_token(self) -> str:
        """Fingerprint of the function, so that editing it does not return results cached for the old code.

        It hashes the source text (or, if that is unavailable, the bytecode and constants) and the default
        argument values. Values the function captures from its environment (closures, globals) are NOT
        tracked: change ``name`` or pass ``cache=False`` to the evaluator if they change."""
        return function_fingerprint(self.func)


def function_fingerprint(func: Callable) -> str:
    """Stable hash of a callable's code (see :meth:`FunctionReader.cache_token`)."""
    h = hashlib.sha256()
    target = func.func if isinstance(func, functools.partial) else func
    h.update(str(getattr(target, "__module__", None) or "").encode())  # None for exec()'d functions
    h.update(str(getattr(target, "__qualname__", type(target).__qualname__)).encode())
    try:
        h.update(inspect.getsource(target).encode("utf-8", errors="replace"))
    except (OSError, TypeError):
        code = getattr(target, "__code__", None)
        if code is not None:
            h.update(code.co_code)
            h.update(repr(code.co_consts).encode())
            h.update(repr(code.co_names).encode())
    h.update(repr(getattr(target, "__defaults__", None)).encode())
    if isinstance(func, functools.partial):  # the bound arguments are part of the function
        h.update(repr((func.args, sorted(func.keywords.items()))).encode())
    return h.hexdigest()[:16]


def as_objective(obj: str | Objective) -> Objective:
    """Convert a meter name into a :class:`MeterReader`; pass :class:`Objective` instances through."""
    if isinstance(obj, Objective):
        return obj
    if isinstance(obj, str):
        return MeterReader(obj)
    raise TypeError(f"objective must be a meter name or an Objective, not {type(obj).__name__}")
