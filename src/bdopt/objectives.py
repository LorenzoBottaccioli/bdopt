"""Objectives: how to read a result out of an EnergyPlus run.

``Problem(parameters, ["Electricity:Facility", ...])`` accepts plain meter names; use
:class:`MeterReader` / :class:`VariableReader` / :class:`FunctionReader` for units, aggregation or
custom post-processing.
"""

from __future__ import annotations

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
        return 1.0 if self.direction == "min" else -1.0

    def idf_outputs(self) -> list[tuple[str, ...]]:
        """``Output:*`` objects (class, fields...) to add to the IDF so the result is reported."""
        return []


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
        return meter_candidates(self.meter)

    def idf_outputs(self) -> list[tuple[str, ...]]:
        # the ``Output:Meter`` key field is called ``Name`` (<9.0) / ``Key Name``: both are field 0
        return [("Output:Meter", c, "RunPeriod") for c in self.candidates]


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

    def idf_outputs(self) -> list[tuple[str, ...]]:
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

    def idf_outputs(self) -> list[tuple[str, ...]]:
        return list(self.outputs)


def as_objective(obj: str | Objective) -> Objective:
    if isinstance(obj, Objective):
        return obj
    if isinstance(obj, str):
        return MeterReader(obj)
    raise TypeError(f"objective must be a meter name or an Objective, not {type(obj).__name__}")
