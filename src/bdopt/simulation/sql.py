"""Read results from ``eplusout.sql``.

The ``ReportDataDictionary`` / ``ReportData`` tables have the same layout in every EnergyPlus
version from 8.7 to 25.x (checked on 8.7, 9.0, 9.6, 23.1, 24.2 and 25.2), so no per-version branch
is needed here.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..objectives import ENERGY_UNITS, FunctionReader, MeterReader, Objective, VariableReader

_AGG = {"sum": np.sum, "max": np.max, "mean": np.mean}
_FREQ_LABEL = {
    "runperiod": "Run Period",
    "annual": "Annual",
    "monthly": "Monthly",
    "daily": "Daily",
    "hourly": "Hourly",
    "timestep": "Zone Timestep",
    "detailed": "HVAC System Timestep",
}


class MissingOutput(KeyError):
    pass


@dataclass
class RunResults:
    """Query interface over one run's ``eplusout.sql``."""

    sql_path: Path

    def _connect(self) -> sqlite3.Connection:
        if not Path(self.sql_path).is_file():
            raise FileNotFoundError(f"{self.sql_path} not found (did EnergyPlus fail?)")
        return sqlite3.connect(f"file:{self.sql_path}?mode=ro", uri=True)

    def series(
        self, name: str, is_meter: bool, key: str | None = None, frequency: str = "Run Period"
    ) -> tuple[np.ndarray, str | None]:
        """Values and units of a meter/variable at a reporting frequency (empty array if absent)."""
        q = (
            "SELECT d.ReportDataDictionaryIndex, d.Units FROM ReportDataDictionary d "
            "WHERE lower(d.Name)=lower(?) AND d.IsMeter=? AND d.ReportingFrequency=?"
        )
        args: list = [name, 1 if is_meter else 0, frequency]
        if key not in (None, "*"):
            q += " AND lower(d.KeyValue)=lower(?)"
            args.append(key)
        with self._connect() as con:
            rows = con.execute(q, args).fetchall()
            if not rows:
                return np.array([]), None
            ids = [r[0] for r in rows]
            marks = ",".join("?" * len(ids))
            vals = con.execute(
                f"SELECT Value FROM ReportData WHERE ReportDataDictionaryIndex IN ({marks})", ids
            ).fetchall()
        return np.array([v[0] for v in vals], dtype=float), rows[0][1]

    def meter(self, name: str, frequency: str = "Run Period") -> np.ndarray:
        return self.series(name, True, None, frequency)[0]

    def variable(self, key: str, name: str, frequency: str = "Run Period") -> np.ndarray:
        return self.series(name, False, key, frequency)[0]


def read_objective(obj: Objective, run: RunResults, warnings: list[str]) -> float:
    """Value of one objective for one run."""
    if isinstance(obj, FunctionReader):
        return float(obj.func(run))
    if isinstance(obj, MeterReader):
        total, found = 0.0, False
        for cand in obj.candidates:
            vals, units = run.series(cand, True, None, "Run Period")
            if vals.size == 0:
                continue
            found = True
            v = float(_AGG[obj.aggregate](vals))
            if units == "J":
                v *= ENERGY_UNITS[obj.unit]
            total += v
        if found:
            return total
        msg = (
            f"meter {obj.meter!r} not reported by EnergyPlus (no consumption, or unknown name)"
        )
        if obj.missing == "error":
            raise MissingOutput(msg)
        warnings.append(msg)
        return 0.0 if obj.missing == "zero" else float("nan")
    if isinstance(obj, VariableReader):
        label = _FREQ_LABEL.get(obj.frequency.lower(), obj.frequency)
        vals, _ = run.series(obj.variable, False, obj.key_value, label)
        if vals.size:
            return float(_AGG[obj.aggregate](vals))
        msg = f"variable {obj.variable!r} (key {obj.key_value!r}) not reported by EnergyPlus"
        if obj.missing == "error":
            raise MissingOutput(msg)
        warnings.append(msg)
        return 0.0 if obj.missing == "zero" else float("nan")
    raise TypeError(f"unsupported objective {obj!r}")
