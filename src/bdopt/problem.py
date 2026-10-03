"""A design problem = parameters + objectives (+ optional constraints)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .eplus.idd import IDD
from .idf import IDF
from .objectives import Objective, as_objective
from .parameters import Parameter, ParameterError, python_value

Constraint = Callable[[Mapping[str, Any]], float]


class Problem:
    """Parameters, objectives and constraints of a design study.

    ``constraints`` are functions of a parameter row (``{name: value}``) returning ``g``, feasible when
    ``g <= 0``.

    All numerical work happens in the unit cube ``[0, 1]^d``: :meth:`encode` / :meth:`decode` convert
    between that and real-valued parameter tables (:class:`pandas.DataFrame`, one column per
    parameter, categorical columns hold the option itself).
    """

    def __init__(
        self,
        parameters: Sequence[Parameter],
        objectives: Sequence[str | Objective],
        constraints: Sequence[Constraint] = (),
    ):
        self.parameters = list(parameters)
        self.objectives = [as_objective(o) for o in objectives]
        self.constraints = list(constraints)
        names = self.parameter_names
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ParameterError(f"duplicate parameter names: {dup}")
        if set(names) & set(self.objective_names):
            raise ParameterError("a parameter and an objective share a name")
        if not self.parameters:
            raise ParameterError("a Problem needs at least one parameter")
        if not self.objectives:
            raise ParameterError("a Problem needs at least one objective")

    # -- names / shapes --------------------------------------------------------------------------
    @property
    def parameter_names(self) -> list[str]:
        """Names of the parameters, in order."""
        return [p.name for p in self.parameters]

    @property
    def objective_names(self) -> list[str]:
        """Names of the objectives, in order."""
        return [o.name for o in self.objectives]

    def names(self) -> list[str]:
        """Parameter names followed by objective names."""
        return self.parameter_names + self.objective_names

    @property
    def n_var(self) -> int:
        """Number of parameters (dimension of the unit cube)."""
        return len(self.parameters)

    @property
    def n_obj(self) -> int:
        """Number of objectives."""
        return len(self.objectives)

    @property
    def signs(self) -> np.ndarray:
        """``+1`` to minimise, ``-1`` to maximise, per objective."""
        return np.array([o.sign for o in self.objectives])

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Real-valued bounds per parameter (categorical: option index ``0..n-1``)."""
        lo, hi = zip(*(p.value_descriptor.bounds for p in self.parameters), strict=True)
        return np.array(lo), np.array(hi)

    # -- unit cube <-> real values ---------------------------------------------------------------
    def decode(self, U: np.ndarray) -> pd.DataFrame:
        """Unit-cube points ``(n, d)`` -> parameter table."""
        U = np.atleast_2d(np.asarray(U, dtype=float))
        if U.shape[1] != self.n_var:
            raise ValueError(f"expected {self.n_var} columns, got {U.shape[1]}")
        cols = {p.name: p.value_descriptor.decode(U[:, i]) for i, p in enumerate(self.parameters)}
        return pd.DataFrame(cols)

    def encode(self, df: pd.DataFrame | Mapping[str, Any]) -> np.ndarray:
        """Parameter table -> unit-cube points ``(n, d)`` (categorical -> bin centre)."""
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame({k: [v] for k, v in df.items()})
        missing = [n for n in self.parameter_names if n not in df.columns]
        if missing:
            raise KeyError(f"missing parameter columns: {missing}")
        cols = [p.value_descriptor.encode(df[p.name].to_numpy()) for p in self.parameters]
        return np.column_stack(cols).astype(float)

    def snap(self, U: np.ndarray) -> np.ndarray:
        """Move discrete coordinates (integer/categorical) to the centre of their bin."""
        U = np.array(np.atleast_2d(U), dtype=float, copy=True)
        for i, p in enumerate(self.parameters):
            if p.is_discrete:
                U[:, i] = p.value_descriptor.encode(p.value_descriptor.decode(U[:, i]))
        return np.clip(U, 0.0, 1.0)

    @property
    def discrete_mask(self) -> np.ndarray:
        """Boolean array telling which parameters are integer or categorical."""
        return np.array([p.is_discrete for p in self.parameters])

    # -- constraints -----------------------------------------------------------------------------
    def violation(self, df: pd.DataFrame) -> np.ndarray:
        """``(n, n_constraints)`` array of ``g`` values (feasible if ``<= 0``)."""
        if not self.constraints:
            return np.zeros((len(df), 0))
        rows = df[self.parameter_names].to_dict("records")
        return np.array([[float(c(r)) for c in self.constraints] for r in rows])

    def feasible(self, df: pd.DataFrame) -> np.ndarray:
        """Boolean mask of the rows of ``df`` that satisfy every constraint."""
        g = self.violation(df)
        return np.all(g <= 0, axis=1) if g.size else np.ones(len(df), dtype=bool)

    # -- model editing ---------------------------------------------------------------------------
    def validate(self, idf: IDF, idd: IDD) -> None:
        """Raise if a selector matches nothing in ``idf`` (typo in a class/object/field name)."""
        for p in self.parameters:
            try:
                p.selector.validate(idf, idd)
            except (KeyError, ParameterError) as e:
                raise ParameterError(f"parameter {p.name!r}: {e.args[0] if e.args else e}") from None

    def apply(self, idf: IDF, row: Mapping[str, Any], idd: IDD) -> None:
        """Write one parameter row into ``idf`` (in place)."""
        for p in self.parameters:
            p.selector.apply(idf, python_value(row[p.name]), idd)

    def __repr__(self) -> str:
        return f"Problem({len(self.parameters)} parameters, objectives={self.objective_names})"
