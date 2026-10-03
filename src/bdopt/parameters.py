"""Design parameters: *where* a value is written in the IDF (selector) and *which values* it takes
(descriptor).

Every descriptor can ``encode`` a real value into ``[0, 1]`` and ``decode`` it back. Samplers,
adaptive designs, surrogates and optimizers all work in that unit cube, which is what makes
categorical parameters (e.g. glazing type) first-class instead of a special case.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .eplus.aliases import resolve_class, resolve_field
from .eplus.idd import IDD, norm
from .idf import IDF, IDFObject


class ParameterError(ValueError):
    """Invalid parameter definition, or a selector that does not match the model."""
    pass


# -- value descriptors ---------------------------------------------------------------------------
class RangeParameter:
    """Continuous values in ``[min_val, max_val]``."""

    kind = "continuous"

    def __init__(self, min_val: float, max_val: float):
        if not max_val > min_val:
            raise ParameterError(f"max_val ({max_val}) must be greater than min_val ({min_val})")
        self.min, self.max = float(min_val), float(max_val)

    min_val = property(lambda self: self.min, doc="Lower bound (alias of ``min``).")
    max_val = property(lambda self: self.max, doc="Upper bound (alias of ``max``).")

    @property
    def bounds(self) -> tuple[float, float]:
        """``(min, max)``."""
        return self.min, self.max

    def decode(self, u: np.ndarray) -> np.ndarray:
        """Map unit-cube coordinates ``u`` in [0, 1] to values in ``[min, max]``."""
        return self.min + np.clip(u, 0.0, 1.0) * (self.max - self.min)

    def encode(self, v) -> np.ndarray:
        """Map values in ``[min, max]`` to unit-cube coordinates."""
        return (np.asarray(v, dtype=float) - self.min) / (self.max - self.min)

    def __repr__(self) -> str:
        return f"RangeParameter({self.min}, {self.max})"


class IntegerParameter(RangeParameter):
    """Integer values in ``[min_val, max_val]`` (both included)."""

    kind = "integer"

    def __init__(self, min_val: int, max_val: int):
        super().__init__(int(min_val), int(max_val))

    def decode(self, u: np.ndarray) -> np.ndarray:
        """Map unit-cube coordinates to integers; each integer owns an equal-width bin."""
        n = int(self.max - self.min) + 1
        idx = np.minimum(np.floor(np.clip(u, 0.0, 1.0) * n), n - 1)
        return (self.min + idx).astype(int)

    def encode(self, v) -> np.ndarray:
        """Map integers to the centre of their bin in the unit cube."""
        n = int(self.max - self.min) + 1
        return (np.asarray(v, dtype=float) - self.min + 0.5) / n

    def __repr__(self) -> str:
        return f"IntegerParameter({int(self.min)}, {int(self.max)})"


class CategoryParameter:
    """One of a list of options (strings or numbers), e.g. construction names."""

    kind = "categorical"

    def __init__(self, options: Sequence[Any]):
        self.options = list(options)
        if len(self.options) < 2:
            raise ParameterError("a CategoryParameter needs at least two options")
        if len({str(o) for o in self.options}) != len(self.options):
            raise ParameterError("CategoryParameter options must be distinct")

    @property
    def bounds(self) -> tuple[float, float]:
        """``(0, n_options - 1)``: the option index range."""
        return 0.0, float(len(self.options) - 1)

    def index(self, u: np.ndarray) -> np.ndarray:
        """Index of the option selected by unit-cube coordinates ``u``."""
        n = len(self.options)
        return np.minimum(np.floor(np.clip(u, 0.0, 1.0) * n), n - 1).astype(int)

    def decode(self, u: np.ndarray) -> np.ndarray:
        """Map unit-cube coordinates to options (object array)."""
        arr = np.empty(np.shape(u), dtype=object)
        idx = self.index(np.asarray(u, dtype=float))
        flat = arr.reshape(-1)
        for i, k in enumerate(idx.reshape(-1)):
            flat[i] = self.options[k]
        return arr

    def encode(self, v) -> np.ndarray:
        """Map options to the centre of their bin in the unit cube.

        Raises
        ------
        ParameterError
            If a value is not one of the options.
        """
        lookup = {str(o): i for i, o in enumerate(self.options)}
        vals = np.atleast_1d(np.asarray(v, dtype=object))
        try:
            idx = np.array([lookup[str(x)] for x in vals], dtype=float)
        except KeyError as e:
            raise ParameterError(f"{e.args[0]!r} is not one of {self.options}") from None
        return (idx + 0.5) / len(self.options)

    def __repr__(self) -> str:
        return f"CategoryParameter({self.options})"


# -- selectors -----------------------------------------------------------------------------------
class FieldSelector:
    """Addresses one field of IDF objects: ``(class_name, object_name, field_name)``.

    ``object_name='*'`` selects every object of the class. ``where`` restricts the selection, either
    ``{"Surface Type": "Window"}`` (field name -> required value) or ``callable(IDFObject) -> bool``.
    Class and field names are resolved against the IDD of the EnergyPlus version in use, so the same
    selector works across versions (see :mod:`bdopt.eplus.aliases`).
    """

    def __init__(
        self,
        class_name: str,
        object_name: str | None = "*",
        field_name: str | None = None,
        *,
        field_index: int | None = None,
        where: dict[str, Any] | Callable[[IDFObject], bool] | None = None,
    ):
        if field_name is None and field_index is None:
            raise ParameterError("give field_name (or field_index)")
        self.class_name, self.object_name = class_name, object_name
        self.field_name, self.field_index, self.where = field_name, field_index, where

    @property
    def default_name(self) -> str:
        """Name used for a parameter that is not given one."""
        return f"{self.class_name}:{self.object_name}:{self.field_name or self.field_index}"

    def _resolve(self, idd: IDD) -> tuple[str, int]:
        if self.field_name is not None:
            return resolve_field(idd, self.class_name, self.field_name)
        return resolve_class(idd, self.class_name), int(self.field_index)  # type: ignore[arg-type]

    def targets(self, idf: IDF, idd: IDD) -> list[IDFObject]:
        """IDF objects selected by this selector in ``idf`` (resolved against ``idd``)."""
        cls, _ = self._resolve(idd)
        pred = self.where
        if isinstance(pred, dict):
            conds = []
            for fname, want in pred.items():
                _, i = resolve_field(idd, self.class_name, fname)
                conds.append((i, norm(str(want))))

            def pred(o: IDFObject, _c=conds) -> bool:  # type: ignore[misc]
                return all(norm(o.get(i)) == w for i, w in _c)

        return idf.find(cls, self.object_name, pred)

    def validate(self, idf: IDF, idd: IDD) -> int:
        """Check that the selector matches at least one object; return how many.

        Raises
        ------
        ParameterError
            If no object matches (typo in the object name, or an absent class).
        KeyError
            If the class or field does not exist in this EnergyPlus version (with suggestions).
        """
        n = len(self.targets(idf, idd))
        if n == 0:
            raise ParameterError(
                f"selector {self.default_name!r} matches no object in the model "
                f"(class {self.class_name!r}, object {self.object_name!r}, where={self.where!r})"
            )
        return n

    def apply(self, idf: IDF, value, idd: IDD) -> None:
        """Write ``value`` into the selected field of every selected object (in place)."""
        _, idx = self._resolve(idd)
        objs = self.targets(idf, idd)
        if not objs:
            self.validate(idf, idd)
        for o in objs:
            o.set(idx, value)

    def __repr__(self) -> str:
        return (
            f"FieldSelector({self.class_name!r}, {self.object_name!r}, "
            f"{self.field_name or self.field_index!r})"
        )


class GenericSelector:
    """Arbitrary edit: ``setter(idf, value)`` is called with the value of the parameter."""

    def __init__(self, setter: Callable[[IDF, Any], None], name: str = "generic"):
        self.setter, self.default_name = setter, name

    def validate(self, idf: IDF, idd: IDD) -> int:
        """Always valid (the setter is arbitrary code); returns 1."""
        return 1

    def apply(self, idf: IDF, value, idd: IDD) -> None:
        """Call the setter with ``idf`` and ``value``."""
        self.setter(idf, value)

    def __repr__(self) -> str:
        return f"GenericSelector({self.default_name!r})"


class Parameter:
    """A design variable: ``Parameter(selector, value_descriptor, name)``.

    ``value_descriptor`` may also be a ``(min, max)`` tuple (a range) or a list of options (a category).
    """

    def __init__(self, selector, value_descriptor=None, name: str | None = None):
        vd = value_descriptor
        if vd is None:
            raise ParameterError("a Parameter needs a value_descriptor")
        if isinstance(vd, tuple) and len(vd) == 2:
            vd = RangeParameter(*vd)
        elif isinstance(vd, list):
            vd = CategoryParameter(vd)
        if not hasattr(vd, "encode"):
            raise ParameterError(f"unsupported value descriptor {vd!r}")
        self.selector, self.value_descriptor = selector, vd
        self.name = name or selector.default_name

    @property
    def kind(self) -> str:
        """``"continuous"``, ``"integer"`` or ``"categorical"``."""
        return self.value_descriptor.kind

    @property
    def is_discrete(self) -> bool:
        """True for integer and categorical parameters."""
        return self.kind != "continuous"

    def __repr__(self) -> str:
        return f"Parameter({self.name!r}, {self.value_descriptor!r})"


def python_value(v):
    """numpy scalar -> plain python value (for IDF formatting)."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and math.isnan(v):
        raise ParameterError("NaN parameter value")
    return v
