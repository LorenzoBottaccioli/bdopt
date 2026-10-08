"""Design parameters: *where* a value is written in the IDF (selector) and *which values* it takes
(descriptor).

Every descriptor can ``encode`` a real value into ``[0, 1]`` and ``decode`` it back. Samplers,
adaptive designs, surrogates and optimizers all work in that unit cube, which is what makes
categorical parameters (e.g. glazing type) first-class instead of a special case.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
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

    ``object_name='*'`` selects every object of the class; a list of names (``["Win 1", "Win 2"]``)
    selects exactly those objects, and every listed name must exist in the model. ``where`` restricts the
    selection, either ``{"Surface Type": "Window"}`` (field name -> required value) or
    ``callable(IDFObject) -> bool``. Class and field names are resolved against the IDD of the
    EnergyPlus version in use, so the same selector works across versions (see
    :mod:`bdopt.eplus.aliases`). To write one value into several classes or fields use
    :class:`MultiSelector`.
    """

    def __init__(
        self,
        class_name: str,
        object_name: str | Sequence[str] | None = "*",
        field_name: str | None = None,
        *,
        field_index: int | None = None,
        where: dict[str, Any] | Callable[[IDFObject], bool] | None = None,
    ):
        if field_name is None and field_index is None:
            raise ParameterError("give field_name (or field_index)")
        if isinstance(object_name, (list, tuple, set, frozenset)):
            object_name = list(object_name)
            if not object_name:
                raise ParameterError("object_name is an empty list: it would select nothing")
        self.class_name, self.object_name = class_name, object_name
        self.field_name, self.field_index, self.where = field_name, field_index, where

    @property
    def _object_label(self) -> str:
        if isinstance(self.object_name, list):
            return self.object_name[0] if len(self.object_name) == 1 else f"{len(self.object_name)} objects"
        return str(self.object_name)

    @property
    def default_name(self) -> str:
        """Name used for a parameter that is not given one."""
        return f"{self.class_name}:{self._object_label}:{self.field_name or self.field_index}"

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
        found = self.targets(idf, idd)
        if not found:
            raise ParameterError(
                f"selector {self.default_name!r} matches no object in the model "
                f"(class {self.class_name!r}, object {self.object_name!r}, where={self.where!r})"
            )
        if isinstance(self.object_name, list):  # a listed name that is missing is a typo, not "fewer targets"
            present = {o.name.strip().lower() for o in found}
            missing = [n for n in self.object_name if str(n).strip().lower() not in present]
            if missing:
                shown = ", ".join(repr(n) for n in missing[:5]) + (" ..." if len(missing) > 5 else "")
                raise ParameterError(
                    f"selector {self.default_name!r}: {len(missing)} of {len(self.object_name)} listed "
                    f"{self.class_name} objects do not exist in the model (or fail 'where'): {shown}"
                )
        return len(found)

    def apply(self, idf: IDF, value, idd: IDD) -> None:
        """Write ``value`` into the selected field of every selected object (in place)."""
        _, idx = self._resolve(idd)
        objs = self.targets(idf, idd)
        if not objs:
            self.validate(idf, idd)
        for o in objs:
            o.set(idx, value)

    def __repr__(self) -> str:
        obj = self._object_label if isinstance(self.object_name, list) else repr(self.object_name)
        return f"FieldSelector({self.class_name!r}, {obj}, {self.field_name or self.field_index!r})"


class MultiSelector:
    """One design variable that writes the same value in several places.

    The places can be different objects, classes and fields, e.g. one thermostat set point written into
    the ``Field 4`` of several ``Schedule:Compact`` objects, or one glazing type written into the
    construction of every window of a facade. The parameter is a single column of the design table, so
    samplers, surrogates and optimizers see one variable, not one per target::

        sel = MultiSelector([
            FieldSelector("Schedule:Compact", "Heating Z1", field_index=4),
            FieldSelector("Schedule:Compact", "Heating Z2", field_index=4),
            FieldSelector("ThermostatSetpoint:SingleHeating", "*", "Setpoint Temperature Schedule Name"),
        ])
        Parameter(sel, RangeParameter(18, 22), name="Heating set point")

    Use :meth:`from_targets` to build it from a list of targets.
    """

    def __init__(self, selectors: Sequence[Any], name: str | None = None):
        self.selectors = list(selectors)
        if not self.selectors:
            raise ParameterError("a MultiSelector needs at least one selector")
        for s in self.selectors:
            if not (hasattr(s, "apply") and hasattr(s, "validate")):
                raise ParameterError(f"{s!r} is not a selector (it needs validate() and apply())")
        self._name = name

    @classmethod
    def from_targets(
        cls,
        targets: Sequence[str | Mapping[str, Any]],
        *,
        class_name: str | None = None,
        field_name: str | None = None,
        field_index: int | None = None,
        name: str | None = None,
    ) -> MultiSelector:
        """Build a selector from a list of targets.

        Each target is an object name (a string; ``class_name`` and ``field_name``/``field_index`` give the
        rest) or a mapping with the keys ``class``, ``object``, ``field`` / ``field_index`` (missing keys
        fall back to the arguments). Consecutive targets that share class and field are merged into one
        :class:`FieldSelector` with a list of object names.
        """
        grouped: list[dict[str, Any]] = []
        for t in targets:
            spec = {"object": t} if isinstance(t, str) else dict(t)
            cls_ = spec.get("class", spec.get("class_name", class_name))
            fld = spec.get("field", spec.get("field_name", field_name))
            idx = spec.get("field_index", field_index if fld is None else None)
            if cls_ is None or (fld is None and idx is None):
                raise ParameterError(f"target {t!r} needs a class and a field (or field_index)")
            obj = spec.get("object", spec.get("object_name", "*"))
            if grouped and (grouped[-1]["class"], grouped[-1]["field"], grouped[-1]["index"]) == (cls_, fld, idx) \
                    and isinstance(grouped[-1]["objects"], list) and obj != "*":
                grouped[-1]["objects"].append(obj)
            else:
                grouped.append({"class": cls_, "field": fld, "index": idx,
                                "objects": [obj] if obj != "*" else "*", "where": spec.get("where")})
        sels = [
            FieldSelector(g["class"], g["objects"], g["field"], field_index=g["index"], where=g["where"])
            for g in grouped
        ]
        return cls(sels, name=name)

    @property
    def default_name(self) -> str:
        """Name used for a parameter that is not given one."""
        if self._name:
            return self._name
        first = self.selectors[0].default_name
        return first if len(self.selectors) == 1 else f"{first} (+{len(self.selectors) - 1} more)"

    def targets(self, idf: IDF, idd: IDD) -> list[IDFObject]:
        """Every object edited by this selector (objects reached by several members appear once)."""
        seen: dict[int, IDFObject] = {}
        for s in self.selectors:
            if hasattr(s, "targets"):
                for o in s.targets(idf, idd):
                    seen.setdefault(id(o), o)
        return list(seen.values())

    def validate(self, idf: IDF, idd: IDD) -> int:
        """Check every member; return the total number of objects. The error names the failing member."""
        total = 0
        for i, s in enumerate(self.selectors, 1):
            try:
                total += s.validate(idf, idd)
            except ParameterError as e:
                raise ParameterError(f"target {i} of {len(self.selectors)}: {e}") from None
            except KeyError as e:
                raise KeyError(f"target {i} of {len(self.selectors)} ({s!r}): {e.args[0] if e.args else e}") from None
        return total

    def apply(self, idf: IDF, value, idd: IDD) -> None:
        """Write ``value`` in every place (in place)."""
        for s in self.selectors:
            s.apply(idf, value, idd)

    def __repr__(self) -> str:
        return f"MultiSelector({len(self.selectors)} selectors, first={self.selectors[0]!r})"


def selector_from_spec(spec: Mapping[str, Any]) -> FieldSelector | MultiSelector:
    """Selector from a configuration entry (the form used in YAML/JSON studies).

    ``{class, object, field | field_index, where}`` gives a :class:`FieldSelector` (``object`` is ``"*"``
    by default and may be a list of names). ``{targets: [...], class, field}`` gives a
    :class:`MultiSelector`: each target is an object name or a mapping with its own ``class`` / ``object`` /
    ``field`` / ``field_index``, falling back to the entry's ``class`` and ``field``.
    """
    if "targets" in spec:
        return MultiSelector.from_targets(
            spec["targets"], class_name=spec.get("class"), field_name=spec.get("field"),
            field_index=spec.get("field_index"), name=spec.get("name"),
        )
    if "class" not in spec:
        raise ParameterError(f"a parameter needs 'class' (and 'field'), or 'targets': {dict(spec)}")
    return FieldSelector(
        spec["class"], spec.get("object", "*"), spec.get("field"),
        field_index=spec.get("field_index"), where=spec.get("where"),
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


class NullSelector:
    """A selector that edits nothing: for design spaces evaluated by a Python function
    (:class:`~bdopt.FunctionEvaluator`) instead of EnergyPlus."""

    def __init__(self, name: str = "x"):
        self.default_name = name

    def validate(self, idf: IDF, idd: IDD) -> int:
        """Always valid; returns 1."""
        return 1

    def apply(self, idf: IDF, value, idd: IDD) -> None:
        """Does nothing."""

    def __repr__(self) -> str:
        return f"NullSelector({self.default_name!r})"


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
