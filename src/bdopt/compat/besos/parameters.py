from __future__ import annotations

from ...parameters import (  # noqa: F401
    CategoryParameter,
    FieldSelector,
    GenericSelector,
    IntegerParameter,
    Parameter,
    RangeParameter,
)


def expand_plist(plist: dict) -> list[Parameter]:
    """``expand_plist({'Class': {'Field': (min, max), ...}})`` as in BESOS (object name ``'*'``)."""
    out = []
    for class_name, fields in plist.items():
        for field_name, bounds in fields.items():
            out.append(
                Parameter(
                    FieldSelector(class_name, "*", field_name),
                    RangeParameter(*bounds) if isinstance(bounds, tuple) else CategoryParameter(list(bounds)),
                    name=f"{class_name} {field_name}",
                )
            )
    return out


def wwr(*args, **kwargs):
    raise NotImplementedError(
        "wwr (window-to-wall ratio through geometry edits) is not provided by bdopt; "
        "use a GenericSelector with your own edit function"
    )
