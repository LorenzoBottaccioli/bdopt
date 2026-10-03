"""Names that changed between EnergyPlus versions (8.7 ... 25.x).

The tables were derived by diffing the IDDs of EnergyPlus 8.7, 9.0, 9.6, 23.1, 24.2 and 25.2
(``tools/idd_diff.py`` regenerates the report for any pair of versions). Only *renames* of the
same quantity are listed; fields that were added or removed need no alias because fields are always
addressed by name, never by position.

With an alias a notebook written for one version keeps working on the others::

    FieldSelector("Lights", "*", "Watts per Zone Floor Area")   # also works on 24.2+
"""

from __future__ import annotations

import difflib
import re

from .idd import IDD, norm
from .version import Version

#: Groups of equivalent field names. ``classes=None`` means every class.
#: ``(classes, names)`` -- any member of ``names`` is accepted for any other member.
FIELD_ALIAS_GROUPS: list[tuple[set[str] | None, tuple[str, ...]]] = [
    (None, ("Watts per Zone Floor Area", "Watts per Floor Area")),  # Lights, *Equipment (24.2)
    (
        {"zoneinfiltration:designflowrate"},
        ("Flow per Zone Floor Area", "Flow Rate per Zone Floor Area", "Flow Rate per Floor Area"),
    ),
    (
        {"zoneventilation:designflowrate"},
        ("Flow Rate per Zone Floor Area", "Flow per Zone Floor Area", "Flow Rate per Floor Area"),
    ),
    (
        {"zoneinfiltration:designflowrate"},
        ("Flow per Exterior Surface Area", "Flow Rate per Exterior Surface Area"),
    ),
    (
        None,
        (
            "Zone or ZoneList Name",
            "Zone or ZoneList or Space or SpaceList Name",
        ),
    ),
    ({"zoneinfiltration:effectiveleakagearea"}, ("Zone Name", "Zone or Space Name")),
    ({"output:meter", "output:meter:cumulative", "output:meter:meterfileonly"}, ("Name", "Key Name")),
]

#: Classes that were renamed between versions (first match present in the IDD wins).
CLASS_ALIAS_GROUPS: list[tuple[str, ...]] = [
    ("WindowShadingControl", "WindowProperty:ShadingControl"),  # renamed in 9.0
]


def class_candidates(class_name: str) -> list[str]:
    """A class name followed by its known renamed equivalents."""
    out = [class_name]
    for group in CLASS_ALIAS_GROUPS:
        if norm(class_name) in {norm(g) for g in group}:
            out += [g for g in group if norm(g) != norm(class_name)]
    return out


def resolve_class(idd: IDD, class_name: str) -> str:
    """Canonical class name for this IDD, applying class aliases."""
    for cand in class_candidates(class_name):
        if cand in idd:
            return idd.canonical_class(cand)
    close = difflib.get_close_matches(class_name, list(idd), n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise KeyError(f"class {class_name!r} does not exist in EnergyPlus {idd.version}.{hint}")


def resolve_field(idd: IDD, class_name: str, field_name: str) -> tuple[str, int]:
    """Return ``(canonical class, field index)`` for a class/field in this IDD version."""
    cls = resolve_class(idd, class_name)
    idx = idd.field_index(cls, field_name)
    if idx is not None:
        return cls, idx
    wanted = norm(field_name)
    for classes, names in FIELD_ALIAS_GROUPS:
        if classes is not None and norm(cls) not in classes:
            continue
        if wanted in {norm(n) for n in names}:
            for alt in names:
                idx = idd.field_index(cls, alt)
                if idx is not None:
                    return cls, idx
    fields = idd.fields(cls)
    close = difflib.get_close_matches(field_name, fields, n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise KeyError(
        f"field {field_name!r} does not exist in class {cls!r} of EnergyPlus {idd.version}.{hint}"
    )


# -- meters --------------------------------------------------------------------------------------
_DISTRICT = re.compile(r"(?<![A-Za-z])District(Heating|Cooling)(Water|Steam)?(?![A-Za-z])")

#: First EnergyPlus version in which ``DistrictHeating`` is split into ``DistrictHeatingWater`` and
#: ``DistrictHeatingSteam`` (IDD classes ``DistrictHeating:Water`` / ``:Steam``). Measured by running a model
#: with district heating on 8.7, 9.0, 9.6, 22.1, 22.2, 23.1, 23.2, 24.1, 24.2 and 25.2: up to 23.1 only
#: ``DistrictHeating:Facility`` exists, from 23.2 only ``DistrictHeatingWater:Facility`` (and ``...Steam``).
DISTRICT_HEATING_SPLIT = Version(23, 2, 0)


def meter_candidates(name: str, version: Version | None = None, has_steam: bool = True) -> list[str]:
    """Meter names to request for ``name`` on a given EnergyPlus version.

    ``DistrictHeating:Facility`` is ``DistrictHeatingWater:Facility`` (and ``...Steam``) from 23.2. Asking
    EnergyPlus for a meter that does not exist makes it print a warning, so with a known ``version`` only the
    names valid there are returned (the steam meter only if ``has_steam``, i.e. the model has a steam district
    heating plant). With ``version=None`` every variant is returned, which is what is needed to *read* results.
    """
    m = _DISTRICT.search(name)
    if not m or m.group(1) != "Heating":
        return [name]
    pre, post, kind = name[: m.start()], name[m.end() :], m.group(2)
    variants = {"": f"{pre}DistrictHeating{post}", "Water": f"{pre}DistrictHeatingWater{post}",
                "Steam": f"{pre}DistrictHeatingSteam{post}"}
    if version is None:
        return list(variants.values())
    if version < DISTRICT_HEATING_SPLIT:
        return [variants[""]]
    if kind == "Steam":
        return [variants["Steam"]]
    if kind == "Water":
        return [variants["Water"]]
    return [variants["Water"], *([variants["Steam"]] if has_steam else [])]
