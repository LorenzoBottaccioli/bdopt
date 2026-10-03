"""Minimal parser for ``Energy+.idd``.

Only what bdopt needs: object class names and the ordered list of field names per class, so that
a parameter can address ``(class, field name)`` independently of the EnergyPlus version.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterator
from pathlib import Path

_FIELD_LINE = re.compile(r"^\s*([AN]\d+)\s*[,;]\s*(?:\\field\s+(.*))?$")
_FIELD_START = re.compile(r"^\s*[AN]\d+\s*[,;]")
_FIELD_NAME_ONLY = re.compile(r"^\s+\\field\s+(.*)$")


def norm(name: str) -> str:
    """Normalise a class/field name for case- and whitespace-insensitive comparison."""
    return re.sub(r"\s+", " ", name.strip()).lower()


class IDD:
    """Parsed Input Data Dictionary: ``{class name: [field names]}``."""

    def __init__(self, classes: dict[str, list[str]], version: str | None = None, path=None):
        self._classes = classes
        self._lookup = {norm(k): k for k in classes}
        self.version = version
        self.path = path

    @classmethod
    def from_text(cls, text: str, path=None) -> IDD:
        """Parse IDD text."""
        classes: dict[str, list[str]] = {}
        version = None
        current: list[str] | None = None
        pending_unnamed = False
        for raw in text.splitlines():
            line = raw.rstrip()
            if not line:
                continue
            if line.startswith("!"):
                if version is None and line.startswith("!IDD_Version"):
                    version = line.split(None, 1)[1].strip() if " " in line else None
                continue
            if line[0] not in " \t\\" and not _FIELD_START.match(line):
                name = re.split(r"[,;]", line, maxsplit=1)[0].strip()
                if name:
                    current = classes.setdefault(name, [])
                    pending_unnamed = False
                continue
            if current is None:
                continue
            m = _FIELD_LINE.match(line)
            if m:
                fname = (m.group(2) or "").strip()
                current.append(fname)
                pending_unnamed = not fname
                continue
            if pending_unnamed:
                m = _FIELD_NAME_ONLY.match(line)
                if m:
                    current[-1] = m.group(1).strip()
                    pending_unnamed = False
        return cls(classes, version, path)

    @classmethod
    @functools.lru_cache(maxsize=16)
    def load(cls, path: str | Path) -> IDD:
        p = Path(path)
        return cls.from_text(p.read_text(encoding="latin-1"), path=p)

    # -- queries -------------------------------------------------------------------------------
    def __contains__(self, class_name: str) -> bool:
        return norm(class_name) in self._lookup

    def __iter__(self) -> Iterator[str]:
        return iter(self._classes)

    def canonical_class(self, class_name: str) -> str:
        """Class name as spelled in the IDD (lookup is case/whitespace-insensitive)."""
        try:
            return self._lookup[norm(class_name)]
        except KeyError:
            raise KeyError(f"class {class_name!r} not in IDD {self.version}") from None

    def fields(self, class_name: str) -> list[str]:
        """Ordered field names of a class."""
        return self._classes[self.canonical_class(class_name)]

    def field_index(self, class_name: str, field_name: str) -> int | None:
        """Index of ``field_name`` in objects of ``class_name`` (``None`` if absent)."""
        target = norm(field_name)
        for i, f in enumerate(self.fields(class_name)):
            if norm(f) == target:
                return i
        return None
