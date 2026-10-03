"""A small, version-agnostic IDF model.

IDF files are plain text: ``Class, field1, field2, ...;`` with ``!`` comments. This module parses
them without needing an IDD (field *names* are resolved separately through :class:`bdopt.eplus.IDD`),
so a model can be loaded, edited and written for any EnergyPlus version, and there is no global
state shared between versions (a known limitation of IDD-based editors).

Comments are not preserved when a model is written back; the simulation result does not depend on them.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .eplus.version import Version


@dataclass
class IDFObject:
    """One IDF object: its class name and its field values (as strings)."""

    class_name: str
    fields: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.fields[0] if self.fields else ""

    def get(self, index: int, default: str = "") -> str:
        return self.fields[index] if index < len(self.fields) else default

    def set(self, index: int, value) -> None:
        while len(self.fields) <= index:
            self.fields.append("")
        self.fields[index] = _fmt(value)

    def to_text(self) -> str:
        if not self.fields:
            return f"{self.class_name};"
        lines = [f"{self.class_name},"]
        for i, f in enumerate(self.fields):
            sep = ";" if i == len(self.fields) - 1 else ","
            lines.append(f"    {f}{sep}")
        return "\n".join(lines)


def _fmt(value) -> str:
    """Format a field value for an IDF."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return repr(value) if value != int(value) or abs(value) > 1e15 else f"{value:.1f}"
    return str(value)


def _tokenize(text: str) -> Iterator[list[str]]:
    """Yield ``[class, field, field, ...]`` token lists from IDF text."""
    buf: list[str] = []
    cur: list[str] = []
    for line in text.splitlines():
        bang = line.find("!")
        if bang >= 0:
            line = line[:bang]
        for ch in line:
            if ch == ",":
                buf.append("".join(cur).strip())
                cur = []
            elif ch == ";":
                buf.append("".join(cur).strip())
                cur = []
                yield buf
                buf = []
            else:
                cur.append(ch)
        cur.append(" ")
    tail = "".join(cur).strip()
    if buf or tail:  # object without terminating ';' at EOF
        buf.append(tail)
        yield buf


class IDF:
    """An EnergyPlus input model."""

    def __init__(self, objects: Iterable[IDFObject] = (), path: Path | None = None):
        self.objects: list[IDFObject] = list(objects)
        self.path = path

    # -- io ------------------------------------------------------------------------------------
    @classmethod
    def from_text(cls, text: str, path: Path | None = None) -> IDF:
        objs = [IDFObject(toks[0], toks[1:]) for toks in _tokenize(text) if toks and toks[0]]
        return cls(objs, path)

    @classmethod
    def from_file(cls, path: str | Path) -> IDF:
        p = Path(path)
        return cls.from_text(p.read_text(encoding="latin-1"), p)

    def to_text(self) -> str:
        return "\n\n".join(o.to_text() for o in self.objects) + "\n"

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_text(), encoding="latin-1", errors="replace")
        return p

    def copy(self) -> IDF:
        return IDF(copy.deepcopy(self.objects), self.path)

    # -- queries -------------------------------------------------------------------------------
    @property
    def version(self) -> Version | None:
        for o in self.objects:
            if o.class_name.lower() == "version":
                try:
                    return Version.parse(o.get(0))
                except ValueError:
                    return None
        return None

    def by_class(self, class_name: str) -> list[IDFObject]:
        key = class_name.lower()
        return [o for o in self.objects if o.class_name.lower() == key]

    def classes(self) -> set[str]:
        return {o.class_name for o in self.objects}

    def has_class_prefix(self, prefix: str) -> bool:
        p = prefix.lower()
        return any(o.class_name.lower().startswith(p) for o in self.objects)

    def find(
        self,
        class_name: str,
        object_name: str | None = "*",
        where: Callable[[IDFObject], bool] | None = None,
    ) -> list[IDFObject]:
        """Objects of a class, filtered by name (``'*'``/``None`` = all, case-insensitive)."""
        objs = self.by_class(class_name)
        if object_name not in (None, "*"):
            want = str(object_name).strip().lower()
            objs = [o for o in objs if o.name.strip().lower() == want]
        if where is not None:
            objs = [o for o in objs if where(o)]
        return objs

    # -- edits ---------------------------------------------------------------------------------
    def add(self, class_name: str, *fields) -> IDFObject:
        obj = IDFObject(class_name, [_fmt(f) for f in fields])
        self.objects.append(obj)
        return obj

    def remove(self, objs: Iterable[IDFObject]) -> None:
        ids = {id(o) for o in objs}
        self.objects = [o for o in self.objects if id(o) not in ids]

    def __len__(self) -> int:
        return len(self.objects)

    def __repr__(self) -> str:
        return f"IDF({len(self.objects)} objects, version={self.version})"


_SAFE = re.compile(r"[^A-Za-z0-9_.\-]+")


def safe_name(text: str) -> str:
    """Make ``text`` usable as a file/dir name."""
    return _SAFE.sub("_", text)
