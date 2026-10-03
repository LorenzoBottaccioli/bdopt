"""EnergyPlus version handling (8.7 ... 25.x)."""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

_VERSION_RE = re.compile(r"(\d+)[.\-_](\d+)(?:[.\-_](\d+))?")


@functools.total_ordering
@dataclass(frozen=True)
class Version:
    """A ``major.minor.patch`` EnergyPlus version.

    ``patch`` is ``None`` when it was not given (``"9.6"``); such a version matches any patch level
    of the same ``major.minor`` and compares equal to patch ``0``.
    """

    major: int
    minor: int
    patch: int | None = None

    @classmethod
    def parse(cls, text: str | Version) -> Version:
        """Parse ``9.6``, ``9.6.0``, ``V9-6-0``, ``9.6.0-f420c06a69``, ``EnergyPlus-9-6-0`` ..."""
        if isinstance(text, Version):
            return text
        m = _VERSION_RE.search(str(text))
        if not m:
            raise ValueError(f"cannot parse EnergyPlus version from {text!r}")
        patch = int(m.group(3)) if m.group(3) is not None else None
        return cls(int(m.group(1)), int(m.group(2)), patch)

    @property
    def _key(self) -> tuple[int, int, int]:
        return (self.major, self.minor, self.patch or 0)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._key == other._key

    def __hash__(self) -> int:
        return hash(self._key)

    def __lt__(self, other: Version) -> bool:
        return self._key < other._key

    def matches(self, spec: str | Version) -> bool:
        """True if this (installed) version satisfies ``spec`` (patch only compared if given)."""
        spec = Version.parse(spec)
        if (self.major, self.minor) != (spec.major, spec.minor):
            return False
        return spec.patch is None or (self.patch or 0) == spec.patch

    @property
    def short(self) -> str:
        """``"major.minor"``, e.g. ``"9.6"``."""
        return f"{self.major}.{self.minor}"

    @property
    def dashed(self) -> str:
        """``9-6-0``, as used in EnergyPlus install directories and transition programs."""
        return f"{self.major}-{self.minor}-{self.patch or 0}"

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch or 0}"

    def __repr__(self) -> str:
        return f"Version('{self}')"


#: Oldest EnergyPlus version supported by bdopt.
MIN_SUPPORTED = Version(8, 7, 0)
