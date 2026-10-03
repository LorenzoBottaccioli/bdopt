"""Locate EnergyPlus installations (any OS, any version >= 8.7).

Resolution order for :meth:`EnergyPlus.find`:

1. an explicit ``path=``;
2. ``BDOPT_EPLUS_PATH`` (``os.pathsep``-separated roots or parent directories), ``ENERGYPLUS_DIR``,
   ``EPLUS_PATH`` (path of the executable) and ``ENERGYPLUS_INSTALL_DIR``;
3. the ``energyplus`` found on ``PATH``;
4. standard install locations (``/usr/local``, ``/opt``, ``/Applications``, ``C:\\``, ``~/.bdopt/eplus`` ...).

No assumption is made about Docker or about a particular path.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .idd import IDD
from .version import MIN_SUPPORTED, Version

_IS_WINDOWS = sys.platform.startswith("win")
_EXE_NAMES = ("energyplus.exe", "EnergyPlus.exe") if _IS_WINDOWS else ("energyplus", "EnergyPlus")
_IDD_VERSION = re.compile(r"!IDD_Version\s+(\S+)")


class EnergyPlusNotFound(RuntimeError):
    """No (matching) EnergyPlus installation could be found."""


@dataclass(frozen=True)
class EnergyPlus:
    """One EnergyPlus installation."""

    root: Path
    version: Version
    executable: Path
    idd_path: Path

    # -- construction --------------------------------------------------------------------------
    @classmethod
    def from_root(cls, root: str | Path) -> EnergyPlus | None:
        """Describe the installation at ``root`` or return ``None`` if it is not one."""
        root = Path(root)
        idd = root / "Energy+.idd"
        if not idd.is_file():
            return None
        exe = next((root / n for n in _EXE_NAMES if (root / n).is_file()), None)
        if exe is None:  # e.g. only ``energyplus-9.6.0`` is present
            cands = sorted(root.glob("energyplus-*")) + sorted(root.glob("energyplus*.exe"))
            exe = next((c for c in cands if c.is_file() and c.suffix != ".1"), None)
        if exe is None:
            return None
        try:
            with open(idd, encoding="latin-1") as fh:
                head = "".join(fh.readline() for _ in range(5))
            m = _IDD_VERSION.search(head)
            version = Version.parse(m.group(1)) if m else Version.parse(root.name)
        except (OSError, ValueError):
            return None
        return cls(root.resolve(), version, exe.resolve(), idd.resolve())

    @classmethod
    def find(
        cls,
        version: str | Version | None = None,
        path: str | Path | None = None,
        search_paths: list[str | Path] | None = None,
    ) -> EnergyPlus:
        """Find one installation (the newest matching ``version``, or the newest overall)."""
        if path is not None:
            ep = cls.from_root(path)
            if ep is None:
                found = find_installations([path])
                ep = max(found, key=lambda e: e.version) if found else None
            if ep is None:
                raise EnergyPlusNotFound(f"no EnergyPlus installation at {path}")
            if version is not None and not ep.version.matches(version):
                raise EnergyPlusNotFound(f"{path} is EnergyPlus {ep.version}, not {version}")
            return ep
        installs = find_installations(search_paths)
        return select(installs, version)

    # -- helpers -------------------------------------------------------------------------------
    @property
    def idd(self) -> IDD:
        return IDD.load(self.idd_path)

    @property
    def updater_dir(self) -> Path:
        return self.root / "PreProcess" / "IDFVersionUpdater"

    def __str__(self) -> str:
        return f"EnergyPlus {self.version} ({self.root})"


def _env_roots() -> list[Path]:
    roots: list[Path] = []
    for var in ("BDOPT_EPLUS_PATH", "ENERGYPLUS_DIR", "ENERGYPLUS_INSTALL_DIR"):
        for item in os.environ.get(var, "").split(os.pathsep):
            if item.strip():
                roots.append(Path(item.strip()))
    exe = os.environ.get("EPLUS_PATH", "").strip()
    if exe:
        roots.append(Path(exe).resolve().parent)
    return roots


def _standard_parents() -> list[Path]:
    home = Path.home()
    parents = [home / ".bdopt" / "eplus", home]
    if _IS_WINDOWS:
        parents += [Path("C:/"), Path(os.environ.get("ProgramFiles", "C:/Program Files"))]
    elif sys.platform == "darwin":
        parents += [Path("/Applications"), Path("/usr/local")]
    else:
        parents += [Path("/usr/local"), Path("/opt"), Path("/opt/eplus")]
    return parents


def _scan(parent: Path, depth: int = 2) -> list[EnergyPlus]:
    """``parent`` itself, or directories below it (up to ``depth``) that are installations."""
    out: list[EnergyPlus] = []
    ep = EnergyPlus.from_root(parent)
    if ep:
        return [ep]
    if depth <= 0 or not parent.is_dir():
        return out
    try:
        children = sorted(parent.iterdir())
    except OSError:
        return out
    for child in children:
        if not child.is_dir():
            continue
        if child.name.lower().startswith("energyplus") or depth > 1 and child.parent == parent:
            if child.name.lower().startswith(("energyplus", "eplus")) or child.name[0] in "0123456789":
                out += _scan(child, depth - 1)
            elif depth > 1:
                out += _scan(child, depth - 1)
    return out


def find_installations(search_paths: list[str | Path] | None = None) -> list[EnergyPlus]:
    """All EnergyPlus >= 8.7 installations found, sorted by version (oldest first)."""
    candidates: list[Path] = [Path(p) for p in (search_paths or [])]
    candidates += _env_roots()
    which = shutil.which("energyplus")
    if which:
        candidates.append(Path(which).resolve().parent)
    candidates += _standard_parents()
    seen: dict[Path, EnergyPlus] = {}
    for c in candidates:
        for ep in _scan(c):
            if ep.version >= MIN_SUPPORTED:
                seen.setdefault(ep.root, ep)
    return sorted(seen.values(), key=lambda e: (e.version, str(e.root)))


def select(installs: list[EnergyPlus], version: str | Version | None = None) -> EnergyPlus:
    """Pick the newest installation matching ``version`` (``None``: newest)."""
    if not installs:
        raise EnergyPlusNotFound(
            "no EnergyPlus installation found. Install one (e.g. `python tools/install_eplus.py 9.6`), "
            "or set ENERGYPLUS_DIR / BDOPT_EPLUS_PATH, or pass EnergyPlus(path=...)."
        )
    if version is None:
        return installs[-1]
    ok = [e for e in installs if e.version.matches(version)]
    if not ok:
        have = ", ".join(sorted({e.version.short for e in installs}))
        raise EnergyPlusNotFound(f"EnergyPlus {version} not found (installed: {have})")
    return ok[-1]
