"""Upgrade an IDF to a newer EnergyPlus version with the ``IDFVersionUpdater`` tools shipped in every
EnergyPlus installation (``Transition-V9-6-0-to-V22-1-0`` ...).

An installation only contains the transitions of the last few releases (EnergyPlus 9.6 covers 7.2 ->
9.6, 25.2 covers 9.0 -> 25.2), so the chain is assembled across *all* installations found.
Downgrading is not possible and raises an error.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .install import EnergyPlus
from .version import Version

_TRANSITION = re.compile(r"^Transition-V(\d+-\d+-\d+)-to-V(\d+-\d+-\d+)(?:\.exe)?$")
_IS_WINDOWS = sys.platform.startswith("win")


class TransitionError(RuntimeError):
    pass


def _edges(installs: list[EnergyPlus]) -> dict[tuple[int, int], list[tuple[Path, Version]]]:
    """``{(major, minor) of source: [(transition executable, target version)]}``."""
    edges: dict[tuple[int, int], list[tuple[Path, Version]]] = {}
    for ep in installs:
        d = ep.updater_dir
        if not d.is_dir():
            continue
        for f in d.iterdir():
            m = _TRANSITION.match(f.name)
            if not m:
                continue
            src, dst = Version.parse(m.group(1)), Version.parse(m.group(2))
            edges.setdefault((src.major, src.minor), []).append((f, dst))
    return edges


def plan(source: Version, target: Version, installs: list[EnergyPlus]) -> list[Path]:
    """Ordered transition executables taking ``source`` to ``target`` (``[]`` if equal)."""
    if (source.major, source.minor) == (target.major, target.minor):
        return []
    if source > target:
        raise TransitionError(f"cannot downgrade an IDF from {source.short} to {target.short}")
    edges = _edges(installs)
    steps: list[Path] = []
    cur = (source.major, source.minor)
    goal = (target.major, target.minor)
    while cur != goal:
        options = [e for e in edges.get(cur, []) if (e[1].major, e[1].minor) <= goal]
        if not options:
            raise TransitionError(
                f"no transition available from IDF version {cur[0]}.{cur[1]} towards "
                f"{goal[0]}.{goal[1]} in the installed EnergyPlus versions; install an EnergyPlus "
                "version that ships the missing step (e.g. a release in between)"
            )
        exe, dst = max(options, key=lambda e: e[1])
        steps.append(exe)
        cur = (dst.major, dst.minor)
    return steps


def upgrade_text(
    idf_text: str, source: Version, target: Version, installs: list[EnergyPlus], timeout: float = 600
) -> str:
    """Return ``idf_text`` upgraded from ``source`` to ``target`` version."""
    steps = plan(source, target, installs)
    if not steps:
        return idf_text
    with tempfile.TemporaryDirectory(prefix="bdopt_transition_") as tmp:
        work = Path(tmp) / "model.idf"
        work.write_text(idf_text, encoding="latin-1", errors="replace")
        for exe in steps:
            proc = subprocess.run(
                [str(exe), str(work)],
                cwd=exe.parent,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode != 0:
                raise TransitionError(f"{exe.name} failed ({proc.returncode}): {proc.stdout[-500:]}")
            # the updater rewrites the file in place (and leaves *.idfnew / *.idfold next to it)
        return work.read_text(encoding="latin-1")
