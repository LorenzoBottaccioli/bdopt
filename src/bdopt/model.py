"""Loading models."""

from __future__ import annotations

from pathlib import Path

from .idf import IDF


def load_idf(path: str | Path) -> IDF:
    """Read an ``.idf`` file (any EnergyPlus version)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    return IDF.from_file(p)
