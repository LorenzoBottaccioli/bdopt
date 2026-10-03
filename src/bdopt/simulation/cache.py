"""On-disk cache of EnergyPlus results.

The key hashes everything that determines a result: the edited IDF text, the weather file content,
the EnergyPlus version and the requested outputs. Re-running a study (or resuming one that crashed)
therefore never repeats a simulation that already finished.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_key(idf_text: str, epw_digest: str, eplus_version: str, signature: str, flags: str = "") -> str:
    h = hashlib.sha256()
    for part in (idf_text, epw_digest, eplus_version, signature, flags):
        h.update(part.encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()


class RunCache:
    """``key -> {objective name: value}`` stored as one small JSON file per key."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return None

    def put(self, key: str, values: dict) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(values, fh)
            os.replace(tmp, p)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def __len__(self) -> int:
        return sum(1 for _ in self.dir.glob("*/*.json"))
