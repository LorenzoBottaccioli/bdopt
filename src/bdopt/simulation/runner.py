"""Run EnergyPlus.

* :class:`LocalRunner` -- the default: ``energyplus`` as a subprocess of an installation found on the
  machine. Works on Linux, Windows and macOS with any version >= 8.7 (the command line used,
  ``energyplus -w EPW -d OUTDIR [-x] IDF``, is the same in all of them).
* :class:`CommandRunner` -- same, but the command is prefixed, e.g. ``["docker", "exec", "energyplus"]``
  or ``["ssh", "node"]``, with host -> remote path translation. This is how a platform that keeps
  EnergyPlus in a container can still use the library; ``bdopt`` itself does not depend on Docker.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..eplus.install import EnergyPlus


@dataclass
class RunOutcome:
    """Result of one EnergyPlus execution."""

    ok: bool
    run_dir: Path
    returncode: int | None
    elapsed: float
    errors: list[str] = field(default_factory=list)
    warnings: int = 0
    severe: int = 0

    @property
    def sql_path(self) -> Path:
        return self.run_dir / "eplusout.sql"


_COUNTS = re.compile(r"(\d+)\s+Warning;\s+(\d+)\s+Severe Errors")


def parse_err_file(path: Path) -> tuple[list[str], int, int, bool]:
    """``(fatal/severe messages, n warnings, n severe, completed successfully)`` from ``eplusout.err``."""
    if not path.is_file():
        return [], 0, 0, False
    text = path.read_text(errors="replace")
    msgs = [ln.strip() for ln in text.splitlines() if "** Fatal" in ln or "** Severe" in ln]
    m = _COUNTS.search(text)
    warn, sev = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    return msgs, warn, sev, "EnergyPlus Completed Successfully" in text


class Runner:
    """Common logic; subclasses decide how the command line is built."""

    def __init__(self, eplus: EnergyPlus, timeout: float | None = None, extra_args: tuple[str, ...] = ()):
        self.eplus = eplus
        self.timeout = timeout
        self.extra_args = tuple(extra_args)

    # to override ----------------------------------------------------------------------------
    def command(self, idf: Path, epw: Path | None, run_dir: Path, expand: bool, design_day: bool) -> list[str]:
        raise NotImplementedError

    def _args(self, idf, epw, run_dir, expand, design_day) -> list[str]:
        args: list[str] = []
        if epw is not None:
            args += ["-w", str(epw)]
        args += ["-d", str(run_dir)]
        if expand:
            args.append("-x")
        if design_day:
            args.append("-D")
        args += list(self.extra_args)
        args.append(str(idf))
        return args

    # ----------------------------------------------------------------------------------------
    def run(
        self,
        idf_text: str,
        epw: str | Path | None,
        run_dir: str | Path,
        expand_objects: bool = False,
        design_day: bool = False,
    ) -> RunOutcome:
        run_dir = Path(run_dir).resolve()
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True)
        idf_path = run_dir / "in.idf"
        idf_path.write_text(idf_text, encoding="latin-1", errors="replace")
        epw_path = Path(epw).resolve() if epw is not None else None
        cmd = self.command(idf_path, epw_path, run_dir, expand_objects, design_day)
        t0 = time.perf_counter()
        errors: list[str] = []
        rc: int | None = None
        try:
            proc = subprocess.run(
                cmd,
                cwd=run_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                errors="replace",
            )
            rc = proc.returncode
            (run_dir / "stdout.txt").write_text(proc.stdout + proc.stderr, errors="replace")
        except subprocess.TimeoutExpired:
            errors.append(f"timeout after {self.timeout}s")
        except OSError as e:
            errors.append(f"cannot execute {cmd[0]!r}: {e}")
        elapsed = time.perf_counter() - t0
        msgs, warn, sev, completed = parse_err_file(run_dir / "eplusout.err")
        ok = rc == 0 and completed and (run_dir / "eplusout.sql").is_file()
        if not ok:
            errors += msgs[-5:] or ([f"EnergyPlus exited with code {rc}"] if rc not in (0, None) else [])
            if rc == 0 and not completed:
                errors.append("EnergyPlus did not complete successfully")
            if rc == 0 and completed and not (run_dir / "eplusout.sql").is_file():
                errors.append("no eplusout.sql produced")
        return RunOutcome(ok, run_dir, rc, elapsed, errors, warn, sev)


class LocalRunner(Runner):
    """Run an EnergyPlus installation found on this machine."""

    def command(self, idf, epw, run_dir, expand, design_day) -> list[str]:
        return [str(self.eplus.executable), *self._args(idf, epw, run_dir, expand, design_day)]


class CommandRunner(Runner):
    """Run EnergyPlus through a command prefix (container, remote host, wrapper script ...).

    ``path_map`` translates host paths to the paths seen by the command, e.g.
    ``{"/data/runs": "/runs", "/data/weather": "/weather"}``; ``executable`` is the EnergyPlus
    executable *as seen by the command* (default: the local one). ``eplus`` still has to describe a
    local copy of at least ``Energy+.idd`` (used to resolve field names)."""

    def __init__(
        self,
        prefix: list[str],
        eplus: EnergyPlus,
        executable: str | None = None,
        path_map: dict[str | Path, str] | None = None,
        timeout: float | None = None,
        extra_args: tuple[str, ...] = (),
    ):
        super().__init__(eplus, timeout, extra_args)
        self.prefix = list(prefix)
        self.executable = executable or str(eplus.executable)
        self.path_map = {str(Path(k)): v for k, v in (path_map or {}).items()}

    def map_path(self, p: Path) -> str:
        s = str(p)
        for host, remote in sorted(self.path_map.items(), key=lambda kv: -len(kv[0])):
            if s == host or s.startswith(host + os.sep):
                return remote + s[len(host) :].replace(os.sep, "/")
        return s

    def command(self, idf, epw, run_dir, expand, design_day) -> list[str]:
        args = self._args(
            self.map_path(idf),
            self.map_path(epw) if epw is not None else None,
            self.map_path(run_dir),
            expand,
            design_day,
        )
        return [*self.prefix, self.executable, *args]
