"""Evaluators: turn a table of parameter values into a table of objective values."""

from __future__ import annotations

import shutil
import threading
import warnings
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .eplus import EnergyPlus, EnergyPlusNotFound, find_installations, select, upgrade_text
from .eplus.version import Version
from .idf import IDF
from .problem import Problem
from .simulation.cache import RunCache, file_digest, make_key
from .simulation.runner import LocalRunner, Runner
from .simulation.sql import RunResults, read_objective


class VersionMismatch(RuntimeError):
    """The model's version differs from the EnergyPlus engine and ``auto_upgrade`` is off."""


@dataclass
class RunRecord:
    """One evaluated row (kept in ``evaluator.history``)."""

    key: str
    cached: bool
    ok: bool
    elapsed: float = 0.0
    run_dir: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    eplus_warnings: int = 0  #: ``Warning`` messages EnergyPlus wrote to ``eplusout.err``
    eplus_severe: int = 0  #: ``Severe`` errors EnergyPlus wrote (a run can finish with some)
    parameters: dict | None = None  #: the design that was simulated (parameter name -> value)


class Evaluator:
    """Base class: ``evaluate(df)`` maps parameter rows to objective rows."""

    problem: Problem

    def evaluate(self, df: pd.DataFrame, keep_input: bool = False) -> pd.DataFrame:
        """Objective values for a table of parameter values.

        Parameters
        ----------
        df : pandas.DataFrame
            One column per parameter (names as in the problem), one row per design.
        keep_input : bool
            Also return the parameter columns.

        Returns
        -------
        pandas.DataFrame
            One column per objective, same index as ``df``.
        """
        raise NotImplementedError

    def evaluate_unit(self, U: np.ndarray) -> np.ndarray:
        """Objective values ``(n, n_obj)`` for unit-cube points (what optimizers/samplers use)."""
        df = self.problem.decode(U)
        return self.evaluate(df)[self.problem.objective_names].to_numpy(dtype=float)

    def __call__(self, row: Mapping | pd.Series | list) -> list[float]:
        if not isinstance(row, Mapping | pd.Series):
            row = dict(zip(self.problem.parameter_names, row, strict=True))
        out = self.evaluate(pd.DataFrame([dict(row)]))
        return out.iloc[0].tolist()


class FunctionEvaluator(Evaluator):
    """Evaluate with a python function (a cheap analytic model, a surrogate ...).

    ``func(row)`` gets ``{parameter name: value}`` and returns one value per objective; with
    ``vectorized=True`` it gets the whole parameter table and returns ``(n, n_obj)``.
    """

    def __init__(self, problem: Problem, func: Callable, vectorized: bool = False):
        self.problem, self.func, self.vectorized = problem, func, vectorized

    def evaluate(self, df: pd.DataFrame, keep_input: bool = False) -> pd.DataFrame:
        """Evaluate every row with the python function (see class docstring)."""
        names = self.problem.objective_names
        if self.vectorized:
            arr = np.asarray(self.func(df[self.problem.parameter_names]), dtype=float)
        else:
            rows = df[self.problem.parameter_names].to_dict("records")
            arr = np.array([np.atleast_1d(self.func(r)) for r in rows], dtype=float)
        out = pd.DataFrame(arr.reshape(len(df), len(names)), columns=names, index=df.index)
        return pd.concat([df, out], axis=1) if keep_input else out


class EnergyPlusEvaluator(Evaluator):
    """Run EnergyPlus for every parameter row and read the objectives.

    Parameters
    ----------
    problem, building, epw:
        the problem, the model (:class:`~bdopt.idf.IDF` or an ``.idf`` path) and the weather file.
    eplus:
        which EnergyPlus to use: an :class:`~bdopt.EnergyPlus`, a version string (``"9.6"``),
        a path, or ``None`` to pick the installed version that matches the model.
    auto_upgrade:
        if no installed version matches the model, upgrade the model to the newest installed version
        with the ``IDFVersionUpdater`` tools (upgrade only; downgrades are impossible).
    out_dir, err_dir:
        where run directories go (removed after a successful run unless ``keep_dirs``); failed runs
        are kept under ``err_dir`` (default ``out_dir/failed``).
    processes:
        parallel EnergyPlus processes.
    cache:
        ``True`` (``out_dir/cache``), a directory, or ``False``.
    runner:
        a custom :class:`~bdopt.simulation.Runner` (e.g. :class:`~bdopt.simulation.CommandRunner`).
    on_error:
        ``'nan'`` (default: failed rows give NaN and a warning) or ``'raise'``.
    on_severe:
        what to do with a run that finishes but reports *severe* errors (e.g. a zone whose warm-up did not
        converge): ``'warn'`` (default: keep the result and raise a ``UserWarning`` quoting the messages),
        ``'nan'`` (treat the run as failed: NaN, run directory kept in ``err_dir``) or ``'ignore'``.
    """

    def __init__(
        self,
        problem: Problem,
        building: IDF | str | Path,
        epw: str | Path | None,
        eplus: EnergyPlus | str | Version | Path | None = None,
        out_dir: str | Path = "bdopt_runs",
        err_dir: str | Path | None = None,
        processes: int = 1,
        cache: bool | str | Path = True,
        keep_dirs: bool = False,
        timeout: float | None = None,
        runner: Runner | None = None,
        auto_upgrade: bool = False,
        design_day: bool = False,
        progress: bool = False,
        on_error: str = "nan",
        on_severe: str = "warn",
        search_paths: list[str | Path] | None = None,
    ):
        self.problem = problem
        self.epw = Path(epw).resolve() if epw is not None else None
        if self.epw is not None and not self.epw.is_file():
            raise FileNotFoundError(f"weather file not found: {self.epw}")
        if on_error not in ("nan", "raise"):
            raise ValueError("on_error must be 'nan' or 'raise'")
        if on_severe not in ("warn", "nan", "ignore"):
            raise ValueError("on_severe must be 'warn', 'nan' or 'ignore'")
        self.on_severe = on_severe
        self.out_dir = Path(out_dir).resolve()
        self.err_dir = Path(err_dir).resolve() if err_dir else self.out_dir / "failed"
        self.processes = max(1, int(processes))
        self.keep_dirs, self.timeout, self.design_day = keep_dirs, timeout, design_day
        self.progress = progress
        self.on_error = on_error
        self.out_dir.mkdir(parents=True, exist_ok=True)

        idf = building if isinstance(building, IDF) else IDF.from_file(building)
        self.runner = runner or LocalRunner(
            self._pick_engine(eplus, idf, auto_upgrade, search_paths), timeout=timeout
        )
        self.eplus = self.runner.eplus
        self.idf = self._prepare_model(idf, auto_upgrade, search_paths)
        problem.validate(self.idf, self.eplus.idd)

        self._expand = self.idf.has_class_prefix("HVACTemplate:")
        self._signature = "|".join(
            f"{type(o).__name__}:{o.name}:{o.idf_outputs(self.eplus.version, self.idf)}:{getattr(o, 'unit', '')}:"
            f"{getattr(o, 'aggregate', '')}:{getattr(o, 'missing', '')}"
            for o in problem.objectives
        )
        if cache is True:
            self.cache: RunCache | None = RunCache(self.out_dir / "cache")
        elif cache:
            self.cache = RunCache(cache)
        else:
            self.cache = None
        self._epw_digest = file_digest(self.epw) if self.epw else "no-epw"

        self.history: list[RunRecord] = []
        self.n_simulations = 0
        self.n_cache_hits = 0
        self.simulation_time = 0.0
        self._lock = threading.Lock()

    # -- engine / model -------------------------------------------------------------------------
    @staticmethod
    def _pick_engine(eplus, idf: IDF, auto_upgrade: bool, search_paths) -> EnergyPlus:
        if isinstance(eplus, EnergyPlus):
            return eplus
        if isinstance(eplus, Path) or (isinstance(eplus, str) and Path(eplus).exists()):
            return EnergyPlus.find(path=eplus)
        installs = find_installations(search_paths)
        if eplus is not None:  # a version string
            return select(installs, eplus)
        v = idf.version
        if v is None:
            return select(installs, None)
        match = [e for e in installs if (e.version.major, e.version.minor) == (v.major, v.minor)]
        if match:
            return match[-1]
        newer = [e for e in installs if e.version > v]
        if auto_upgrade and newer:
            return newer[-1]
        have = ", ".join(sorted({e.version.short for e in installs})) or "none"
        raise EnergyPlusNotFound(
            f"the model is for EnergyPlus {v.short} but no such version is installed (found: {have}). "
            "Install it (tools/install_eplus.py), pass eplus=..., or use auto_upgrade=True to convert "
            "the model to the newest installed version."
        )

    def _prepare_model(self, idf: IDF, auto_upgrade: bool, search_paths) -> IDF:
        idf = idf.copy()
        v, e = idf.version, self.eplus.version
        if v is not None and (v.major, v.minor) != (e.major, e.minor):
            if not auto_upgrade:
                raise VersionMismatch(
                    f"model version {v.short} != EnergyPlus {e.short}; use auto_upgrade=True or an "
                    "EnergyPlus of the model's version"
                )
            idf = self._upgraded(idf, v, e, search_paths)
        elif v is None:
            warnings.warn("model has no Version object; assuming it matches the EnergyPlus version", stacklevel=2)
        self._add_outputs(idf)
        return idf

    def _upgraded(self, idf: IDF, v: Version, e: Version, search_paths) -> IDF:
        """Upgrade ``idf`` to ``e``; the (slow: ~30 s per transition step on a large model) result is
        cached on disk so it is only paid once per model and target version."""
        src = idf.to_text()
        digest = make_key(src, "", f"{v}->{e}", "upgrade")[:24]
        cached = self.out_dir / "upgraded" / f"{digest}.idf"
        if cached.is_file():
            return IDF.from_text(cached.read_text(encoding="latin-1"), idf.path)
        text = upgrade_text(src, v, e, find_installations(search_paths))
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(text, encoding="latin-1", errors="replace")
        return IDF.from_text(text, idf.path)

    def _add_outputs(self, idf: IDF) -> None:
        """Add ``Output:SQLite`` and the ``Output:Meter`` / ``Output:Variable`` objects the objectives need.

        Nothing is added that EnergyPlus would complain about: an existing identical request is kept, and an
        ``Output:Meter:MeterFileOnly`` for the same meter and frequency (which EnergyPlus warns about when the
        same meter is also requested with ``Output:Meter``) is replaced by the ``Output:Meter`` we add.
        """
        if not idf.by_class("Output:SQLite"):
            idf.add("Output:SQLite", "SimpleAndTabular")

        def key_of(o) -> tuple[str, tuple[str, ...]]:
            return o.class_name.lower(), tuple(f.strip().lower() for f in o.fields[:3])

        have = {key_of(o) for o in idf.objects if o.class_name.lower().startswith("output:")}
        for obj in self.problem.objectives:
            for cls, *fields in obj.idf_outputs(self.eplus.version, idf):
                key = (cls.lower(), tuple(str(f).strip().lower() for f in fields[:3]))
                if key in have:
                    continue
                if cls.lower() == "output:meter":
                    meter, freq = key[1][0], key[1][1] if len(key[1]) > 1 else ""
                    idf.remove([
                        o for o in idf.by_class("Output:Meter:MeterFileOnly")
                        if o.get(0).strip().lower() == meter and o.get(1).strip().lower() == freq
                    ])
                idf.add(cls, *fields)
                have.add(key)

    # -- evaluation -----------------------------------------------------------------------------
    def _row_model(self, row: Mapping) -> str:
        idf = self.idf.copy()
        self.problem.apply(idf, row, self.eplus.idd)
        return idf.to_text()

    def _simulate(self, key: str, text: str) -> tuple[dict | None, RunRecord]:
        run_dir = self.out_dir / f"run_{key[:12]}"
        outcome = self.runner.run(text, self.epw, run_dir, self._expand, self.design_day)
        rec = RunRecord(key, False, outcome.ok, outcome.elapsed, str(run_dir), list(outcome.errors),
                        eplus_warnings=outcome.warnings, eplus_severe=outcome.severe)
        severe_msg = ""
        if outcome.ok and outcome.severe:
            severe_msg = (
                f"EnergyPlus finished run {run_dir.name} with {outcome.severe} severe error(s): "
                + " | ".join(outcome.severe_messages[:2])
            )
            if self.on_severe == "warn":
                rec.warnings.append(severe_msg)
            elif self.on_severe == "nan":
                rec.ok = False
                rec.errors.append(severe_msg)
        with self._lock:
            self.n_simulations += 1
            self.simulation_time += outcome.elapsed
        if not rec.ok:
            self.err_dir.mkdir(parents=True, exist_ok=True)
            dest = self.err_dir / run_dir.name
            shutil.rmtree(dest, ignore_errors=True)
            shutil.move(str(run_dir), dest)
            rec.run_dir = str(dest)
            return None, rec
        results = RunResults(outcome.sql_path)
        values: dict[str, float] = {}
        try:
            for obj in self.problem.objectives:
                values[obj.name] = read_objective(obj, results, rec.warnings)
        except Exception as e:  # noqa: BLE001 - report as a failed row, keep the batch going
            rec.ok = False
            rec.errors.append(f"reading results: {e}")
            return None, rec
        finally:
            if not self.keep_dirs and rec.ok:
                shutil.rmtree(run_dir, ignore_errors=True)
        return values, rec

    def evaluate(
        self,
        df: pd.DataFrame,
        keep_input: bool = False,
        processes: int | None = None,
        keep_dirs: bool | None = None,
    ) -> pd.DataFrame:
        """Run EnergyPlus for every row of ``df`` and read the objectives.

        Identical rows are simulated once, finished simulations are taken from the cache, the rest run in
        parallel (``processes``). A failed run gives NaN for its row (and a warning), or raises with
        ``on_error="raise"``.

        Parameters
        ----------
        df : pandas.DataFrame
            One column per parameter, one row per design.
        keep_input : bool
            Also return the parameter columns.
        processes : int, optional
            Override the number of parallel EnergyPlus processes for this call.
        keep_dirs : bool, optional
            Keep every run directory (default: only failed runs are kept).

        Returns
        -------
        pandas.DataFrame
            One column per objective, same index as ``df``.
        """
        if keep_dirs is not None:
            self.keep_dirs = keep_dirs
        names = self.problem.objective_names
        rows = df[self.problem.parameter_names].to_dict("records")
        texts = [self._row_model(r) for r in rows]
        keys = [
            make_key(t, self._epw_digest, str(self.eplus.version), self._signature,
                     f"x{int(self._expand)}d{int(self.design_day)}")
            for t in texts
        ]
        results: dict[str, dict | None] = {}
        todo: dict[str, str] = {}
        cached_recs: list[RunRecord] = []
        for k, t in zip(keys, texts, strict=True):
            if k in results or k in todo:
                continue
            hit = self.cache.get(k) if self.cache is not None else None
            if hit is not None and all(n in hit for n in names):
                sev = int(hit.get("_eplus_severe", 0))
                rec = RunRecord(k, True, True, eplus_warnings=int(hit.get("_eplus_warnings", 0)), eplus_severe=sev)
                if sev:
                    msg = f"cached result of a run that had {sev} severe EnergyPlus error(s)"
                    if self.on_severe == "nan":
                        rec.ok = False
                        rec.errors.append(msg + " (on_severe='nan')")
                    elif self.on_severe == "warn":
                        rec.warnings.append(msg)
                results[k] = hit if rec.ok else None
                cached_recs.append(rec)
                self.n_cache_hits += 1
                self.history.append(rec)
            else:
                todo[k] = t
        n_workers = max(1, processes or self.processes)
        items = list(todo.items())
        bar = None
        if self.progress and items:
            try:
                from tqdm.auto import tqdm

                bar = tqdm(total=len(items), desc="EnergyPlus")
            except ImportError:
                bar = None

        def work(item):
            k, t = item
            values, rec = self._simulate(k, t)
            if bar is not None:
                bar.update(1)
            return k, values, rec

        if n_workers == 1 or len(items) <= 1:
            outs = [work(i) for i in items]
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                outs = list(pool.map(work, items))
        if bar is not None:
            bar.close()
        failed: list[RunRecord] = []
        row_of = dict(zip(keys, rows, strict=True))
        for rec in cached_recs:
            rec.parameters = dict(row_of[rec.key])
            if not rec.ok:
                failed.append(rec)
        for k, values, rec in outs:
            rec.parameters = dict(row_of[k])
            self.history.append(rec)
            results[k] = values
            if values is not None and self.cache is not None:
                self.cache.put(k, {**values, "_eplus_severe": rec.eplus_severe, "_eplus_warnings": rec.eplus_warnings})
            if values is None:
                failed.append(rec)
        if failed:
            msg = "; ".join(f"{r.run_dir or r.key[:12]}: {' | '.join(r.errors[:2])}" for r in failed[:3])
            if self.on_error == "raise":
                raise RuntimeError(f"{len(failed)} EnergyPlus run(s) failed: {msg}")
            warnings.warn(f"{len(failed)} EnergyPlus run(s) failed, result is NaN: {msg}", stacklevel=2)
        for rec in [*(r for _, _, r in outs), *cached_recs]:
            for w in rec.warnings:
                warnings.warn(w, stacklevel=2)
        arr = np.array(
            [[(results[k] or {}).get(n, np.nan) for n in names] for k in keys], dtype=float
        ).reshape(len(keys), len(names))
        out = pd.DataFrame(arr, columns=names, index=df.index)
        return pd.concat([df, out], axis=1) if keep_input else out

    @property
    def failures(self) -> list[RunRecord]:
        """Records of the runs that failed (``errors`` has EnergyPlus' messages)."""
        return [r for r in self.history if not r.ok]

    def stats(self) -> dict:
        """Counters: simulations run, cache hits, failures, simulation seconds, EnergyPlus version."""
        return {
            "simulations": self.n_simulations,
            "cache_hits": self.n_cache_hits,
            "failed": len(self.failures),
            "simulation_seconds": round(self.simulation_time, 2),
            "eplus_version": str(self.eplus.version),
            "energyplus_warnings": sum(r.eplus_warnings for r in self.history),
            "energyplus_severe": sum(r.eplus_severe for r in self.history),
        }
