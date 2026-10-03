"""End-to-end study driven by a configuration (dict, JSON or YAML file).

The configuration is plain data, so a web backend can build it from user choices instead of injecting
code into notebooks::

    study = DesignStudy.from_config("study.yaml")
    results = study.run()
    results.save("out/")

See ``examples/study.yaml`` for every option.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .adaptive import AdaptiveResult, AdaptiveSampler
from .eplus import EnergyPlus, find_installations, select
from .evaluator import EnergyPlusEvaluator
from .objectives import MeterReader, Objective, VariableReader
from .optimize import ParetoFront, nsga2, nsga3
from .parameters import (
    CategoryParameter,
    FieldSelector,
    IntegerParameter,
    Parameter,
    ParameterError,
    RangeParameter,
)
from .problem import Problem
from .sampling import lhs_unit
from .simulation.runner import CommandRunner

_SURROGATES = ("gp", "mlp", "torch_mlp")


def load_config(source: dict | str | Path) -> tuple[dict, Path]:
    """``(config dict, base directory for relative paths)`` from a dict or a .json/.yaml file."""
    if isinstance(source, dict):
        return source, Path.cwd()
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise ImportError("YAML configs need pyyaml: pip install 'bdopt[yaml]'") from e
        return yaml.safe_load(text), path.resolve().parent
    return json.loads(text), path.resolve().parent


def parameter_from_config(spec: dict) -> Parameter:
    """One entry of the ``parameters`` list (see ``examples/study.yaml``)."""
    kind = spec.get("type")
    if kind is None:
        kind = "category" if "options" in spec else "integer" if spec.get("integer") else "range"
    if kind == "range":
        vd = RangeParameter(spec["min"], spec["max"])
    elif kind == "integer":
        vd = IntegerParameter(spec["min"], spec["max"])
    elif kind in ("category", "categorical"):
        vd = CategoryParameter(spec["options"])
    else:
        raise ParameterError(f"unknown parameter type {kind!r}")
    sel = FieldSelector(
        spec["class"], spec.get("object", "*"), spec.get("field"),
        field_index=spec.get("field_index"), where=spec.get("where"),
    )
    return Parameter(sel, vd, name=spec.get("name"))


def objective_from_config(spec: str | dict) -> str | Objective:
    if isinstance(spec, str):
        return spec
    direction = spec.get("direction", "min")
    if "meter" in spec:
        return MeterReader(spec["meter"], spec.get("name"), spec.get("unit", "J"),
                           spec.get("aggregate", "sum"), spec.get("missing", "zero"), direction)
    if "variable" in spec:
        return VariableReader(spec.get("key", "*"), spec["variable"], spec.get("name"),
                              spec.get("aggregate", "sum"), spec.get("frequency", "RunPeriod"),
                              spec.get("missing", "nan"), direction)
    raise ParameterError(f"objective needs 'meter' or 'variable': {spec}")


def make_surrogate(spec: dict | str | None):
    from .surrogate import GPSurrogate, MLPSurrogate

    if spec is None:
        return GPSurrogate()
    if isinstance(spec, str):
        spec = {"type": spec}
    kind, params = spec.get("type", "gp"), spec.get("params", {})
    if kind == "gp":
        return GPSurrogate(**params)
    if kind == "mlp":
        return MLPSurrogate(**params)
    if kind == "torch_mlp":
        from .surrogate import TorchMLP

        return TorchMLP(**params)
    raise ValueError(f"surrogate type must be one of {_SURROGATES}")


@dataclass
class StudyResults:
    data: pd.DataFrame
    test: pd.DataFrame
    history: pd.DataFrame | None = None
    metrics: pd.DataFrame | None = None
    surrogate: Any = None
    pareto: ParetoFront | None = None
    best: pd.Series | None = None
    verification: pd.DataFrame | None = None
    stats: dict = field(default_factory=dict)

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(d / "train.csv", index=False)
        if len(self.test):
            self.test.to_csv(d / "test.csv", index=False)
        if self.history is not None:
            self.history.to_csv(d / "history.csv", index=False)
        if self.metrics is not None:
            self.metrics.to_csv(d / "metrics.csv")
        if self.pareto is not None:
            self.pareto.df.to_csv(d / "pareto.csv", index=False)
        if self.best is not None:
            self.best.to_frame("value").to_csv(d / "best.csv")
        if self.verification is not None:
            self.verification.to_csv(d / "verification.csv", index=False)
        if self.surrogate is not None:
            self.surrogate.save(d / "surrogate.pkl")
        (d / "stats.json").write_text(json.dumps(self.stats, indent=2, default=str))
        return d


class DesignStudy:
    """Sampling -> surrogate -> multi-objective optimization, as configured."""

    def __init__(self, config: dict, base_dir: str | Path | None = None):
        self.cfg = config
        self.base = Path(base_dir) if base_dir else Path.cwd()
        self.problem = self._build_problem()
        self._evaluator: EnergyPlusEvaluator | None = None

    @classmethod
    def from_config(cls, source: dict | str | Path) -> DesignStudy:
        cfg, base = load_config(source)
        return cls(cfg, base)

    def _path(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.base / p

    def _build_problem(self) -> Problem:
        cfg = self.cfg
        params = [parameter_from_config(s) for s in cfg["parameters"]]
        objs = [objective_from_config(o) for o in cfg["objectives"]]
        return Problem(params, objs)

    # -- evaluator -------------------------------------------------------------------------------
    def _eplus(self) -> EnergyPlus:
        e = self.cfg.get("eplus", {})
        extra = [self._path(p) for p in e.get("search_paths", [])]
        if e.get("path"):
            return EnergyPlus.find(version=e.get("version"), path=self._path(e["path"]))
        return select(find_installations(extra), e.get("version"))

    @property
    def evaluator(self) -> EnergyPlusEvaluator:
        if self._evaluator is None:
            cfg = self.cfg
            e = cfg.get("eplus", {})
            ev = cfg.get("evaluation", {})
            m = cfg["model"]
            runner = None
            r = cfg.get("runner")
            if r and r.get("type", "local") == "command":
                runner = CommandRunner(
                    r["prefix"], self._eplus(), r.get("executable"), r.get("path_map"),
                    timeout=ev.get("timeout"),
                )
            out = self._path(cfg.get("output_dir", "bdopt_study")) / "runs"
            from .idf import IDF

            idf = IDF.from_file(self._path(m["idf"]))
            self._evaluator = EnergyPlusEvaluator(
                self.problem, idf, self._path(m["epw"]),
                eplus=None if runner else (self._eplus() if e.get("version") or e.get("path") else None),
                out_dir=ev.get("out_dir", out), processes=ev.get("processes", 1),
                cache=ev.get("cache", True), keep_dirs=ev.get("keep_dirs", False),
                timeout=ev.get("timeout"), runner=runner, auto_upgrade=e.get("auto_upgrade", False),
                design_day=ev.get("design_day", False), progress=ev.get("progress", False),
                on_error=ev.get("on_error", "nan"),
                search_paths=[self._path(p) for p in e.get("search_paths", [])] or None,
            )
        return self._evaluator

    # -- stages ----------------------------------------------------------------------------------
    def sample(self) -> AdaptiveResult | tuple[pd.DataFrame, pd.DataFrame]:
        s = self.cfg.get("sampling", {})
        method = s.get("method", "adaptive")
        seed = s.get("seed", 0)
        surrogate = make_surrogate(self.cfg.get("surrogate"))
        if method == "adaptive":
            if s.get("engine", "native") == "harlow":
                from .adaptive import HarlowAdaptiveSampler

                sampler = HarlowAdaptiveSampler(
                    self.problem, self.evaluator, s.get("adaptive_method", "FuzzyLolaVoronoi"),
                    s.get("harlow_surrogate", "BatchIndependentGaussianProcess"), seed,
                )
                return sampler.run(
                    s.get("n_initial"), s.get("n_test"), s.get("max_iter", 40), s.get("batch", 1),
                    s.get("tol", 0.01), verbose=s.get("verbose", True),
                )
            sampler = AdaptiveSampler(
                self.problem, self.evaluator, s.get("adaptive_method", "flola_voronoi"), surrogate, seed
            )
            return sampler.run(
                n_initial=s.get("n_initial"), n_test=s.get("n_test"), max_iter=s.get("max_iter", 40),
                batch=s.get("batch", 1), tol=s.get("tol", 0.05),
                max_evaluations=s.get("max_evaluations"), verbose=s.get("verbose", True),
                checkpoint_dir=self._path(self.cfg.get("output_dir", "bdopt_study")) / "checkpoint",
            )
        # static design
        from . import sampling as smp

        n = s.get("n_samples", 5 * self.problem.n_var)
        fn = {"lhs": smp.lhs, "sobol": smp.sobol, "random": smp.random}[method]
        train = self.evaluator.evaluate(fn(self.problem, n, seed=seed), keep_input=True).dropna()
        n_test = s.get("n_test", 0)
        test = (
            self.evaluator.evaluate(
                self.problem.decode(lhs_unit(self.problem, n_test, seed=[seed, 2])), keep_input=True
            ).dropna()
            if n_test
            else pd.DataFrame(columns=self.problem.names())
        )
        return train, test

    def run(self, optimize: bool = True) -> StudyResults:
        sampled = self.sample()
        if isinstance(sampled, AdaptiveResult):
            data, test, history = sampled.data, sampled.test, sampled.history
        else:
            (data, test), history = sampled, None
        surrogate = make_surrogate(self.cfg.get("surrogate")).fit(data, self.problem)
        metrics = surrogate.score(test) if len(test) else surrogate.cross_validate(data, self.problem)
        res = StudyResults(data, test, history, metrics, surrogate)
        o = self.cfg.get("optimization")
        if optimize and o is not None:
            algo = nsga3 if o.get("algorithm", "nsga2") == "nsga3" else nsga2
            kw = {k: o[k] for k in ("pop_size", "n_gen", "seed") if k in o}
            if algo is nsga3 and "n_partitions" in o:
                kw["n_partitions"] = o["n_partitions"]
            model = self.evaluator if o.get("use") == "energyplus" else surrogate
            res.pareto = algo(self.problem, model, **kw)
            res.best = res.pareto.closest_to_ideal(o.get("weights"), o.get("normalize", True))
            nv = int(o.get("verify", 0))
            if nv and model is surrogate:
                pick = res.pareto.df.sample(min(nv, len(res.pareto)), random_state=0)
                real = self.evaluator.evaluate(pick[self.problem.parameter_names], keep_input=True)
                pred = surrogate.predict(pick[self.problem.parameter_names])
                for n in self.problem.objective_names:
                    real[f"{n} (surrogate)"] = pred[n].to_numpy()
                res.verification = real
        res.stats = self.evaluator.stats()
        return res
