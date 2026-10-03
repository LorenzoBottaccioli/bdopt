#!/usr/bin/env python3
"""Convert a BESOS / harlow surrogate notebook (like the platform's ``Besos_Example_N`` /
``Harlow_Example_N``) into a bdopt notebook.

    python tools/besos_to_bdopt_notebook.py "Test 1/Harlow_Example_1.ipynb" "Test 1/bdopt_Example_1.ipynb"

The *design space* cells (model, selectors, ranges, objectives) are carried over, adapted where the
API differs; everything after them (sampling, scaling, harlow glue, surrogate comparison, NSGA-II)
is regenerated with the bdopt flow, so there are no hand-copied bounds, scalers or patches.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

WINDOWS_BLOCK = '''# One categorical design variable shared by every window of the model.
# (The BESOS notebook created one variable per window, all called 'Windows types'; the model
#  has {n_windows} windows, which would be {n_windows} identically named variables.)
win_arr = ['single_glazing', 'double_glazing', 'triple_glazing']
windowsParam = Parameter(
    selector=FieldSelector(class_name='FenestrationSurface:Detailed', object_name='*',
                           field_name='Construction Name', where={{'Surface Type': 'Window'}}),
    value_descriptor=CategoryParameter(win_arr),
    name='Windows types',
)'''


def _code_cells(nb) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def extract_design_cells(src: Path) -> tuple[str, str, list[str]]:
    """``(idf name, epw name, parameter-definition cells)`` from a BESOS/harlow notebook."""
    nb = json.loads(src.read_text(encoding="utf-8"))
    code = _code_cells(nb)
    i0 = next(i for i, c in enumerate(code) if "get_building" in c)
    i1 = next(i for i, c in enumerate(code) if "EPProblem(" in c)
    head = code[i0]
    idf = re.search(r"get_building\(\s*[\"']([^\"']+)[\"']", head).group(1)
    epw = re.search(r"epw_file\s*=\s*[\"']([^\"']+)[\"']", head).group(1)
    cells = []
    for c in code[i0 + 1 : i1]:
        if "epw()" in c or "meteo" in c:  # weather summary through the `epw` package: not needed
            continue
        cells.append(c)
    objectives = next(c for c in code[i0:i1 + 1] if "objectives" in c and "=" in c)
    obj = re.search(r"objectives\s*=\s*(\[[^\]]*\])", objectives).group(1)
    return idf, epw, [c for c in cells if "objectives" not in c or "EPProblem" in c] + [f"objectives = {obj}"], 


def adapt(cells: list[str], n_windows: int) -> list[str]:
    out = []
    for c in cells:
        if "windowsParameters" in c and "for fenestration" in c:
            c = WINDOWS_BLOCK.format(n_windows=n_windows) + "\n"
            out.append(c)
            continue
        c = re.sub(r"\+\s*windowsParameters\s*\+", "+ [windowsParam] +", c)
        c = c.replace("value_descriptors=", "value_descriptor=").replace("value_descriptors =", "value_descriptor =")
        if "problem.names()" in c:
            continue
        out.append(c.rstrip())
    return out


IMPORTS = '''import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bdopt import (CategoryParameter, EnergyPlusEvaluator, FieldSelector, IntegerParameter,
                   Parameter, Problem, RangeParameter, load_idf)
from bdopt.adaptive import AdaptiveSampler
from bdopt.optimize import nsga2
from bdopt.surrogate import GPSurrogate, MLPSurrogate, TorchMLP
from bdopt import viz'''

SETTINGS = '''# Budget of the study (EnergyPlus simulations are the expensive part).
N_INITIAL = {n_initial}    # first Latin-hypercube simulations
N_TEST = {n_test}          # held-out simulations used to measure the surrogate error
MAX_ITER = {max_iter}      # adaptive iterations (batch of new simulations each)
BATCH = 1
TOL = 0.05                 # stop when the test error (in std units of each objective) is below this
PROCESSES = {processes}    # parallel EnergyPlus runs'''

EVALUATOR = '''problem = Problem(parameters, objectives)       # parameters + objectives; bounds/order/scaling are derived from it
print(problem)

# EnergyPlus is found automatically (the installed version that matches the model); results are
# cached on disk, so re-running a cell or resuming after a crash never repeats a simulation.
evaluator = EnergyPlusEvaluator(problem, building, epw_file, out_dir="outputdir", processes=PROCESSES)
print(evaluator.eplus)'''

SAMPLING = '''sampler = AdaptiveSampler(problem, evaluator, method="flola_voronoi", seed=0)
result = sampler.run(n_initial=N_INITIAL, n_test=N_TEST, max_iter=MAX_ITER, batch=BATCH, tol=TOL)
print(result)
result.data.to_pickle("adaptive_samples.pkl")     # EnergyPlus runs (inputs + outputs), in real units
result.test.to_pickle("test_samples.pkl")
print(evaluator.stats())
result.history'''

EXPLORE = '''viz.convergence(result.history, tol=TOL)
viz.pairplot(result.data, problem)
viz.correlation_heatmap(result.data, problem)
result.data.describe()'''

MODELS = '''# Same data, three surrogate models; no manual scaling: inputs live in [0,1], outputs are standardised inside.
models = {
    "Gaussian process": GPSurrogate(),
    "MLP (scikit-learn)": MLPSurrogate(),
    "MLP (PyTorch)": TorchMLP(hidden=(32, 64, 32), n_ensemble=3),
}
scores = {}
for name, model in models.items():
    model.fit(result.data, problem)
    scores[name] = model.score(result.test)
    print(name); display(scores[name][["rmse_pct", "cv_rmse", "nmbe", "r2"]].round(3))'''

COMPARE = '''summary = pd.DataFrame({name: s["rmse_pct"] for name, s in scores.items()})
display(summary.round(3))                       # RMSE as % of the mean, per objective
best_name = summary.mean().idxmin()
surrogate = models[best_name]
print("selected surrogate:", best_name)
viz.parity(result.test[problem.objective_names], surrogate.predict(result.test))'''

OPTIMIZE = '''# Multi-objective optimisation (NSGA-II) on the surrogate: thousands of designs in seconds.
front = nsga2(problem, surrogate, pop_size=100, n_gen=100, seed=0)
print(len(front), "non-dominated designs")
front.df'''

RESULT = '''best = front.closest_to_ideal()                 # closest to the ideal point after min-max scaling
display(best.to_frame("selected design"))
viz.pareto(front, background=result.data, highlight=best)
viz.parallel_coordinates(front.df, problem)

# Check the selected design with the real simulator.
x = best[problem.parameter_names].to_frame().T.reset_index(drop=True)
pd.DataFrame({"EnergyPlus": evaluator.evaluate(x).iloc[0], "surrogate": surrogate.predict(x).iloc[0]})'''


def build(src: Path, dst: Path, models_dir: str, n_initial: int, n_test: int, max_iter: int, processes: int) -> None:
    idf, epw, cells = extract_design_cells(src)
    from bdopt import IDF

    n_windows = 0
    idf_path = Path(models_dir) / idf
    if idf_path.exists():
        n_windows = len([o for o in IDF.from_file(idf_path).idfobjects["FenestrationSurface:Detailed"] if o.obj[2] == "Window"])
    cells = adapt(cells, n_windows)
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    md = new_markdown_cell
    code = new_code_cell
    nb.cells = [
        md(f"# Surrogate modelling and design optimisation with bdopt\n\nModel `{idf}`, weather `{epw}`.\n\n"
           "The notebook builds a surrogate of the EnergyPlus model with **adaptive sampling** (simulations are "
           "placed where the surrogate needs them), compares surrogate models and runs a multi-objective "
           "optimisation on the best one. It uses the `bdopt` library (EnergyPlus 8.7+, no BESOS, no TensorFlow)."),
        code(IMPORTS),
        md("## Budget"), code(SETTINGS.format(n_initial=n_initial, n_test=n_test, max_iter=max_iter, processes=processes)),
        md("## Model and weather"),
        code(f'building = load_idf("{idf}")\nepw_file = "{epw}"\nprint(building)'),
        md("## Design space\nWhere each design variable is written in the IDF (`FieldSelector`) and the values it can take "
           "(`RangeParameter`, `CategoryParameter`, `IntegerParameter`)."),
        *[code(c) for c in cells],
        md("## Problem and evaluator"), code(EVALUATOR),
        md("## Adaptive sampling"), code(SAMPLING),
        md("## Samples"), code(EXPLORE),
        md("## Surrogate models"), code(MODELS), code(COMPARE),
        md("## Optimisation"), code(OPTIMIZE),
        md("## Selected design"), code(RESULT),
    ]
    nbformat.write(nb, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("dest")
    ap.add_argument("--models-dir", default=None, help="directory with the IDF (to count windows)")
    ap.add_argument("--n-initial", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--processes", type=int, default=1)
    a = ap.parse_args()
    src = Path(a.source)
    build(src, Path(a.dest), a.models_dir or str(src.parent), a.n_initial, a.n_test, a.max_iter, a.processes)
    print(f"written {a.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
