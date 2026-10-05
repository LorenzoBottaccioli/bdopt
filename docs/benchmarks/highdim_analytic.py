"""Adaptive strategies on analytic responses with 15 and 20 parameters (a few of them matter, one localised bump).

    python docs/benchmarks/highdim_analytic.py --out highdim_analytic.csv

For each dimension, seed and strategy: ``n_initial`` = 3 d Latin-hypercube points + 10 held-out, then adaptive batches
of 5 up to ``budget`` training points; the error is the RMSE of a Gaussian process on 3000 independent points divided
by the standard deviation of the response (``nrmse``). A plain Latin hypercube with ``budget`` points is the baseline.
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from bdopt import FunctionEvaluator, NullSelector, Parameter, Problem, RangeParameter, sampling
from bdopt.adaptive import AdaptiveSampler
from bdopt.surrogate import GPSurrogate

warnings.filterwarnings("ignore")
METHODS = ["flola_voronoi", "lola_voronoi", "cv_voronoi", "voronoi", "nipv", "qbc", "loocv", "lipschitz", "idw",
           "greedy", "cvt", "soas", "sobol_seq", "halton", "mixed", "latin_hypercube", "random"]


def response(row):
    """Smooth trend in 5 parameters, a narrow bump in the first two, tiny effect of the rest."""
    x = [row[f"x{i}"] for i in range(len(row))]
    bump = np.exp(-(((x[0] - 0.7) ** 2 + (x[1] - 0.3) ** 2) / 0.02))
    smooth = 0.5 * x[0] + 0.3 * x[1] - 0.2 * x[2] + 0.2 * np.sin(2 * np.pi * x[3]) + 0.15 * x[4] ** 2
    return [smooth + bump + 0.01 * sum(x[5:])]


def gp():
    return GPSurrogate(n_restarts=2, max_extra_restarts=4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="highdim_analytic.csv")
    ap.add_argument("--dims", default="15,20")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--methods", default=",".join(METHODS))
    args = ap.parse_args()
    out = Path(args.out)
    done = set()
    if out.exists():
        old = pd.read_csv(out)
        done = set(zip(old["d"], old["seed"], old["method"], strict=True))
    for d in [int(x) for x in args.dims.split(",")]:
        problem = Problem([Parameter(NullSelector(f"x{i}"), RangeParameter(0, 1), name=f"x{i}") for i in range(d)], ["f"])
        ev = FunctionEvaluator(problem, response)
        big = sampling.lhs(problem, 3000, seed=99)
        truth = ev.evaluate(big).to_numpy()

        def err(model, problem=problem, big=big, truth=truth):
            p = model.predict(big).to_numpy()
            return float(np.sqrt(((p - truth) ** 2).mean() / truth.var()))

        for seed in [int(s) for s in args.seeds.split(",")]:
            for method in ["plain_lhs", *args.methods.split(",")]:
                if (d, seed, method) in done:
                    continue
                t0 = time.time()
                if method == "plain_lhs":
                    train = ev.evaluate(sampling.lhs(problem, args.budget, seed=seed), keep_input=True)
                    e, n = err(gp().fit(train, problem)), len(train)
                else:
                    n0 = 3 * d
                    r = AdaptiveSampler(problem, ev, method, gp(), seed=seed).run(
                        n_initial=n0, n_test=10, max_iter=(args.budget - n0) // 5, batch=5, tol=0, verbose=False)
                    e, n = err(r.surrogate), len(r.data)
                pd.DataFrame([{"d": d, "seed": seed, "method": method, "n_train": n, "nrmse": e,
                               "seconds": round(time.time() - t0, 1)}]).to_csv(out, mode="a", header=not out.exists(), index=False)
                print(f"d={d} seed={seed} {method:16s} n={n} nrmse={e:.3f} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
