"""Benchmark on the seven platform tests, with real EnergyPlus runs.

    python docs/benchmarks/platform_benchmark.py ROOT --out platform_bench

For every test: an independent test set of Latin-hypercube simulations, a plain Latin hypercube of the same
size as each adaptive run, one adaptive run per method, and every surrogate trained on the Latin hypercube.
Results are appended to ``OUT/results.csv`` (finished simulations are cached, so the script can be
interrupted and restarted). ``ROOT`` is the platform's folder ``Surrogate modeling examples``.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from platform_tests import build  # noqa: E402

from bdopt import EnergyPlusEvaluator, sampling  # noqa: E402
from bdopt.adaptive import AdaptiveSampler  # noqa: E402
from bdopt.surrogate import GPSurrogate, make_surrogate  # noqa: E402

METHODS = ["flola_voronoi", "lola_voronoi", "cv_voronoi", "nipv", "probabilistic", "latin_hypercube"]
SURROGATES = ["gp", "multitask_gp", "deep_kernel_gp", "random_forest", "extra_trees", "gradient_boosting",
              "svr", "kernel_ridge", "knn", "polynomial", "rbf", "mlp", "torch_mlp", "mc_dropout_mlp", "auto"]


def error(model, test: pd.DataFrame, problem) -> dict:
    s = model.score(test)
    return {"nrmse": float(s["nrmse"].mean()), "rmse_pct": float(s["rmse_pct"].mean()), "r2": float(s["r2"].mean())}


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="platform_bench")
    ap.add_argument("--tests", default="1,2,3,4,5,6,7")
    ap.add_argument("--n-initial", type=int, default=15)
    ap.add_argument("--n-test", type=int, default=5)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--n-independent", type=int, default=25)
    ap.add_argument("--processes", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    budget = args.n_initial + args.batch * args.iters
    rows_adaptive, rows_surrogate = [], []
    for t in [int(x) for x in args.tests.split(",")]:
        problem, idf, epw = build(t, args.root)
        ev = EnergyPlusEvaluator(problem, idf, epw, out_dir=out / f"test{t}", processes=args.processes)
        t0 = time.time()
        test = sampling.lhs(problem, args.n_independent, seed=1000 + t)
        test = test.join(ev.evaluate(test))
        lhs = sampling.lhs(problem, budget, seed=t)
        lhs = lhs.join(ev.evaluate(lhs))
        print(f"[test {t}] independent set and Latin hypercube done ({time.time() - t0:.0f} s)", flush=True)
        gp = GPSurrogate().fit(lhs, problem)
        rows_adaptive.append({"test": t, "method": "plain_lhs", "n_train": len(lhs), **error(gp, test, problem)})
        for m in METHODS:
            r = AdaptiveSampler(problem, ev, m, GPSurrogate(), seed=t).run(
                n_initial=args.n_initial, n_test=args.n_test, max_iter=args.iters, batch=args.batch,
                tol=0, verbose=False,
            )
            e = error(r.surrogate, test, problem)
            rows_adaptive.append({"test": t, "method": m, "n_train": len(r.data), **e})
            print(f"[test {t}] {m:16s} n={len(r.data)} nrmse={e['nrmse']:.4f} rmse%={e['rmse_pct']:.2f}", flush=True)
            pd.DataFrame(rows_adaptive).to_csv(out / "adaptive.csv", index=False)
        for k in SURROGATES:
            t1 = time.time()
            try:
                e = error(make_surrogate(k).fit(lhs, problem), test, problem)
            except Exception as exc:  # report, do not hide
                e = {"nrmse": np.nan, "rmse_pct": np.nan, "r2": np.nan}
                print(f"[test {t}] surrogate {k} failed: {exc}", flush=True)
            rows_surrogate.append({"test": t, "surrogate": k, "n_train": len(lhs), "fit_s": time.time() - t1, **e})
            pd.DataFrame(rows_surrogate).to_csv(out / "surrogates.csv", index=False)
        print(f"[test {t}] finished ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
