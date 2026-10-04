"""Full benchmark on the seven platform tests, with real EnergyPlus runs.

    python docs/benchmarks/platform_benchmark.py ROOT --out platform_bench

For every test, seed and budget: the independent test set (shared by all seeds), a plain Latin hypercube with the
budget's number of points, one adaptive run per method (all of ``bdopt.adaptive.METHODS``) and every surrogate
trained on the plain Latin hypercube. Rows are appended to ``OUT/adaptive.csv`` and ``OUT/surrogates.csv`` as
they are produced, and finished simulations are cached, so an interrupted run can be restarted with the same
command. ``ROOT`` is the platform's folder ``Surrogate modeling examples``.
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
from bdopt.adaptive import METHODS, AdaptiveSampler  # noqa: E402
from bdopt.surrogate import GPSurrogate, make_surrogate  # noqa: E402

SURROGATES = ["gp", "multitask_gp", "deep_kernel_gp", "random_forest", "extra_trees", "gradient_boosting",
              "svr", "kernel_ridge", "knn", "polynomial", "rbf", "mlp", "torch_mlp", "mc_dropout_mlp", "auto"]


def error(model, test: pd.DataFrame) -> dict:
    s = model.score(test)
    return {"nrmse": float(s["nrmse"].mean()), "rmse_pct": float(s["rmse_pct"].mean()), "r2": float(s["r2"].mean())}


def done(path: Path, **key) -> set:
    if not path.exists():
        return set()
    d = pd.read_csv(path)
    for k, v in key.items():
        d = d[d[k] == v]
    return set(d["method" if "method" in d else "surrogate"])


def append(path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="platform_bench")
    ap.add_argument("--tests", default="6,7,1,4,3,2,5")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--budgets", default="35,60")
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--n-test", type=int, default=5)
    ap.add_argument("--n-independent", type=int, default=25)
    ap.add_argument("--processes", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fa, fs = out / "adaptive2.csv", out / "surrogates2.csv"
    tests = [int(x) for x in args.tests.split(",")]
    problems, evaluators, independent = {}, {}, {}
    for t in tests:
        problems[t], idf, epw = build(t, args.root)
        evaluators[t] = EnergyPlusEvaluator(problems[t], idf, epw, out_dir=out / f"test{t}", processes=args.processes)
        x = sampling.lhs(problems[t], args.n_independent, seed=1000 + t)
        independent[t] = x.join(evaluators[t].evaluate(x))
    for budget in [int(b) for b in args.budgets.split(",")]:
        n0 = round(0.43 * budget)
        iters = (budget - n0) // args.batch
        for seed in [int(s) for s in args.seeds.split(",")]:
            for t in tests:
                t0 = time.time()
                pr, ev, test = problems[t], evaluators[t], independent[t]
                key = {"test": t, "budget": budget, "seed": seed}
                lhs = sampling.lhs(pr, budget, seed=seed * 100 + t)
                lhs = lhs.join(ev.evaluate(lhs))
                have = done(fa, **key)
                if "plain_lhs" not in have:
                    append(fa, {**key, "method": "plain_lhs", "n_train": len(lhs), **error(GPSurrogate().fit(lhs, pr), test)})
                for m in METHODS:
                    if m in have:
                        continue
                    r = AdaptiveSampler(pr, ev, m, GPSurrogate(), seed=seed * 100 + t).run(
                        n_initial=n0, n_test=args.n_test, max_iter=iters, batch=args.batch, tol=0, verbose=False)
                    e = error(r.surrogate, test)
                    append(fa, {**key, "method": m, "n_train": len(r.data), **e})
                    print(f"[B{budget} s{seed} t{t}] {m:16s} n={len(r.data)} nrmse={e['nrmse']:.4f}", flush=True)
                have_s = done(fs, **key)
                for k in SURROGATES:
                    if k in have_s:
                        continue
                    t1 = time.time()
                    try:
                        e = error(make_surrogate(k).fit(lhs, pr), test)
                    except Exception as exc:  # report, do not hide
                        e = {"nrmse": np.nan, "rmse_pct": np.nan, "r2": np.nan}
                        print(f"[B{budget} s{seed} t{t}] surrogate {k} failed: {exc}", flush=True)
                    append(fs, {**key, "surrogate": k, "n_train": len(lhs), "fit_s": time.time() - t1, **e})
                print(f"[B{budget} s{seed} t{t}] finished ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
