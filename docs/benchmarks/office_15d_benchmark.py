"""Adaptive strategies on the 15-parameter EnergyPlus office problem (examples/office_15d.py).

    python docs/benchmarks/office_15d_benchmark.py --out office15 --seeds 0,1 --budget 80

Per seed: an independent Latin-hypercube test set (shared by all seeds), a plain Latin hypercube with ``budget`` points
and one adaptive run per method (3 d initial points + 10 held-out, batches of 5 up to ``budget`` training points). The
error is the mean ``nrmse`` of a Gaussian process over the three objectives on the independent set. Rows are appended to
``OUT/results.csv``; finished simulations are cached, so the script can be restarted.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from office_15d import build  # noqa: E402

from bdopt import EnergyPlusEvaluator, sampling  # noqa: E402
from bdopt.adaptive import AdaptiveSampler  # noqa: E402
from bdopt.surrogate import GPSurrogate  # noqa: E402

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="office15")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--budget", type=int, default=80)
    ap.add_argument("--methods", default="flola_voronoi,lola_voronoi,nipv,qbc,latin_hypercube")
    ap.add_argument("--n-independent", type=int, default=30)
    ap.add_argument("--processes", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "results.csv"
    done = set()
    if csv.exists():
        old = pd.read_csv(csv)
        done = set(zip(old["seed"], old["method"], strict=True))
    pr, idf, epw = build()
    ev = EnergyPlusEvaluator(pr, idf, epw, out_dir=out / "runs", processes=args.processes)
    test = sampling.lhs(pr, args.n_independent, seed=1000)
    test = test.join(ev.evaluate(test))
    n0 = 3 * pr.n_var
    for seed in [int(s) for s in args.seeds.split(",")]:
        for method in ["plain_lhs", *args.methods.split(",")]:
            if (seed, method) in done:
                continue
            t0 = time.time()
            if method == "plain_lhs":
                train = sampling.lhs(pr, args.budget, seed=seed)
                train = train.join(ev.evaluate(train))
                model, n = GPSurrogate().fit(train, pr), len(train)
            else:
                r = AdaptiveSampler(pr, ev, method, GPSurrogate(), seed=seed).run(
                    n_initial=n0, n_test=10, max_iter=(args.budget - n0) // 5, batch=5, tol=0, verbose=False)
                model, n = r.surrogate, len(r.data)
            s = model.score(test)
            row = {"seed": seed, "method": method, "n_train": n, "nrmse": float(s["nrmse"].mean()),
                   "rmse_pct": float(s["rmse_pct"].mean()), "r2": float(s["r2"].mean()),
                   "seconds": round(time.time() - t0)}
            pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
            print(f"seed={seed} {method:16s} n={n} nrmse={row['nrmse']:.4f} rmse%={row['rmse_pct']:.2f}", flush=True)


if __name__ == "__main__":
    main()
