"""Regenerate the platform benchmark tables of docs/benchmarks.md from the CSV files.

    python docs/benchmarks/make_tables.py            # prints Markdown
    python docs/benchmarks/make_tables.py --write    # replaces the block between the BENCH markers in docs/benchmarks.md

Rows are averaged over every seed available for the chosen budget; ``±`` is the standard deviation over seeds (blank
with a single seed). The ``mean`` column averages the seven tests.
"""

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent


def table(df: pd.DataFrame, key: str, budget: int, seeds=None) -> str:
    d = df[df.budget == budget]
    if seeds is not None:
        d = d[d.seed.isin(seeds)]
    per_seed = d.groupby([key, "test", "seed"]).nrmse.mean().reset_index()
    cell = per_seed.groupby([key, "test"]).nrmse.mean().unstack()
    overall = per_seed.groupby([key, "seed"]).nrmse.mean().groupby(key).agg(["mean", "std"])
    cell["mean"] = overall["mean"]
    order = cell["mean"].sort_values().index
    cols = [f"test {c}" for c in cell.columns[:-1]] + ["mean (± std over seeds)"]
    out = ["| " + " | ".join([key, *cols]) + " |", "|" + "---|" * (len(cols) + 1)]
    for k in order:
        std = overall.loc[k, "std"]
        mean_txt = f"{overall.loc[k, 'mean']:.3f}" + (f" ± {std:.3f}" if pd.notna(std) else "")
        out.append("| " + " | ".join([str(k), *[f"{cell.loc[k, c]:.3f}" for c in cell.columns[:-1]], mean_txt]) + " |")
    return "\n".join(out)


def block(budget: int = 35) -> str:
    a = pd.read_csv(HERE / "platform_adaptive_full.csv")
    s = pd.read_csv(HERE / "platform_surrogates_full.csv")
    seeds = sorted(a[a.budget == budget].seed.unique())
    # only seeds finished on all seven tests
    done = [x for x in seeds if a[(a.budget == budget) & (a.seed == x)].test.nunique() == 7
            and s[(s.budget == budget) & (s.seed == x)].test.nunique() == 7]
    note = f"Seeds used: {', '.join(map(str, done))} (budget {budget} points; error = nrmse of a Gaussian process / of the named surrogate)."
    return (f"{note}\n\n**Adaptive strategies**\n\n" + table(a.rename(columns={"method": "strategy"}), "strategy", budget, done)
            + "\n\n**Surrogates** (trained on the plain Latin hypercube)\n\n" + table(s, "surrogate", budget, done))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--budget", type=int, default=35)
    args = ap.parse_args()
    text = block(args.budget)
    if args.write:
        p = HERE.parent / "benchmarks.md"
        s = p.read_text()
        new = f"<!-- BENCH:start -->\n{text}\n<!-- BENCH:end -->"
        if "<!-- BENCH:start -->" in s:
            a, b = s.index("<!-- BENCH:start -->"), s.index("<!-- BENCH:end -->") + len("<!-- BENCH:end -->")
            s = s[:a] + new + s[b:]
        else:
            raise SystemExit("markers missing in docs/benchmarks.md")
        p.write_text(s)
    else:
        print(text)
