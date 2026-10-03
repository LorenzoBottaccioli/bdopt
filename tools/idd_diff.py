#!/usr/bin/env python3
"""Compare the IDDs of two EnergyPlus versions: renamed/added/removed fields per object class.

    python tools/idd_diff.py /opt/eplus/EnergyPlus-9-6-0 /opt/eplus/EnergyPlus-25-2-0 [--class Lights ...]

Use it when a new EnergyPlus version appears, to decide whether ``bdopt/eplus/aliases.py`` needs a new
entry. A "rename" is reported when both versions have the same number of fields and the names differ at
the same position (a heuristic: a field inserted in the middle shows up as add + remove instead).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bdopt.eplus.idd import IDD, norm


def load(arg: str) -> IDD:
    p = Path(arg)
    return IDD.load(p / "Energy+.idd" if p.is_dir() else p)


def diff(a: IDD, b: IDD, classes: list[str] | None = None) -> list[str]:
    lines: list[str] = []
    names = classes or sorted(set(a) & set(b))
    for c in names:
        if c not in a or c not in b:
            lines.append(f"{c}: only in {'second' if c in b else 'first'} IDD")
            continue
        fa, fb = a.fields(c), b.fields(c)
        sa, sb = {norm(x) for x in fa}, {norm(x) for x in fb}
        if len(fa) == len(fb):
            for i, (x, y) in enumerate(zip(fa, fb, strict=True)):
                if norm(x) != norm(y):
                    lines.append(f"{c}[{i}] renamed: {x!r} -> {y!r}")
        else:
            for x in fa:
                if norm(x) not in sb:
                    lines.append(f"{c}: removed field {x!r}")
            for y in fb:
                if norm(y) not in sa:
                    lines.append(f"{c}: added field {y!r}")
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    if not classes:
        lines += [f"class removed: {c}" for c in only_a] + [f"class added: {c}" for c in only_b]
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("first")
    ap.add_argument("second")
    ap.add_argument("--class", dest="classes", action="append")
    args = ap.parse_args()
    a, b = load(args.first), load(args.second)
    print(f"# {a.version} -> {b.version}")
    out = diff(a, b, args.classes)
    print("\n".join(out) if out else "no differences")
    return 0


if __name__ == "__main__":
    sys.exit(main())
