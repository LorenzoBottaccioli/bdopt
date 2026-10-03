"""Command line: ``bdopt run study.yaml``, ``bdopt eplus list``."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .eplus import find_installations


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``bdopt`` command; returns the process exit code."""
    ap = argparse.ArgumentParser(prog="bdopt", description="Building design optimization with EnergyPlus")
    ap.add_argument("--version", action="version", version=f"bdopt {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ep = sub.add_parser("eplus", help="EnergyPlus installations")
    ep_sub = ep.add_subparsers(dest="eplus_cmd", required=True)
    ls = ep_sub.add_parser("list", help="list the EnergyPlus versions found on this machine")
    ls.add_argument("--path", action="append", default=[], help="extra directory to search")

    run = sub.add_parser("run", help="run a study from a YAML/JSON configuration")
    run.add_argument("config")
    run.add_argument("--out", help="output directory (default: output_dir of the config)")
    run.add_argument("--no-optimize", action="store_true", help="stop after sampling and surrogate")

    args = ap.parse_args(argv)
    if args.cmd == "eplus":
        installs = find_installations(args.path)
        if not installs:
            print("no EnergyPlus installation found (>= 8.7). See tools/install_eplus.py", file=sys.stderr)
            return 1
        for e in installs:
            print(f"{e.version}\t{e.root}")
        return 0
    if args.cmd == "run":
        from .study import DesignStudy

        study = DesignStudy.from_config(args.config)
        if args.out:
            study.cfg["output_dir"] = args.out
        res = study.run(optimize=not args.no_optimize)
        out = study._path(study.cfg.get("output_dir", "bdopt_study"))
        res.save(out)
        print(f"results written to {out}")
        if res.best is not None:
            print("selected design:")
            print(res.best.to_string())
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
