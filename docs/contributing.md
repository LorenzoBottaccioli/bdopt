# Contributing

Bug reports, questions and pull requests are welcome in the
[repository](https://github.com/lorenzobottaccioli/buidlingdesginoptimization).

## Reporting a problem

Include the bdopt version (`bdopt --version`), the output of `bdopt eplus list`, the EnergyPlus version of your
model, and a minimal script. For a failed simulation, the `eplusout.err` of the run in `failed/` is the single most
useful file.

## Development setup

```bash
git clone https://github.com/lorenzobottaccioli/buidlingdesginoptimization.git
cd buidlingdesginoptimization
pip install -e ".[dev,docs]"
```

## Tests

```bash
pytest -m "not eplus and not slow"                          # no EnergyPlus needed (a few minutes)
BDOPT_EPLUS_PATH=/opt/eplus pytest -m eplus                 # every installed EnergyPlus version
BDOPT_TEST_EPLUS_VERSIONS=9.6,24.2 pytest -m eplus -k "not reference"
pytest -m slow                                              # model upgrades, benchmark-style checks
```

| Marker | Meaning |
|---|---|
| `eplus` | needs an EnergyPlus installation; parametrised over the installed versions |
| `slow` | takes minutes (model upgrade through many versions, adaptive-vs-LHS comparison) |

Install several EnergyPlus versions side by side with `python tools/install_eplus.py 9.6 24.2 25.2`.

## Style

`ruff check src tests tools` must pass (line length 100). Public classes and functions have numpy-style
docstrings; the API reference is generated from them.

## Documentation

```bash
sphinx-build -W -b html docs docs/_build/html
```

`-W` turns warnings into errors, which is what CI does. The tutorials in `docs/tutorials/` are stored with
their outputs and are not re-executed by the build; regenerate them with `python docs/tutorials/build_notebooks.py`
(the EnergyPlus tutorial needs EnergyPlus 9.6). The figures come from `python docs/make_figures.py`.
Code blocks marked `<!-- runnable -->` in the documentation are executed by `tests/test_docs_snippets.py`.

## Supporting a new EnergyPlus version

1. install it (`tools/install_eplus.py`) and run `python tools/idd_diff.py OLD_ROOT NEW_ROOT` between it and the
   previous release;
2. add the renamed fields, classes or meters it reports to `src/bdopt/eplus/aliases.py`;
3. add the version to the tables in `tools/install_eplus.py` and to the CI matrix in `.github/workflows/ci.yml`;
4. run `pytest -m eplus` against it.

## Design rules

* Everything numerical works in the unit cube; parameters know how to encode and decode themselves.
* Anything that depends on the EnergyPlus version lives in `bdopt/eplus/`.
* No global state: two EnergyPlus versions must be usable in one process.
* A failure must be loud and early (a misspelt name) or contained and visible (a failed run), never silent.
