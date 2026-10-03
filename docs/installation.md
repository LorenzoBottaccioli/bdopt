# Installation

bdopt needs **Python 3.10 or newer** and an **EnergyPlus installation** (8.7 or newer). Nothing else: it
runs on a laptop, a server or a container, and does not depend on Docker.

## The package

```bash
pip install "bdopt[torch,viz] @ git+https://github.com/lorenzobottaccioli/buidlingdesginoptimization.git"
```

`pip install bdopt` alone gives sampling, adaptive sampling, scikit-learn surrogates and NSGA-II. Extras:

| Extra | Adds |
|---|---|
| `torch` | {class}`~bdopt.surrogate.TorchMLP` (PyTorch; use the CPU build to avoid a multi-GB download) |
| `viz` | the plots in {mod}`bdopt.viz` (matplotlib) |
| `yaml` | YAML study files (JSON works without it) |
| `harlow` | the optional {class}`~bdopt.adaptive.HarlowAdaptiveSampler` bridge (pulls in TensorFlow) |
| `docs` | building this documentation |
| `dev` | tests and linting |

## EnergyPlus

Any release from 8.7 works; install the one that matches your model (or let bdopt upgrade the model, see
{doc}`guide/energyplus-versions`). Either use the official installer from
[energyplus.net](https://energyplus.net/downloads), or the helper shipped with bdopt, which needs no
administrator rights:

```bash
python tools/install_eplus.py 9.6            # -> ~/.bdopt/eplus/EnergyPlus-9-6-0
python tools/install_eplus.py 8.7 25.2       # several versions side by side
python tools/install_eplus.py --list         # known versions
```

bdopt finds installations in this order:

1. `EnergyPlus.find(path=...)` / the `eplus=` argument of the evaluator;
2. the environment variables `BDOPT_EPLUS_PATH` (several folders separated by `os.pathsep`), `ENERGYPLUS_DIR`,
   `ENERGYPLUS_INSTALL_DIR`, `EPLUS_PATH` (path of the executable);
3. the `energyplus` on `PATH`;
4. the usual folders: `~/.bdopt/eplus`, `/usr/local`, `/opt`, `/opt/eplus`, `/Applications`, `C:\`, `C:\Program Files`.

Check what bdopt can see:

```bash
bdopt eplus list
bdopt eplus list --path /some/folder
```

```text
8.7.0    /opt/eplus/EnergyPlus-8-7-0
9.6.0    /opt/eplus/EnergyPlus-9.6.0-f420c06a69-Linux-Ubuntu20.04-x86_64
25.2.0   /opt/eplus/EnergyPlus-25.2.0-cf7368216c-Linux-Ubuntu24.04-x86_64
```

:::{note}
`tools/install_eplus.py` downloads the official release assets. The 8.7 Windows and macOS releases only ship
installers (`.exe`, `.dmg`); on Windows the script runs the installer silently, on macOS install 8.7 by hand.
:::

## Development install

```bash
git clone https://github.com/lorenzobottaccioli/buidlingdesginoptimization.git
cd buidlingdesginoptimization
pip install -e ".[dev]"
pytest -m "not eplus and not slow"        # no EnergyPlus needed
```

## Optional: a container

`docker/Dockerfile` builds an image with one EnergyPlus version and bdopt
(`docker build -f docker/Dockerfile --build-arg EPLUS_VERSION=9.6 -t bdopt:eplus-9.6 .`). It is a convenience;
the library works the same without it. To use an EnergyPlus that lives in a container from a bdopt that runs
outside, see {class}`~bdopt.simulation.CommandRunner`.
