# Platform migration patch

`0001-replace-besos-with-bdopt.patch` is the commit that moves the Smart Building Educational Platform's
surrogate-modelling part from BESOS/harlow/TensorFlow to bdopt. It could not be pushed to
`andresorehida1/Smart_Builing_Educational_Platform_Dockers` (the Claude GitHub App has no access to that
repository), so it is kept here. Apply it in a clone of the platform repository, on top of commit `fac9649`:

```bash
git am --keep-cr /path/to/0001-replace-besos-with-bdopt.patch   # --keep-cr: several platform files use CRLF
```

What it changes:

* `backend/notebook_templates/Surrogate_template.ipynb`, `backend/surrogate_notebooks/SurrogateNotebook.ipynb`:
  imports from `bdopt` instead of `besos` / `harlow` / TensorFlow.
* `backend/inject_data.py`: the injected cell uses `load_idf(...)`.
* `Surrogate modeling examples/Test N/bdopt_Example_N.ipynb` (7 notebooks) + `README_bdopt.md`: generated with
  `tools/besos_to_bdopt_notebook.py`; each was executed end to end with a reduced budget on EnergyPlus 9.6.
* `conda-environment/surrogate/` (Dockerfile + vendored `install_eplus.py`) and `docker-compose.yml`: a surrogate image
  with Python 3.11 and EnergyPlus inside. **The images were never built** (no Docker daemon was available).
* `conda-environment/conda_environment.yml`: `besos`, `harlow`, `tensorflow-probability` removed.

The Dockerfile installs bdopt from `git+https://github.com/lorenzobottaccioli/buidlingdesginoptimization.git`
(build argument `BDOPT_SPEC`); change it if the repository is private or the package is published elsewhere.
