"""Sphinx configuration for the bdopt documentation.

Build:  pip install -e ".[docs]" && sphinx-build -W -b html docs docs/_build/html
The notebooks in docs/tutorials are stored with their outputs (they were executed against real
simulations) and are *not* re-executed by the build.
"""

import os
from importlib.metadata import version as _version

project = "bdopt"
author = "bdopt contributors"
copyright = "2026, bdopt contributors"
release = _version("bdopt")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]
# Cross-references into numpy/pandas/scikit-learn docs need network access: enabled in CI.
if os.environ.get("DOCS_INTERSPHINX"):
    extensions.append("sphinx.ext.intersphinx")
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
        "pandas": ("https://pandas.pydata.org/docs/", None),
        "sklearn": ("https://scikit-learn.org/stable/", None),
        "pymoo": ("https://pymoo.org/", None),
    }

exclude_patterns = ["_build", "make_figures.py", "**.ipynb_checkpoints"]
source_suffix = {".md": "myst-nb", ".ipynb": "myst-nb", ".rst": "restructuredtext"}
root_doc = "index"

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "attrs_inline", "fieldlist"]
myst_heading_anchors = 3
nb_execution_mode = "off"
nb_number_source_lines = False

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True}
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = False

html_theme = "furo"
html_title = f"bdopt {release}"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/lorenzobottaccioli/buidlingdesginoptimization",
    "source_branch": "main",
    "source_directory": "docs/",
}
suppress_warnings = ["myst.header"]

# Broken cross-references are errors. References into other projects (pandas, numpy ...) can only be
# resolved with intersphinx (network), so without it they are ignored.
nitpicky = True
nitpick_ignore_regex = [(r"py:class", r"Constraint")]   # alias of Callable[[Mapping[str, Any]], float]
if not os.environ.get("DOCS_INTERSPHINX"):
    nitpick_ignore_regex += [
        (r"py:(class|obj|data|func|exc|mod)", r"(pandas|numpy|np|pd|pathlib|collections|typing|sklearn|pymoo|torch|"
                                              r"matplotlib|scipy|builtins|nbformat)\..*"),
        (r"py:(class|obj|data)", r"(Path|DataFrame|Series|ndarray|NDArray|Callable|Mapping|Sequence|Iterable|"
                                 r"Generator|Any|optional|array_like|callable|str|int|float|bool|list|dict|tuple|"
                                 r"set|None|Figure|Axes|nn\.Module|Repair|PymooProblem)"),
    ]
