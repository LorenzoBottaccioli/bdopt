from __future__ import annotations

from ...idf import IDF
from ...model import load_idf


def get_building(building_path: str, *args, **kwargs) -> IDF:
    """``besos.eppy_funcs.get_building``: load an IDF (extra BESOS arguments are ignored)."""
    return load_idf(building_path)


def _unsupported(name: str, hint: str):
    def f(*a, **k):
        raise NotImplementedError(f"besos.eppy_funcs.{name} is not provided by bdopt. {hint}")

    f.__name__ = name
    return f


get_idf = get_building
get_idf_version = _unsupported("get_idf_version", "Use IDF.version.")
