"""Surrogate models (scikit-learn always available; PyTorch if installed)."""

from .base import Surrogate
from .metrics import regression_metrics
from .sklearn_models import GPSurrogate, MLPSurrogate

__all__ = ["GPSurrogate", "MLPSurrogate", "Surrogate", "TorchMLP", "regression_metrics"]


def __getattr__(name: str):
    if name == "TorchMLP":
        try:
            from .torch_mlp import TorchMLP
        except ImportError as e:  # pragma: no cover
            raise ImportError("TorchMLP needs PyTorch: pip install 'bdopt[torch]'") from e
        return TorchMLP
    raise AttributeError(name)
