"""Surrogate models (scikit-learn always available; PyTorch models if torch is installed)."""

from .base import Surrogate
from .metrics import regression_metrics
from .sklearn_models import (
    AdaBoostSurrogate,
    AutoSurrogate,
    BaggingSurrogate,
    DecisionTreeSurrogate,
    ELMSurrogate,
    ExtraTreesSurrogate,
    GPSurrogate,
    GradientBoostingSurrogate,
    HistGradientBoostingSurrogate,
    KernelRidgeSurrogate,
    KNNSurrogate,
    LassoPolynomialSurrogate,
    LightGBMSurrogate,
    MLPSurrogate,
    PCESurrogate,
    PolynomialSurrogate,
    RandomForestSurrogate,
    RBFSurrogate,
    StackingSurrogate,
    SVRSurrogate,
    UniversalKrigingSurrogate,
    XGBoostSurrogate,
)

#: name -> class, for configuration files and :func:`make_surrogate`
SURROGATES = {
    "gp": GPSurrogate,
    "mlp": MLPSurrogate,
    "random_forest": RandomForestSurrogate,
    "extra_trees": ExtraTreesSurrogate,
    "gradient_boosting": GradientBoostingSurrogate,
    "svr": SVRSurrogate,
    "kernel_ridge": KernelRidgeSurrogate,
    "knn": KNNSurrogate,
    "polynomial": PolynomialSurrogate,
    "rbf": RBFSurrogate,
    "auto": AutoSurrogate,
    "decision_tree": DecisionTreeSurrogate,
    "hist_gradient_boosting": HistGradientBoostingSurrogate,
    "adaboost": AdaBoostSurrogate,
    "lightgbm": LightGBMSurrogate,
    "xgboost": XGBoostSurrogate,
    "lasso_polynomial": LassoPolynomialSurrogate,
    "pce": PCESurrogate,
    "elm": ELMSurrogate,
    "universal_kriging": UniversalKrigingSurrogate,
    "bagging": BaggingSurrogate,
    "stacking": StackingSurrogate,
}
_TORCH = {
    "torch_mlp": "TorchMLP",
    "mc_dropout_mlp": "MCDropoutMLP",
    "multitask_gp": "MultiTaskGPSurrogate",
    "deep_kernel_gp": "DeepKernelGPSurrogate",
    "deep_ensemble": "DeepEnsembleMLP",
    "sparse_gp": "SparseGPSurrogate",
}

__all__ = [
    "SURROGATES",
    "AdaBoostSurrogate",
    "AutoSurrogate",
    "BaggingSurrogate",
    "DeepEnsembleMLP",
    "DeepKernelGPSurrogate",
    "DecisionTreeSurrogate",
    "ELMSurrogate",
    "ExtraTreesSurrogate",
    "GPSurrogate",
    "GradientBoostingSurrogate",
    "HistGradientBoostingSurrogate",
    "KNNSurrogate",
    "KernelRidgeSurrogate",
    "LassoPolynomialSurrogate",
    "LightGBMSurrogate",
    "MCDropoutMLP",
    "MLPSurrogate",
    "MultiTaskGPSurrogate",
    "PCESurrogate",
    "PolynomialSurrogate",
    "RBFSurrogate",
    "RandomForestSurrogate",
    "SVRSurrogate",
    "SparseGPSurrogate",
    "StackingSurrogate",
    "Surrogate",
    "TorchMLP",
    "UniversalKrigingSurrogate",
    "XGBoostSurrogate",
    "make_surrogate",
    "regression_metrics",
]


def _torch_class(name: str):
    try:
        from . import torch_mlp, torch_models
    except ImportError as e:  # pragma: no cover
        raise ImportError(f"{name} needs PyTorch: pip install 'bdopt[torch]'") from e
    return getattr(torch_mlp, name, None) or getattr(torch_models, name)


def make_surrogate(kind: str = "gp", **params) -> Surrogate:
    """Surrogate from its short name (``gp``, ``mlp``, ``random_forest``, ``extra_trees``,
    ``gradient_boosting``, ``svr``, ``kernel_ridge``, ``knn``, ``polynomial``, ``rbf``, ``auto``, and with
    PyTorch ``torch_mlp``, ``mc_dropout_mlp``, ``multitask_gp``, ``deep_kernel_gp``)."""
    if kind in SURROGATES:
        return SURROGATES[kind](**params)
    if kind in _TORCH:
        return _torch_class(_TORCH[kind])(**params)
    raise ValueError(f"unknown surrogate {kind!r}; choose from {sorted([*SURROGATES, *_TORCH])}")


def __getattr__(name: str):
    if name in _TORCH.values():
        return _torch_class(name)
    raise AttributeError(name)
