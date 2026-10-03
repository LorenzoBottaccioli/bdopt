"""Adaptive sampling."""

from .sampler import METHODS, AdaptiveResult, AdaptiveSampler

__all__ = ["METHODS", "AdaptiveResult", "AdaptiveSampler", "HarlowAdaptiveSampler"]


def __getattr__(name: str):
    if name == "HarlowAdaptiveSampler":
        from .harlow_bridge import HarlowAdaptiveSampler

        return HarlowAdaptiveSampler
    raise AttributeError(name)
