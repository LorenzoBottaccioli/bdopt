"""EnergyPlus discovery and everything that depends on the EnergyPlus version."""

from .aliases import meter_candidates, resolve_class, resolve_field
from .idd import IDD
from .install import EnergyPlus, EnergyPlusNotFound, find_installations, select
from .transition import TransitionError, plan, upgrade_text
from .version import MIN_SUPPORTED, Version

__all__ = [
    "IDD",
    "MIN_SUPPORTED",
    "EnergyPlus",
    "EnergyPlusNotFound",
    "TransitionError",
    "Version",
    "find_installations",
    "meter_candidates",
    "plan",
    "resolve_class",
    "resolve_field",
    "select",
    "upgrade_text",
]
