"""
nr-catalog-tools is a toolkit for interfacing with gravitational-wave
catalogs generated via Numerical Relativity simulations
"""

from __future__ import absolute_import

from . import lvc, maya, metadata, registry, rit, sxs, utils, waveform
from .maya import MayaCatalog
from .rit import RITCatalog
from .sxs import SXSCatalog
from .registry import get_catalog, list_catalogs, register_catalog
from .waveform import WaveformModes, apply_wigner_rotation_to_mode_dict
from .metadata import (
    RIT_KEYS,
    SXS_KEYS,
    MAYA_KEYS,
    CANONICAL_TO_CATALOG,
    CANONICAL_TO_PYCBC,
    PYCBC_KEYS,
)

__all__ = [
    # Catalogs
    "MayaCatalog",
    "RITCatalog",
    "SXSCatalog",
    # Registry
    "register_catalog",
    "get_catalog",
    "list_catalogs",
    # Waveform
    "WaveformModes",
    "apply_wigner_rotation_to_mode_dict",
    # Metadata key mappings
    "RIT_KEYS",
    "SXS_KEYS",
    "MAYA_KEYS",
    "CANONICAL_TO_CATALOG",
    "CANONICAL_TO_PYCBC",
    "PYCBC_KEYS",
]


try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError

    __version__ = _pkg_version("nr-catalog-tools")
except PackageNotFoundError:
    __version__ = "unknown"
