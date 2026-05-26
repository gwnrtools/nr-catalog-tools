"""nr-catalog-tools: unified Python interface to NR BBH waveform catalogs.

Provides stable, PyCBC-compatible access to the SXS, RIT, and MAYA/GT
numerical-relativity binary black-hole waveform catalogs for use in
LVK data-analysis pipelines, waveform-model calibration, and cross-catalog
accuracy studies.

Public API
----------
Catalogs:
    RITCatalog   -- RIT catalog (web-scraped metadata, HDF5 waveforms)
    SXSCatalog   -- SXS catalog (via the ``sxs`` package, Zenodo-backed)
    MayaCatalog  -- MAYA/GT catalog (pickled metadata, HDF5 waveforms)

Waveform:
    WaveformModes                   -- ndarray-like waveform object with
                                       physical-unit scaling and frame tools
    apply_wigner_rotation_to_mode_dict -- rotate a mode dict via Wigner D-matrices

Registry:
    register_catalog  -- decorator to register a new catalog class
    get_catalog       -- look up a registered catalog class by tag
    list_catalogs     -- return the set of all registered tags

Metadata key mappings:
    RIT_KEYS, SXS_KEYS, MAYA_KEYS  -- canonical → catalog key dicts
    CANONICAL_TO_CATALOG            -- unified cross-catalog lookup
    CANONICAL_TO_PYCBC              -- canonical → PyCBC parameter name
    PYCBC_KEYS                      -- PyCBC output parameter names

Example
-------
>>> import nrcatalogtools as nrcat
>>> cat = nrcat.RITCatalog.load()
>>> wfm = cat.get("RIT:BBH:0001-n100-id3")
>>> hp, hc = wfm.get_td_waveform(total_mass=60., distance=100.,
...                                inclination=0., coa_phase=0.)
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
