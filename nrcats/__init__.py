"""nrcats: unified Python interface to NR BBH waveform catalogs.

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
>>> import nrcats as nrcat
>>> cat = nrcat.RITCatalog.load()
>>> wfm = cat.get("RIT:BBH:0001-n100-id3")
>>> h = wfm.get_td_waveform(total_mass=60., distance=100.,
...                         inclination=0., coa_phase=0.)
>>> hp, hc = h.real(), h.imag()      # h+ + i hx, as LALSimulation
"""

from __future__ import absolute_import

from . import lvc, maya, metadata, registry, rit, sxs, utils, waveform
from .maya import MayaCatalog
from .rit import RITCatalog
from .sxs import SXSCatalog
from .registry import get_catalog, list_catalogs, register_catalog
from .waveform import WaveformModes, apply_wigner_rotation_to_mode_dict
from .classification import NRCatalogClassifier
from .metadata import (
    RIT_KEYS,
    SXS_KEYS,
    MAYA_KEYS,
    CANONICAL_TO_CATALOG,
    CANONICAL_TO_PYCBC,
    PYCBC_KEYS,
)


def load_catalog(name: str, **kwargs):
    """Load a catalog by name tag.

    Parameters
    ----------
    name : str
        One of ``'SXS'``, ``'RIT'``, or ``'MAYA'`` (case-insensitive).
    **kwargs
        Forwarded to the catalog's ``load()`` class method.

    Returns
    -------
    CatalogBase subclass instance
    """
    tag = name.upper()
    if tag == "SXS":
        return SXSCatalog.load(download=False, **kwargs)
    elif tag == "RIT":
        return RITCatalog.load(**kwargs)
    elif tag == "MAYA":
        return MayaCatalog.load(**kwargs)
    else:
        raise ValueError(f"Unknown catalog '{name}'. Supported: 'SXS', 'RIT', 'MAYA'.")


def filter_by_surrogate_prior(
    catalog,
    total_mass: float = 40.0,
    q_max: float = 4.0,
    chi_max: float = 0.8,
    verbose: bool = False,
) -> list:
    """Return the subset of simulation names within the NRSur7dq4 prior volume.

    NRSur7dq4 is valid for q ∈ [1, 4] and |χ₁|, |χ₂| ≤ 0.8.  Simulations
    whose parameters cannot be retrieved (metadata errors) are silently skipped.

    Parameters
    ----------
    catalog : CatalogBase
        Loaded catalog object.
    total_mass : float, optional
        Total mass for parameter extraction (default 40 M☉).
    q_max : float, optional
        Maximum allowed mass ratio (default 4).
    chi_max : float, optional
        Maximum allowed spin magnitude (default 0.8).
    verbose : bool, optional
        Print progress to stdout.

    Returns
    -------
    list[str]
        Simulation name tags that pass the prior cuts.
    """
    from .surrogate import check_surrogate_prior

    passing = []
    sims = catalog.simulations_list
    for i, sim_name in enumerate(sims):
        if verbose and i % 50 == 0:
            print(f"  Checking {i}/{len(sims)}: {sim_name}")
        try:
            params = catalog.get_parameters(sim_name, total_mass=total_mass)
        except Exception as exc:
            if verbose:
                print(f"  Skipping {sim_name}: {exc}")
            continue
        if check_surrogate_prior(params, q_max=q_max, chi_max=chi_max):
            passing.append(sim_name)
    return passing


__all__ = [
    # Catalogs
    "MayaCatalog",
    "RITCatalog",
    "SXSCatalog",
    # Classifier
    "NRCatalogClassifier",
    # Registry
    "register_catalog",
    "get_catalog",
    "list_catalogs",
    # Catalog helpers
    "load_catalog",
    "filter_by_surrogate_prior",
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

    __version__ = _pkg_version("nrcats")
except PackageNotFoundError:
    __version__ = "unknown"
