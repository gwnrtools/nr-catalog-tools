"""Tests for catalog-decoupling invariants.

These tests verify inheritance and to_sxs() contracts without requiring
any cached catalog data, so they are safe to run in a clean CI environment.
"""

from unittest.mock import MagicMock

import pytest
import sxs

import nrcatalogtools as nrcat
from nrcatalogtools.catalog import CatalogBase

_HAS_SXS_SIMULATIONS = hasattr(sxs, "Simulations")


def _make_rit_catalog():
    """Build a minimal RITCatalog without touching disk or network."""
    return nrcat.RITCatalog(catalog={"simulations": {}}, helper=MagicMock())


def test_rit_catalog_not_sxs_catalog():
    """Verify that RITCatalog does not inherit from sxs.Catalog."""
    ritcat = _make_rit_catalog()
    assert isinstance(ritcat, CatalogBase)
    # We allow the check to pass if sxs.Catalog is fully removed in future sxs versions,
    # but for sxs >= 2024.0.0 where it's deprecated, it must be False.
    if hasattr(sxs, "Catalog"):
        assert not isinstance(ritcat, sxs.Catalog)


@pytest.mark.skipif(
    not _HAS_SXS_SIMULATIONS,
    reason="sxs.Simulations not available (requires sxs >= 2025)",
)
def test_sxs_catalog_to_sxs():
    """Verify that SXSCatalog.to_sxs() returns the live sxs.Simulations object."""
    fake_sims = sxs.Simulations({})
    sxscat = nrcat.SXSCatalog(simulations_dict={})
    sxscat._sxs_simulations = fake_sims

    sxs_result = sxscat.to_sxs()
    assert isinstance(sxs_result, sxs.Simulations)


@pytest.mark.skipif(
    not _HAS_SXS_SIMULATIONS,
    reason="sxs.Simulations not available (requires sxs >= 2025)",
)
def test_rit_catalog_to_sxs():
    """Verify that RITCatalog.to_sxs() returns a valid sxs.Simulations instance."""
    ritcat = _make_rit_catalog()
    sxs_sims = ritcat.to_sxs()
    assert isinstance(sxs_sims, sxs.Simulations)
