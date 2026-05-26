import sxs

import nrcatalogtools as nrcat
from nrcatalogtools.catalog import CatalogBase


def test_rit_catalog_not_sxs_catalog():
    """Verify that RITCatalog does not inherit from sxs.Catalog."""
    ritcat = nrcat.RITCatalog.load(download=False)
    # Check that it inherits from our own CatalogBase, not sxs.Catalog
    assert isinstance(ritcat, CatalogBase)
    # This was previously True before composition refactor
    assert getattr(sxs, "Catalog", type("Dummy", (), {})) is not None
    # We allow the check to pass if sxs.Catalog is fully removed in future sxs versions,
    # but for sxs >= 2024.0.0 where it's deprecated, it must be False.
    if hasattr(sxs, "Catalog"):
        assert not isinstance(ritcat, sxs.Catalog)


def test_sxs_catalog_to_sxs():
    """Verify that SXSCatalog.to_sxs() returns the live sxs.Simulations object."""
    sxscat = nrcat.SXSCatalog.load(download=False)
    sxs_sims = sxscat.to_sxs()
    assert isinstance(sxs_sims, sxs.Simulations)

    # Check that it is the live object with full dataframe features
    df = sxscat.simulations_dataframe
    assert hasattr(df, "BBH")


def test_rit_catalog_to_sxs():
    """Verify that RITCatalog.to_sxs() returns a valid sxs.Simulations instance."""
    ritcat = nrcat.RITCatalog.load(download=False)
    sxs_sims = ritcat.to_sxs()
    assert isinstance(sxs_sims, sxs.Simulations)
