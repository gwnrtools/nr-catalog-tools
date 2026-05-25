"""Tests for RITCatalog module-level singleton caching."""

from unittest.mock import MagicMock, patch

import nrcatalogtools.rit as rit_module


def test_rit_singleton_returned_on_download_false():
    """load(download=False) returns the existing singleton without rebuilding."""
    sentinel = MagicMock()
    orig = rit_module._rit_catalog_singleton
    try:
        rit_module._rit_catalog_singleton = sentinel
        result = rit_module.RITCatalog.load(download=False)
        assert result is sentinel
    finally:
        rit_module._rit_catalog_singleton = orig


def test_rit_singleton_returned_on_download_none():
    """load(download=None) returns the existing singleton without rebuilding."""
    sentinel = MagicMock()
    orig = rit_module._rit_catalog_singleton
    try:
        rit_module._rit_catalog_singleton = sentinel
        result = rit_module.RITCatalog.load(download=None)
        assert result is sentinel
    finally:
        rit_module._rit_catalog_singleton = orig


def test_rit_load_download_true_replaces_singleton():
    """load(download=True) ignores the cached singleton and creates a new one."""
    import pandas as pd

    sentinel = MagicMock()
    orig = rit_module._rit_catalog_singleton

    # Build a fake DataFrame large enough to pass the
    # acceptable_scraping_fraction check (0.7 * 2000 = 1400 rows).
    fake_df = pd.DataFrame(
        {"simulation_name": [f"RIT:BBH:{i:04d}" for i in range(1400)]}
    )
    fake_helper = MagicMock()
    fake_helper.read_metadata_df_from_disk.return_value = fake_df

    try:
        rit_module._rit_catalog_singleton = sentinel
        with patch("nrcatalogtools.rit.RITCatalogHelper", return_value=fake_helper):
            # Patch __init__ so cls(catalog=...) doesn't touch the filesystem.
            with patch.object(rit_module.RITCatalog, "__init__", return_value=None):
                result = rit_module.RITCatalog.load(download=True)
        # The singleton must have been replaced by a new object.
        assert result is not sentinel
        assert rit_module._rit_catalog_singleton is not sentinel
    finally:
        rit_module._rit_catalog_singleton = orig
