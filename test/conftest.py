"""
pytest configuration for nr-catalog-tools tests.

Markers
-------
requires_data
    Test needs at least one waveform HDF5 / simulation file cached on disk.
    Safe to run in CI once the data-download step has completed.

cross_catalog
    Test compares results across all three catalogs simultaneously.
    Skipped automatically (via fixture) when any one catalog is unavailable.
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_data: test needs waveform data cached locally (HDF5 / tar.gz)",
    )
    config.addinivalue_line(
        "markers",
        "cross_catalog: test compares waveforms across RIT, SXS, and MAYA catalogs",
    )
