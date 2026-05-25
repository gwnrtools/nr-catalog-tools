"""Tests for get_source_parameters_from_metadata catalog_type validation."""

import pytest

from nrcatalogtools.metadata import get_source_parameters_from_metadata


def test_missing_catalog_type_raises_value_error():
    """metadata without 'catalog_type' must raise ValueError."""
    meta = {
        "relaxed_mass_ratio_1_over_2": 1.0,
        "relaxed_chi1x": 0.0,
        "relaxed_chi1y": 0.0,
        "relaxed_chi1z": 0.0,
        "relaxed_chi2x": 0.0,
        "relaxed_chi2y": 0.0,
        "relaxed_chi2z": 0.0,
        "freq_start_22": 0.01,
    }
    with pytest.raises(ValueError, match="catalog_type"):
        get_source_parameters_from_metadata(meta)


def test_unknown_catalog_type_raises_value_error():
    """An unrecognised catalog_type value must raise ValueError."""
    meta = {"catalog_type": "BOGUS"}
    with pytest.raises(ValueError, match="Unknown catalog_type"):
        get_source_parameters_from_metadata(meta)


def test_valid_rit_metadata_returns_pycbc_compatible_dict():
    """Valid RIT metadata returns all expected PyCBC parameter keys."""
    meta = {
        "catalog_type": "RIT",
        "relaxed_mass_ratio_1_over_2": 1.0,
        "relaxed_chi1x": 0.0,
        "relaxed_chi1y": 0.0,
        "relaxed_chi1z": 0.3,
        "relaxed_chi2x": 0.0,
        "relaxed_chi2y": 0.0,
        "relaxed_chi2z": -0.3,
        "freq_start_22": 0.01,
    }
    params = get_source_parameters_from_metadata(meta, total_mass=60.0)
    for key in (
        "mass1",
        "mass2",
        "spin1x",
        "spin1y",
        "spin1z",
        "spin2x",
        "spin2y",
        "spin2z",
        "f_lower",
    ):
        assert key in params, f"Missing PyCBC key: {key}"
