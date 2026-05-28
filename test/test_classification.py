"""Unit tests for the NRCatalogClassifier class."""

import pytest
from unittest.mock import MagicMock, patch

from nrcatalogtools import NRCatalogClassifier


def test_classifier_initialization_defaults():
    """Test that classifier initializes with correct default thresholds."""
    classifier = NRCatalogClassifier()
    assert classifier.spin_threshold == 0.001
    assert classifier.ecc_threshold == 0.005
    assert len(classifier.CATEGORY_MAPPING) == 6


def test_classifier_initialization_custom():
    """Test that classifier respects custom thresholds."""
    classifier = NRCatalogClassifier(spin_threshold=0.05, ecc_threshold=0.01)
    assert classifier.spin_threshold == 0.05
    assert classifier.ecc_threshold == 0.01


def test_category_mapping_keys_and_values():
    """Test that category keys are exactly a-f and map to correct values."""
    mapping = NRCatalogClassifier.CATEGORY_MAPPING
    assert list(mapping.keys()) == ["a", "b", "c", "d", "e", "f"]
    assert mapping["a"] == "non-spinning eccentric"
    assert mapping["b"] == "non-spinning non-eccentric"
    assert mapping["c"] == "aligned-spin eccentric"
    assert mapping["d"] == "aligned-spin non-eccentric"
    assert mapping["e"] == "precessing-spin eccentric"
    assert mapping["f"] == "precessing-spin non-eccentric"


@pytest.mark.parametrize(
    "catalog,meta,expected",
    [
        # SXS Non-spinning, Non-eccentric
        (
            "SXS",
            {
                "reference_eccentricity": 0.0002,
                "reference_dimensionless_spin1": [1e-4, -2e-4, 5e-4],
                "reference_dimensionless_spin2": [3e-5, 0.0, 9e-4],
            },
            "non-spinning non-eccentric",
        ),
        # SXS Non-spinning, Eccentric
        (
            "SXS",
            {
                "reference_eccentricity": 0.02,
                "reference_dimensionless_spin1": [1e-4, 0.0, 0.0],
                "reference_dimensionless_spin2": [0.0, 0.0, 0.0],
            },
            "non-spinning eccentric",
        ),
        # SXS Aligned-spin, Non-eccentric
        (
            "SXS",
            {
                "reference_eccentricity": 0.001,
                "reference_dimensionless_spin1": [0.0, 0.0, 0.3],
                "reference_dimensionless_spin2": [0.0, 0.0, -0.4],
            },
            "aligned-spin non-eccentric",
        ),
        # SXS Aligned-spin, Eccentric
        (
            "SXS",
            {
                "reference_eccentricity": 0.0051,
                "reference_dimensionless_spin1": [1e-4, 2e-5, 0.1],
                "reference_dimensionless_spin2": [0.0, 0.0, 0.0],
            },
            "aligned-spin eccentric",
        ),
        # SXS Precessing-spin, Non-eccentric
        (
            "SXS",
            {
                "reference_eccentricity": 0.0,
                "reference_dimensionless_spin1": [0.1, 0.0, 0.0],
                "reference_dimensionless_spin2": [0.0, 0.0, 0.0],
            },
            "precessing-spin non-eccentric",
        ),
        # SXS Precessing-spin, Eccentric
        (
            "SXS",
            {
                "reference_eccentricity": 0.006,
                "reference_dimensionless_spin1": [0.0, 0.2, 0.1],
                "reference_dimensionless_spin2": [0.0, 0.0, 0.0],
            },
            "precessing-spin eccentric",
        ),
        # RIT Aligned-spin, Non-eccentric
        (
            "RIT",
            {
                "eccentricity": 0.002,
                "relaxed-chi1x": 0.0,
                "relaxed-chi1y": 0.0,
                "relaxed-chi1z": 0.5,
                "relaxed-chi2x": 0.0,
                "relaxed-chi2y": 0.0,
                "relaxed-chi2z": 0.0,
            },
            "aligned-spin non-eccentric",
        ),
        # RIT Non-spinning, Eccentric (with string eccentricity)
        (
            "RIT",
            {
                "eccentricity": "~0.008",
                "relaxed-chi1x": 0.0,
                "relaxed-chi1y": 0.0,
                "relaxed-chi1z": 0.0,
                "relaxed-chi2x": 0.0,
                "relaxed-chi2y": 0.0,
                "relaxed-chi2z": 0.0,
            },
            "non-spinning eccentric",
        ),
        # MAYA Precessing-spin, Non-eccentric
        (
            "MAYA",
            {
                "eccentricity": 0.0001,
                "a1x": 0.2,
                "a1y": 0.0,
                "a1z": 0.0,
                "a2x": 0.0,
                "a2y": 0.0,
                "a2z": 0.0,
            },
            "precessing-spin non-eccentric",
        ),
    ],
)
def test_simulation_classification_logic(catalog, meta, expected):
    """Test that the core threshold and category logic functions correctly."""
    classifier = NRCatalogClassifier()

    mock_catalog = MagicMock()
    mock_catalog.get_metadata.return_value = meta

    with patch.object(classifier, "load_catalog", return_value=mock_catalog):
        res = classifier.classify_simulation(catalog, "TEST_SIM")
        assert res == expected


def test_classify_all_and_caching():
    """Test that classify_all precomputes categories and caches them correctly."""
    classifier = NRCatalogClassifier()

    mock_catalog = MagicMock()
    mock_catalog.simulations_list = ["SIM1", "SIM2"]

    # Mock classify_simulation to return pre-selected classes
    classifications = {
        "SIM1": "non-spinning non-eccentric",
        "SIM2": "aligned-spin eccentric",
    }

    def mock_classify(cat, sim):
        return classifications[sim]

    with patch.object(classifier, "load_catalog", return_value=mock_catalog):
        with patch.object(classifier, "classify_simulation", side_effect=mock_classify):
            classifier.classify_all("SXS")

            # Verify cache was populated correctly
            cache = classifier._classifications["SXS"]
            assert "non-spinning non-eccentric" in cache
            assert "aligned-spin eccentric" in cache
            assert "SIM1" in cache["non-spinning non-eccentric"]
            assert "SIM2" in cache["aligned-spin eccentric"]

            # Verify total count across mapping categories matches simulations count
            total_cached_sims = sum(len(sim_list) for sim_list in cache.values())
            assert total_cached_sims == 2


def test_get_simulations_reloading_and_validations():
    """Test the dynamic get_simulations API including query key/value resolving."""
    classifier = NRCatalogClassifier()

    # Pre-populate classification cache to bypass classify_all logic
    cache = {cat: [] for cat in classifier.CATEGORY_MAPPING.values()}
    cache["aligned-spin non-eccentric"] = ["SXS:BBH:0001", "SXS:BBH:0002"]
    cache["non-spinning eccentric"] = ["SXS:BBH:0003"]
    classifier._classifications["SXS"] = cache

    # Query with short category tag
    res_short = classifier.get_simulations("SXS", "d")
    assert res_short == ["SXS:BBH:0001", "SXS:BBH:0002"]

    # Query with full category name
    res_full = classifier.get_simulations("SXS", "non-spinning eccentric")
    assert res_full == ["SXS:BBH:0003"]

    # Query invalid category raises ValueError
    with pytest.raises(ValueError, match="Unknown category"):
        classifier.get_simulations("SXS", "invalid_cat")


def test_nrsur_calibration_filtering():
    """Test filtering simulations by NRSur7dq4 training/calibration membership."""
    classifier = NRCatalogClassifier()

    # Pre-populate classification cache and mock training set
    cache = {cat: [] for cat in classifier.CATEGORY_MAPPING.values()}
    cache["precessing-spin non-eccentric"] = [
        "SXS:BBH:0001",
        "SXS:BBH:0002",
        "SXS:BBH:0003",
    ]
    classifier._classifications["SXS"] = cache

    mock_calibration_set = {"SXS:BBH:0001", "SXS:BBH:0003"}

    with patch.object(
        classifier, "load_nrsur_calibration_sims", return_value=mock_calibration_set
    ):
        filtered_sims = classifier.get_simulations(
            "SXS", "f", only_nrsur_calibration=True
        )
        assert filtered_sims == ["SXS:BBH:0001", "SXS:BBH:0003"]

        # only_nrsur_calibration is only supported for SXS
        with pytest.raises(
            ValueError, match="only_nrsur_calibration=True is only supported"
        ):
            classifier.get_simulations("RIT", "f", only_nrsur_calibration=True)
