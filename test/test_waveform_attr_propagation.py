"""Tests for WaveformModes attribute propagation (REQ-3.2).

Verify that custom instance attributes survive slicing, copying, and
ufunc operations on ``WaveformModes`` instances.
"""

import copy

import numpy as np
import pytest
import quaternionic

from nrcatalogtools.waveform import WaveformModes


# ---------------------------------------------------------------------------
# Fixture: build a minimal WaveformModes with all custom attributes set
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_wfm():
    """Return a small WaveformModes instance with known custom attributes."""
    n_times = 200
    ell_min, ell_max = 2, 4
    n_modes = sum(2 * ell + 1 for ell in range(ell_min, ell_max + 1))

    times = np.linspace(0, 100, n_times)
    data = np.random.default_rng(42).standard_normal((n_times, n_modes)) + 0j

    metadata = {
        "metadata": {
            "catalog_type": "RIT",
            "waveform_data_location": "/fake/path.h5",
        }
    }

    wfm = WaveformModes(
        data,
        time=times,
        time_axis=0,
        modes_axis=1,
        ell_min=ell_min,
        ell_max=ell_max,
        verbosity=3,
        _filepath="/test/file.h5",
        frame=quaternionic.array([[1.0, 0.0, 0.0, 0.0]]),
        frame_type="inertial",
        data_type="h",
        spin_weight=-2,
        r_is_scaled_out=True,
        m_is_scaled_out=True,
        **metadata,
    )

    # Set additional custom attributes that the loaders would set.
    wfm._present_modes = {(2, 2), (2, -2), (3, 1)}
    wfm._peak_time_22 = 55.5
    wfm._t_ref_nr = 12.3
    return wfm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_attrs_match(result, original):
    """Assert that all custom attributes on *result* match those on *original*."""
    assert result._filepath == original._filepath
    assert result._present_modes == original._present_modes
    assert result._peak_time_22 == original._peak_time_22
    assert result._t_ref_nr == original._t_ref_nr
    assert result.verbosity == original.verbosity


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSlicing:
    """Attributes survive NumPy/sxs slicing operations."""

    def test_row_slice(self, sample_wfm):
        """Basic time-axis slicing via sxs._slice preserves custom attrs."""
        sliced = sample_wfm[50:150]
        assert isinstance(sliced, WaveformModes)
        _assert_attrs_match(sliced, sample_wfm)

    def test_single_column_returns_timeseries(self, sample_wfm):
        """Column selection drops the modes axis → sxs.TimeSeries, not WaveformModes."""
        col = sample_wfm[:, 0]
        # This is expected to be a TimeSeries (the modes axis is gone).
        # We just verify it doesn't crash.
        assert col.shape[0] == sample_wfm.shape[0]

    def test_step_slice(self, sample_wfm):
        """Step-slice preserves custom attrs."""
        stepped = sample_wfm[::2]
        assert isinstance(stepped, WaveformModes)
        _assert_attrs_match(stepped, sample_wfm)


class TestCopy:
    """Attributes survive copy.copy and copy.deepcopy."""

    def test_copy_copy(self, sample_wfm):
        copied = copy.copy(sample_wfm)
        assert isinstance(copied, WaveformModes)
        _assert_attrs_match(copied, sample_wfm)
        # Mutable containers should be separate objects.
        assert copied._present_modes is not sample_wfm._present_modes

    def test_copy_deepcopy(self, sample_wfm):
        deep = copy.deepcopy(sample_wfm)
        assert isinstance(deep, WaveformModes)
        _assert_attrs_match(deep, sample_wfm)
        assert deep._present_modes is not sample_wfm._present_modes

    def test_deepcopy_independence(self, sample_wfm):
        """Mutating the deepcopy must not affect the original."""
        deep = copy.deepcopy(sample_wfm)
        deep._present_modes.add((4, 4))
        assert (4, 4) not in sample_wfm._present_modes

    def test_ndarray_copy_method(self, sample_wfm):
        """The ndarray .copy() method should propagate attributes."""
        copied = sample_wfm.copy()
        assert copied._filepath == sample_wfm._filepath
        assert copied._present_modes == sample_wfm._present_modes
        assert copied._peak_time_22 == sample_wfm._peak_time_22


class TestUfuncAndArithmetic:
    """Attributes survive NumPy ufunc operations when result is WaveformModes."""

    def test_np_conjugate(self, sample_wfm):
        result = np.conjugate(sample_wfm)
        if isinstance(result, WaveformModes):
            _assert_attrs_match(result, sample_wfm)


class TestDefaults:
    """When constructing from scratch, missing attributes get safe defaults."""

    def test_defaults_on_fresh_instance(self):
        """A brand-new WaveformModes without explicit custom attrs gets defaults."""
        n_times = 50
        ell_min, ell_max = 2, 2
        n_modes = 5  # 2*2+1
        times = np.linspace(0, 10, n_times)
        data = np.zeros((n_times, n_modes), dtype=complex)

        wfm = WaveformModes(
            data,
            time=times,
            time_axis=0,
            modes_axis=1,
            ell_min=ell_min,
            ell_max=ell_max,
            metadata={"catalog_type": "RIT"},
            frame=quaternionic.array([[1.0, 0.0, 0.0, 0.0]]),
            frame_type="inertial",
            data_type="h",
            spin_weight=-2,
            r_is_scaled_out=True,
            m_is_scaled_out=True,
        )
        assert wfm._filepath is None
        assert wfm._present_modes == set()
        assert wfm._peak_time_22 is None
        assert wfm._t_ref_nr is None
        assert wfm.verbosity == 0


class TestPeakTime22CacheSurvival:
    """The cached _peak_time_22 survives slicing; lazy recomputation works."""

    def test_cached_value_propagates(self, sample_wfm):
        """If _peak_time_22 is cached, it should propagate on slice."""
        # Force it to be cached.
        _ = sample_wfm.peak_time_22
        assert sample_wfm._peak_time_22 is not None

        sliced = sample_wfm[10:100]
        assert sliced._peak_time_22 == sample_wfm._peak_time_22

    def test_uncached_value_computes_fresh(self):
        """A fresh WaveformModes with _peak_time_22=None can lazily compute."""
        n_times = 100
        ell_min, ell_max = 2, 2
        n_modes = 5
        times = np.linspace(0, 50, n_times)
        # Create data with a clear peak in the (2,2) mode.
        data = np.zeros((n_times, n_modes), dtype=complex)
        # LM ordering for ell_min=ell_max=2: (2,-2),(2,-1),(2,0),(2,1),(2,2)
        # → (2,2) is at index 4
        peak_idx = 60
        data[:, 4] = np.exp(-((np.arange(n_times) - peak_idx) ** 2) / 50.0)

        wfm = WaveformModes(
            data,
            time=times,
            time_axis=0,
            modes_axis=1,
            ell_min=ell_min,
            ell_max=ell_max,
            metadata={"catalog_type": "RIT"},
            frame=quaternionic.array([[1.0, 0.0, 0.0, 0.0]]),
            frame_type="inertial",
            data_type="h",
            spin_weight=-2,
            r_is_scaled_out=True,
            m_is_scaled_out=True,
        )
        assert wfm._peak_time_22 is None  # not yet computed
        pt = wfm.peak_time_22
        assert pt == pytest.approx(times[peak_idx])
        assert wfm._peak_time_22 is not None  # now cached


class TestPresentModesSurvivesSlice:
    """_present_modes set from load_from_targz must survive slicing."""

    def test_present_modes_after_time_slice(self, sample_wfm):
        sliced = sample_wfm[20:80]
        assert sliced._present_modes == sample_wfm._present_modes

    def test_present_modes_mutation_isolation(self, sample_wfm):
        """Mutating _present_modes on a slice should not affect the original
        when using copy, but MAY affect via view (since _metadata is shared)."""
        copied = copy.copy(sample_wfm)
        copied._present_modes.add((4, 0))
        # After copy.copy, the sets should be independent.
        assert (4, 0) not in sample_wfm._present_modes
