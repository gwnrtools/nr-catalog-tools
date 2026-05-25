"""Tests for WaveformModes.get_mode() delta_t parameter handling."""

import numpy as np
import pytest
import quaternionic

from nrcatalogtools.waveform import WaveformModes


def _make_minimal_wfm(n_times=500, dt=1.0):
    """Return a synthetic WaveformModes (ell=2 only, constant data).

    All modes are constant complex-valued 1+0j so interpolation is trivially
    stable.  The time axis runs from 0 to (n_times-1)*dt in dimensionless M
    units.
    """
    from sxs.waveforms.format_handlers.nrar import (
        h,
        translate_data_type_to_spin_weight,
        translate_data_type_to_sxs_string,
    )

    times = np.arange(n_times, dtype=float) * dt
    n_modes = 5  # ell=2, m in {-2,-1,0,1,2}
    data = np.ones((n_times, n_modes), dtype=complex)

    w_attributes = {
        "_filepath": "",
        "metadata": {"catalog_type": "RIT"},
        "history": "",
        "frame": quaternionic.array([[1.0, 0.0, 0.0, 0.0]]),
        "frame_type": "inertial",
        "data_type": h,
        "r_is_scaled_out": True,
        "m_is_scaled_out": True,
    }
    w_attributes["spin_weight"] = translate_data_type_to_spin_weight(
        w_attributes["data_type"]
    )
    w_attributes["data_type"] = translate_data_type_to_sxs_string(
        w_attributes["data_type"]
    )

    return WaveformModes(
        data,
        time=times,
        time_axis=0,
        modes_axis=1,
        ell_min=2,
        ell_max=2,
        **w_attributes,
    )


def test_get_mode_both_delta_t_params_raises_value_error():
    """Passing both delta_t_seconds and delta_t_Msun must raise ValueError."""
    wfm = _make_minimal_wfm()
    with pytest.raises(ValueError, match="only one"):
        wfm.get_mode(2, 2, delta_t_seconds=1.0 / 4096, delta_t_Msun=0.5)


def test_get_mode_deprecated_delta_t_issues_warning():
    """The legacy delta_t keyword must trigger a DeprecationWarning."""
    wfm = _make_minimal_wfm(n_times=500, dt=1.0)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        wfm.get_mode(2, 2, total_mass=100.0, delta_t=1.0 / 4096)


@pytest.mark.requires_data
def test_get_mode_delta_t_seconds_and_msun_produce_different_lengths():
    """delta_t_seconds and delta_t_Msun yield different sample counts.

    For total_mass=100 M_sun: 1 M_sun ≈ 4.93e-6 s, so delta_t_Msun=1/4096
    maps to a much shorter physical step than delta_t_seconds=1/4096.  The
    resulting time series therefore have different lengths.
    """
    wfm = _make_minimal_wfm(n_times=2000, dt=1.0)
    total_mass = 100.0
    h_secs = wfm.get_mode(2, 2, total_mass=total_mass, delta_t_seconds=1.0 / 4096)
    h_msun = wfm.get_mode(2, 2, total_mass=total_mass, delta_t_Msun=1.0 / 4096)
    assert len(h_secs) != len(h_msun)
