"""compute_mode_match is a complex match: independent of a constant phase of the mode."""

import numpy as np
import pytest
from pycbc.types import TimeSeries

from nrcats.waveform.matching import compute_mode_match

DELTA_T = 1.0 / 4096


def _mode(duration, emm=2, both_signs=0.0, seed_phase=0.3):
    """A chirping mode e^{-i m phi(t)} with an optional counter-rotating part."""
    time = np.arange(0.0, duration, DELTA_T)
    orbital_phase = 2 * np.pi * (15.0 * time + 20.0 * time**2) + seed_phase
    amplitude = 1e-21 * (1 + 2 * time / duration)
    mode = amplitude * np.exp(-1j * emm * orbital_phase)
    if both_signs:
        mode = mode + both_signs * amplitude * np.exp(1j * emm * orbital_phase + 0.7j)
    return TimeSeries(mode, delta_t=DELTA_T)


@pytest.mark.parametrize("both_signs", [0.0, 0.2])
@pytest.mark.parametrize("duration", [4.0, 0.5])
def test_match_is_independent_of_a_constant_phase(both_signs, duration):
    """A mode against itself times e^{i phi_0} matches to 1.  The real-part filter
    gave mismatches up to 1e-3 here when the mode had content at both signs of
    frequency."""
    mode = _mode(duration, both_signs=both_signs)
    for constant_phase in (0.0, 0.9, np.pi / 2, 2.0):
        rotated = TimeSeries(np.asarray(mode) * np.exp(1j * constant_phase), delta_t=DELTA_T)
        assert 1 - compute_mode_match(mode, rotated, 20.0) < 1e-10


def test_match_is_independent_of_a_time_shift():
    mode = _mode(4.0)
    shifted = TimeSeries(np.roll(np.asarray(mode), 37), delta_t=DELTA_T)
    assert 1 - compute_mode_match(mode, shifted, 20.0) < 1e-6


def test_different_modes_are_not_matched():
    """The phase maximisation does not hide a real difference: a different chirp rate."""
    mode = _mode(4.0)
    time = np.arange(0.0, 4.0, DELTA_T)
    other = 1e-21 * (1 + 2 * time / 4.0) * np.exp(-2j * 2 * np.pi * (15.0 * time + 21.0 * time**2))
    assert compute_mode_match(mode, TimeSeries(other, delta_t=DELTA_T), 20.0) < 0.9


def test_single_sign_mode_agrees_with_the_real_part_filter():
    """For a mode with content at one sign of frequency only, the real part carries
    everything, so pycbc's real-series match gives the same value."""
    from pycbc.filter import match
    from pycbc.psd import aLIGOZeroDetHighPower

    mode_a = _mode(4.0)
    mode_b = _mode(4.0, seed_phase=1.1)
    time = np.arange(0.0, 4.0, DELTA_T)
    mode_b = TimeSeries(np.asarray(mode_b) * (1 + 0.05 * np.sin(2 * np.pi * 0.7 * time)), delta_t=DELTA_T)
    complex_match = compute_mode_match(mode_a, mode_b, 20.0)
    assert 0.9 < complex_match < 1 - 1e-6

    real_a, real_b = mode_a.real(), mode_b.real()
    n_fft = 1
    while n_fft < len(real_a):
        n_fft <<= 1
    real_a.resize(n_fft)
    real_b.resize(n_fft)
    psd = aLIGOZeroDetHighPower(n_fft // 2 + 1, 1.0 / (n_fft * DELTA_T), 20.0)
    real_match, _ = match(real_a, real_b, psd=psd, low_frequency_cutoff=20.0, subsample_interpolation=True)
    assert complex_match == pytest.approx(real_match, abs=2e-3)
