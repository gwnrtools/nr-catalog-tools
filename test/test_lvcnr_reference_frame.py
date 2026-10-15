"""Reference epoch and frame of LVCNR files, checked against LAL.

LAL's NR routine (``XLALSimInspiralNRWaveformGetHplusHcross``) builds the
LAL source frame from ``LNhat`` and ``nhat``:

- for LVCNR format 1, and whenever ``fRef <= 0``, from the vectors stored as
  file attributes, which hold at the first sample of the stored waveform;
- for format 2 and above with ``fRef > 0``, by interpolating the
  ``LNhat*-vs-time`` and ``position*-vs-time`` groups at the time where
  ``Omega-vs-time`` reaches ``pi * M * fRef``.

The unit tests below exercise the interpolation helpers on small synthetic
LVCNR files whose time series are linear in time, so a natural cubic spline
reproduces them exactly. The reference-time tests use a synthetic chirp whose
orbital frequency is known in closed form. The LAL-equivalence tests compare
``WaveformModes.get_td_waveform`` with LAL on cached MAYA files, with time
alignment only: no phase or polarization maximization.
"""

from pathlib import Path

import h5py
import lal
import numpy as np
import pytest

from nrcats import lvc
from nrcats.waveform import WaveformModes

MTSUN_PI = lal.MTSUN_SI * lal.PI

# Synthetic dynamics: every time series is linear in t on [T0, T1].
T0, T1 = -1000.0, 0.0
OMEGA0, OMEGA_SLOPE = 0.02, 2e-5  # Omega(t) = OMEGA0 + OMEGA_SLOPE * (t - T0)


def _omega_of_t(t):
    return OMEGA0 + OMEGA_SLOPE * (np.asarray(t) - T0)


# Linear coefficients (value at T0, slope) of the vector time series.
_LINEAR_SERIES = {
    # LNhat stays within 3e-4 of unit length and nearly normal to the
    # separation, as for real dynamics.
    "LNhatx-vs-time": (0.05, 3e-5),
    "LNhaty-vs-time": (-0.02, 1e-5),
    "LNhatz-vs-time": (0.9986, -1.9e-6),
    "position1x-vs-time": (4.0, -1e-3),
    "position1y-vs-time": (1.0, 4e-3),
    "position1z-vs-time": (-0.2, 0.0),
    "position2x-vs-time": (-6.0, 2e-3),
    "position2y-vs-time": (-1.5, -5e-3),
    "position2z-vs-time": (0.25, 1e-6),
}


def _linear(key, t):
    v0, slope = _LINEAR_SERIES[key]
    return v0 + slope * (np.asarray(t) - T0)


def _write_group(h5_file, name, x, y):
    group = h5_file.create_group(name)
    group.create_dataset("X", data=np.asarray(x, dtype=float))
    group.create_dataset("Y", data=np.asarray(y, dtype=float))
    group.create_dataset("deg", data=3)
    group.create_dataset("tol", data=1e-6)


def _frame_attrs_at(t):
    """LNhat and nhat attributes of the synthetic dynamics at time t."""
    ln = np.array([_linear(f"LNhat{c}-vs-time", t) for c in "xyz"])
    dpos = np.array(
        [
            _linear(f"position1{c}-vs-time", t) - _linear(f"position2{c}-vs-time", t)
            for c in "xyz"
        ]
    )
    nh = dpos / np.linalg.norm(dpos)
    return ln, nh


def _make_dynamics_file(path, fmt=2, omega_y=None, attrs_scale=1.0):
    """A synthetic LVCNR file with *-vs-time groups and frame attributes."""
    x = np.linspace(T0, T1, 41)
    ln, nh = _frame_attrs_at(T0)
    with h5py.File(path, "w") as f:
        f.attrs["Format"] = fmt
        f.attrs["f_lower_at_1MSUN"] = OMEGA0 / MTSUN_PI
        f.attrs["Omega"] = OMEGA0
        for i, c in enumerate("xyz"):
            f.attrs[f"LNhat{c}"] = attrs_scale * ln[i]
            f.attrs[f"nhat{c}"] = attrs_scale * nh[i]
        _write_group(
            f, "Omega-vs-time", x, _omega_of_t(x) if omega_y is None else omega_y
        )
        for key in _LINEAR_SERIES:
            _write_group(f, key, x, _linear(key, x))
    return path


@pytest.fixture
def dynamics_file(tmp_path):
    return _make_dynamics_file(tmp_path / "dyn.h5")


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------


def test_read_lvcnr_group_returns_knots(dynamics_file):
    with h5py.File(dynamics_file, "r") as f:
        x, y = lvc.read_lvcnr_group(f, "Omega-vs-time")
    np.testing.assert_allclose(x, np.linspace(T0, T1, 41))
    np.testing.assert_allclose(y, _omega_of_t(x))


def test_ref_time_and_freq_are_inverse_and_in_lal_units(dynamics_file):
    t_known = -437.25
    f_known = _omega_of_t(t_known) / MTSUN_PI  # Hz at 1 Msun
    with h5py.File(dynamics_file, "r") as f:
        assert lvc.get_ref_time_from_ref_freq(f, f_known) == pytest.approx(
            t_known, abs=1e-8
        )
        assert lvc.get_ref_freq_from_ref_time(f, t_known) == pytest.approx(
            f_known, rel=1e-12
        )


def test_interp_ref_values_from_groups(dynamics_file):
    t_ref = -123.4
    with h5py.File(dynamics_file, "r") as f:
        vals = lvc.get_interp_ref_values_from_h5_file(f, list(_LINEAR_SERIES), t_ref)
    for key, val in vals.items():
        assert val == pytest.approx(_linear(key, t_ref), abs=1e-12)


def test_frame_by_interp_points_from_body2_to_body1(dynamics_file):
    t_ref = -600.0
    ln, nh = _frame_attrs_at(t_ref)
    with h5py.File(dynamics_file, "r") as f:
        params = lvc.compute_lal_source_frame_by_interp(
            f, list(_LINEAR_SERIES), t_ref
        )
    np.testing.assert_allclose([params[f"nhat{c}"] for c in "xyz"], nh, atol=1e-12)
    np.testing.assert_allclose([params[f"LNhat{c}"] for c in "xyz"], ln, atol=1e-12)


def test_interpolation_refuses_out_of_range_and_non_monotonic(tmp_path):
    path = _make_dynamics_file(tmp_path / "dyn.h5")
    with h5py.File(path, "r") as f:
        with pytest.raises(ValueError, match="outside"):
            lvc.get_interp_ref_values_from_h5_file(f, ["LNhatx-vs-time"], T1 + 1)
        with pytest.raises(ValueError, match="outside"):
            lvc.get_ref_time_from_ref_freq(f, 10 * OMEGA0 / MTSUN_PI)

    # Omega dipping after the start cannot be inverted (LAL fails too).
    x = np.linspace(T0, T1, 41)
    omega = _omega_of_t(x)
    omega[1] = omega[0] - 1e-6
    path = _make_dynamics_file(tmp_path / "dip.h5", omega_y=omega)
    with h5py.File(path, "r") as f:
        with pytest.raises(ValueError, match="strictly increasing"):
            lvc.get_ref_time_from_ref_freq(f, omega[5] / MTSUN_PI)


def test_check_fref_follows_lal(tmp_path):
    path = _make_dynamics_file(tmp_path / "dyn.h5")
    f_low = OMEGA0 / MTSUN_PI
    with h5py.File(path, "a") as f:
        # f_lower_at_1MSUN a little below the start of Omega-vs-time
        f.attrs["f_lower_at_1MSUN"] = 0.99 * f_low
        assert lvc.check_fref(f, -1) == -1
        assert lvc.check_fref(f, 1.2 * f_low) == 1.2 * f_low
        # Between f_lower and the first Omega sample: use the attributes.
        assert lvc.check_fref(f, 0.995 * f_low) == -1
        with pytest.raises(ValueError, match="lower than the start frequency"):
            lvc.check_fref(f, 0.9 * f_low)


def test_lvcnr_ref_time_ignores_fref_for_format_1(tmp_path):
    f_ref = _omega_of_t(-300.0) / MTSUN_PI
    path = _make_dynamics_file(tmp_path / "f1.h5", fmt=1)
    with h5py.File(path, "r") as f:
        assert lvc.get_lvcnr_format(f) == 1
        assert lvc.get_lvcnr_ref_time(f, f_ref) is None
    path = _make_dynamics_file(tmp_path / "f3.h5", fmt=3)
    with h5py.File(path, "r") as f:
        assert lvc.get_lvcnr_ref_time(f, f_ref) == pytest.approx(-300.0, abs=1e-8)
        assert lvc.get_lvcnr_ref_time(f, None) is None
        assert lvc.get_lvcnr_ref_time(f, -1) is None


def test_check_interp_req_without_any_reference_time(dynamics_file):
    with h5py.File(dynamics_file, "r") as f:
        assert lvc.check_interp_req(f, {}, ref_time=None) == (False, None)
        assert lvc.check_interp_req(f, {}, ref_time=10.0) == (True, None)


def test_check_interp_req_reads_rit_relaxed_time():
    md = {"relaxed-time": 30}
    assert lvc.check_interp_req(None, md, ref_time=None) == (False, 30.0)
    assert lvc.check_interp_req(None, md, ref_time=30.0) == (False, 30.0)
    assert lvc.check_interp_req(None, md, ref_time=50.0) == (True, 30.0)
    # an integer ref_time is a reference time too
    assert lvc.check_interp_req(None, md, ref_time=50) == (True, 30.0)
    assert lvc.get_ref_time_from_metadata(md) == 30


# ---------------------------------------------------------------------------
# Rotation angles
# ---------------------------------------------------------------------------


def _angles(path, **kw):
    with h5py.File(path, "r") as f:
        return lvc.get_nr_to_lal_rotation_angles(f, {}, 1.1, 0.7, **kw)


def test_rotation_angles_normalise_the_frame_vectors(tmp_path):
    unit = _angles(_make_dynamics_file(tmp_path / "a.h5"))
    scaled = _angles(_make_dynamics_file(tmp_path / "b.h5", attrs_scale=1.0005))
    for key in ("theta", "psi", "alpha"):
        assert scaled[key] == pytest.approx(unit[key], abs=1e-12)
    with pytest.raises(ValueError, match="not unity"):
        _angles(_make_dynamics_file(tmp_path / "c.h5", attrs_scale=1.01))


def test_rotation_angles_at_fref_use_the_interpolated_frame(tmp_path):
    """f_ref > 0 on a format 2 file gives the frame at t(f_ref)."""
    t_ref, total_mass = -250.0, 50.0
    f_ref_hz = _omega_of_t(t_ref) / MTSUN_PI / total_mass
    dyn = _make_dynamics_file(tmp_path / "dyn.h5")
    at_fref = _angles(dyn, f_ref=f_ref_hz, total_mass=total_mass)
    assert at_fref["t_ref"] == pytest.approx(t_ref, abs=1e-8)
    at_tref = _angles(dyn, t_ref=t_ref)

    # A format 1 file whose attributes are the frame at t_ref.
    snap = tmp_path / "snap.h5"
    ln, nh = _frame_attrs_at(t_ref)
    with h5py.File(snap, "w") as f:
        f.attrs["Format"] = 1
        for i, c in enumerate("xyz"):
            f.attrs[f"LNhat{c}"] = ln[i]
            f.attrs[f"nhat{c}"] = nh[i]
    expected = _angles(snap)
    for key in ("theta", "psi", "alpha"):
        assert at_fref[key] == pytest.approx(expected[key], abs=1e-10)
        assert at_tref[key] == pytest.approx(expected[key], abs=1e-10)

    # The same f_ref on a format 1 file is ignored, as in LAL.
    _make_dynamics_file(tmp_path / "f1.h5", fmt=1)
    ignored = _angles(tmp_path / "f1.h5", f_ref=f_ref_hz, total_mass=total_mass)
    default = _angles(tmp_path / "f1.h5")
    for key in ("theta", "psi", "alpha"):
        assert ignored[key] == default[key]
    assert ignored["f_ref"] is None
    assert at_fref["f_ref"] == f_ref_hz


# ---------------------------------------------------------------------------
# Reference time from a synthetic chirp
# ---------------------------------------------------------------------------

CHIRP_T = np.arange(-800.0, 60.0, 0.25)
CHIRP_PEAK = 0.0
CHIRP_W0, CHIRP_K = 0.02, 1e-4  # orbital frequency before the peak


def _chirp_orbital_phase(t):
    """Orbital phase with Omega = CHIRP_W0 + CHIRP_K (t - t[0]) before the peak,
    a burst of junk at the start and a noisy ringdown after the peak."""
    t = np.asarray(t)
    s = t - t[0]
    phase = CHIRP_W0 * s + 0.5 * CHIRP_K * s**2
    # junk radiation: frequency briefly overshoots any reference value
    phase += 0.3 * np.exp(-((s / 15.0) ** 2)) * np.sin(s / 2.0)
    # after the peak the phase runs backwards and fast
    post = t > CHIRP_PEAK
    phase[post] -= 3.0 * (t[post] - CHIRP_PEAK)
    return phase


def _write_chirp_file(path, omega_ref, fmt=None):
    phase22 = -2 * _chirp_orbital_phase(CHIRP_T)  # h22 ~ exp(-2 i Phi)
    amp22 = 0.1 / (1 + ((CHIRP_T - CHIRP_PEAK) / 30.0) ** 2)
    with h5py.File(path, "w") as f:
        if fmt is not None:
            f.attrs["Format"] = fmt
        f.attrs["Omega"] = omega_ref
        for em, sign in ((2, 1), (-2, -1)):
            _write_group(f, f"amp_l2_m{em}", CHIRP_T, amp22)
            _write_group(f, f"phase_l2_m{em}", CHIRP_T, sign * phase22)
        # WaveformModes needs every m of an ell; make the others negligible.
        for em in (-1, 0, 1):
            _write_group(f, f"amp_l2_m{em}", CHIRP_T, 1e-8 * amp22)
            _write_group(f, f"phase_l2_m{em}", CHIRP_T, 0 * phase22)
    return path


def test_reference_time_is_where_the_orbital_frequency_reaches_omega(tmp_path):
    t_known = -500.0
    omega_ref = CHIRP_W0 + CHIRP_K * (t_known - CHIRP_T[0])
    w = WaveformModes.load_from_h5(str(_write_chirp_file(tmp_path / "c.h5", omega_ref)))
    assert w.peak_time_22 == pytest.approx(CHIRP_PEAK, abs=0.5)
    t_ref = w.t_ref_nr
    assert isinstance(t_ref, float)
    assert t_ref < w.peak_time_22
    assert t_ref == pytest.approx(t_known, abs=0.5)


def test_reference_time_of_a_format_1_file_is_the_first_sample(tmp_path):
    omega_ref = CHIRP_W0 + CHIRP_K * 300.0
    w = WaveformModes.load_from_h5(
        str(_write_chirp_file(tmp_path / "c.h5", omega_ref, fmt=1))
    )
    assert w.t_ref_nr == pytest.approx(float(w.time[0]))


# ---------------------------------------------------------------------------
# Equivalence with LAL on cached MAYA files
# ---------------------------------------------------------------------------

MAYA_DATA = Path.home() / ".cache" / "MAYA" / "data"
TOTAL_MASS, DISTANCE, DELTA_T = 60.0, 100.0, 1.0 / 4096


def _lal_rom_available():
    """LAL needs the SEOBNRv4ROM data file to run format 1 files."""
    import lalsimulation as lalsim

    try:
        lalsim.SimIMRSEOBNRv4ROMTimeOfFrequency(
            20.0, 30 * lal.MSUN_SI, 30 * lal.MSUN_SI, 0.0, 0.0
        )
    except RuntimeError:
        return False
    return True


def _time_aligned_overlap(x, y):
    """Complex overlap of x and y at the best time shift; no phase maximization."""
    n = 1 << int(np.ceil(np.log2(len(x) + len(y))))
    corr = np.fft.ifft(np.fft.fft(x, n) * np.conj(np.fft.fft(y, n)))
    k = np.argmax(np.abs(corr))
    return corr[k] / (np.linalg.norm(x) * np.linalg.norm(y))


@pytest.mark.requires_data
@pytest.mark.parametrize(
    "sim, f_ref",
    [
        ("GT0001", None),  # format 1, tilted frame
        ("MAYA1070", None),  # format 3
        ("MAYA1060", 30.0),  # format 2, frame interpolated at f_ref
        ("MAYA1060", 40.0),
    ],
)
def test_get_td_waveform_matches_lal_without_phase_maximization(sim, f_ref):
    import lalsimulation as lalsim

    path = MAYA_DATA / f"{sim}.h5"
    if not path.exists():
        pytest.skip(f"{path} not cached")
    with h5py.File(path, "r") as f:
        fmt = lvc.get_lvcnr_format(f)
        m1, m2 = float(f.attrs["mass1"]), float(f.attrs["mass2"])
        f_low = float(f.attrs["f_lower_at_1MSUN"]) / TOTAL_MASS
    if fmt < 2 and not _lal_rom_available():
        pytest.skip("LAL cannot find the SEOBNRv4ROM data it needs for format 1")
    m1, m2 = TOTAL_MASS * m1 / (m1 + m2), TOTAL_MASS * m2 / (m1 + m2)
    lal_f_ref = -1.0 if f_ref is None else f_ref
    spins = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(
        lal_f_ref, TOTAL_MASS, str(path)
    )

    w = WaveformModes.load_from_h5(str(path))
    for inclination in (0.4, 1.6, 2.6):
        for coa_phase in (0.3, 2.0, 4.5):
            h_nrcats = np.asarray(
                w.get_td_waveform(
                    total_mass=TOTAL_MASS,
                    distance=DISTANCE,
                    inclination=inclination,
                    coa_phase=coa_phase,
                    delta_t_seconds=DELTA_T,
                    f_ref=f_ref,
                )
            )
            # get_td_waveform maps coa_phase to phi_ref; give LAL the same.
            phi_ref = w.get_obs_phi_ref_from_obs_coa_phase(
                coa_phase, f_ref=None if f_ref is None else f_ref * TOTAL_MASS
            )
            hp, hc = lalsim.SimInspiralNRWaveformGetHplusHcross(
                float(phi_ref),
                inclination,
                DELTA_T,
                m1 * lal.MSUN_SI,
                m2 * lal.MSUN_SI,
                DISTANCE * 1e6 * lal.PC_SI,
                f_low * 1.001,
                lal_f_ref,
                *spins,
                str(path),
                None,
            )
            overlap = _time_aligned_overlap(h_nrcats, hp.data.data + 1j * hc.data.data)
            assert abs(overlap) > 0.9999, (inclination, coa_phase, abs(overlap))
            assert abs(np.angle(overlap)) < 1e-3, (
                inclination,
                coa_phase,
                np.angle(overlap),
            )


@pytest.mark.requires_data
def test_fref_changes_the_frame_of_a_precessing_file():
    """Guards against f_ref being silently ignored on format 2 files."""
    path = MAYA_DATA / "MAYA1060.h5"
    if not path.exists():
        pytest.skip(f"{path} not cached")
    w = WaveformModes.load_from_h5(str(path))
    default = w.get_angles(1.6, 2.0)
    at_fref = w.get_angles(1.6, 2.0, f_ref=40.0 * TOTAL_MASS)
    assert at_fref["t_ref"] > w.t_ref_nr + 100
    assert abs(at_fref["alpha"] - default["alpha"]) > 1e-2
