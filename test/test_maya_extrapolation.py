"""Extrapolation of finite-radius Psi4 to null infinity (nrcats.extrapolate) and the
MAYA native-file route (nrcats.maya_native, MayaCatalog.get_extrapolated)."""

import os

import numpy as np
import pytest

from nrcats import frames
from nrcats.extrapolate import extrapolate_psi4_to_strain, resample_uniformly, tortoise_radius
from nrcats.maya_native import choose_extraction_radius, reference_metadata_from_tracks

NATIVE_CACHE = os.path.expanduser("~/.cache/MAYA/native")


def test_extrapolation_recovers_the_strain_at_a_large_radius():
    """At a very large radius the extrapolation corrections vanish, so the result is
    the double time integral of Psi4, on the time axis shifted by r_*.  The modes are
    periodic on the window, where the FFI is exact; on a chirp that starts and ends
    abruptly it is not (about 1% here), which is a property of the FFI."""
    delta_t = 0.25
    coordinate_time = np.arange(0.0, 3000.0, delta_t)
    window = len(coordinate_time) * delta_t
    orbital_frequency = 2 * np.pi * 14 / window
    extraction_radius = 1e6
    strain_modes, rpsi4_modes = {}, {}
    for (ell, emm) in [(2, 2), (2, -2), (3, 3), (2, 0)]:
        strain_modes[(ell, emm)] = (0.1 / ell) * np.exp(-1j * emm * orbital_frequency * coordinate_time)
        rpsi4_modes[(ell, emm)] = -((emm * orbital_frequency) ** 2) * strain_modes[(ell, emm)]
    retarded_time, extrapolated = extrapolate_psi4_to_strain(
        coordinate_time, rpsi4_modes, extraction_radius, total_mass=1.0, initial_orbital_frequency=orbital_frequency,
        taper_duration=0,
    )
    areal_radius = extraction_radius * (1 + 0.5 / extraction_radius) ** 2
    np.testing.assert_allclose(retarded_time, coordinate_time - tortoise_radius(areal_radius, 1.0))
    middle = slice(len(coordinate_time) // 4, 3 * len(coordinate_time) // 4)
    for mode_index in [(2, 2), (2, -2), (3, 3)]:
        expected = strain_modes[mode_index][middle]
        assert np.max(np.abs(extrapolated[mode_index][middle] - expected) / np.abs(expected)) < 1e-4
    assert not np.any(extrapolated[(2, 0)])


def test_planck_taper_window():
    from nrcats.extrapolate import planck_taper_window

    time = np.arange(0.0, 100.0, 0.5)
    window = planck_taper_window(time, 10.0)
    assert window[0] == pytest.approx(0, abs=1e-12) and window[-1] == pytest.approx(0, abs=1e-12)
    assert np.all(window[(time >= 10.0) & (time <= 89.5)] == 1)
    assert np.all(np.diff(window[time < 10.0]) >= 0)
    assert np.all(planck_taper_window(time, 0) == 1)
    with pytest.raises(ValueError):
        planck_taper_window(time, 60.0)


def test_extrapolation_requires_an_orbital_frequency():
    time = np.arange(0.0, 10.0, 0.5)
    with pytest.raises(ValueError):
        extrapolate_psi4_to_strain(time, {(2, 2): np.ones_like(time, dtype=complex)}, 100.0, 1.0, None)


def test_resampling_onto_a_uniform_grid():
    time = np.sort(np.concatenate([np.linspace(0, 10, 41), [3.33, 7.77]]))
    uniform_time, uniform_data = resample_uniformly(time, np.exp(1j * time), delta_t=0.25)
    assert np.allclose(np.diff(uniform_time), 0.25)
    np.testing.assert_allclose(uniform_data, np.exp(1j * uniform_time), atol=1e-4)


def test_choose_extraction_radius():
    radii = [30.0, 60.0, 100.0, 145.0, 200.0]
    assert choose_extraction_radius(radii) == 145.0
    assert choose_extraction_radius(radii, 60.0) == 60.0
    assert choose_extraction_radius([200.0, 300.0]) == 200.0
    with pytest.raises(ValueError):
        choose_extraction_radius(radii, 75.0)


def _circular_tracks(heavier_object, orbital_frequency=0.02, separation=10.0, spin_track_end=1000.0):
    """Two holes on circular orbits about the origin, counter-clockwise about +z."""
    time = np.arange(0.0, 1000.0, 0.5)
    masses = {heavier_object: 0.6, 1 - heavier_object: 0.4}
    tracks = {}
    for object_index, sign in ((heavier_object, 1.0), (1 - heavier_object, -1.0)):
        radius = separation * masses[1 - object_index]
        phase = orbital_frequency * time
        spin_time = np.where(time <= spin_track_end, 1.0, np.nan)
        tracks[object_index] = {
            "time": time,
            "x": sign * radius * np.cos(phase),
            "y": sign * radius * np.sin(phase),
            "z": np.zeros_like(time),
            "vx": -sign * radius * orbital_frequency * np.sin(phase),
            "vy": sign * radius * orbital_frequency * np.cos(phase),
            "vz": np.zeros_like(time),
            "Sx": spin_time * 0.1 * masses[object_index] ** 2,
            "Sy": spin_time * 0.0,
            "Sz": spin_time * 0.3 * masses[object_index] ** 2,
        }
    attrs = {"horizon mass 0": masses[0], "horizon mass 1": masses[1]}
    return {"attrs": attrs, "tracks": tracks}


@pytest.mark.parametrize("heavier_object", [0, 1])
def test_reference_metadata_from_tracks_labels_the_heavier_hole_as_body_1(heavier_object):
    native_data = _circular_tracks(heavier_object)
    metadata = reference_metadata_from_tracks(native_data, 100.0)
    assert metadata["reference_mass1"] == pytest.approx(0.6)
    phase = 0.02 * 100.0
    np.testing.assert_allclose(metadata["reference_position1"], [4 * np.cos(phase), 4 * np.sin(phase), 0], atol=1e-6)
    np.testing.assert_allclose(metadata["reference_orbital_frequency"], [0, 0, 0.02], atol=1e-8)
    np.testing.assert_allclose(metadata["reference_dimensionless_spin1"], [0.1, 0, 0.3], atol=1e-8)
    assert metadata["reference_spin_time"] == pytest.approx(100.0)


def test_reference_metadata_uses_the_last_spins_when_the_spin_track_ends_early():
    native_data = _circular_tracks(0, spin_track_end=20.0)
    metadata = reference_metadata_from_tracks(native_data, 75.0)
    assert metadata["reference_spin_time"] == pytest.approx(20.0)
    np.testing.assert_allclose(metadata["reference_dimensionless_spin2"], [0.1, 0, 0.3], atol=1e-8)


@pytest.fixture(scope="module")
def maya():
    from nrcats import MayaCatalog

    try:
        return MayaCatalog.load(download=False)
    except Exception as exc:  # pragma: no cover - depends on the local cache
        pytest.skip(f"MAYA catalog not available: {exc}")


@pytest.mark.requires_data
@pytest.mark.skipif(not os.path.exists(os.path.join(NATIVE_CACHE, "GT0370_r100.00.npz")),
                    reason="cached GT0370 native extract not available")
def test_get_extrapolated_reproduces_the_october_2026_extrapolation(maya):
    """GT0370 (2,2) mismatch with NRSur7dq4 was 2.03e-4 from the analysis script
    (results/maya_table.csv, column E), against 2.3e-3 for the LVCNR file."""
    from nrcats.surrogate import generate_surrogate_modes
    from nrcats.waveform.matching import compute_mode_match, mode_f_lower
    from pycbc.types import TimeSeries
    from scipy.interpolate import CubicSpline

    from nrcats import utils

    total_mass, delta_t_seconds = 40.0, 1.0 / 16384
    wfm = maya.get("GT0370", quantity="extrapolated", frame="simulation")
    assert wfm.sim_metadata["extrapolation"]["extraction_radius"] == 100.0
    assert wfm._metadata["maya_azimuth_corrected"]
    params = maya.get_parameters("GT0370", total_mass=total_mass, frame=None)
    surrogate_modes, f_lower = generate_surrogate_modes(params, total_mass=total_mass,
                                                        delta_t_seconds=delta_t_seconds, modes=[(2, 2)])
    time = np.asarray(wfm.t, dtype=float)
    keep = time >= time[0] + 250.0
    seconds_per_mass = utils.time_to_physical(total_mass)
    strain_22 = np.asarray(wfm.data[:, wfm.index(2, 2)])[keep]
    physical_time = np.arange(time[keep][0], time[keep][-1], delta_t_seconds / seconds_per_mass)
    amplitude = CubicSpline(time[keep], np.abs(strain_22))(physical_time)
    phase = CubicSpline(time[keep], np.unwrap(np.angle(strain_22)))(physical_time)
    series = TimeSeries(amplitude * np.exp(1j * phase), delta_t=delta_t_seconds, epoch=physical_time[0] * seconds_per_mass)
    mismatch = 1 - compute_mode_match(series, surrogate_modes[(2, 2)], mode_f_lower(f_lower, 2))
    assert mismatch == pytest.approx(2.03e-4, rel=0.05)


@pytest.mark.requires_data
@pytest.mark.skipif(not os.path.exists(os.path.join(NATIVE_CACHE, "MAYA1060_r100.00.npz")),
                    reason="cached MAYA1060 native extract not available")
def test_extrapolated_lal_frame_agrees_with_the_waveform_frame(maya):
    """Precessing MAYA1060: the frame from the native tracks and the frame derived from
    the extrapolated modes agree once the modes are rotated by pi about z.  With the
    dynamics paired as in MAYA's LVCNR export the orbital phases differ by about
    0.09 rad (0.31 rad if the dynamics at tau were paired with u = tau)."""
    from nrcats.frame_waveform import frame_consistency

    t_ref = 200.0
    lal_wfm = maya.get_extrapolated("MAYA1060", t_ref=t_ref, extraction_radius=100.0)
    waveform_wfm = maya.get_extrapolated("MAYA1060", frame="waveform", t_ref=t_ref, extraction_radius=100.0)
    tracks_frame = frames.frame_of(lal_wfm)
    assert tracks_frame.name == "lal" and tracks_frame.t_ref == pytest.approx(t_ref)
    consistency = frame_consistency(tracks_frame, frames.frame_of(waveform_wfm))
    assert consistency["l_hat_angle_deg"] < 5.0
    assert abs(consistency["phase_offset_rad"]) < 0.2
    assert lal_wfm.sim_metadata["reference_dynamics_time"] == pytest.approx(t_ref - 100.0 + 108.78, abs=0.05)
