"""Tests for nrcats.frames: the rotation into the LAL source frame."""

import os

import numpy as np
import pytest
import quaternionic

from nrcats import frames

sxs = pytest.importorskip("sxs")


def _aligned_modes(n_t=400, ell_max=4):
    """A toy non-precessing waveform with L along z: h_lm ~ exp(-i m Phi) for m != 0."""
    t = np.linspace(0, 400, n_t)
    phi = 0.05 * t + 1e-5 * t**2
    lm = [(l, m) for l in range(2, ell_max + 1) for m in range(-l, l + 1)]
    data = np.zeros((n_t, len(lm)), dtype=complex)
    for j, (l, m) in enumerate(lm):
        amp = 1.0 / l**2 * (1 + 0.1 * m)
        data[:, j] = amp * np.exp(-1j * m * phi) * (1 + 0.01 * t / 400)
    return sxs.WaveformModes(data, time=t, time_axis=0, modes_axis=1, ell_min=2, ell_max=ell_max,
                             spin_weight=-2, data_type="h")


def test_triad_is_orthonormal_and_right_handed():
    L = np.array([0.1, 0.2, 0.97]); L /= np.linalg.norm(L)
    n = np.array([1.0, 0.05, -0.1]); n /= np.linalg.norm(n)      # unit, but not perpendicular to L
    n, m, L = frames.orthonormal_triad(L, n)
    M = np.column_stack([n, m, L])
    np.testing.assert_allclose(M.T @ M, np.eye(3), atol=1e-12)
    assert np.linalg.det(M) == pytest.approx(1.0)


def test_triad_rejects_non_unit_vectors():
    with pytest.raises(ValueError):
        frames.orthonormal_triad([0, 0, 2.0], [1, 0, 0])


def test_triad_rotor_maps_axes_onto_the_triad():
    L = np.array([0.3, -0.2, 0.9]); L /= np.linalg.norm(L)
    n = np.array([1.0, 0.4, 0.0]); n -= n.dot(L) * L; n /= np.linalg.norm(n)
    R = frames.triad_rotor(L, n)
    rot = lambda v: (R * quaternionic.array([0, *v]) * R.conjugate()).vector
    np.testing.assert_allclose(rot([0, 0, 1]), L, atol=1e-12)
    np.testing.assert_allclose(rot([1, 0, 0]), n, atol=1e-12)


def test_rotate_modes_undoes_a_known_rotation():
    """Turn an aligned waveform so that L and n point elsewhere; rotating into its frame restores it."""
    w0 = _aligned_modes()
    R0 = quaternionic.array.from_axis_angle(np.array([0.3, -0.7, 0.2]))
    # sxs: rotate(R) moves features by R^-1, so rotate(R0^-1) moves z to R0 z R0^-1.
    w_sim = w0.rotate(R0.conjugate())
    L_sim = (R0 * quaternionic.array([0, 0, 0, 1]) * R0.conjugate()).vector
    n_sim = (R0 * quaternionic.array([0, 1, 0, 0]) * R0.conjugate()).vector
    frame = frames.frame_from_vectors(L_sim, n_sim)
    w_lal = frames.rotate_modes(w_sim, frame)
    np.testing.assert_allclose(np.asarray(w_lal), np.asarray(w0), atol=1e-10)


def test_simulation_frame_is_the_identity():
    w0 = _aligned_modes()
    out = frames.rotate_modes(w0, frames.ReferenceFrame())
    np.testing.assert_allclose(np.asarray(out), np.asarray(w0))


def test_vector_to_frame():
    frame = frames.frame_from_vectors([0, 0, 1], [0, 1, 0])     # n along y: x_frame = y_sim
    np.testing.assert_allclose(frames.vector_to_frame([0, 1, 0], frame), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(frames.vector_to_frame([-1, 0, 0], frame), [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(frames.vector_to_frame([0, 0, 1], frame), [0, 0, 1], atol=1e-12)


GT0001 = os.path.expanduser("~/.cache/MAYA/data/GT0001.h5")


@pytest.mark.requires_data
@pytest.mark.skipif(not os.path.exists(GT0001), reason="cached MAYA GT0001 LVCNR file not available")
@pytest.mark.parametrize("inclination,phi_ref", [(0.3, 0.4), (1.0, 2.5), (1.6, 5.0), (2.5, 1.2)])
def test_lal_frame_strain_equals_the_lvcnr_route(inclination, phi_ref):
    """Rotating into the file's LAL frame and evaluating at (iota, pi/2 - phi_ref) reproduces the
    existing route through get_nr_to_lal_rotation_angles, which matches LALSimulation exactly."""
    import h5py
    from nrcats import lvc
    from nrcats.waveform import WaveformModes

    wfm = WaveformModes.load_from_h5(GT0001, metadata={})
    with h5py.File(GT0001, "r") as f:
        frame = frames.frame_from_lvcnr_attrs(f.attrs)
        angles = lvc.get_nr_to_lal_rotation_angles(f, {}, inclination, phi_ref)
    h_route = np.asarray(wfm.evaluate([angles["theta"], angles["psi"], angles["alpha"]]))
    w_lal = frames.rotate_modes(wfm, frame)
    h_frame = np.asarray(w_lal.evaluate(list(frames.lal_frame_angles(inclination, phi_ref))))
    np.testing.assert_allclose(h_frame, h_route, rtol=0, atol=1e-8 * np.max(np.abs(h_route)))


LVCNR_FILES = [os.path.expanduser(f"~/.cache/MAYA/data/{s}.h5") for s in ("GT0001", "MAYA1060", "MAYA1070")]


@pytest.mark.requires_data
@pytest.mark.parametrize("path", [p for p in LVCNR_FILES if os.path.exists(p)] or [pytest.param(None, marks=pytest.mark.skip(reason="no cached LVCNR files"))])
def test_reference_spins_match_lalsimulation(path):
    """Spins projected onto the file's LAL frame equal LAL's own NR spin routine (no reference frequency)."""
    import h5py
    lalsim = pytest.importorskip("lalsimulation")
    with h5py.File(path, "r") as f:
        md = frames.reference_metadata_from_lvcnr(f)
    s = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(-1, 60.0, path)
    np.testing.assert_allclose(md["chi1"], s[:3], atol=1e-6)
    np.testing.assert_allclose(md["chi2"], s[3:], atol=1e-6)
    np.testing.assert_allclose(md["L_hat"], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(md["n_hat"], [1, 0, 0], atol=1e-12)
    assert md["mass1"] >= md["mass2"]
