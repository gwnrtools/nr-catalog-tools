"""SXS strain sign, LAL source frame from SXS metadata, and get_td_waveform.

Three defects in the SXS path, each with a test here:

1. **Strain sign.**  SXS strain sidecars record "Overall sign change in complex
   strain h." in ``version_info.spec_version_hist``.  With that sign, SXS modes
   run through LAL's NR routine disagree with LAL waveform models by an overall
   sign; RIT and MAYA need no change.  ``SXSCatalog.get`` now multiplies the
   modes by -1 when the record is present (``correct_strain_sign=True``).
2. **n_hat direction.**  LAL's ``nhat`` points from body 2 to body 1, and in
   SXS metadata body 1 (object A) is the heavier hole.
   ``compute_lal_source_frame_from_sxs_metadata`` used ``Pos2 - Pos1``.
3. **No LVCNR file.**  ``get_angles`` and ``_compute_reference_time`` opened
   ``self.filepath`` as an LVCNR file, so ``get_td_waveform`` failed for every
   SXS waveform.  They now use ``reference_time`` and the metadata frame.

Measured on SXS:BBH:0792 (q = 2, precessing, tilted frame), M = 60 Msun, time
alignment only, no phase or polarization maximization:

* against ``SimInspiralNRWaveformGetHplusHcross`` on the sxs package's own
  LVCNR export of the same simulation (modes ell <= 4 on both sides, sign
  flip undone): Re(overlap) 0.9975 to 0.9993, residual phase within 0.04 rad,
  over 4 inclinations x 4 phases.  The export's frame vectors agree with the
  metadata ones to better than 1e-3 degrees, since the export starts at
  reference_time.  With ``nhat`` reversed (the old code) the same grid gives
  |overlap| 0.22 to 0.66.
* against NRSur7dq4 through LAL at the LAL-frame spins and
  ``f_ref = |Omega_ref| / (pi M)``: overlap 0.993 to 0.9996 in modulus with
  phase -0.15 to +0.06 rad after the flip, and phase near pi without it.

Markers:
  requires_data — loads SXS:BBH:0792 from the local sxs cache.
"""

import json
import os

import numpy as np
import pytest

from nrcats import lvc
from nrcats.sxs import strain_sign_change_record

SIM = "SXS:BBH:0792"

# LVCNR export of SXS:BBH:0792 made with sxs.utilities.lvcnr.convert_simulation.
LVCNR_0792 = os.environ.get(
    "NRCATS_SXS_0792_LVCNR", "/tmp/nrcmp/sxs_lvc/out/SXS_BBH_0792_Res3.h5"
)

TOTAL_MASS = 60.0
DISTANCE = 100.0


# ── helpers ───────────────────────────────────────────────────────────────────


def _frame_vectors(params):
    ln = np.array([params["LNhatx"], params["LNhaty"], params["LNhatz"]])
    nh = np.array([params["nhatx"], params["nhaty"], params["nhatz"]])
    return ln, nh


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _synthetic_metadata(R=np.eye(3), offset=(0.0, 0.0, 0.0), omega_sign=1.0):
    """Body 1 heavier and on +x, orbit counter-clockwise about +z, then rotated."""
    offset = np.asarray(offset, dtype=float)
    return {
        "reference_time": 100.0,
        "reference_mass1": 0.6,
        "reference_mass2": 0.4,
        "reference_position1": R @ np.array([4.0, 0.0, 0.0]) + offset,
        "reference_position2": R @ np.array([-6.0, 0.0, 0.0]) + offset,
        # A component along the separation must not change the result.
        "reference_orbital_frequency": R @ np.array([0.003, 0.0, omega_sign * 0.02]),
    }


def _overlap(x, y, max_shift=64):
    """Complex overlap at the best time shift within +-max_shift samples.

    Time alignment only: the phase of the returned number is not maximized
    over.  The longer series is cut to the shorter one about the peak.
    """
    px, py = np.argmax(np.abs(x)), np.argmax(np.abs(y))
    lo = min(px, py)
    hi = min(len(x) - px, len(y) - py)
    x = np.array(x[px - lo : px + hi])
    y = np.array(y[py - lo : py + hi])
    n = 1 << int(np.ceil(np.log2(2 * len(x))))
    C = np.fft.ifft(np.fft.fft(x, n) * np.conj(np.fft.fft(y, n)))
    window = np.r_[0:max_shift, n - max_shift : n]
    k = window[np.argmax(np.abs(C[window]))]
    return C[k] / (np.linalg.norm(x) * np.linalg.norm(y))


# ── n_hat / L_hat from metadata (no data needed) ─────────────────────────────


def test_nhat_points_from_body2_to_body1():
    """nhat = (x_1 - x_2)/|x_1 - x_2| and LNhat along the orbital angular momentum."""
    ln, nh = _frame_vectors(
        lvc.compute_lal_source_frame_from_sxs_metadata(
            _synthetic_metadata(offset=(1.0, -2.0, 0.5))
        )
    )
    np.testing.assert_allclose(nh, [1.0, 0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(ln, [0.0, 0.0, 1.0], atol=1e-14)


def test_lnhat_follows_the_sense_of_rotation():
    ln, nh = _frame_vectors(
        lvc.compute_lal_source_frame_from_sxs_metadata(
            _synthetic_metadata(omega_sign=-1.0)
        )
    )
    np.testing.assert_allclose(nh, [1.0, 0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(ln, [0.0, 0.0, -1.0], atol=1e-14)


def test_frame_rotates_with_the_binary():
    """A rotated configuration gives the rotated triad, orthonormal."""
    R = _rotation([0.3, -1.0, 0.7], 1.1)
    ln, nh = _frame_vectors(
        lvc.compute_lal_source_frame_from_sxs_metadata(_synthetic_metadata(R=R))
    )
    np.testing.assert_allclose(nh, R @ [1.0, 0.0, 0.0], atol=1e-13)
    np.testing.assert_allclose(ln, R @ [0.0, 0.0, 1.0], atol=1e-13)
    assert abs(ln @ nh) < 1e-13
    assert abs(np.linalg.norm(ln) - 1) < 1e-13 and abs(np.linalg.norm(nh) - 1) < 1e-13


# ── sidecar record (no data needed) ───────────────────────────────────────────


class _FakeSim:
    def __init__(self, directory, strain_path):
        self.__file__ = str(directory)
        self.strain_path = strain_path


_SIGN_ENTRY = [
    "ef51849550a1d8a5bbdd810c7debf0dd839e86dd",
    "Overall sign change in complex strain h.",
]


def test_sign_record_found_in_strain_sidecar(tmp_path):
    (tmp_path / "Lev3:Strain_N2.json").write_text(
        json.dumps({"version_info": {"spec_version_hist": [_SIGN_ENTRY]}})
    )
    record = strain_sign_change_record(_FakeSim(tmp_path, ("Lev3:Strain_N2", "/")))
    assert record == _SIGN_ENTRY[1]


def test_sign_record_found_in_extra_waveforms_group(tmp_path):
    group = {"version_info": {"spec_version_hist": [_SIGN_ENTRY]}}
    (tmp_path / "Lev3:ExtraWaveforms.json").write_text(
        json.dumps({"/Strain_N3.dir": group, "/Psi4_N3.dir": {}})
    )
    sim = _FakeSim(tmp_path, ("Lev3:ExtraWaveforms", "/Strain_N3.dir"))
    assert strain_sign_change_record(sim) == _SIGN_ENTRY[1]


def test_no_record_means_no_flip(tmp_path):
    (tmp_path / "Lev3:Strain_N2.json").write_text(
        json.dumps(
            {"version_info": {"spec_version_hist": [["abc", "Something else."]]}}
        )
    )
    assert (
        strain_sign_change_record(_FakeSim(tmp_path, ("Lev3:Strain_N2", "/"))) is None
    )
    # Missing sidecar: no record either.
    assert (
        strain_sign_change_record(_FakeSim(tmp_path, ("Lev3:Strain_N4", "/"))) is None
    )


# ── real data: SXS:BBH:0792 ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def sxs_pair():
    """(flipped, raw) WaveformModes for SIM, skipping if not cached."""
    from nrcats import load_catalog

    try:
        cat = load_catalog("SXS")
        flipped = cat.get(SIM, download=False)
        raw = cat.get(SIM, download=False, correct_strain_sign=False)
    except Exception as exc:  # noqa: BLE001 - any failure means "no data here"
        pytest.skip(f"{SIM} unavailable: {type(exc).__name__}: {exc}")
    return flipped, raw


@pytest.mark.requires_data
def test_sign_flip_applied_and_recorded(sxs_pair):
    flipped, raw = sxs_pair
    assert flipped.metadata["strain_sign_flipped"] is True
    assert "sign change" in flipped.metadata["strain_sign_record"].lower()
    assert raw.metadata["strain_sign_flipped"] is False
    np.testing.assert_array_equal(np.asarray(flipped.data), -np.asarray(raw.data))


@pytest.mark.requires_data
def test_reference_time_and_angles_without_lvcnr_file(sxs_pair):
    wfm, _ = sxs_pair
    assert wfm.t_ref_nr == float(wfm.metadata["reference_time"])
    angles = wfm.get_angles(inclination=0.7, coa_phase=0.3)
    assert all(np.isfinite(angles[k]) for k in ("theta", "psi", "alpha"))
    # The metadata defines the frame at one epoch only.
    with pytest.raises(ValueError, match="reference_time"):
        wfm.get_angles(inclination=0.7, coa_phase=0.3, f_ref=20.0)


def _lal_frame_parameters(wfm):
    """Masses, LAL-frame spins and f_ref (Hz) at the SXS reference epoch."""
    import lal

    md = wfm.metadata
    ln, nh = _frame_vectors(lvc.compute_lal_source_frame_from_sxs_metadata(md))
    S1, S2 = lvc.transform_spins_nr_to_lal(
        md["reference_dimensionless_spin1"], md["reference_dimensionless_spin2"], nh, ln
    )
    m1, m2 = md["reference_mass1"], md["reference_mass2"]
    mass1 = TOTAL_MASS * m1 / (m1 + m2)
    f_ref = np.linalg.norm(md["reference_orbital_frequency"]) / (
        np.pi * TOTAL_MASS * lal.MTSUN_SI
    )
    return mass1, TOTAL_MASS - mass1, list(S1), list(S2), f_ref


@pytest.mark.requires_data
def test_sign_against_nrsur7dq4(sxs_pair):
    """Flipped SXS polarizations match NRSur7dq4 in sign; unflipped ones do not.

    Time alignment only.  The model check is sensitive to the model's own
    f_ref / phi_ref semantics, so the tolerance is loose: it only has to tell
    +1 from -1.
    """
    import lal
    import lalsimulation as ls

    flipped, raw = sxs_pair
    dt = 1.0 / 8192
    mass1, mass2, S1, S2, f_ref = _lal_frame_parameters(flipped)
    t_ref = float(flipped.metadata["reference_time"])
    phi_at_zero = flipped.get_obs_phi_ref_from_obs_coa_phase(0.0)

    for inclination, phi_ref in ((0.8, 1.0), (1.57, 2.5)):
        try:
            hp, hc = ls.SimInspiralChooseTDWaveform(
                mass1 * lal.MSUN_SI, mass2 * lal.MSUN_SI, *S1, *S2,
                DISTANCE * 1e6 * lal.PC_SI, inclination, phi_ref,
                0.0, 0.0, 0.0, dt, 0.0, f_ref, None, ls.NRSur7dq4,
            )  # fmt: skip
        except Exception as exc:  # noqa: BLE001 - model data not installed
            pytest.skip(f"NRSur7dq4 unavailable through LAL: {exc}")
        model = hp.data.data + 1j * hc.data.data

        overlaps = {}
        for name, wfm in (("flipped", flipped), ("raw", raw)):
            h = wfm.get_td_waveform(
                total_mass=TOTAL_MASS, distance=DISTANCE, inclination=inclination,
                coa_phase=phi_ref - phi_at_zero, delta_t_seconds=dt, t_relax=t_ref,
            )  # fmt: skip
            overlaps[name] = _overlap(np.asarray(h), model, max_shift=40)

        assert overlaps["flipped"].real > 0.9, (inclination, phi_ref, overlaps)
        assert overlaps["raw"].real < -0.9, (inclination, phi_ref, overlaps)


@pytest.mark.requires_data
def test_equivalence_with_lal_on_sxs_lvcnr_export(sxs_pair):
    """get_td_waveform (SXS backend) equals LAL's NR routine on the LVCNR export.

    The export copies the modes unchanged, so LAL's output is compared with
    -h.  Only modes with ell <= 4 are kept on the nrcats side, matching the
    export.  Time alignment only.
    """
    import h5py
    import lal
    import lalsimulation as ls

    if not os.path.isfile(LVCNR_0792):
        pytest.skip(f"LVCNR export not found: {LVCNR_0792}")

    flipped, _ = sxs_pair
    wfm = flipped.copy()
    wfm.ndarray[:, wfm.index(4, 4) + 1 :] = 0.0

    with h5py.File(LVCNR_0792) as f:
        m1, m2 = float(f.attrs["mass1"]), float(f.attrs["mass2"])
        t_start = float(f.attrs["NR_start_time"])
        f_lower = float(f.attrs["f_lower_at_1MSUN"]) / TOTAL_MASS * 1.001
    mass1 = TOTAL_MASS * m1 / (m1 + m2)
    mass2 = TOTAL_MASS - mass1
    spins = ls.SimInspiralNRWaveformGetSpinsFromHDF5File(-1, TOTAL_MASS, LVCNR_0792)
    dt = 1.0 / 4096

    for inclination in (0.3, 1.6, 2.5):
        for coa_phase in (0.4, 3.5):
            h = wfm.get_td_waveform(
                total_mass=TOTAL_MASS, distance=DISTANCE, inclination=inclination,
                coa_phase=coa_phase, delta_t_seconds=dt, t_relax=t_start,
            )  # fmt: skip
            phi_ref = wfm.get_obs_phi_ref_from_obs_coa_phase(coa_phase)
            hp, hc = ls.SimInspiralNRWaveformGetHplusHcross(
                phi_ref, inclination, dt, mass1 * lal.MSUN_SI, mass2 * lal.MSUN_SI,
                DISTANCE * 1e6 * lal.PC_SI, f_lower, -1, *spins, LVCNR_0792, None,
            )  # fmt: skip
            c = _overlap(np.asarray(h), -(hp.data.data + 1j * hc.data.data))
            assert c.real > 0.995, (inclination, coa_phase, c)
            assert abs(np.angle(c)) < 0.1, (inclination, coa_phase, c)
