"""Catalog-level frame handling: get(frame=...), get_reference_metadata, get_parameters(frame=...)."""

import os

import numpy as np
import pytest

from nrcats import frames

MAYA_CACHE = os.path.expanduser("~/.cache/MAYA/data")
HAVE = {s: os.path.exists(os.path.join(MAYA_CACHE, f"{s}.h5")) for s in ("GT0001", "MAYA1060", "MAYA1070", "GT0587")}
# Precessing MAYA runs whose LVCNR dynamics (LNhat, positions, spins) and waveform modes
# disagree by a reflection through the orbital plane (measured October 2026; under study).
MAYA_HANDEDNESS_SUSPECT = {"GT0001", "MAYA1060", "MAYA1070"}


@pytest.fixture(scope="module")
def maya():
    from nrcats import MayaCatalog

    try:
        return MayaCatalog.load(download=False)
    except Exception as exc:  # pragma: no cover - depends on the local cache
        pytest.skip(f"MAYA catalog not available: {exc}")


def _strain(wfm, inc, coa):
    return np.asarray(wfm.get_td_waveform(total_mass=60.0, distance=100.0, inclination=inc,
                                          coa_phase=coa, delta_t_seconds=1 / 4096))


def _overlap(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    return np.vdot(b, a) / (np.linalg.norm(a) * np.linalg.norm(b))


def _sxs_or_skip():
    from nrcats import SXSCatalog

    try:
        cat = SXSCatalog.load(download=False)
        cat.get("SXS:BBH:0792", download=False, frame="simulation")
    except Exception as exc:  # pragma: no cover - depends on the local cache
        pytest.skip(f"SXS:BBH:0792 not available offline: {exc}")
    return cat


@pytest.mark.requires_data
@pytest.mark.parametrize("sim", [s for s in ("GT0001", "MAYA1060") if HAVE[s]] or [pytest.param("none", marks=pytest.mark.skip(reason="no cached MAYA files"))])
def test_get_records_the_lal_frame(maya, sim):
    wfm = maya.get(sim)
    fr = frames.frame_of(wfm)
    assert fr.name == "lal" and fr.method == "catalog"
    assert fr.t_ref == pytest.approx(wfm.t_ref_nr)
    raw = maya.get(sim, frame="simulation")
    assert frames.frame_of(raw).name == "simulation"


@pytest.mark.requires_data
def test_get_returns_modes_in_the_lal_frame():
    """In the LAL frame the radiation axis of the modes at t_ref is +z and the
    (2,2) phase puts n_hat on +x (precessing SXS run, trusted conventions)."""
    from nrcats.frame_waveform import waveform_frame

    cat = _sxs_or_skip()
    wfm = cat.get("SXS:BBH:0792", download=False)
    fr = frames.frame_of(wfm)
    assert fr.name == "lal"
    derived = waveform_frame(wfm, fr.t_ref, strip_memory=True,
                             memory_integration_start=wfm.sim_metadata["memory_integration_start_time"])
    assert np.degrees(np.arccos(np.clip(derived.L_hat[2], -1, 1))) < 2.0
    assert abs(np.arctan2(derived.n_hat[1], derived.n_hat[0])) < 0.3


@pytest.mark.requires_data
@pytest.mark.parametrize("sim", [s for s in ("GT0001", "MAYA1060") if HAVE[s]] or [pytest.param("none", marks=pytest.mark.skip(reason="no cached MAYA files"))])
@pytest.mark.parametrize("inc,coa", [(0.3, 0.4), (1.2, 2.0), (2.4, 5.0)])
def test_lal_frame_and_simulation_frame_give_the_same_strain(maya, sim, inc, coa):
    """Rotating at load and evaluating at (iota, pi/2 - phi_ref) equals the LVCNR route
    that rotates inside get_angles (which reproduces LALSimulation)."""
    h_lal = _strain(maya.get(sim), inc, coa)
    h_sim = _strain(maya.get(sim, frame="simulation"), inc, coa)
    ov = _overlap(h_lal, h_sim)
    assert abs(ov) > 1 - 1e-8
    assert abs(np.angle(ov)) < 1e-5


@pytest.mark.requires_data
@pytest.mark.skipif(not HAVE["MAYA1060"], reason="cached MAYA1060 (Format 2) not available")
def test_t_ref_moves_the_frame_on_format_2_and_is_refused_on_format_1(maya):
    wfm0 = maya.get("MAYA1060")
    t1 = frames.frame_of(wfm0).t_ref + 300.0
    wfm1 = maya.get("MAYA1060", t_ref=t1)
    f0, f1 = frames.frame_of(wfm0), frames.frame_of(wfm1)
    assert f1.t_ref == pytest.approx(t1)
    # a precessing run: the orbital plane moves between the two epochs
    assert np.degrees(np.arccos(np.clip(f0.L_hat @ f1.L_hat, -1, 1))) > 0.01
    if HAVE["GT0001"]:
        with pytest.raises(ValueError):
            maya.get("GT0001", t_ref=frames.frame_of(maya.get("GT0001")).t_ref + 50.0)


@pytest.mark.requires_data
@pytest.mark.parametrize("sim", [s for s in ("GT0001", "MAYA1060", "MAYA1070") if HAVE[s]] or [pytest.param("none", marks=pytest.mark.skip(reason="no cached MAYA files"))])
def test_reference_metadata_matches_lalsimulation(maya, sim):
    lalsim = pytest.importorskip("lalsimulation")
    md = maya.get_reference_metadata(sim)
    s = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(-1, 60.0, os.path.join(MAYA_CACHE, f"{sim}.h5"))
    np.testing.assert_allclose(md["chi1"], s[:3], atol=1e-6)
    np.testing.assert_allclose(md["chi2"], s[3:], atol=1e-6)
    p = maya.get_parameters(sim, total_mass=60.0, frame="lal")
    np.testing.assert_allclose([p["spin1x"], p["spin1y"], p["spin1z"]], s[:3], atol=1e-6)
    assert p["frame"] == "lal" and p["f_ref"] > 0


@pytest.mark.requires_data
@pytest.mark.skipif(not HAVE["MAYA1060"], reason="cached MAYA1060 not available")
@pytest.mark.xfail(reason="MAYA precessing files: dynamics and waveform differ by a reflection "
                          "through the orbital plane (see docs/frame_conventions.md)", strict=False)
def test_waveform_frame_agrees_with_the_catalog_frame(maya):
    from nrcats.frame_waveform import frame_consistency

    raw = maya.get("MAYA1060", frame="simulation")
    t_ref = raw.t_ref_nr + 200.0                          # past the junk
    fa = frames.catalog_frame(raw, t_ref=t_ref)
    wf = maya.get("MAYA1060", frame="waveform", t_ref=t_ref)
    c = frame_consistency(fa, frames.frame_of(wf))
    assert c["l_hat_angle_deg"] < 5.0


@pytest.mark.requires_data
def test_sxs_catalog_and_waveform_frames_agree():
    """On a precessing SXS run (trusted conventions) the metadata triad and the waveform-derived
    triad at reference_time agree within the physical floor of the design note."""
    from nrcats.frame_waveform import frame_consistency

    cat = _sxs_or_skip()
    raw = cat.get("SXS:BBH:0792", download=False, frame="simulation")
    fa = frames.catalog_frame(raw)
    fb = frames.frame_of(cat.get("SXS:BBH:0792", download=False, frame="waveform"))
    c = frame_consistency(fa, fb)
    assert c["l_hat_angle_deg"] < 2.0
    assert abs(c["phase_offset_rad"]) < 0.3


@pytest.mark.requires_data
@pytest.mark.skipif(not HAVE["MAYA1060"], reason="cached MAYA1060 (Format 2) not available")
def test_reference_metadata_at_another_epoch_matches_lalsimulation(maya):
    """Spins at f_ref = 40 Hz for 60 Msun, i.e. at a t_ref other than the file's default."""
    import h5py

    lalsim = pytest.importorskip("lalsimulation")
    from nrcats.lvc import get_lvcnr_ref_time

    path = os.path.join(MAYA_CACHE, "MAYA1060.h5")
    with h5py.File(path, "r") as h5:
        t_ref = get_lvcnr_ref_time(h5, 40.0 * 60.0)
    assert t_ref is not None and abs(t_ref - maya.get("MAYA1060").t_ref_nr) > 1.0
    md = maya.get_reference_metadata("MAYA1060", t_ref=t_ref)
    s = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(40.0, 60.0, path)
    np.testing.assert_allclose(md["chi1"], s[:3], atol=1e-5)
    np.testing.assert_allclose(md["chi2"], s[3:], atol=1e-5)


@pytest.mark.requires_data
@pytest.mark.skipif(not HAVE["MAYA1060"], reason="cached MAYA1060 not available")
def test_simulation_frame_with_t_ref_is_refused(maya):
    with pytest.raises(ValueError):
        maya.get("MAYA1060", frame="simulation", t_ref=maya.get("MAYA1060").t_ref_nr + 100.0)


@pytest.mark.requires_data
@pytest.mark.skipif(not HAVE["MAYA1060"], reason="cached MAYA1060 not available")
def test_rotating_after_loading_marks_the_frame_unknown(maya):
    import quaternionic

    wfm = maya.get("MAYA1060")
    rotated = wfm.rotate(quaternionic.array.from_axis_angle([0.0, 0.3, 0.0]))
    assert frames.frame_of(rotated).name == "unknown"
    with pytest.raises(ValueError):
        rotated.get_angles(0.5, 0.2)
    with pytest.raises(ValueError):
        frames.put_in_frame(rotated, "lal")
    # modes already in the LAL frame cannot be put in a frame a second time
    with pytest.raises(ValueError):
        frames.put_in_frame(wfm, "lal")


@pytest.mark.requires_data
@pytest.mark.skipif(not HAVE["MAYA1060"], reason="cached MAYA1060 not available")
def test_slicing_keeps_the_lal_and_simulation_strain_equal(maya):
    lal_w, sim_w = maya.get("MAYA1060"), maya.get("MAYA1060", frame="simulation")
    # drop samples on both sides of t_ref, keeping t_ref inside
    i_ref = int(np.argmin(np.abs(np.asarray(lal_w.t) - lal_w.t_ref_nr)))
    i0, i1 = i_ref // 2, len(lal_w.t) - 200
    h_lal = _strain(lal_w[i0:i1], 1.0, 0.7)
    h_sim = _strain(sim_w[i0:i1], 1.0, 0.7)
    ov = _overlap(h_lal, h_sim)
    assert abs(ov) > 1 - 1e-8
    assert abs(np.angle(ov)) < 1e-5


@pytest.mark.requires_data
@pytest.mark.skipif(not HAVE["MAYA1060"], reason="cached MAYA1060 not available")
def test_f_ref_matching_the_frame_is_accepted(maya):
    import lal

    wfm = maya.get("MAYA1060")
    fr = frames.frame_of(wfm)
    f_ref = fr.f_ref_dimless / lal.MTSUN_SI
    assert wfm.get_angles(0.5, 0.2, f_ref=f_ref) == wfm.get_angles(0.5, 0.2)
    with pytest.raises(ValueError):
        wfm.get_angles(0.5, 0.2, f_ref=1.2 * f_ref)


def test_near_equal_masses_are_accepted_within_the_tolerance():
    from types import SimpleNamespace

    def fake(m1, m2):
        fr = frames.frame_from_vectors([0, 0, 1.0], [0, 1.0, 0], t_ref=0.0)
        md = {"reference_mass1": m1, "reference_mass2": m2,
              "reference_dimensionless_spin1": [0, 0, 0.1], "reference_dimensionless_spin2": [0, 0, 0]}
        return SimpleNamespace(sim_metadata=md, t_ref_nr=0.0, _metadata={"reference_frame": fr},
                               _has_lvcnr_file=lambda: False), fr

    w, fr = fake(0.4998, 0.5002)          # numerically equal-mass run, labels swapped by noise
    assert frames.reference_metadata(w, fr)["q"] == pytest.approx(0.4998 / 0.5002)
    w, fr = fake(0.45, 0.55)
    with pytest.raises(ValueError):
        frames.reference_metadata(w, fr)
