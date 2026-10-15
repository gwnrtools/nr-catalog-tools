"""Tests for nrcats.frame_waveform: the LAL frame read from the waveform, and frame_consistency.

The truth is NRSur7dq4's own coorbital frame, built from its returned dynamics
(``q_copr`` and ``orbphase``): at any time the surrogate's frame has L_hat along
the coprecessing z axis and n_hat (body 2 to body 1, body 1 heavier) at azimuth
``orbphase`` about it.  The modes are turned into an arbitrary orientation with
a known rotor, using the convention verified in test_frames.py.
"""

import numpy as np
import pytest
import quaternionic

from nrcats import frame_waveform as fw
from nrcats import frames

sxs = pytest.importorskip("sxs")
gwsurrogate = pytest.importorskip("gwsurrogate")

# An arbitrary orientation of the simulation's coordinates.
R0 = quaternionic.array.from_axis_angle(np.array([0.9, -1.7, 0.4]))

CASES = {
    "q2_aligned": (2.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    "q3_aligned": (3.0, [0.0, 0.0, 0.4], [0.0, 0.0, -0.3]),
    "q2_precessing": (2.0, [0.5, 0.2, 0.3], [-0.2, 0.4, 0.1]),
    "q1_nonspinning": (1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
}


def _rotate_vector(R, v):
    return (R * quaternionic.array([0, *v]) * R.conjugate()).vector


@pytest.fixture(scope="module")
def surrogate():
    try:
        return gwsurrogate.LoadSurrogate("NRSur7dq4")
    except Exception as exc:  # data file not downloaded
        pytest.skip(f"NRSur7dq4 not available: {exc}")


@pytest.fixture(scope="module")
def waveforms(surrogate):
    """Per case: (times, modes rotated by R0, coorbital rotor at each time in the rotated coordinates)."""
    out = {}
    lm = [(ell, m) for ell in range(2, 5) for m in range(-ell, ell + 1)]
    for name, (q, chiA, chiB) in CASES.items():
        t, h, dyn = surrogate(q, chiA, chiB, dt=1.0, f_low=0, f_ref=0.01,
                              precessing_opts={"return_dynamics": True})
        w = sxs.WaveformModes(np.column_stack([h[k] for k in lm]), time=t, time_axis=0, modes_axis=1,
                              ell_min=2, ell_max=4, spin_weight=-2, data_type="h")
        # rotate(R0^-1) moves a feature along v to R0 v R0^-1 (see test_frames.py).
        w_sim = w.rotate(R0.conjugate())
        rotors = R0 * quaternionic.array(dyn["q_copr"].T) * quaternionic.array.from_axis_angle(
            np.column_stack([np.zeros_like(t), np.zeros_like(t), dyn["orbphase"]]))
        out[name] = (np.asarray(t), w_sim, rotors)
    return out


def _truth(waveforms, name, t_ref):
    t, w_sim, rotors = waveforms[name]
    j = int(np.argmin(np.abs(t - t_ref)))
    R = rotors[j]
    truth = frames.frame_from_vectors(_rotate_vector(R, [0, 0, 1]), _rotate_vector(R, [1, 0, 0]), t_ref=t[j])
    return t[j], w_sim, truth


@pytest.mark.parametrize("name,l_tol_deg,phase_tol", [
    ("q2_aligned", 1.0, 0.1),
    ("q3_aligned", 1.0, 0.1),
    # Measured: under 0.2 deg and 0.01 rad; the truth axis is NRSur7dq4's
    # unsmoothed coprecessing axis, so the one-orbit average costs a little.
    ("q2_precessing", 1.0, 0.1),
])
@pytest.mark.parametrize("t_ref", [-3000.0, -1500.0, -600.0])
def test_recovers_the_known_frame(waveforms, name, l_tol_deg, phase_tol, t_ref):
    t_ref, w_sim, truth = _truth(waveforms, name, t_ref)
    frame = fw.waveform_frame(w_sim, t_ref)
    assert frame.name == "waveform" and frame.method == "waveform"
    assert frame.t_ref == t_ref
    c = fw.frame_consistency(truth, frame)
    assert c["l_hat_angle_deg"] < l_tol_deg
    assert abs(c["phase_offset_rad"]) < phase_tol          # also checks the pi branch
    assert frame.extra["pi_ambiguous"] is False
    assert frame.extra["pi_resolved_by"] == "h33"
    assert c["consistent"]


def test_reference_frequency_matches_the_surrogate(waveforms, surrogate):
    """NRSur7dq4 puts its reference epoch where the coprecessing (2,2) frequency is f_ref."""
    t, w_sim, _ = waveforms["q2_aligned"]
    # the surrogate's frame is the coorbital frame at the reference epoch, so
    # its orbital phase there is zero; find that epoch from the dynamics
    _, _, dyn = surrogate(2.0, [0, 0, 0], [0, 0, 0], dt=1.0, f_low=0, f_ref=0.01,
                          precessing_opts={"return_dynamics": True})
    t_ref = t[int(np.argmin(np.abs(dyn["orbphase"])))]
    frame = fw.waveform_frame(w_sim, t_ref)
    assert frame.f_ref_dimless == pytest.approx(0.01, rel=0.02)


def test_equal_mass_nonspinning_is_pi_ambiguous(waveforms):
    t_ref, w_sim, truth = _truth(waveforms, "q1_nonspinning", -1500.0)
    frame = fw.waveform_frame(w_sim, t_ref)
    assert frame.extra["pi_ambiguous"] is True
    assert frame.extra["pi_resolved_by"] is None
    c = fw.frame_consistency(truth, frame)
    assert c["pi_ambiguous"] is True
    assert c["l_hat_angle_deg"] < 1.0
    # n_hat is right up to pi
    assert min(abs(c["phase_offset_rad"]), np.pi - abs(c["phase_offset_rad"])) < 0.1


def test_h21_resolves_the_branch_without_h33(waveforms):
    """With only l = 2 the fallback to h21 still picks body 1 as the heavier."""
    t_ref, w_sim, truth = _truth(waveforms, "q3_aligned", -1500.0)
    frame = fw.waveform_frame(w_sim, t_ref, ell_max=2)
    assert frame.extra["pi_resolved_by"] == "h21"
    c = fw.frame_consistency(truth, frame)
    assert abs(c["phase_offset_rad"]) < 0.1


def test_strip_memory_undoes_added_memory(waveforms):
    from sxs.waveforms.memory import add_memory

    t_ref, w_sim, _ = _truth(waveforms, "q2_precessing", -600.0)
    clean = fw.waveform_frame(w_sim, t_ref)
    w_mem = add_memory(w_sim)
    with_mem = fw.waveform_frame(w_mem, t_ref)
    stripped = fw.waveform_frame(w_mem, t_ref, strip_memory=True)
    assert fw.frame_consistency(clean, with_mem)["l_hat_angle_deg"] > 1e-2   # memory tilts the axis
    c = fw.frame_consistency(clean, stripped)
    assert c["l_hat_angle_deg"] < 1e-3
    assert abs(c["phase_offset_rad"]) < 1e-4


def test_strip_memory_needs_sxs_modes():
    with pytest.raises(TypeError):
        fw.waveform_frame(object(), 0.0, strip_memory=True)


def test_t_ref_outside_the_waveform_raises(waveforms):
    t, w_sim, _ = waveforms["q2_aligned"]
    with pytest.raises(ValueError):
        fw.waveform_frame(w_sim, t[-1] + 10.0)


# frame_consistency on constructed frames ------------------------------------

def _tilted_frame(**extra):
    L = _rotate_vector(R0, [0, 0, 1])
    n = _rotate_vector(R0, [1, 0, 0])
    f = frames.frame_from_vectors(L, n, t_ref=0.0)
    f.extra.update(extra)
    return f


def test_consistency_of_identical_frames_is_zero():
    f = _tilted_frame()
    c = fw.frame_consistency(f, f)
    assert c["l_hat_angle_deg"] == pytest.approx(0.0, abs=1e-6)
    assert c["phase_offset_rad"] == pytest.approx(0.0, abs=1e-12)
    assert c["consistent"] and not c["pi_ambiguous"]


@pytest.mark.parametrize("angle", [0.05, -0.2, 1.3, -2.9, np.pi])
def test_consistency_measures_a_rotation_about_L(angle):
    a = _tilted_frame()
    n_b = np.cos(angle) * a.n_hat + np.sin(angle) * a.m_hat
    b = frames.frame_from_vectors(a.L_hat, n_b)
    c = fw.frame_consistency(a, b)
    assert c["l_hat_angle_deg"] == pytest.approx(0.0, abs=1e-6)
    assert c["phase_offset_rad"] == pytest.approx(fw._wrap(angle), abs=1e-10)
    assert c["phase_ok"] == (abs(angle) <= fw.PHASE_TOL_RAD)


def test_consistency_measures_a_tilt_and_carries_the_ambiguity_flag():
    a = _tilted_frame()
    tilt = np.radians(3.0)
    # tilt L towards n by 3 degrees; n follows so the triad stays orthonormal
    L_b = np.cos(tilt) * a.L_hat + np.sin(tilt) * a.n_hat
    n_b = np.cos(tilt) * a.n_hat - np.sin(tilt) * a.L_hat
    b = frames.frame_from_vectors(L_b, n_b)
    b.extra["pi_ambiguous"] = True
    c = fw.frame_consistency(a, b)
    assert c["l_hat_angle_deg"] == pytest.approx(3.0, abs=1e-8)
    assert c["phase_offset_rad"] == pytest.approx(0.0, abs=1e-10)
    assert not c["l_hat_ok"] and c["phase_ok"] and not c["consistent"]
    assert c["pi_ambiguous"]
