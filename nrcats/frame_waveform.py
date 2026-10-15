"""The LAL source frame read from the waveform alone ("method B"), and its comparison with another.

:func:`waveform_frame` builds the LAL frame at an epoch ``t_ref`` from the modes,
with no metadata:

* ``L_hat`` is the dominant l = 2 radiation axis (the z axis of the waveform's
  coprecessing frame), with its sign fixed by the sense of rotation
  (``reference_frame._coprecessing_axis``), averaged over about one orbit
  centred on ``t_ref``.
* ``n_hat`` comes from the orbital phase.  The modes are turned so that
  ``L_hat`` is ``z`` (with an arbitrary ``x``), and the (2,2) mode gives the
  orbital phase ``Phi`` modulo ``pi`` through ``h22 ~ A exp(-2 i Phi)``.  The
  odd-m modes, which change sign under ``Phi -> Phi + pi``, pick the branch for
  which body 1 is the heavier hole: ``h33``, or ``h21`` when the data hold no
  ``h33``.  Where the mode is too weak to decide (nearly equal masses; equal
  masses with symmetric spins is the exact case) the (2,2) branch is kept and
  the frame records ``extra["pi_ambiguous"] = True``.

:func:`frame_consistency` compares two frames at the same epoch, typically the
catalog's frame (method A) with this one: the angle between their ``L_hat``,
the orbital-phase offset between their ``n_hat`` about ``L_hat``, and flags
against thresholds.  The two need not agree exactly even when both are right;
see "Two orientation methods" in ``docs/frame_conventions.md``.

SXS strain carries a constant (2,0) memory offset.  It tilts an axis built from
the l = 2 modes, so it has to be removed first: pass ``strip_memory=True`` (for
``sxs.WaveformModes`` input; this calls ``sxs.waveforms.memory.remove_memory``),
or pass modes from which the memory has already been removed.
"""

from __future__ import annotations

import numpy as np

from . import frames
from .reference_frame import _coprecessing_axis, _lm_list, _mode_block, gw_frequency

__all__ = [
    "PHASE_CONSTANTS",
    "ODD_MODE_THRESHOLD",
    "L_HAT_TOL_DEG",
    "PHASE_TOL_RAD",
    "waveform_frame",
    "frame_consistency",
]

# Phase constants c_lm in h_lm = |h_lm| exp(i (c_lm - m Phi)) for a binary whose
# frame at the epoch is the LAL frame (L_hat along z, n_hat along x from body 2
# to body 1, body 1 heavier, Phi = 0).
#
# Calibrated on NRSur7dq4, whose frame at its reference epoch is that frame: we
# evaluated it with return_dynamics=True (dt = 1 M, f_ref = 0.01 cycles/M),
# rotated the inertial modes into the coorbital frame given by q_copr and
# orbphase at t = -3000, -2000, -1000 and -300 M, and read the mode phases, for
# q = 1.2, 1.5, 2, 3 non-spinning, (q, chiA_z, chiB_z) = (4, 0.5, -0.3) and
# (2, 0.8, 0.8), and two precessing cases (q = 2, chiA = [0.5, 0.2, 0.3],
# chiB = [-0.2, 0.4, 0.1]; q = 3, chiA = [0.7, 0, 0], chiB = [0, 0.5, 0]).
# Measured: c22 = 0 to 1e-3 for aligned spins (within 0.05 for precessing),
# c33 between -1.69 and -1.60, c21 between +1.48 and +1.56.  These are the PN
# values -pi/2 and +pi/2 (h33 ~ -i delta, h21 ~ +i delta, with delta = (m1 -
# m2)/M > 0), so those are used.  Only the sign of the projection onto exp(i
# c_lm) is used for the odd modes, so errors of a few tenths of a radian do not
# matter.  Caveat: for q = 1 with antisymmetric aligned spins (0.6, -0.6) the
# spin term reverses h21 (c21 = -1.6), so h21 is only a fallback for data
# without h33.
PHASE_CONSTANTS = {(2, 2): 0.0, (3, 3): -np.pi / 2, (2, 1): np.pi / 2}

# |h_lm| / |h22| (projected onto the expected phase) below which an odd mode
# cannot decide the pi branch.  NRSur7dq4: |h33/h22| is 1e-4 at q = 1 and 0.02
# at q = 1.2 (roughly 0.22 delta in the inspiral), so this resolves q above
# about 1.03.
ODD_MODE_THRESHOLD = 3e-3

# Default thresholds for frame_consistency, from docs/frame_conventions.md.
L_HAT_TOL_DEG = 2.0
PHASE_TOL_RAD = 0.3


def _wrap(phase):
    """Wrap to (-pi, pi]."""
    out = -((-np.asarray(phase, dtype=float) + np.pi) % (2 * np.pi) - np.pi)
    return float(out) if np.ndim(out) == 0 else out


def _window_mean(t, x, t_ref, half_width):
    """Mean of ``x`` over ``|t - t_ref| <= half_width`` (trapezoid in t)."""
    sel = np.abs(t - t_ref) <= half_width
    if np.count_nonzero(sel) < 2:
        j = int(np.argmin(np.abs(t - t_ref)))
        return np.asarray(x)[j]
    ts, xs = t[sel], np.asarray(x)[sel]
    return np.trapezoid(xs, ts, axis=0) / (ts[-1] - ts[0])


def _as_sxs(data, t, ell_max):
    import sxs

    return sxs.WaveformModes(
        np.asarray(data, dtype=complex), time=np.asarray(t, dtype=float), time_axis=0,
        modes_axis=1, ell_min=2, ell_max=ell_max, spin_weight=-2, data_type="h",
    )


def _strip_memory(wfm, integration_start_time=None):
    import sxs
    from sxs.waveforms.memory import remove_memory

    if not isinstance(wfm, sxs.WaveformModes):
        raise TypeError(
            "strip_memory=True needs an sxs.WaveformModes; remove the memory "
            "before calling waveform_frame for other inputs."
        )
    return remove_memory(wfm, integration_start_time=integration_start_time)


def waveform_frame(
    wfm,
    t_ref,
    ell_max=4,
    smooth_window=None,
    strip_memory=False,
    memory_integration_start=None,
    odd_mode_threshold=ODD_MODE_THRESHOLD,
):
    """LAL source frame at ``t_ref`` derived from the waveform modes ("method B").

    Parameters
    ----------
    wfm : sxs.WaveformModes (or nrcats WaveformModes)
        Strain modes in the simulation's inertial frame, with the LAL/SXS sign
        convention (``h22 ~ exp(-2 i Phi)`` for an orbit about ``+z``) and body 1
        heavier.  SXS strain carries a constant (2,0) memory offset that tilts
        the l = 2 radiation axis: pass ``strip_memory=True`` or modes with the
        memory already removed.
    t_ref : float
        Epoch on the waveform's own time axis (M).
    ell_max : int
        Highest l used for the radiation axis.
    smooth_window : float, optional
        Width (M) of the window over which ``L_hat`` and the odd-mode
        projections are averaged.  Defaults to one orbital period at ``t_ref``
        (``2 / f22``).  Pass 0 for no averaging.
    strip_memory : bool
        Remove the electric null memory first with
        ``sxs.waveforms.memory.remove_memory``; ``memory_integration_start`` is
        passed on as its ``integration_start_time`` (for SXS, the sidecar's
        ``integration_start_time``).
    odd_mode_threshold : float
        Smallest projected ``|h_lm / h22|`` of an odd-m mode that is trusted to
        resolve the ``pi`` ambiguity of ``n_hat``.

    Returns
    -------
    frames.ReferenceFrame
        ``name="waveform"``, ``method="waveform"``, with ``L_hat`` and ``n_hat``
        in the input's coordinates, ``t_ref`` and ``f_ref_dimless`` (the (2,2)
        frequency at ``t_ref`` in the coprecessing frame, cycles/M).  ``extra``
        holds ``pi_ambiguous``, ``pi_resolved_by`` (``"h33"``, ``"h21"`` or
        ``None``), the projected odd-mode amplitudes ``odd_h33`` and ``odd_h21``
        (positive when the chosen branch agrees with them),
        ``coprecessing_branch_flipped`` and ``smooth_window``.
    """
    if strip_memory:
        wfm = _strip_memory(wfm, memory_integration_start)

    data, lmin, lmax = _mode_block(wfm, ell_max)
    t = np.asarray(wfm.time, dtype=float)
    t_ref = float(t_ref)
    if not (t[0] <= t_ref <= t[-1]):
        raise ValueError(f"t_ref = {t_ref} is outside the waveform's time span [{t[0]}, {t[-1]}].")

    # Radiation axis and the coprecessing (2,2) frequency.
    tw, axis, flipped, w_copr = _coprecessing_axis(t, data, lmin, lmax)
    lm = _lm_list(lmin, lmax)
    h22_copr = np.asarray(w_copr.data)[:, lm.index((2, 2))]
    f22 = gw_frequency(tw, h22_copr)
    j = int(np.argmin(np.abs(tw - t_ref)))
    if smooth_window is None:
        # One orbit, from the frequency averaged over a few samples around t_ref.
        f_loc = float(np.median(f22[max(j - 5, 0): j + 6]))
        smooth_window = 2.0 / f_loc if f_loc > 0 else 0.0
    half = 0.5 * float(smooth_window)
    f_ref_dimless = float(_window_mean(tw, f22, t_ref, half)) if half > 0 else float(np.interp(t_ref, tw, f22))

    L = _window_mean(tw, axis, t_ref, half) if half > 0 else axis[j]
    L = np.asarray(L, dtype=float)
    L = L / np.linalg.norm(L)

    # Temporary frame with L along z and any x; read the orbital phase there.
    seed = np.array([1.0, 0.0, 0.0]) if abs(L[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_tmp = seed - np.dot(seed, L) * L
    x_tmp /= np.linalg.norm(x_tmp)
    y_tmp = np.cross(L, x_tmp)
    tmp = frames.frame_from_vectors(L, x_tmp, method="waveform", name="waveform")

    # Only the samples near t_ref are needed; keep a few extra for interpolation.
    pad = max(half, 3.0 * float(np.median(np.diff(t))))
    sel = np.abs(t - t_ref) <= pad + 1e-12
    if np.count_nonzero(sel) < 2:
        sel[max(j - 1, 0): j + 2] = True
    ts = t[sel]
    w_tmp = frames.rotate_modes(_as_sxs(data[sel], ts, lmax), tmp)
    arr = np.asarray(w_tmp.data)
    h22 = arr[:, lm.index((2, 2))]
    phase22 = np.unwrap(np.angle(h22))
    Phi_t = -(phase22 - PHASE_CONSTANTS[(2, 2)]) / 2.0
    Phi = float(np.interp(t_ref, ts, Phi_t))

    # Odd-m projections: h_lm exp(i (m Phi - c_lm)) / |h22| is real and positive
    # on the branch where body 1 is the heavier, and flips sign under Phi -> Phi + pi.
    proj = {}
    for (ell, m) in ((3, 3), (2, 1)):
        if (ell, m) not in lm:
            proj[(ell, m)] = 0.0
            continue
        r = arr[:, lm.index((ell, m))] * np.exp(1j * (m * Phi_t - PHASE_CONSTANTS[(ell, m)])) / np.abs(h22)
        val = _window_mean(ts, r.real, t_ref, half) if half > 0 else float(np.interp(t_ref, ts, r.real))
        proj[(ell, m)] = float(val)

    # h21 is used only when h33 is not in the data.  Where h33 is present but
    # weak, the masses are nearly equal and h21 is dominated by its spin term,
    # which is not tied to the mass ordering: on NRSur7dq4 at q = 1 with
    # chiA = [0.5, 0.2, 0.3], chiB = [-0.2, 0.4, 0.1], h21/h22 reaches 0.01 and
    # its sign relative to n_hat changes as the spins precess.
    key = (3, 3) if (3, 3) in lm else (2, 1)
    resolved_by = None
    if key in lm and abs(proj[key]) >= odd_mode_threshold:
        resolved_by = "h33" if key == (3, 3) else "h21"
        if proj[key] < 0:
            Phi += np.pi
            proj = {k: -v for k, v in proj.items()}

    n = np.cos(Phi) * x_tmp + np.sin(Phi) * y_tmp
    frame = frames.frame_from_vectors(
        L, n, t_ref=t_ref, f_ref_dimless=f_ref_dimless, method="waveform", name="waveform",
        source="waveform modes: l=2 radiation axis and (2,2) orbital phase"
        + ("; memory removed" if strip_memory else ""),
    )
    frame.extra.update({
        "pi_ambiguous": resolved_by is None,
        "pi_resolved_by": resolved_by,
        "odd_h33": proj[(3, 3)],
        "odd_h21": proj[(2, 1)],
        "coprecessing_branch_flipped": bool(flipped),
        "smooth_window": float(smooth_window),
    })
    return frame


def frame_consistency(frame_a, frame_b, l_hat_tol_deg=L_HAT_TOL_DEG, phase_tol_rad=PHASE_TOL_RAD):
    """Compare two LAL frames given in the same simulation coordinates.

    Returns a dict with

    * ``l_hat_angle_deg``: angle between the two ``L_hat``;
    * ``phase_offset_rad``: azimuth of ``frame_b.n_hat`` about ``frame_a.L_hat``
      measured from ``frame_a.n_hat`` (both projected into the plane normal to
      ``frame_a.L_hat``), wrapped to (-pi, pi];
    * ``l_hat_ok``, ``phase_ok`` and ``consistent`` against the thresholds;
    * ``pi_ambiguous``: true if either frame carries that flag; a phase offset
      near ``pi`` is then expected rather than an error;
    * ``t_ref_a``, ``t_ref_b`` and the thresholds used.

    The frames should belong to the same epoch; the comparison does not
    check that.
    """
    La, na = np.asarray(frame_a.L_hat, float), np.asarray(frame_a.n_hat, float)
    Lb, nb = np.asarray(frame_b.L_hat, float), np.asarray(frame_b.n_hat, float)
    angle = float(np.degrees(np.arccos(np.clip(np.dot(La, Lb), -1.0, 1.0))))
    ma = np.cross(La, na)
    nb_perp = nb - np.dot(nb, La) * La
    offset = _wrap(np.arctan2(np.dot(nb_perp, ma), np.dot(nb_perp, na)))
    ambiguous = bool(frame_a.extra.get("pi_ambiguous", False) or frame_b.extra.get("pi_ambiguous", False))
    l_ok = angle <= l_hat_tol_deg
    p_ok = abs(offset) <= phase_tol_rad
    return {
        "l_hat_angle_deg": angle,
        "phase_offset_rad": offset,
        "l_hat_ok": bool(l_ok),
        "phase_ok": bool(p_ok),
        "consistent": bool(l_ok and p_ok),
        "pi_ambiguous": ambiguous,
        "t_ref_a": frame_a.t_ref,
        "t_ref_b": frame_b.t_ref,
        "l_hat_tol_deg": float(l_hat_tol_deg),
        "phase_tol_rad": float(phase_tol_rad),
    }
