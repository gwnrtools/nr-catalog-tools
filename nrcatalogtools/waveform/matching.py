"""Standalone waveform matching and rotation helpers.

These are module-level functions (not bound to WaveformModes) so they can
be unit-tested and used independently of the class.
"""

import numpy as np
import spherical
from sxs import TimeSeries as sxs_TimeSeries


def apply_wigner_rotation_to_mode_dict(mode_dict, R, ell_max=4):
    """Apply a Wigner rotation to a dictionary of spherical harmonic modes.

    This is useful for rotating the output of ``gwsurrogate`` or
    ``pycbc.waveform.get_td_waveform_modes`` (which return dicts) into the
    NR source frame before computing mode-by-mode matches.

    The rotation is applied mode-by-mode via Wigner D-matrices:

        h'_{ℓm}(t) = Σ_{m'} D^{(ℓ)}_{m'm}(R) h_{ℓm'}(t)

    where R ∈ SO(3) is a unit quaternion and D^{(ℓ)} is the (2ℓ+1)×(2ℓ+1)
    Wigner D-matrix for angular momentum ℓ.

    Parameters
    ----------
    mode_dict : dict
        Keys are ``(l, m)`` integer tuples; values are complex
        ``pycbc.types.TimeSeries`` objects (or 1-D numpy arrays of matching
        length).
    R : quaternionic.array
        Unit quaternion representing the rotation.
    ell_max : int, optional
        Maximum ℓ to include (default 4).

    Returns
    -------
    dict
        Rotated mode dictionary with the same ``(l, m)`` keys.
    """
    wigner = spherical.Wigner(ell_max)

    by_ell = {}
    for (ell, m), val in mode_dict.items():
        if ell > ell_max:
            continue
        by_ell.setdefault(ell, {})[m] = val

    rotated = {}
    for ell, m_dict in by_ell.items():
        m_vals = list(range(-ell, ell + 1))
        first = next(iter(m_dict.values()))
        is_timeseries = hasattr(first, "delta_t")
        n = len(first)

        block = np.zeros((n, 2 * ell + 1), dtype=complex)
        for i, mv in enumerate(m_vals):
            if mv in m_dict:
                block[:, i] = np.asarray(m_dict[mv])

        D = wigner.D(R, ell)
        rotated_block = block @ D.T

        for i, mv in enumerate(m_vals):
            if is_timeseries:
                rotated[(ell, mv)] = type(first)(
                    rotated_block[:, i], delta_t=first.delta_t, epoch=first.start_time
                )
            else:
                rotated[(ell, mv)] = rotated_block[:, i]

    return rotated


def interpolate_in_amp_phase(obj, new_time, k=3, kind=None):
    """Interpolate in amplitude and phase using a variety of methods.

    Parameters
    ----------
    obj : sxs.TimeSeries
        Complex waveform time series.
    new_time : array_like
        New time axis to interpolate onto.
    k : int, optional
        Spline order for ``InterpolatedUnivariateSpline`` (default 3).
    kind : str, optional
        Alternative interpolation: ``'linear'``, ``'quadratic'``, ``'cubic'``,
        or ``'CubicSpline'``.  When specified, ``k`` is ignored.

    Returns
    -------
    sxs.TimeSeries
        Interpolated complex waveform on ``new_time``.
    """
    from waveformtools.waveformtools import interp_resam_wfs

    resam_data = interp_resam_wfs(
        wavf_data=np.array(obj),
        old_taxis=obj.time,
        new_taxis=new_time,
        k=k,
        kind=kind,
    )

    resam_data = sxs_TimeSeries(resam_data, new_time)

    metadata = obj._metadata.copy()
    metadata["time"] = new_time
    metadata["time_axis"] = obj.time_axis

    return type(obj)(resam_data, **metadata)
