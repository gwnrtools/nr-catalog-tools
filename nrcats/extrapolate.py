"""Extrapolate finite-radius :math:`\\Psi_4` to null infinity and integrate to strain.

The numerics are waveformtools': the areal radius of the extraction sphere
(``r_to_ra_conversion``), the perturbative extrapolation of Nakano et al.
(2015), Eq. 29 (``waveextract_to_inf_perturbative_twop5_order``), and the
fixed-frequency integration (``fixed_frequency_integrator``).  This module
puts them in the order and on the time axes nrcats needs; it does not depend
on any catalog.

The result lives on the retarded time :math:`u = t - r_*(R)`, with
:math:`R` the areal radius of the extraction sphere and
:math:`r_* = R + 2M \\ln(R/2M - 1)` the tortoise coordinate.

These operations need waveformtools >= 2026.10.15, which fixed the sign of
the FFI at negative frequencies and the second-order terms of the
extrapolation.
"""

import logging

import numpy as np
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)

# Fraction of |m| times the initial orbital frequency used as the FFI cutoff.
# Low enough to keep the start of each mode, high enough to suppress the
# drift that double integration amplifies at low frequency.
DEFAULT_FFI_CUTOFF_FRACTION = 0.75
# Length of the Planck taper applied to each end of r * Psi4 before the
# extrapolation and the FFI, in units of the total mass.  Finite-radius Psi4
# is close to zero at both ends (before the signal reaches the sphere and
# after the ringdown), so the taper removes only the residual step that the
# FFT would otherwise treat as a discontinuity.
DEFAULT_TAPER_DURATION = 20.0


def tortoise_radius(areal_radius, total_mass):
    """Schwarzschild tortoise coordinate :math:`r_* = R + 2M \\ln(R/2M - 1)`."""
    return areal_radius + 2 * total_mass * np.log(areal_radius / (2 * total_mass) - 1)


def resample_uniformly(time, data, delta_t=None):
    """Resample complex ``data`` onto a uniform grid over ``time``.

    Parameters
    ----------
    time : 1d array
           Increasing sample times.
    data : 1d complex array
           Values at ``time``.
    delta_t : float, optional
              Spacing of the new grid. Defaults to the median spacing of
              ``time``.

    Returns
    -------
    uniform_time, uniform_data : 1d arrays
    """
    time = np.asarray(time, dtype=float)
    if delta_t is None:
        delta_t = float(np.median(np.diff(time)))
    num_samples = int(np.floor((time[-1] - time[0]) / delta_t)) + 1
    uniform_time = time[0] + np.arange(num_samples) * delta_t
    if len(uniform_time) == len(time) and np.allclose(uniform_time, time, rtol=0, atol=1e-9 * delta_t):
        return uniform_time, np.asarray(data, dtype=complex)
    real_part = CubicSpline(time, np.real(data))(uniform_time)
    imag_part = CubicSpline(time, np.imag(data))(uniform_time)
    return uniform_time, real_part + 1j * imag_part


def planck_taper_window(time, taper_duration):
    """Planck-taper window rising over the first and falling over the last
    ``taper_duration`` of ``time`` (McKechan, Robinson & Sathyaprakash 2010)."""
    time = np.asarray(time, dtype=float)
    window = np.ones_like(time)
    if taper_duration is None or taper_duration <= 0:
        return window
    if 2 * taper_duration >= time[-1] - time[0]:
        raise ValueError(f"taper_duration {taper_duration} is too long for data of length {time[-1] - time[0]}.")
    for distance_from_edge in (time - time[0], time[-1] - time):
        in_taper = distance_from_edge < taper_duration
        fraction = np.clip(distance_from_edge[in_taper] / taper_duration, 1e-12, 1 - 1e-12)
        exponent = np.clip(1 / fraction - 1 / (1 - fraction), -700, 700)
        window[in_taper] *= 1 / (np.exp(exponent) + 1)
    return window


def retardation(extraction_radius, total_mass, background_spin=0.0):
    """Shift :math:`r_*(R)` from coordinate time at the extraction sphere to retarded time."""
    from waveformtools.extrapolate import r_to_ra_conversion

    return tortoise_radius(r_to_ra_conversion(extraction_radius, mass=total_mass, spin=background_spin), total_mass)


def extrapolate_psi4_to_strain(
    time,
    rpsi4_modes,
    extraction_radius,
    total_mass,
    initial_orbital_frequency,
    background_spin=0.0,
    ffi_cutoff_fraction=DEFAULT_FFI_CUTOFF_FRACTION,
    delta_t=None,
    taper_duration=DEFAULT_TAPER_DURATION,
):
    """Strain at null infinity from :math:`r\\Psi_4` modes at one finite radius.

    Each mode with :math:`m \\neq 0` is extrapolated with the Nakano et al.
    (2015) formula and integrated twice with the FFI, with cutoff
    ``ffi_cutoff_fraction * |m| * initial_orbital_frequency``.  The
    :math:`m = 0` modes are returned as zeros: their content is mostly the
    non-oscillatory memory, which the FFI cannot recover (the same choice as
    the MAYA LVCNR export).

    Parameters
    ----------
    time : 1d array
           Coordinate time of the extraction, in units of the total mass.
    rpsi4_modes : dict
                  ``{(ell, emm): r * Psi4_lm}`` on ``time``, scaled by the
                  coordinate radius of the extraction sphere.
    extraction_radius : float
                        Coordinate radius of the extraction sphere.
    total_mass : float
                 Total (horizon) mass in the units of ``time``.
    initial_orbital_frequency : float
                                Orbital angular frequency at the start of
                                the simulation, used for the FFI cutoff.
    background_spin : float, optional
                      Kerr spin parameter :math:`a` (dimensionful) of the
                      background in the extrapolation formula. Defaults to 0,
                      which needs no neighbouring modes.
    ffi_cutoff_fraction : float, optional
                          FFI cutoff as a fraction of
                          ``|m| * initial_orbital_frequency``.
    delta_t : float, optional
              Spacing of the uniform grid on which the modes are processed.
              Defaults to the median spacing of ``time``.
    taper_duration : float, optional
                     Length of the Planck taper at each end of
                     :math:`r\Psi_4`, applied before the extrapolation and
                     the FFI. ``0`` or None disables it. Defaults to
                     ``DEFAULT_TAPER_DURATION``.

    Returns
    -------
    retarded_time : 1d array
                    :math:`u = t - r_*(R)` on the uniform grid.
    strain_modes : dict
                   ``{(ell, emm): h_lm}`` at null infinity on
                   ``retarded_time``, scaled by the distance.
    """
    from waveformtools.extrapolate import (
        r_to_ra_conversion,
        waveextract_to_inf_perturbative_twop5_order,
    )
    from waveformtools.integrate import fixed_frequency_integrator

    if initial_orbital_frequency is None or not initial_orbital_frequency > 0:
        raise ValueError(f"initial_orbital_frequency must be positive, not {initial_orbital_frequency}.")

    areal_radius = r_to_ra_conversion(extraction_radius, mass=total_mass, spin=background_spin)
    uniform_modes = {}
    for mode_index, mode_data in rpsi4_modes.items():
        uniform_time, uniform_modes[mode_index] = resample_uniformly(time, mode_data, delta_t)
    taper_window = planck_taper_window(uniform_time, taper_duration)
    uniform_modes = {mode_index: taper_window * mode_data for mode_index, mode_data in uniform_modes.items()}
    grid_spacing = float(uniform_time[1] - uniform_time[0])

    strain_modes = {}
    for (ell, emm), rpsi4_mode in uniform_modes.items():
        if emm == 0:
            strain_modes[(ell, emm)] = np.zeros_like(rpsi4_mode)
            continue
        cutoff_frequency = ffi_cutoff_fraction * abs(emm) * initial_orbital_frequency
        neighbour_modes = {}
        if background_spin != 0:
            neighbour_modes = {
                "rPsi4_lp1m": uniform_modes.get((ell + 1, emm)),
                "rPsi4_lm1m": uniform_modes.get((ell - 1, emm)),
            }
        rpsi4_at_infinity = waveextract_to_inf_perturbative_twop5_order(
            rpsi4_mode,
            grid_spacing,
            areal_radius=areal_radius,
            mass=total_mass,
            spin=background_spin,
            ell=ell,
            emm=emm,
            omega0=cutoff_frequency,
            **neighbour_modes,
        )
        _, strain_modes[(ell, emm)] = fixed_frequency_integrator(
            rpsi4_at_infinity, grid_spacing, omega0=cutoff_frequency, order=2
        )

    retarded_time = uniform_time - tortoise_radius(areal_radius, total_mass)
    logger.info(
        f"Extrapolated {len(strain_modes)} modes from coordinate radius {extraction_radius} "
        f"(areal radius {areal_radius:.3f}, retardation {retarded_time[0] - uniform_time[0]:.3f})"
    )
    return retarded_time, strain_modes
