# Copyright (C) 2023 Aditya Vijaykumar, Md Arif, Prayush Kumar
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
import lal
import numpy as np


def time_to_physical(M):
    """Factor to convert time from dimensionless units to SI units

    parameters
    ----------
    M: mass of system in the units of solar mass

    Returns
    -------
    converting factor
    """

    return M * lal.MTSUN_SI


def amp_to_physical(M, D):
    """Factor to rescale strain to mass M and distance D convert from
    dimensionless units to SI units

    parameters
    ----------
    M: mass of the system in units of solar mass
    D: Luminosity distance in units of megaparsecs

    Returns
    -------
    Scaling factor
    """

    return lal.G_SI * M * lal.MSUN_SI / (lal.C_SI**2 * D * 1e6 * lal.PC_SI)


def dlm(ell, m, theta):
    """Wigner d function
    parameters
    ----------
    ell: lvalue
    m: mvalue
    theta: theta angle, e. g in GW, inclination angle iota

    Returns:
    value of d^{ell m}(theta)
    """
    kmin = max(0, m - 2)
    kmax = min(ell + m, ell - 2)
    d = 0
    for k in range(kmin, kmax + 1):
        numerator = np.sqrt(
            float(
                np.math.factorial(ell + m) * np.math.factorial(ell - m) *
                np.math.factorial(ell + 2) * np.math.factorial(ell - 2)))
        denominator = (np.math.factorial(k - m + 2) *
                       np.math.factorial(ell + m - k) *
                       np.math.factorial(ell - k - 2))
        d += (((-1)**k / np.math.factorial(k)) * (numerator / denominator) *
              (np.cos(theta / 2))**(2 * ell + m - 2 * k - 2) *
              (np.sin(theta / 2))**(2 * k - m + 2))
    return d


def ylm(ell, m, theta, phi):
    """
    parameters:
    -----------
    ell: lvalue
    m: mvalue
    theta: theta angle, e. g in GW, inclination angle iota
    phi: phi angle, e. g. in GW, orbital phase

    Returns:
    --------
    ylm_s(theta, phi)
    """
    return (np.sqrt((2 * ell + 1) / (4 * np.pi)) * dlm(ell, m, theta) *
            np.exp(1j * m * phi))
