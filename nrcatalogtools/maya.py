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
import glob
import h5py
from nrcatalogtools.utils import (time_to_physical, amp_to_physical, ylm)
import numpy as np
import os
from pycbc.types import TimeSeries
from scipy.interpolate import InterpolatedUnivariateSpline


def mode_maya_or_rit(catalog,
                     sim_id,
                     res_tag=None,
                     mode=[2, 2],
                     delta_t=0.1,
                     M_fed=50,
                     dMpc=1,
                     data_dir="/home1/aditya.vijaykumar/nr-simulations/",
                     physical_units=False,
                     k=3):
    """
    catalog: str, Name of the catalog. RIT or MAYA
    sim_id: str, simulation id. For RIT it is like BBH-0100. For MAYA, it is
    like GT0100.
    res_tag: str, required for RIT. Example n100. Default is None.
    mode: list of two elements, mode of gravitational waves. Default is [2, 2]
    delta_t: float, time step for the timeseries in dimensionless units.
    Default is 0.1
    M_fed: float, Fiducial total mass of the system. Default is 50
    dMpc: float, distance in the units of Mpc. Default is 1
    data_dir: str, path to directory where NR waveforms are stored.
    physical_units: Boolean, whethere to return the starin and time in physical
    units. Default is True.
    k: degree for interpolation. Default is 3.

    returns:
    times: time array
    hlm: complex strain
    """
    # Fetch data file name
    if catalog == "MAYA":
        file_name = f"{sim_id}.h5"
        file_regex = os.path.join(data_dir, "**/", file_name)
    elif catalog == "RIT":
        if res_tag is None:
            raise Exception("For RIT catalog a res_tag is required.")
        # file_name = f"ExtrapStrain_RIT-{sim_id}-{res_tag}.h5"
        file_regex = os.path.join(
            data_dir,
            "ExtrapStrain_RIT-{0}-{1}.h5".format('*' + sim_id.split(':')[-1],
                                                 res_tag))
    else:
        raise Exception("Catalog must be one of ['MAYA','RIT']")

    file_names = glob.glob(file_regex, recursive=True)
    if len(file_names) == 0:
        raise IOError("File {} not found in dir {} using REGEX: {}".format(
            file_name, data_dir, os.path.join(data_dir, "**/", file_name)))
    file_name = file_names[0]

    # Read in data from file and prepare mode time series
    with h5py.File(file_name, 'r') as data:
        ell, m = mode
        amp_data = data[f"amp_l{ell}_m{m}"]
        amp_time = amp_data["X"][:]
        amp = amp_data["Y"][:]
        amp_interp = InterpolatedUnivariateSpline(amp_time, amp, k=k)
        phase_data = data[f"phase_l{ell}_m{m}"]
        phase_time = phase_data["X"][:]
        phase = phase_data["Y"][:]
    phase_interp = InterpolatedUnivariateSpline(phase_time, phase, k=k)
    times = np.arange(max(amp_time[0], phase_time[0]),
                      min(amp_time[-1], phase_time[-1]), delta_t)
    hlm = amp_interp(times) * np.exp(1j * phase_interp(times))
    if physical_units:
        times = times * time_to_physical(M_fed)
        hlm = hlm * amp_to_physical(M_fed, dMpc)
    return times, hlm


def td_maya_or_rit(catalog,
                   sim_id,
                   res_tag=None,
                   modes=[[2, 2]],
                   delta_t=0.1,
                   M_fed=50,
                   dMpc=1,
                   data_dir="/home1/aditya.vijaykumar/nr-simulations/",
                   physical_units=False,
                   k=3,
                   theta=0,
                   phi=0):

    for idx, mode in enumerate(modes):
        ell, m = mode
        times, hlm = mode_maya_or_rit(catalog, sim_id, res_tag, mode, delta_t,
                                      M_fed, dMpc, data_dir, physical_units, k)
        if idx == 0:
            h = np.zeros(len(times), dtype="complex128")
        h += ylm(ell, m, theta, phi) * hlm

    if physical_units:
        delta_t = delta_t * time_to_physical(M_fed)

    hp = TimeSeries(np.real(h), delta_t, epoch=times[0])
    hc = TimeSeries(-np.imag(h), delta_t, epoch=times[0])
    return hp, hc
