import lal
import numpy as np
import os
import h5py
from sxs import WaveformModes as sxs_WaveformModes

from . import utils


class WaveformModes(sxs_WaveformModes):
    def __new__(cls,
                data,
                time=None,
                time_axis=0,
                modes_axis=1,
                ell_min=2,
                ell_max=4,
                verbosity=0,
                **w_attributes) -> None:
        self = super().__new__(cls,
                               data,
                               time=time,
                               time_axis=time_axis,
                               modes_axis=modes_axis,
                               ell_min=ell_min,
                               ell_max=ell_max,
                               **w_attributes)
        self.verbosity = verbosity
        return self

    @classmethod
    def _load(cls,
              data,
              time=None,
              time_axis=0,
              modes_axis=1,
              ell_min=2,
              ell_max=4,
              verbosity=0,
              **w_attributes):
        if time is None:
            time = np.arange(0, len(data[:, 0]))
        return cls(data,
                   time=time,
                   time_axis=time_axis,
                   modes_axis=modes_axis,
                   ell_min=ell_min,
                   ell_max=ell_max,
                   verbosity=verbosity,
                   **w_attributes)

    @classmethod
    def load_from_h5(cls, file_path_or_open_file, metadata={}, verbosity=0):
        """Method to load SWSH waveform modes from RIT or MAYA catalogs
        from HDF5 file.

        Args:
            file_path_or_open_file (str or open file): Either the path to an
                HDF5 file containing waveform data, or an open file pointer to
                the same.
            metadata (dict): Dictionary containing metadata (Note that keys
                will be NR group specific)
            verbosity (int, optional): Verbosity level with which to
                print messages during execution. Defaults to 0.

        Raises:
            RuntimeError: If inputs are invalid, or if no mode found in
                input file.

        Returns:
            WaveformModes: Object containing time-series of SWSH modes.
        """
        import quaternionic
        from scipy.interpolate import InterpolatedUnivariateSpline
        from scipy.stats import mode as stat_mode
        from sxs.waveforms.nrar import (h, translate_data_type_to_spin_weight,
                                        translate_data_type_to_sxs_string)

        if type(file_path_or_open_file) == h5py._hl.files.File:
            h5_file = file_path_or_open_file
            close_input_file = False
            nr_group = 'UNKNOWN'
        elif os.path.exists(file_path_or_open_file):
            h5_file = h5py.File(file_path_or_open_file, "r")
            close_input_file = True
            file_path_str = str(file_path_or_open_file)
            for tag in utils.nr_group_tags:
                if utils.nr_group_tags[tag] in file_path_str:
                    nr_group = utils.nr_group_tags[tag]
        else:
            raise RuntimeError(
                f"Could not use or open {file_path_or_open_file}")

        ELL_MIN, ELL_MAX = 2, 10
        ell_min, ell_max = 99, -1
        LM = []
        t_min, t_max, dt = -1e99, 1e99, 1
        mode_data = {}
        for ell in range(ELL_MIN, ELL_MAX + 1):
            for em in range(-ell, ell + 1):
                afmt = f"amp_l{ell}_m{em}"
                pfmt = f"phase_l{ell}_m{em}"
                if afmt not in h5_file or pfmt not in h5_file:
                    continue
                amp_time = h5_file[afmt]["X"][:]
                amp = h5_file[afmt]["Y"][:]
                phase_time = h5_file[pfmt]["X"][:]
                phase = h5_file[pfmt]["Y"][:]
                mode_data[(ell, em)] = [amp_time, amp, phase_time, phase]
                # get the minimum time and maximum time stamps for all modes
                t_min = max(t_min, amp_time[0], phase_time[0])
                t_max = min(t_max, amp_time[-1], phase_time[-1])
                dt = min(dt,
                         stat_mode(np.diff(amp_time), keepdims=True)[0][0],
                         stat_mode(np.diff(phase_time), keepdims=True)[0][0])
                ell_min = min(ell_min, ell)
                ell_max = max(ell_max, ell)
                LM.append([ell, em])
        if close_input_file:
            h5_file.close()
        if len(LM) == 0:
            raise RuntimeError(
                f"We did not find even one mode in the file. Perhaps the "
                f" format `amp_l?_m?` and `phase_l?_m?` is not the "
                f"nomenclature of datagroups in the input file?")

        times = np.arange(t_min, t_max + 0.5 * dt, dt)
        data = np.empty((len(times), len(LM)), dtype=complex)
        for idx, (ell, em) in enumerate(LM):
            amp_time, amp, phase_time, phase = mode_data[(ell, em)]
            amp_interp = InterpolatedUnivariateSpline(amp_time, amp)
            phase_interp = InterpolatedUnivariateSpline(phase_time, phase)
            data[:, idx] = amp_interp(times) * np.exp(1j * phase_interp(times))

        w_attributes = {}
        w_attributes["metadata"] = metadata
        w_attributes["history"] = ""
        w_attributes["frame"] = quaternionic.array([[1., 0., 0., 0.]])
        w_attributes["frame_type"] = "inertial"
        w_attributes["data_type"] = h
        w_attributes["spin_weight"] = translate_data_type_to_spin_weight(
            w_attributes["data_type"])
        w_attributes["data_type"] = translate_data_type_to_sxs_string(
            w_attributes["data_type"])
        w_attributes["r_is_scaled_out"] = True
        w_attributes["m_is_scaled_out"] = True
        # w_attributes["ells"] = ell_min, ell_max

        return cls(data,
                   time=times,
                   time_axis=0,
                   modes_axis=1,
                   ell_min=ell_min,
                   ell_max=ell_max,
                   verbosity=verbosity,
                   **w_attributes)

    def get_mode(self, ell, em):
        return self[f"Y_l{ell}_m{em}.dat"]

    def get_polarizations(self, inclination, coa_phase):
        """Sum over modes data and return plus and cross GW polarizations

        Args:
            inclination (float): Inclination angle between the line-of-sight
                orbital angular momentum vector [radians]
            coa_phase (float): Coalesence orbital phase [radians]

        Returns:
            Tuple(numpy.ndarray): Numpy Arrays containing polarizations
                time-series
        """
        polarizations = self.evaluate([inclination, coa_phase])
        return polarizations

    def get_td_waveform(self,
                        total_mass,
                        distance,
                        inclination,
                        coa_phase,
                        delta_t=None):
        """Sum over modes data and return plus and cross GW polarizations,
        rescaled appropriately for a compact-object binary with given
        total mass and distance from GW detectors.

        Returns:
            Tuple(numpy.ndarray): Numpy Arrays containing polarizations
                time-series

        Args:
            total_mass (_type_): _description_
            distance (_type_): _description_
            inclination (float): Inclination angle between the line-of-sight
                orbital angular momentum vector [radians]
            coa_phase (float): Coalesence orbital phase [radians]
            delta_t (_type_, optional): _description_. Defaults to None.

        Returns:
            pycbc.TimeSeries(numpy.complex128): Complex polarizations
                stored in `pycbc` container `TimeSeries`
        """
        if delta_t is None:
            from scipy.stats import mode as stat_mode
            delta_t = stat_mode(np.diff(self.time), keepdims=True)[0][0]
        m_secs = utils.time_to_physical(total_mass)
        # we assume that we generally do not sample at a rate below 128Hz.
        # Therefore, depending on the numerical value of dt, we deduce whether
        # dt is in dimensionless units or in seconds.
        if delta_t > 1. / 128:
            new_time = np.arange(min(self.time), max(self.time), delta_t)
        else:
            new_time = np.arange(min(self.time), max(self.time),
                                 delta_t / m_secs)
        h = self.interpolate(new_time).evaluate([inclination, coa_phase
                                                 ]) * utils.amp_to_physical(
                                                     total_mass, distance)
        h.time *= m_secs
        return self.to_pycbc(h)

    def to_pycbc(self, input_array=None):
        if input_array is None:
            input_array = self
        from pycbc.types import TimeSeries
        from scipy.stats import mode as stat_mode
        delta_t = stat_mode(np.diff(input_array.time), keepdims=True)[0][0]
        return TimeSeries(np.array(input_array),
                          delta_t=delta_t,
                          dtype=self.ndarray.dtype,
                          epoch=input_array.time[0],
                          copy=True)

    def to_lal(self):
        raise NotImplementedError()

    def to_astropy(self):
        return self.to_pycbc().to_astropy()
