import os

import h5py
import lal
import numpy as np
from sxs import WaveformModes as sxs_WaveformModes
from nrcatalogtools.lvc import GetNRToLALRotationAngles
from . import utils


class WaveformModes(sxs_WaveformModes):
    def __new__(
        cls,
        data,
        time=None,
        time_axis=0,
        modes_axis=1,
        ell_min=2,
        ell_max=4,
        verbosity=0,
        **w_attributes,
    ) -> None:
        self = super().__new__(
            cls,
            data,
            time=time,
            time_axis=time_axis,
            modes_axis=modes_axis,
            ell_min=ell_min,
            ell_max=ell_max,
            **w_attributes,
        )
        self.verbosity = verbosity
        return self

    @classmethod
    def _load(
        cls,
        data,
        time=None,
        time_axis=0,
        modes_axis=1,
        ell_min=2,
        ell_max=4,
        verbosity=0,
        **w_attributes,
    ):
        if time is None:
            time = np.arange(0, len(data[:, 0]))
        return cls(
            data,
            time=time,
            time_axis=time_axis,
            modes_axis=modes_axis,
            ell_min=ell_min,
            ell_max=ell_max,
            verbosity=verbosity,
            **w_attributes,
        )

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
        from sxs.waveforms.nrar import (
            h,
            translate_data_type_to_spin_weight,
            translate_data_type_to_sxs_string,
        )

        if type(file_path_or_open_file) == h5py._hl.files.File:
            h5_file = file_path_or_open_file
            close_input_file = False
        elif os.path.exists(file_path_or_open_file):
            h5_file = h5py.File(file_path_or_open_file, "r")
            close_input_file = True
        else:
            raise RuntimeError(f"Could not use or open {file_path_or_open_file}")

        # Set the file path attribute
        cls._filepath = h5_file.filename
        # Note: to be activate after metdata has been loaded
        # in SXS format.
        cls._metadata_path = cls._metadata["metadata_path"]

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
                dt = min(
                    dt,
                    stat_mode(np.diff(amp_time), keepdims=True)[0][0],
                    stat_mode(np.diff(phase_time), keepdims=True)[0][0],
                )
                ell_min = min(ell_min, ell)
                ell_max = max(ell_max, ell)
                LM.append([ell, em])
        if close_input_file:
            h5_file.close()
        if len(LM) == 0:
            raise RuntimeError(
                "We did not find even one mode in the file. Perhaps the "
                "format `amp_l?_m?` and `phase_l?_m?` is not the "
                "nomenclature of datagroups in the input file?"
            )

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
        w_attributes["frame"] = quaternionic.array([[1.0, 0.0, 0.0, 0.0]])
        w_attributes["frame_type"] = "inertial"
        w_attributes["data_type"] = h
        w_attributes["spin_weight"] = translate_data_type_to_spin_weight(
            w_attributes["data_type"]
        )
        w_attributes["data_type"] = translate_data_type_to_sxs_string(
            w_attributes["data_type"]
        )
        w_attributes["r_is_scaled_out"] = True
        w_attributes["m_is_scaled_out"] = True
        # w_attributes["ells"] = ell_min, ell_max

        return cls(
            data,
            time=times,
            time_axis=0,
            modes_axis=1,
            ell_min=ell_min,
            ell_max=ell_max,
            verbosity=verbosity,
            **w_attributes,
        )

    @property
    def filepath(self):
        """Return the data file path"""
        return self._filepath

    @property
    def sim_metadata(self):
        """Return the simulation metadata dictionary"""
        return self._metadata["metadata"]

    def get_mode(self, ell, em):
        return self[f"Y_l{ell}_m{em}.dat"]

    @property
    def f_lower_at_1Msun(self):
        from pycbc.types import TimeSeries
        from pycbc.waveform import frequency_from_polarizations

        mode22 = self.get_mode(2, 2)
        fr22 = frequency_from_polarizations(
            TimeSeries(mode22[:, 1], delta_t=np.diff(self.time)[0]),
            TimeSeries(-1 * mode22[:, 2], delta_t=np.diff(self.time)[0]),
        )
        return fr22[0] / lal.MTSUN_SI

    def get_polarizations(self, inclination, coa_phase, f_ref=None, t_ref=None):
        """Sum over modes data and return plus and cross GW polarizations

        Args:
            inclination (float): Inclination angle between the line-of-sight
                orbital angular momentum vector [radians]
            coa_phase (float): Coalesence orbital phase [radians]

        Returns:
            Tuple(numpy.ndarray): Numpy Arrays containing polarizations
                time-series
        """

        # Get angles
        angles = self.get_angles(inclination, coa_phase, f_ref, t_ref)

        polarizations = self.evaluate([angles["theta"], angles["psi"], angles["alpha"]])

        return polarizations

    def get_td_waveform(
        self,
        total_mass,
        distance,
        inclination,
        coa_phase,
        delta_t=None,
        f_ref=None,
        t_ref=None,
    ):
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
            FRef (float, optional) : The reference frequency.
            TRef (float, optional) : The reference time.
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
        if delta_t > 1.0 / 128:
            new_time = np.arange(min(self.time), max(self.time), delta_t)
        else:
            new_time = np.arange(min(self.time), max(self.time), delta_t / m_secs)

        # Get angles
        angles = self.get_angles(inclination, coa_phase, f_ref, t_ref)

        h = self.interpolate(new_time).evaluate(
            [angles["theta"], angles["psi"], angles["alpha"]]
        ) * utils.amp_to_physical(total_mass, distance)
        h.time *= m_secs
        return self.to_pycbc(h)

    def get_angles(
        self, inclination, coa_phase, phi_ref=0, f_ref=None, t_ref=None
    ):
        """Get the inclination, azimuthal and polarization angles
        of the observer in the NR source frame.

        Parameters
        ----------
        inclination : float
                      The inclination angle of the observer
                      in the LAL source frame
        coa_phase : float
                    The coalescence phase. This will be
                    the same as reference orbital phase.
        phi_ref  : float, optional
                   The reference orbital phase.
        fref, tref : float, optional
                    The reference frquency and time to define the LAL source frame.
                     Defaults to the available frequency in the data file.

        Returns
        -------
        angles : dict
                 angles : dict
                 The angular corrdinates Theta, Psi,  and the rotation angle Alpha.
                 If available, this also contains the reference time and frequency.
        """
        # Note: 02 May 23 (VP)
        # Presently, coa_phase is not implemented.
        print(
            "Warining! coa_phase is not implemented yet. The reference phase will be used."
        )

        # Compute angles
        with h5py.File(self.filepath) as h5_file:
            # print(H5File.attrs.keys())
            angles = GetNRToLALRotationAngles(
                h5_file=h5_file,
                sim_metadata=self._sim_metadata,
                inclination=inclination,
                phi_ref=phi_ref,
                f_ref=f_ref,
                t_ref=t_ref,
            )

        return angles

    def to_pycbc(self, input_array=None):
        if input_array is None:
            input_array = self
        from pycbc.types import TimeSeries
        from scipy.stats import mode as stat_mode

        delta_t = stat_mode(np.diff(input_array.time), keepdims=True)[0][0]
        return TimeSeries(
            np.array(input_array),
            delta_t=delta_t,
            dtype=self.ndarray.dtype,
            epoch=input_array.time[0],
            copy=True,
        )

    def to_lal(self):
        raise NotImplementedError()

    def to_astropy(self):
        return self.to_pycbc().to_astropy()
