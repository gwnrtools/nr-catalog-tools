import os

import h5py
import lal
import numpy as np
import quaternionic
import spherical
from pycbc.filter import highpass, lowpass
from pycbc.psd import interpolate
from pycbc.types import TimeSeries
from pycbc.waveform import frequency_from_polarizations
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import mode as stat_mode

from nrcatalogtools import metadata as md
from nrcatalogtools import utils
from nrcatalogtools.lvc import (
    check_interp_req,
    get_nr_to_lal_rotation_angles,
    get_ref_vals,
)

from sxs import TimeSeries as sxs_TimeSeries
from sxs import WaveformModes as sxs_WaveformModes
from sxs.waveforms.format_handlers.nrar import (
    h,
    translate_data_type_to_spin_weight,
    translate_data_type_to_sxs_string,
)

ELL_MIN, ELL_MAX = 2, 10


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
        self._t_ref_nr = None
        self._filepath = None
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

        if type(file_path_or_open_file) is h5py._hl.files.File:
            h5_file = file_path_or_open_file
            close_input_file = False
        elif os.path.exists(file_path_or_open_file):
            h5_file = h5py.File(file_path_or_open_file, "r")
            close_input_file = True
        else:
            raise RuntimeError(f"Could not use or open {file_path_or_open_file}")

        # Set the file path attribute
        cls._filepath = h5_file.filename

        # If _metadata is not already a set attribute, then set it here.
        try:
            cls._metadata
        except AttributeError:
            cls._metadata = metadata

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

                try:
                    amp_time = h5_file[afmt]["X"][:]
                    amp = h5_file[afmt]["Y"][:]
                    phase_time = h5_file[pfmt]["X"][:]
                    phase = h5_file[pfmt]["Y"][:]
                except KeyError:
                    # Some RIT files have empty modes stored (*sigh*)
                    # Skip these modes, and print a warning
                    if verbosity > 0:
                        print(
                            f"Skipping mode l={ell}, m={em} for {file_path_or_open_file} "
                            "since columns 'X' and 'Y' not found"
                        )
                    continue
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

    @classmethod
    def load_from_targz(cls, file_path, metadata={}, verbosity=0):
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
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            raise RuntimeError(f"Could not use or open {file_path}")

        import quaternionic
        import re
        import tarfile

        def get_tag(name):
            return os.path.splitext(os.path.splitext(os.path.basename(name))[0])[0]

        def get_el_em_from_filename(filename: str):
            substr = re.search(pattern=r"l\d_m\d", string=filename)
            if substr is None:
                substr = re.search(pattern=r"l\d_m-\d", string=filename)
            elem = substr[0].split("_")
            return (int(elem[0].strip("l")), int(elem[1].strip("m")))

        # Set the file path attribute
        cls._filepath = file_path

        # If _metadata is not already a set attribute, then set it here.
        if not hasattr(cls, "_metadata"):
            cls._metadata = metadata

        ell_min, ell_max = 99, -1
        t_min, t_max, dt = -1e99, 1e99, 1

        file_tag = get_tag(file_path)
        mode_data = {}
        reference_mode_num_for_length = ()
        possible_ascii_extensions = ["asc", "dat", "txt"]

        with tarfile.open(file_path, "r:gz") as tar:
            if verbosity > 4:
                print(f"Opening tarfile: {file_path}")
            for dat_file in tar.getmembers():
                dat_file_name = dat_file.name
                if verbosity > 4:
                    print(f"dat_file_name is: {dat_file_name}")
                if file_tag not in dat_file_name or np.all(
                    [
                        f".{ext}" not in dat_file_name
                        for ext in possible_ascii_extensions
                    ]
                ):
                    if verbosity > 5:
                        print(
                            f"{file_tag} not in {dat_file_name} is {file_tag not in dat_file_name}"
                        )
                        print(
                            "the other flag is: ",
                            np.all(
                                [
                                    f".{ext}" not in dat_file_name
                                    for ext in possible_ascii_extensions
                                ]
                            ),
                        )
                    continue
                ell, em = get_el_em_from_filename(dat_file_name)
                with tar.extractfile(dat_file_name) as f:
                    reference_mode_num_for_length = (ell, em)
                    mode_data[(ell, em)] = np.loadtxt(f)
                    # Convert to row-major form
                    nrows, ncols = np.shape(mode_data[(ell, em)])
                    if nrows < ncols:
                        mode_data[(ell, em)] = mode_data[(ell, em)].T
                    # mode_data[get_tag(dat_file_name)] = np.loadtxt(f)
                # get the minimum time and maximum time stamps for all modes
                t_min = max(t_min, mode_data[(ell, em)][0, 0])
                t_max = min(t_max, mode_data[(ell, em)][-1, 0])
                dt = min(
                    dt,
                    stat_mode(np.diff(mode_data[(ell, em)][:, 0]), keepdims=True)[0][0],
                )
                ell_min = min(ell_min, ell)
                ell_max = max(ell_max, ell)

        # We populate LM here because it has to be ordered, as the WaveformModes
        # class expects an ordered data set.
        LM = []
        for ell in range(ELL_MIN, ELL_MAX + 1):
            for em in range(-ell, ell + 1):
                if (ell, em) in mode_data:
                    LM.append([ell, em])
                else:
                    reference_mode = mode_data[reference_mode_num_for_length]
                    mode_data[(ell, em)] = np.zeros(np.shape(reference_mode))
                    mode_data[(ell, em)][:, 0] = reference_mode[:, 0]  # Time axis
                    LM.append([ell, em])

        if len(LM) == 0:
            raise RuntimeError(
                "We did not find even one mode in the file. Perhaps the "
                "format `amp_l?_m?` and `phase_l?_m?` is not the "
                "nomenclature of datagroups in the input file?"
            )

        times = np.arange(t_min, t_max + 0.5 * dt, dt)
        data = np.empty((len(times), len(LM)), dtype=complex)
        for idx, (ell, em) in enumerate(LM):
            mode_time, mode_real, mode_imag = (
                mode_data[(ell, em)][:, 0],
                mode_data[(ell, em)][:, 1],
                mode_data[(ell, em)][:, 2],
            )
            if verbosity > 5:
                print(f"Interpolating mode {ell}, {em}. Data length: {len(mode_time)}")
            mode_real_interp = InterpolatedUnivariateSpline(mode_time, mode_real)
            mode_imag_interp = InterpolatedUnivariateSpline(mode_time, mode_imag)
            data[:, idx] = mode_real_interp(times) + 1j * mode_imag_interp(times)

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
        if not self._filepath:
            self._filepath = self.sim_metadata["waveform_data_location"]

        return self._filepath

    @property
    def sim_metadata(self):
        """Return the simulation metadata dictionary"""
        return self._metadata["metadata"]

    @property
    def metadata(self):
        """Return the simulation metadata dictionary"""
        return self.sim_metadata

    @property
    def label(self):
        """Return a Latex label that summarizes key simulation details"""
        md = self.metadata

        return (
            f"$q{md['relaxed_mass_ratio_1_over_2']:0.3f}\\_"
            f"\\chi_A{md['relaxed_chi1x']:0.3f}\\_{md['relaxed_chi1y']:0.3f}\\_"
            f"{md['relaxed_chi1z']:0.3f}\\_\\_"
            f"\\chi_B{md['relaxed_chi2x']:0.3f}\\_{md['relaxed_chi2y']:0.3f}\\_"
            f"{md['relaxed_chi2z']:0.3f}$"
        )

    @property
    def label_nolatex(self):
        """Return a Latex label that summarizes key simulation details"""
        md = self.metadata
        return f"""q{md['relaxed_mass_ratio_1_over_2']:0.3f}-sA{
            md['relaxed_chi1x']:0.3f}-{md['relaxed_chi1y']:0.3f}-{
                md['relaxed_chi1z']:0.3f}--sB{md['relaxed_chi2x']:0.3f}-{
                    md['relaxed_chi2y']:0.3f}-{md['relaxed_chi2z']:0.3f}"""

    def get_parameters(self, total_mass=1.0):
        """
        Return the initial physical parameters for the simulation. Only
        quasicircular simulations are supported, orbital eccentricity is ignored

        Args:
            total_mass (float, optional): Total Mass of Binary (solar masses).
                Defaults to 1.0.

        Returns:
            dict: Initial binary parameters with names compatible with PyCBC.
        """
        metadata = self.metadata
        parameters = md.get_source_parameters_from_metadata(
            metadata, total_mass=total_mass
        )
        if "relaxed_mass1" in metadata:
            # RIT Catalog
            if parameters["f_lower"] == -1:
                h = self.get_mode(2, 2, total_mass, distance=1, delta_t=1.0 / 8192)
                fr = frequency_from_polarizations(h.real(), -h.imag())
                parameters.update(f_lower=fr[0])
        elif "GTID" in metadata:
            # GT / MAYA CAtalog
            if parameters["f_lower"] == -1:
                h = self.get_mode(2, 2, total_mass, distance=1, delta_t=1.0 / 8192)
                fr = frequency_from_polarizations(h.real(), -h.imag())
                parameters.update(f_lower=fr[0])
        else:
            # SXS Catalog
            if parameters["f_lower"] == -1:
                delta_t_secs = 1.0 / 8192
                h = self.get_mode(2, 2, total_mass, distance=1, delta_t=delta_t_secs)
                fr = frequency_from_polarizations(h.real(), -h.imag())
                # Get the frequency at reference_time
                reference_time_idx = int(
                    np.round(
                        (metadata["reference_time"] / (total_mass * lal.MTSUN_SI))
                        / delta_t_secs
                    )
                )
                parameters.update(f_lower=fr[reference_time_idx])

        return parameters

    def get_mode_data(self, ell, em):
        return self[f"Y_l{ell}_m{em}.dat"]

    def get_mode(
        self,
        ell,
        em,
        total_mass=1.0,
        distance=1.0,  # Megaparsecs
        delta_t=None,
        to_pycbc=True,
    ):
        """Get individual modes, rescaled appropriately for a compact-object
        binary with given total mass and distance from a GW detector.

        Args:
            ell (int): mode l value
            em (int): mode m value
            total_mass (float, optional): Total Mass (Solar Masses).
                                          Defaults to 1.
            distance (float, optional): Distance to Source (Megaparsecs).
                                        Defaults to 1.
            delta_t (float, optional): Sample rate (in Hz or M).
                                       Defaults to None.
            to_pycbc (bool, optional) : Return `pycbc.types.TimeSeries` or
                `sxs.TimeSeries`. Defaults to True.
        Returns:
            `pycbc.types.TimeSeries(numpy.complex128)` or
                `sxs.TimeSeries(numpy.complex128)`:
                Complex waveform mode time series
        """
        if delta_t is None:
            delta_t = stat_mode(np.diff(self.time), keepdims=True)[0][0]

        # we assume that we generally do not sample at a rate below 128Hz.
        # Therefore, depending on the numerical value of dt, we deduce whether
        # dt is in dimensionless units or in seconds.
        if delta_t > 1.0 / 128:
            m_secs = 1
            new_time = np.arange(min(self.time), max(self.time), delta_t)
        else:
            m_secs = utils.time_to_physical(total_mass)
            new_time = np.arange(min(self.time), max(self.time), delta_t / m_secs)

        # --- OPTIMIZATION: Interpolate only the requested mode ---
        mode_data = self.data[:, self.index_from_ell_m(ell, em)]
        mode_ts = sxs_TimeSeries(mode_data, time=self.time)
        interpolated_mode_ts = mode_ts.interpolate(new_time)

        h_mode_complex = interpolated_mode_ts.data
        h_mode_complex *= utils.amp_to_physical(total_mass, distance)

        # --- OPTIMIZATION: Use cached peak time of (2,2) mode to set epoch ---
        # The epoch is set to shift the time axis so that the peak is at t=0
        peak_time_sec = self.peak_time_22 * m_secs
        start_time_sec = new_time[0] * m_secs
        epoch = start_time_sec - peak_time_sec

        retval = self.to_pycbc(
            input_array=h_mode_complex,
            delta_t=delta_t,
            epoch=epoch,
        )
        if not to_pycbc:
            retval = sxs_TimeSeries(retval.data, time=retval.sample_times)
        return retval

    def f_lower_at_1Msun(self, t=None):
        mode22 = self.get_mode_data(2, 2)
        fr22 = frequency_from_polarizations(
            TimeSeries(mode22[:, 1], delta_t=np.diff(self.time)[0]),
            TimeSeries(-1 * mode22[:, 2], delta_t=np.diff(self.time)[0]),
        )
        # If time value is not provided, return the initial f_lower
        if t is None:
            return float(fr22[0] / lal.MTSUN_SI)
        # Interpolate fr22 as a function of time and evaluate at t
        sample_times = self.time[: len(fr22)]
        interp_fr22 = InterpolatedUnivariateSpline(sample_times, fr22, k=3)
        return float(interp_fr22(t) / lal.MTSUN_SI)

    def get_polarizations(
        self, inclination, coa_phase, f_ref=None, t_ref=None, tol=1e-6
    ):
        """Sum over modes data and return plus and cross GW polarizations

        Args:
            inclination (float): Inclination angle between the line-of-sight
                orbital angular momentum vector [radians]
            coa_phase (float): Coalesence orbital phase [radians]
            tol (float, optional) : The tolerance to allow for
                                    floating point precision errors
                                    in the computation of rotation
                                    angles. Default value is 1e-6.

        Returns:
            Tuple(numpy.ndarray): Numpy Arrays containing polarizations
                time-series
        """

        # Get angles
        angles = self.get_angles(inclination, coa_phase, f_ref, t_ref, tol)

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
        k=3,
        kind=None,
        tol=1e-6,
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
            f_ref (float, optional) : The reference frequency.
            t_ref (float, optional) : The reference time.
            k (int, optional) : The interpolation order to use with
                                `scipy.interpolate.InterpolatedUnivariateSpline`.
                                This is the method used by default with value 3.
                                This parameter `k` is given preference over
                                `kind` (see below).
            kind (str, optional) : The interpolation order to use with
                                    `scipy.interpolate.interp1d`
                                (`linear`, `quadratic`, `cubic`) or
                                `CubicSpline` to use `scipy.interpolate.CubicSpline`.
            tol (float, optional) : The tolerance to allow for
                                    floating point precision errors
                                    in the computation of rotation
                                    angles. Default value is 1e-6.
        Returns:
            pycbc.TimeSeries(numpy.complex128): Complex polarizations
                stored in `pycbc` container `TimeSeries`
        """
        if delta_t is None:
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
        angles = self.get_angles(
            inclination=inclination,
            coa_phase=coa_phase,
            f_ref=f_ref,
            t_ref=t_ref,
            tol=tol,
        )
        h = interpolate_in_amp_phase(
            self.evaluate([angles["theta"], angles["psi"], angles["alpha"]]),
            new_time,
            k=k,
            kind=kind,
        ) * utils.amp_to_physical(total_mass, distance)

        h.time *= m_secs
        # Return conjugated waveform to comply with lal
        return self.to_pycbc(np.conjugate(h))

    def get_angles(self, inclination, coa_phase, f_ref=None, t_ref=None, tol=1e-6):
        """Get the inclination, azimuthal and polarization angles
        of the observer in the NR source frame.

        Parameters
        ----------
        inclination : float
                      The inclination angle of the observer
                      in the LAL source frame
        coa_phase : float
                    The coalescence phase. This will be
                    used to compute the reference orbital phase.
        f_ref, t_ref : float, optional
                    The reference frquency and time to define the LAL source frame.
                     Defaults to the available frequency in the data file.

        tol (float, optional) : The tolerance to allow for
                                    floating point precision errors
                                    in the computation of rotation
                                    angles. Default value is 1e-6.
        Returns
        -------
        angles : dict
                 angles : dict
                 The angular corrdinates Theta, Psi,  and the rotation angle Alpha.
                 If available, this also contains the reference time and frequency.
        """

        # Get observer phi_ref
        obs_phi_ref = self.get_obs_phi_ref_from_obs_coa_phase(
            coa_phase=coa_phase, t_ref=t_ref, f_ref=f_ref
        )

        # Compute angles
        with h5py.File(self.filepath) as h5_file:
            angles = get_nr_to_lal_rotation_angles(
                h5_file=h5_file,
                sim_metadata=self.sim_metadata,
                inclination=inclination,
                phi_ref=obs_phi_ref,
                f_ref=f_ref,
                t_ref=t_ref,
                tol=tol,
            )

        return angles

    def to_pycbc(self, input_array=None, delta_t=None, epoch=None):
        if input_array is None:
            input_array = self
        if epoch is None:
            epoch = input_array.time[0]
        if delta_t is None:
            delta_t = stat_mode(np.diff(input_array.time), keepdims=True)[0][0]
        return TimeSeries(
            np.array(input_array),
            delta_t=delta_t,
            dtype=self.ndarray.dtype,
            epoch=epoch,
            copy=True,
        )

    def get_nr_coa_phase(self):
        """Get the NR coalescence orbital phase from the 2,2 mode."""

        # Get the waveform phase.
        phase_22 = self._get_phase(2, 2)

        waveform_22 = (
            self.get_mode_data(2, 2)[:, 1] + 1j * self.get_mode_data(2, 2)[:, 2]
        )

        # print(len(phase_22), len(waveform_22))
        # Get the localtion of max amplitude.

        maxloc = np.argmax(np.absolute(waveform_22))
        # Compute the orbital phase at max amplitude.
        coa_phase = phase_22[maxloc] / 2

        return coa_phase

    def get_obs_phi_ref_from_obs_coa_phase(self, coa_phase, t_ref=None, f_ref=None):
        """Get the observer reference phase given the observer
        coalescence phase."""

        # Get the NR coalescence phase
        nr_coa_phase = self.get_nr_coa_phase()
        # Get the NR orbital phasing series
        nr_orb_phase_ts = self._get_phase(2, 2) / 2

        # Compute the observer reference phase from
        # this information.

        avail_t_ref = self.t_ref_nr

        # Second, get the NR reference phase
        from scipy.interpolate import interp1d

        nr_phi_ref = interp1d(self.time, nr_orb_phase_ts, kind="cubic")(avail_t_ref)

        # Third, compute the offset in coa_phase
        delta_phi_ref = coa_phase - nr_coa_phase

        # Finally compute the obserer reference phase at NR reference time.
        obs_phi_ref = nr_phi_ref + delta_phi_ref

        return obs_phi_ref

    def to_lal(self):
        raise NotImplementedError()

    def to_astropy(self):
        return self.to_pycbc().to_astropy()

    def _get_phase(self, ell=2, emm=2):
        """Get the phasing of a particular waveform mode."""

        # Get the complex waveform.
        wfm_array = self.get_mode_data(ell, emm)
        waveform_lm_re = wfm_array[:, 1]
        waveform_lm_im = wfm_array[:, 2]
        waveform_lm = waveform_lm_re + 1j * waveform_lm_im
        # Get the waveform phase.
        phase_lm = np.unwrap(np.angle(waveform_lm))
        return phase_lm

    def _compute_reference_time(self):
        """Obtain the reference time from the
        simulation data. Interpolate and get the reference
        time if only reference frequency is given"""

        # To get from available data

        with h5py.File(self.filepath) as h5_file:
            # First, check if interp is required and get the available reference time .
            interp, avail_t_ref = check_interp_req(
                h5_file, self.sim_metadata, ref_time=None
            )

        if avail_t_ref is None:
            ref_omega = None
            # If the reference time is not available,
            # compute from reference phase!
            # Omega is the key of interest.
            try:
                ref_omega = get_ref_vals(self.sim_metadata, req_attrs=["Omega"])[
                    "Omega"
                ]
            except Exception as excep:
                print(
                    "Reference orbital phase not found in simulation metadata."
                    "Proceeding to retrieve from the h5 file..",
                    excep,
                )
                with h5py.File(self.filepath) as h5_file:
                    ref_omega = get_ref_vals(h5_file, req_attrs=["Omega"])["Omega"]
            if ref_omega is None:
                raise KeyError("Could not compute reference omega!")

            nr_orb_phase_ts = self._get_phase(2, 2) / 2

            # Differentiate the phase to get orbital angular frequency
            from waveformtools.differentiate import derivative

            nr_omega_ts = derivative(self.time, nr_orb_phase_ts, method="FD", degree=2)
            # Identify the location in time where nr_omega = ref_omega
            ref_loc = np.argmin(np.absolute(nr_omega_ts - ref_omega))
            avail_t_ref = self.time[ref_loc]

        self._t_ref_nr = avail_t_ref

        # print(avail_t_ref, self._t_ref_nr)
        return avail_t_ref

    @property
    def t_ref_nr(self):
        """Fetch the reference time of a simulation"""

        if not isinstance(self._t_ref_nr, float):
            print("Computing reference time..")
            self._compute_reference_time()

        return self._t_ref_nr

    @property
    def peak_time_22(self):
        """Computes and caches the dimensionless time of the peak amplitude
        of the (2, 2) mode."""
        if hasattr(self, "_peak_time_22"):
            return self._peak_time_22

        from scipy.interpolate import InterpolatedUnivariateSpline

        # Use the raw (non-interpolated) data of the object
        try:
            mode22_idx = self.index_from_ell_m(2, 2)
        except ValueError:
            # Fallback if 2,2 mode is not present, though it is standard.
            self._peak_time_22 = 0.0
            return self._peak_time_22

        mode22_data = self.data[:, mode22_idx]
        amp22 = np.abs(mode22_data)

        # Find the peak of the amplitude vs. time
        x_axis = self.time
        y_axis = amp22

        f = InterpolatedUnivariateSpline(x_axis, y_axis, k=4)
        cr_pts = f.derivative().roots()
        # also check the endpoints of the interval
        cr_pts = np.append(cr_pts, (x_axis[0], x_axis[-1]))
        cr_vals = f(cr_pts)
        max_index = np.argmax(cr_vals)

        self._peak_time_22 = cr_pts[max_index]
        return self._peak_time_22

    def rotated(self, R):
        """
        Rotate the waveform modes.
        Parameters
        ----------
        R : quaternionic.array
            A quaternion representing the rotation.
        Returns
        -------
        WaveformModes
            A new WaveformModes object with the rotated modes.
        """

        # Create a new object for the rotated waveform
        rotated_self = self.copy()

        wigner = spherical.Wigner(self.ell_max)

        # Get the mode indices
        modes = self.LM

        rotated_data = np.zeros_like(self.data)

        for l in range(self.ell_min, self.ell_max + 1):
            if l not in self.ells:
                continue

            # Get the modes for this l
            l_modes_indices = np.where(self.LM[:, 0] == l)[0]
            if len(l_modes_indices) == 0:
                continue

            l_modes = self.data[:, l_modes_indices]

            # Get the Wigner D-matrix for this l
            D = wigner.D(R, l)

            # Apply the rotation
            rotated_l_modes = l_modes @ D

            rotated_data[:, l_modes_indices] = rotated_l_modes

        rotated_self.data = rotated_data

        # Update the frame attribute
        rotated_self.frame = R * self.frame

        return rotated_self

    def match_sphere_averaged(
        self,
        other,
        psd,
        f_lower,
        f_upper=None,
    ):
        """Calculates the match between this waveform and another.

        The match is a measure of similarity between two waveforms, maximized over
        extrinsic parameters (time shift, phase shift, and spatial orientation).
        It is defined as the noise-weighted inner product, normalized bye
        waveform norms.

        The function finds the maximum of the overlap functional, which is given by:

        Overlap(h₁, h₂) = |(h̃₁ | h̃₂)| / sqrt( (h̃₁ | h̃₁) * (h̃₂ | h̃₂) )

        where the noise-weighted inner product (h̃₁ | h̃₂) is defined as:

        (h̃₁ | h̃₂) = 4 * Re ∫[ h̃₁(f) * conj(h̃₂(f; t₀, φ₀, R)) / Sₙ(f) ] df

        The maximization is performed over:
        - t₀ : A relative time shift between the waveforms.
        - φ₀ : A relative phase shift.
        - R  : A relative 3D rotation, parameterized by Euler angles.

        Sₙ(f) is the noise power spectral density (`psd`). The integral is
        approximated as a discrete sum over frequency bins from `f_lower` to
        `f_upper`.

        The calculation is performed by summing the inner products of all common
        spherical harmonic modes (l, m) of the two waveforms.

        The optimization is done by minimizing (1 - Overlap) using the
        'Nelder-Mead' algorithm.

        Parameters
        ----------
        other : Waveform
            The other waveform object to compare against.
        psd : pycbc.types.FrequencySeries
            The one-sided power spectral density (PSD) of the detector noise.
        f_lower : float
            The lower frequency cutoff for the match integral, in Hz.
        f_upper : float, optional
            The upper frequency cutoff for the match integral, in Hz.
            If None, it defaults to the Nyquist frequency of the waveforms.

        Returns
        -------
        float
            The maximum match value, a float between 0 and 1.

        """
        from scipy.optimize import minimize

        def objective_function(x):
            time_shift, phi_c, alpha, beta, gamma = x

            R = quaternionic.array.from_euler_angles(alpha, beta, gamma)
            other_rot = other.rotated(R)

            total_inner_prod = 0.0
            total_norm1_sq = 0.0
            total_norm2_sq = 0.0

            common_modes = set(map(tuple, self.LM)) & set(map(tuple, other_rot.LM))

            for l, m in common_modes:
                h1_mode_ts = self.get_mode(l, m, to_pycbc=True, delta_t=1 / 4096)
                h2_mode_ts = other_rot.get_mode(
                    l, m, to_pycbc=True, delta_t=1 / 4096
                )

                # Align lengths
                if len(h1_mode_ts) > len(h2_mode_ts):
                    h2_mode_ts.resize(len(h1_mode_ts))
                else:
                    h1_mode_ts.resize(len(h2_mode_ts))

                psd.resize(len(h1_mode_ts.to_frequencyseries()))

                # to frequency domain
                h1_tilde = h1_mode_ts.to_frequencyseries(delta_f=psd.delta_f)
                h2_tilde = h2_mode_ts.to_frequencyseries(delta_f=psd.delta_f)

                # Apply phase and time shifts to the second waveform
                h2_tilde *= np.exp(-1j * m * phi_c)

                freqs = h2_tilde.sample_frequencies
                h2_tilde.data *= np.exp(-2j * np.pi * freqs * time_shift)

                df = psd.delta_f
                low_idx = int(f_lower / df) if f_lower else 0
                high_idx = int(np.ceil(f_upper / df)) if f_upper else len(psd)

                h1 = h1_tilde.data[low_idx:high_idx]
                h2 = h2_tilde.data[low_idx:high_idx]
                psd_vals = psd.data[low_idx:high_idx]
                psd_vals[np.isinf(psd_vals)] = 1.0

                total_norm1_sq += 4 * df * np.sum((np.abs(h1) ** 2) / psd_vals)
                total_norm2_sq += 4 * df * np.sum((np.abs(h2) ** 2) / psd_vals)
                total_inner_prod += 4 * df * np.sum((h1 * np.conj(h2)) / psd_vals)

            if total_norm1_sq == 0 or total_norm2_sq == 0:
                return 1.0

            overlap = np.abs(total_inner_prod) / np.sqrt(
                total_norm1_sq * total_norm2_sq
            )

            return 1.0 - overlap

        x0 = [0.0, 0.0, 0.0, 0.0, 0.0]
        result = minimize(objective_function, x0, method="Nelder-Mead")

        return 1.0 - result.fun

    def match_sphere_averaged_bms_maximized(
        self,
        other,
        psd,
        f_lower,
        f_upper=None,
        j_max=1,
    ):
        """Calculates the match between this waveform and another, maximizing
        over BMS supertranslations.

        This function extends `match_sphere_averaged` by also optimizing over
        the coefficients of the BMS supertranslation field, α(θ, φ), applied
        to `self`. This accounts for gauge differences between numerical
        relativity codes, particularly center-of-mass drifts (j=1 modes).

        Parameters
        ----------
        other : Waveform
            The other waveform object to compare against.
        psd : pycbc.types.FrequencySeries
            The one-sided power spectral density (PSD) of the detector noise.
        f_lower : float
            The lower frequency cutoff for the match integral, in Hz.
        f_upper : float, optional
            The upper frequency cutoff for the match integral, in Hz.
            If None, it defaults to the Nyquist frequency of the waveforms.
        j_max : int, optional
            The maximum spherical-harmonic order `j` of the supertranslation
            field to optimize over. Defaults to 1, which corresponds to
            center-of-mass corrections.

        Returns
        -------
        float
            The maximum match value, a float between 0 and 1.

        Notes
        -----
        The supertranslated waveform modes are computed in the frequency domain via:

        h̃'_lm(f) = h̃_lm(f) - Σ_{j,k,p,q} α_jk * G^{lm}_{jk,pq} * h̃̇_pq(f)

        where `α_jk` are the supertranslation coefficients, `G` is the Gaunt
        integral computed by `scri.coupling_coefficients`, and `h̃̇` is the
        Fourier transform of the time-derivative of the waveform mode.
        The optimization is performed over `(t_c, φ_c, R, α_jk)`.
        """
        from scipy.optimize import minimize
        import scri

        # Determine the supertranslation coefficients to be optimized
        alpha_jk_indices = []
        for j in range(1, j_max + 1):
            for k in range(-j, j + 1):
                alpha_jk_indices.append((j, k))
        
        # Pre-compute all frequency-domain modes of `self` on a common grid
        # to handle mode-mixing.
        max_len = 0
        for l, m in self.LM:
            max_len = max(max_len, len(self.get_mode(l, m, to_pycbc=True, delta_t=1/4096)))
        
        # Use a reference mode to define the frequency grid
        ref_mode_ts = self.get_mode(2, 2, to_pycbc=True, delta_t=1/4096)
        ref_mode_ts.resize(max_len)
        ref_fs = ref_mode_ts.to_frequencyseries()
        freqs = ref_fs.sample_frequencies
        delta_f = ref_fs.delta_f

        self_modes_tilde = {}
        self_modes_dot_tilde = {}
        for l, m in self.LM:
            h_ts = self.get_mode(l, m, to_pycbc=True, delta_t=1/4096)
            h_ts.resize(max_len)
            h_tilde = h_ts.to_frequencyseries(delta_f=delta_f)
            self_modes_tilde[(l, m)] = h_tilde
            
            h_dot_tilde = h_tilde.copy()
            h_dot_tilde.data *= 1j * 2 * np.pi * freqs
            self_modes_dot_tilde[(l, m)] = h_dot_tilde


        def objective_function(x):
            # Unpack parameters: 5 for rigid transformations, rest for BMS
            time_shift, phi_c, alpha, beta, gamma = x[:5]
            alpha_jk_values = x[5:]
            alpha_jk_coeffs = dict(zip(alpha_jk_indices, alpha_jk_values))

            R = quaternionic.array.from_euler_angles(alpha, beta, gamma)
            other_rot = other.rotated(R)

            total_inner_prod = 0.0
            total_norm1_sq = 0.0
            total_norm2_sq = 0.0

            common_modes = set(map(tuple, self.LM)) & set(map(tuple, other_rot.LM))

            # Pre-compute supertranslated modes for `self`
            self_modes_tilde_st = {}
            for l, m in common_modes:
                h1_tilde = self_modes_tilde[(l,m)]
                st_correction = np.zeros_like(h1_tilde.data, dtype=complex)
                
                for (j, k), alpha_jk in alpha_jk_coeffs.items():
                    for p, q in self.LM:
                        # G = ∫ Y*_{l,m} Y_{j,k} Y_{p,q} dΩ
                        G = scri.coupling_coefficients(s_prime=-2, l_prime=l, m_prime=m,
                                                     s1=0, l1=j, m1=k,
                                                     s2=-2, l2=p, m2=q)
                        if G == 0:
                            continue
                        h_dot_pq = self_modes_dot_tilde[(p, q)]
                        st_correction += alpha_jk * G * h_dot_pq.data
                
                h1_tilde_st = h1_tilde.copy()
                h1_tilde_st.data -= st_correction
                self_modes_tilde_st[(l,m)] = h1_tilde_st

            for l, m in common_modes:
                h1_tilde = self_modes_tilde_st[(l, m)]
                h2_mode_ts = other_rot.get_mode(l, m, to_pycbc=True, delta_t=1 / 4096)

                # Align lengths
                h2_mode_ts.resize(max_len)
                h2_tilde = h2_mode_ts.to_frequencyseries(delta_f=delta_f)
                
                temp_psd = psd.copy()
                temp_psd.resize(len(h1_tilde))

                # Apply phase and time shifts to the second waveform
                h2_tilde *= np.exp(-1j * m * phi_c)
                h2_tilde.data *= np.exp(-2j * np.pi * freqs * time_shift)

                df = delta_f
                low_idx = int(f_lower / df) if f_lower else 0
                high_idx = int(np.ceil(f_upper / df)) if f_upper else len(temp_psd)

                h1 = h1_tilde.data[low_idx:high_idx]
                h2 = h2_tilde.data[low_idx:high_idx]
                psd_vals = temp_psd.data[low_idx:high_idx]
                psd_vals[np.isinf(psd_vals)] = 1.0

                total_norm1_sq += 4 * df * np.sum((np.abs(h1) ** 2) / psd_vals)
                total_norm2_sq += 4 * df * np.sum((np.abs(h2) ** 2) / psd_vals)
                total_inner_prod += 4 * df * np.sum((h1 * np.conj(h2)) / psd_vals)

            if total_norm1_sq == 0 or total_norm2_sq == 0:
                return 1.0

            overlap = np.abs(total_inner_prod) / np.sqrt(
                total_norm1_sq * total_norm2_sq
            )

            return 1.0 - overlap

        x0 = [0.0] * (5 + len(alpha_jk_indices))
        result = minimize(objective_function, x0, method="Nelder-Mead")

        return 1.0 - result.fun


def interpolate_in_amp_phase(obj, new_time, k=3, kind=None):
    """Interpolate in amplitude and phase
    using a variety of interpolation methods.

    Paramters
    ---------
    obj: sxs.TimeSeries
        The TimeSeries object that holds the complex
        waveform to be interpolated.
    new_time: array_like
          The new time axis to interpolate onto.

    k: int, optional
       The order of interpolation when
        `scipy.interpolated.InterpolatedUnivariateSpline` is used.
        This gets preference over `kind` parameter when both are
        specified. The default is 3.

    kind: str, optional
        The interpolation kind parameter when `scipy.interpolate.interp1d`
        is used. Can be `linear`, `quadratic` or `cubic` for`scipy.interpolate.interp1d`,
        or 'CubicSpline' to use `scipy.interpolate.CubicSpline`. Default is None
        i.e. the parameter `k` will be used instead.
    See Also
    --------
    waveformtools.waveformtools.interp_resam_wfs :
        The function that interpolates in amplitude
        and phases using scipy interpolators.

    scipy.interpolate.CubicSpline:
        One of the possible methods that can
        be used for interpolation.
    scipy.interpolate.interp1d:
        Can be used in linear, quadratic and cubic mode.
    scipy.interpolate.InterpolatedUnivariateSpline:
        Can be used with orders k from 1 to 5.

    Notes
    -----
    These interpolation methods ensure that the
    interpolated function passes through all the
    data points.
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
