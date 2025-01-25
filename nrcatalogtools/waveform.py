import os

import h5py
import lal
import numpy as np
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
        from waveformtools.dataIO import load_RIT_Psi4_data_from_disk

        wftools_modes_array = load_RIT_Psi4_data_from_disk(
            data_file_path=file_path,
            output_modes_array=True,
            resam_type="finest",
        )

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
            wftools_modes_array._modes_data.T[:, ELL_MIN**2 :],
            time=wftools_modes_array.time_axis,
            time_axis=0,
            modes_axis=1,
            ell_min=ELL_MIN,
            ell_max=wftools_modes_array.ell_max,
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
        """Return the initial physical parameters for the simulation. Only for
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

        h = self.interpolate(new_time)

        h_mode = h.get_mode_data(ell, em)
        h_mode[:, 1:] *= utils.amp_to_physical(total_mass, distance)
        h_mode[:, 0] *= m_secs

        # Find peak of 22-mode
        h_mode22 = h.get_mode_data(2, 2)
        h_mode22[:, 0] *= m_secs

        from scipy.interpolate import InterpolatedUnivariateSpline

        x_axis = h_mode22[:, 0]
        y_axis = (h_mode22[:, 1] ** 2 + h_mode22[:, 2] ** 2) ** 0.5

        f = InterpolatedUnivariateSpline(x_axis, y_axis, k=4)
        cr_pts = f.derivative().roots()
        cr_pts = np.append(
            cr_pts, (x_axis[0], x_axis[-1])
        )  # also check the endpoints of the interval
        cr_vals = f(cr_pts)
        max_index = np.argmax(cr_vals)

        epoch = h_mode[0, 0] - cr_pts[max_index]

        retval = self.to_pycbc(
            input_array=h_mode[:, 1] + 1j * h_mode[:, 2],
            delta_t=delta_t,
            epoch=epoch,
        )
        if not to_pycbc:
            retval = sxs_TimeSeries(retval.data, time=retval.sample_times)
        return retval

    @property
    def f_lower_at_1Msun(self):
        mode22 = self.get_mode_data(2, 2)
        fr22 = frequency_from_polarizations(
            TimeSeries(mode22[:, 1], delta_t=np.diff(self.time)[0]),
            TimeSeries(-1 * mode22[:, 2], delta_t=np.diff(self.time)[0]),
        )
        return fr22[0] / lal.MTSUN_SI

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

    def fixed_frequency_integration(self, omega_cutoff="auto", order=1):
        """Integrate the modes array using fixed frequency
        integration technique to obtain the News/strain from
        Psi4

        Parameters
        ----------
        omega_cutoff: float
                      The lower cutoff angular frequency to use
                      for FFI.
        order: int
               Integration order

        Returns
        -------
        obj: sxs_WaveformModes
             The wawveform modes representing the integrated
             quantity

        """

        from waveformtools.integrate import fixed_frequency_integrator

        original_times = self.time
        d_times = np.diff(original_times)

        if not (np.diff(d_times) == 0).all():
            raise ValueError("The time axis is not linearly sampled!")
        else:
            dt = d_times[0]

        idx = 0

        created = False
        for ell in range(self.ell_min, self.ell_max + 1):
            for emm in range(-ell, ell + 1):
                times, integrated_mode = fixed_frequency_integrator(
                    udata_time=np.array(self.get_mode(ell, emm)),
                    delta_t=dt,
                    order=order,
                    omega0=omega_cutoff,
                )

                if not created:
                    integrated_wfmodes_data = np.zeros(
                        (len(times), self.n_modes), dtype=np.complex128
                    )
                    created = True

                integrated_wfmodes_data[:, idx] = integrated_mode
                idx += 1

        w_attributes = {}
        w_attributes["metadata"] = self.metadata
        w_attributes["history"] = f"FFI_{order}"
        w_attributes["frame"] = self.frame
        w_attributes["frame_type"] = self.frame_type
        w_attributes["data_type"] = order * "int_" + self.data_type
        w_attributes["spin_weight"] = self.spin_weight
        w_attributes["r_is_scaled_out"] = True
        w_attributes["m_is_scaled_out"] = True
        # w_attributes["ells"] = ell_min, ell_max

        return WaveformModes(
            integrated_wfmodes_data,
            time=times,
            time_axis=0,
            modes_axis=1,
            ell_min=self.ell_min,
            ell_max=self.ell_max,
            verbosity=self.verbosity,
            **w_attributes,
        )


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
