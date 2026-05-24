import os

import h5py
import lal
import numpy as np
import quaternionic
import spherical
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
        # Extract _filepath before passing w_attributes to the parent so it
        # becomes a per-instance attribute, not a class-level one.
        filepath = w_attributes.pop("_filepath", None)
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
        self._filepath = filepath  # instance attribute, not class attribute
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

        # Record the file path so it can be set as a per-instance attribute
        # in __new__ (passed via w_attributes["_filepath"]).
        h5_filepath = h5_file.filename

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
        w_attributes["_filepath"] = h5_filepath  # stored as instance attr in __new__
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

        # _filepath is passed through w_attributes so __new__ sets it as
        # a per-instance attribute rather than a shared class-level one.

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
        w_attributes["_filepath"] = file_path  # stored as instance attr in __new__
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

    def _get_label_params(self):
        """Extract mass ratio and spin components from metadata in a
        catalog-agnostic way. Returns (q, s1x, s1y, s1z, s2x, s2y, s2z)
        using whichever metadata keys are present (RIT, MAYA, or SXS)."""
        md = self.metadata
        try:
            # RIT catalog
            if "relaxed_mass_ratio_1_over_2" in md:
                q = md["relaxed_mass_ratio_1_over_2"]
                s1x, s1y, s1z = md.get("relaxed_chi1x", float("nan")), md.get("relaxed_chi1y", float("nan")), md.get("relaxed_chi1z", float("nan"))
                s2x, s2y, s2z = md.get("relaxed_chi2x", float("nan")), md.get("relaxed_chi2y", float("nan")), md.get("relaxed_chi2z", float("nan"))
            # MAYA / GT catalog
            elif "q" in md:
                q = md["q"]
                s1x, s1y, s1z = md.get("a1x", float("nan")), md.get("a1y", float("nan")), md.get("a1z", float("nan"))
                s2x, s2y, s2z = md.get("a2x", float("nan")), md.get("a2y", float("nan")), md.get("a2z", float("nan"))
            # SXS catalog
            elif "reference_mass_ratio" in md:
                q = md["reference_mass_ratio"]
                sp1 = md.get("reference_dimensionless_spin1", [float("nan")] * 3)
                sp2 = md.get("reference_dimensionless_spin2", [float("nan")] * 3)
                s1x, s1y, s1z = sp1[0], sp1[1], sp1[2]
                s2x, s2y, s2z = sp2[0], sp2[1], sp2[2]
            else:
                q = float("nan")
                s1x = s1y = s1z = s2x = s2y = s2z = float("nan")
        except Exception:
            q = float("nan")
            s1x = s1y = s1z = s2x = s2y = s2z = float("nan")
        return q, s1x, s1y, s1z, s2x, s2y, s2z

    @property
    def label(self):
        """Return a LaTeX label summarizing key simulation parameters.
        Works for RIT, MAYA, and SXS catalogs."""
        q, s1x, s1y, s1z, s2x, s2y, s2z = self._get_label_params()
        return (
            f"$q{q:0.3f}\\_"
            f"\\chi_A{s1x:0.3f}\\_{s1y:0.3f}\\_{s1z:0.3f}\\_\\_"
            f"\\chi_B{s2x:0.3f}\\_{s2y:0.3f}\\_{s2z:0.3f}$"
        )

    @property
    def label_nolatex(self):
        """Return a plain-text label summarizing key simulation parameters.
        Works for RIT, MAYA, and SXS catalogs."""
        q, s1x, s1y, s1z, s2x, s2y, s2z = self._get_label_params()
        return (
            f"q{q:0.3f}-sA{s1x:0.3f}-{s1y:0.3f}-{s1z:0.3f}"
            f"--sB{s2x:0.3f}-{s2y:0.3f}-{s2z:0.3f}"
        )

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
        """Return a single (ℓ, m) waveform mode, rescaled to physical units.

        The raw NR mode data in ``self`` is in **dimensionless units**
        (amplitude in units of *r·c²/G·M*, time in units of *G·M/c³*).
        This method rescales to physical strain and returns a time series
        aligned so that **t = 0 corresponds to the peak of the (2,2) mode**.

        Parameters
        ----------
        ell : int
            Spherical-harmonic ℓ index.
        em : int
            Spherical-harmonic m index.
        total_mass : float, optional
            Total mass of the binary in **solar masses** (M☉).  Default 1.
        distance : float, optional
            Luminosity distance in **megaparsecs** (Mpc).  Default 1.
        delta_t : float, optional
            Desired sample spacing.  Two conventions are supported:

            * **delta_t > 1/128** — interpreted as **dimensionless M units**
              (typical NR time steps are 0.1–1 M).
            * **delta_t ≤ 1/128** — interpreted as **physical seconds**
              (e.g. ``1/4096`` s for detector-band data).

            If *None*, the native NR time step is used (dimensionless M units).
        to_pycbc : bool, optional
            If *True* (default) return a ``pycbc.types.TimeSeries``; otherwise
            return an ``sxs.TimeSeries``.

        Returns
        -------
        pycbc.types.TimeSeries or sxs.TimeSeries
            Complex time series ``h_{ℓm}(t)`` in **physical strain units**
            (dimensionless).  The ``delta_t`` of the returned series is always
            in **physical seconds**, regardless of the input convention.
            The epoch (``start_time``) is set so that t = 0 is the peak of the
            (2,2) mode amplitude.

        Notes
        -----
        Amplitude scaling follows::

            h_{ℓm}^{phys} = h_{ℓm}^{NR} × (G M_tot / c² r)
                           = h_{ℓm}^{NR} × (M_tot [M☉] × MRSUN_SI) / (distance [Mpc] × MPC_SI)

        To obtain the real (plus) polarization at a given sky location and
        inclination, sum over modes with spin-weight −2 spherical harmonics.
        For a quick (2,2)-only approximation::

            h_plus ≈ mode.real()
        """
        if delta_t is None:
            delta_t = stat_mode(np.diff(self.time), keepdims=True)[0][0]

        # Convert between physical seconds and dimensionless M units.
        # self.time is always in dimensionless M units; new_time must match.
        # We assume delta_t > 1/128 means it is in dimensionless M units
        # (typical NR step ~ 0.1-1 M), and delta_t <= 1/128 means it is in
        # physical seconds (e.g. 1/4096 s for GW detector sampling).
        m_secs = utils.time_to_physical(total_mass)
        if delta_t > 1.0 / 128:
            # delta_t is in dimensionless M units
            dt_dimless = delta_t
            dt_physical = delta_t * m_secs
        else:
            # delta_t is in physical seconds
            dt_physical = delta_t
            dt_dimless = delta_t / m_secs

        new_time = np.arange(min(self.time), max(self.time), dt_dimless)

        # --- OPTIMIZATION: Interpolate only the requested mode ---
        # sxs.WaveformModes uses .index(l, m) to get the column index.
        # Explicitly convert to a writable numpy array (sxs may return a memoryview).
        mode_data = np.array(self.data[:, self.index(ell, em)], dtype=complex)
        mode_ts = sxs_TimeSeries(mode_data, time=self.time)
        interpolated_mode_ts = mode_ts.interpolate(new_time)

        h_mode_complex = np.array(interpolated_mode_ts.data, dtype=complex)
        h_mode_complex *= utils.amp_to_physical(total_mass, distance)

        # --- OPTIMIZATION: Use cached peak time of (2,2) mode to set epoch ---
        # The epoch is set to shift the time axis so that the peak is at t=0.
        # peak_time_22 is in dimensionless M units; convert to physical seconds.
        peak_time_sec = self.peak_time_22 * m_secs
        start_time_sec = new_time[0] * m_secs
        epoch = start_time_sec - peak_time_sec

        retval = self.to_pycbc(
            input_array=h_mode_complex,
            delta_t=dt_physical,
            epoch=epoch,
        )
        if not to_pycbc:
            retval = sxs_TimeSeries(retval.data, time=retval.sample_times)
        return retval

    def f_lower_at_1Msun(self, t=None):
        """Return the instantaneous GW frequency of the (2,2) mode, scaled to
        a total mass of **1 solar mass**.

        The raw NR time axis (``self.time``) is in **dimensionless M units**.
        The returned frequency is likewise for a 1 M☉ system; to obtain the
        physical frequency for a system with total mass *M* (in solar masses),
        divide the result by *M*::

            f_lower_hz = wfm.f_lower_at_1Msun(t=t_dimless) / total_mass_msun

        Parameters
        ----------
        t : float or None, optional
            Evaluation time in **dimensionless M units** (same grid as
            ``self.time``).  If *None* the frequency at the first sample
            (start of the waveform) is returned.

        Returns
        -------
        float
            GW frequency in Hz evaluated at 1 M☉ (i.e. in units of
            ``1 / (M☉ × G/c³)``).  Divide by ``total_mass`` [M☉] to get
            the physical frequency in Hz.

        Notes
        -----
        The frequency is estimated from the (2,2) mode via
        ``pycbc.waveform.utils.frequency_from_polarizations``, using the raw
        (dimensionless) time step ``Δt = np.diff(self.time)[0]``.  The
        conversion to physical units uses ``lal.MTSUN_SI`` (seconds per solar
        mass).

        Examples
        --------
        >>> # Frequency at the start of the waveform for a 60 M☉ binary
        >>> f_start = wfm.f_lower_at_1Msun() / 60.0   # Hz
        >>>
        >>> # Frequency at a specific dimensionless time (e.g. relaxation time)
        >>> t_relax_dimless = wfm.time[0] + metadata["relaxed-time"]
        >>> f_relax = wfm.f_lower_at_1Msun(t=t_relax_dimless) / 60.0  # Hz
        """
        mode22 = self.get_mode_data(2, 2)
        fr22 = frequency_from_polarizations(
            TimeSeries(mode22[:, 1], delta_t=np.diff(self.time)[0]),
            TimeSeries(-1 * mode22[:, 2], delta_t=np.diff(self.time)[0]),
        )
        # Ensure positive frequency regardless of catalog phase convention.
        # SXS stores h_{lm} = A·exp(−iΦ) while RIT/MAYA store A·exp(+iΦ),
        # so frequency_from_polarizations can return negative values for SXS.
        fr22 = np.abs(fr22)
        # If time value is not provided, return the initial f_lower
        if t is None:
            return float(fr22[0] / lal.MTSUN_SI)
        # Interpolate fr22 as a function of time and evaluate at t
        sample_times = self.time[: len(fr22)]
        interp_fr22 = InterpolatedUnivariateSpline(sample_times, fr22, k=3)
        return float(interp_fr22(t) / lal.MTSUN_SI)

    def _get_relaxation_time_dimless(self):
        """Return the relaxation time in dimensionless M units from metadata.

        Tries the following metadata keys in order:
        - ``'relaxed-time'`` (RIT catalog)
        - ``'relaxation_time'`` (SXS catalog)
        - ``'reference_time'`` (SXS catalog fallback)

        Returns 0.0 if no relaxation time is found.
        """
        md = self.sim_metadata
        for key in ("relaxed-time", "relaxation_time", "reference_time"):
            if key in md and md[key] is not None:
                return float(md[key])
        return 0.0

    def trim_to_relaxation_time(self, total_mass, delta_t=1.0 / 4096):
        """Return the (2,2) mode trimmed to start at the relaxation epoch.

        The relaxation time marks the epoch after which the NR simulation has
        settled from its initial-data transient.  Comparisons against waveform
        models should use data starting from this time.

        Parameters
        ----------
        total_mass : float
            Total mass of the binary (solar masses).
        delta_t : float, optional
            Sample spacing in seconds (default 1/4096).

        Returns
        -------
        pycbc.types.TimeSeries
            Complex (2,2) mode starting at the relaxation time.
        """
        t_relax = self._get_relaxation_time_dimless()
        mode = self.get_mode(2, 2, total_mass=total_mass, distance=1.0, delta_t=delta_t)
        t_start = mode.start_time
        t_relax_phys = t_relax * utils.time_to_physical(total_mass)
        idx = 0
        for i, t in enumerate(mode.sample_times):
            if t >= t_start + t_relax_phys:
                idx = i
                break
        return mode[idx:]

    def f_lower_at_relaxation(self, total_mass):
        """Return the GW frequency at the relaxation epoch, in Hz.

        Parameters
        ----------
        total_mass : float
            Total mass of the binary (solar masses).

        Returns
        -------
        float
            The instantaneous GW frequency (from the (2,2) mode) at the
            relaxation time, in Hz.
        """
        t_relax = self._get_relaxation_time_dimless()
        t_eval = self.time[0] + t_relax  # both in dimensionless M units
        return self.f_lower_at_1Msun(t=t_eval) / total_mass

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
        lal_convention=False,
    ):
        """Sum over modes and return GW polarizations rescaled to physical units.

        The waveform modes are summed using spin-weight −2 spherical harmonics
        evaluated at the observer angles (θ, ψ, α) computed from the NR source
        frame, following DCC-T1600045:

            h₊ − i h× = Σ_{ℓm} h_{ℓm} · ⁻²Y_{ℓm}(θ, ψ, α)

        The returned complex PyCBC TimeSeries has the convention:

            retval.real() = h₊
            retval.imag() = h×      (note: positive h×, i.e. conjugated)

        This differs from the LAL convention where ``hp - i*hc`` is returned
        with separate real TimeSeries objects.  Set ``lal_convention=True`` to
        get a complex TimeSeries whose imaginary part equals −h× (matching
        ``lalsimulation`` output), i.e. retval = h₊ − i h×.

        Parameters
        ----------
        total_mass : float
            Total mass of the binary (solar masses).
        distance : float
            Luminosity distance (megaparsecs).
        inclination : float
            Inclination angle between the line-of-sight and the orbital
            angular momentum (radians).
        coa_phase : float
            Coalescence orbital phase (radians).
        delta_t : float, optional
            Sample spacing.  Values > 1/128 are interpreted as dimensionless
            M units; values ≤ 1/128 are interpreted as physical seconds.
            Defaults to the native NR time step.
        f_ref : float, optional
            Reference gravitational-wave frequency (Hz).
        t_ref : float, optional
            Reference time (dimensionless M units).
        k : int, optional
            Spline interpolation order (default 3).
        kind : str, optional
            Alternative interpolation method: ``'linear'``, ``'quadratic'``,
            ``'cubic'``, or ``'CubicSpline'``.
        tol : float, optional
            Floating-point tolerance for rotation angle computation (1e-6).
        lal_convention : bool, optional
            If True, return h₊ − i h× (imaginary part = −h×), matching
            ``lalsimulation`` conventions.  Default False returns h₊ + i h×
            (imaginary part = +h×).

        Returns
        -------
        pycbc.types.TimeSeries (complex128)
            Complex waveform.  With default ``lal_convention=False``:
            ``retval.real() = h₊``, ``retval.imag() = h×``.
        """
        if delta_t is None:
            delta_t = stat_mode(np.diff(self.time), keepdims=True)[0][0]
        m_secs = utils.time_to_physical(total_mass)
        # Values > 1/128 are treated as dimensionless M units; <= 1/128 as seconds.
        if delta_t > 1.0 / 128:
            dt_dimless = delta_t
        else:
            dt_dimless = delta_t / m_secs
        new_time = np.arange(min(self.time), max(self.time), dt_dimless)

        # Get angles
        angles = self.get_angles(
            inclination=inclination,
            coa_phase=coa_phase,
            f_ref=f_ref,
            t_ref=t_ref,
            tol=tol,
        )
        # evaluate() returns h₊ − i h× (sxs convention)
        h = interpolate_in_amp_phase(
            self.evaluate([angles["theta"], angles["psi"], angles["alpha"]]),
            new_time,
            k=k,
            kind=kind,
        ) * utils.amp_to_physical(total_mass, distance)

        h.time *= m_secs

        if lal_convention:
            # Return h₊ − i h× so that real()=h₊, imag()=−h× (LAL convention)
            return self.to_pycbc(h)
        else:
            # Return conjugate: h₊ + i h× so that real()=h₊, imag()=+h×
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

        # Use the raw (non-interpolated) data of the object
        try:
            mode22_idx = self.index(2, 2)
        except ValueError:
            # Fallback if 2,2 mode is not present, though it is standard.
            self._peak_time_22 = 0.0
            return self._peak_time_22

        mode22_data = np.array(self.data[:, mode22_idx], dtype=complex)
        amp22 = np.abs(mode22_data)

        # Find the peak of the amplitude via np.argmax.  The spline-derivative
        # approach is fragile for oscillatory waveforms (many local maxima fool
        # the root finder into picking the wrong one).
        x_axis = np.array(self.time)
        self._peak_time_22 = float(x_axis[np.argmax(amp22)])
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

    def match_single_mode(
        self,
        other,
        ell,
        em,
        psd,
        f_lower,
        delta_t=1.0 / 4096,
        f_upper=None,
    ):
        """Compute the noise-weighted match for a single spherical harmonic mode.

        Maximizes over time shift t₀ and phase shift φ₀:

            M_{ℓm} = max_{t₀,φ₀} |⟨h₁_{ℓm} | h₂_{ℓm}(t₀,φ₀)⟩|
                      / sqrt(⟨h₁_{ℓm}|h₁_{ℓm}⟩ · ⟨h₂_{ℓm}|h₂_{ℓm}⟩)

        The mode-dependent lower frequency cutoff ``f_lower * |m| / 2`` is
        applied automatically (the (ℓ,m) mode oscillates at frequency
        |m|/2 × orbital frequency, so the GW frequency for mode m is
        |m|/2 × f_orb).

        Parameters
        ----------
        other : WaveformModes or dict
            The second waveform.  Can be a ``WaveformModes`` object or a dict
            of ``{(l, m): pycbc.types.TimeSeries}`` as returned by
            ``pycbc.waveform.get_td_waveform_modes``.
        ell : int
            Spherical harmonic ℓ index.
        em : int
            Spherical harmonic m index.
        psd : pycbc.types.FrequencySeries
            One-sided noise power spectral density.
        f_lower : float
            Orbital reference frequency in Hz (the mode cutoff
            ``f_lower * |m| / 2`` is applied internally).
        delta_t : float, optional
            Sample spacing in physical seconds (default 1/4096).
        f_upper : float, optional
            Upper frequency cutoff in Hz (Nyquist if None).

        Returns
        -------
        float
            Match value in [0, 1].
        """
        from pycbc.filter import match as pycbc_match

        h1 = self.get_mode(ell, em, to_pycbc=True, delta_t=delta_t).real()

        if isinstance(other, dict):
            # dict from pycbc.waveform.get_td_waveform_modes: values are (re, im)
            if (ell, em) not in other:
                raise KeyError(f"Mode ({ell}, {em}) not found in other waveform dict.")
            val = other[(ell, em)]
            h2 = val[0] if isinstance(val, (tuple, list)) else val.real()
        else:
            h2 = other.get_mode(ell, em, to_pycbc=True, delta_t=delta_t).real()

        # Pad to same length
        target_len = max(len(h1), len(h2))
        h1.resize(target_len)
        h2.resize(target_len)

        psd_copy = psd.copy()
        psd_copy.resize(len(h1.to_frequencyseries()))

        mode_f_lower = f_lower * abs(em) / 2.0 if em != 0 else f_lower

        mm, _ = pycbc_match(
            h1,
            h2,
            psd=psd_copy,
            low_frequency_cutoff=mode_f_lower,
            high_frequency_cutoff=f_upper,
        )
        return float(mm)

    def match_sphere_averaged(
        self,
        other,
        psd,
        f_lower,
        f_upper=None,
        delta_t=1.0 / 4096,
    ):
        """Calculates the match between this waveform and another.

        The match is a measure of similarity between two waveforms, maximized over
        extrinsic parameters (time shift, phase shift, and spatial orientation).
        It is defined as the noise-weighted inner product, normalized by the
        waveform norms:

        Overlap(h₁, h₂) = |(h̃₁ | h̃₂)| / sqrt( (h̃₁ | h̃₁) · (h̃₂ | h̃₂) )

        The noise-weighted inner product is:

        (h̃₁ | h̃₂) = 4 Re ∫ h̃₁*(f) h̃₂(f; t₀, φ₀, R) / Sₙ(f) df

        Maximization is over:
        - t₀ : time shift
        - φ₀ : phase shift
        - R ∈ SO(3) : 3D rotation (Euler angles α, β, γ)

        Parameters
        ----------
        other : WaveformModes
            The other waveform to compare against.
        psd : pycbc.types.FrequencySeries
            One-sided noise power spectral density.
        f_lower : float
            Lower frequency cutoff in Hz.
        f_upper : float, optional
            Upper frequency cutoff in Hz (Nyquist if None).
        delta_t : float, optional
            Sample spacing in physical seconds (default 1/4096).

        Returns
        -------
        float
            Maximum match value in [0, 1].
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
                h1_mode_ts = self.get_mode(l, m, to_pycbc=True, delta_t=delta_t)
                h2_mode_ts = other_rot.get_mode(
                    l, m, to_pycbc=True, delta_t=delta_t
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
        **BMS supertranslation formula:**

        A supertranslation α(θ,φ) = Σ_{jk} α_{jk} Y_{jk}(θ,φ) acts on the
        strain modes as (arXiv:1002.1727, eq. 11):

            h'_{ℓm} = h_{ℓm} - Σ_{j,k,p,q} α_{jk} G^{ℓm}_{jk;pq} ḣ_{pq}

        where the spin-weighted Gaunt coefficient is:

            G^{ℓm}_{jk;pq} = ∫ ⁻²Ȳ_{ℓm} · Y_{jk} · ⁻²Y_{pq} dΩ

        In the frequency domain, ḣ_{pq}(f) = i·2πf · h̃_{pq}(f), which is the
        multiplication applied in ``self_modes_dot_tilde``.

        The ``scri.coupling_coefficients`` call uses ``s_prime=-2`` for the
        left spin-weight factor (the conjugated ⁻²Y_{ℓm}), ``s1=0`` for the
        scalar Y_{jk}, and ``s2=-2`` for ⁻²Y_{pq}`, which matches the formula.

        The optimization is performed over ``(t_c, φ_c, R, α_jk)`` where R is
        parameterized by Euler angles (α_E, β_E, γ_E) via Nelder-Mead.
        Requires the ``scri`` package (``pip install scri``).
        """
        from scipy.optimize import minimize
        try:
            import scri
        except ImportError as e:
            raise ImportError(
                "The 'scri' package is required for BMS supertranslation optimization. "
                "Install it with: pip install scri"
            ) from e

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
        length).  The dict is typically the output of
        ``pycbc.waveform.get_td_waveform_modes`` or ``gwsurrogate``.
    R : quaternionic.array
        Unit quaternion representing the rotation.
    ell_max : int, optional
        Maximum ℓ to include (default 4).

    Returns
    -------
    dict
        Rotated mode dictionary with the same ``(l, m)`` keys and the same
        value type (TimeSeries or ndarray) as the input.
    """
    wigner = spherical.Wigner(ell_max)

    # Group modes by ℓ
    by_ell = {}
    for (l, m), val in mode_dict.items():
        if l > ell_max:
            continue
        by_ell.setdefault(l, {})[m] = val

    rotated = {}
    for l, m_dict in by_ell.items():
        # Build matrix of shape (n_times, 2l+1) in m order: -l … +l
        m_vals = list(range(-l, l + 1))

        # Determine output type and length from the first available mode
        first = next(iter(m_dict.values()))
        is_timeseries = hasattr(first, "delta_t")
        n = len(first)

        # Assemble input block; zero-pad missing m values
        block = np.zeros((n, 2 * l + 1), dtype=complex)
        for i, mv in enumerate(m_vals):
            if mv in m_dict:
                block[:, i] = np.asarray(m_dict[mv])

        # D-matrix shape: (2l+1, 2l+1), rows = output m, cols = input m
        D = wigner.D(R, l)  # shape (2l+1, 2l+1)

        # rotated_block[t, m_out] = Σ_{m_in} D[m_out, m_in] block[t, m_in]
        rotated_block = block @ D.T  # (n, 2l+1)

        for i, mv in enumerate(m_vals):
            if is_timeseries:
                rotated[(l, mv)] = type(first)(
                    rotated_block[:, i], delta_t=first.delta_t, epoch=first.start_time
                )
            else:
                rotated[(l, mv)] = rotated_block[:, i]

    return rotated


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
