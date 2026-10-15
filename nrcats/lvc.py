"""Frame-rotation helpers and LVCNR format utilities.

This module provides the low-level functions needed to:

1. **Read LVCNR HDF5 files** – parse mode data and strain from the LVK
   NR format (``get_modes_from_lvcnr_file``, ``get_strain_from_lvcnr_file``).

2. **Compute the LAL source frame** – derive the NR orbital normal
   (``LNhat``) and binary-separation unit vector (``nhat``) from HDF5
   file attributes, time-series data, or SXS-style simulation metadata
   (``compute_lal_source_frame_from_sxs_metadata``,
   ``compute_lal_source_frame_by_interp``).

3. **Rotate NR waveforms into the LAL frame** – compute the SO(3) rotation
   angles (``theta``, ``psi``, ``alpha``) needed to project the NR
   waveform onto the detector line of sight at a given inclination and
   reference phase (``get_nr_to_lal_rotation_angles``).

4. **Build LAL mode dictionaries** – create LAL parameter dictionaries
   with a specified set of (ell, m) modes for use with LALSimulation
   waveform routines (``get_lal_mode_dictionary``,
   ``get_lal_mode_dictionary_from_lmax``).

Public functions
----------------
get_lal_mode_dictionary(mode_array)
get_lal_mode_dictionary_from_lmax(lmax)
get_modes_from_lvcnr_file(path_to_file, Mtot, distance, srate, lmax, f_low)
get_strain_from_lvcnr_file(path_to_file, Mtot, distance, inclination, phi_ref, srate, mode_array)
check_interp_req(h5_file, metadata, ref_time, avail_ref_time)
get_lvcnr_format(h5_file)
read_lvcnr_group(h5_file, name)
check_fref(h5_file, f_ref)
get_ref_freq_from_ref_time(h5_file, ref_time)
get_ref_time_from_ref_freq(h5_file, ref_freq)
get_lvcnr_ref_time(h5_file, f_ref)
check_nr_attrs(sim_metadata_object, req_attrs)
get_interp_ref_values_from_h5_file(h5_file, req_ts_attrs, ref_time)
get_ref_vals(sim_metadata_object, req_attrs)
compute_lal_source_frame_from_sxs_metadata(sim_metadata)
compute_lal_source_frame_by_interp(h5_file, req_ts_attrs, t_ref)
normalize_metadata(sim_metadata)
get_ref_time_from_metadata(sim_metadata)
transform_spins_nr_to_lal(nrSpin1, nrSpin2, n_hat, ln_hat)
get_nr_to_lal_rotation_angles(h5_file, sim_metadata, inclination, phi_ref, f_ref, t_ref, tol, total_mass)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import h5py
import lal
import lalsimulation as lalsim
import numpy as np
from pycbc.pnutils import mtotal_eta_to_mass1_mass2
from pycbc.types import TimeSeries
from scipy.interpolate import CubicSpline


def get_lal_mode_dictionary(mode_array: list) -> object:
    """
    Get LALDict with all specified modes.


    Parameters
    ----------
    mode_array: list of modes, eg [[2,2], [3,3]]

    Returns
    -------
    waveform_dictionary: LALDict with all modes included

    """
    waveform_dictionary = lal.CreateDict()
    mode_array_lal = lalsim.SimInspiralCreateModeArray()
    for mode in mode_array:
        lalsim.SimInspiralModeArrayActivateMode(mode_array_lal, mode[0], mode[1])
    lalsim.SimInspiralWaveformParamsInsertModeArray(waveform_dictionary, mode_array_lal)

    return waveform_dictionary


def get_lal_mode_dictionary_from_lmax(lmax: int) -> object:
    r"""
    Get LALDict with modes derived from `lmax`.


    Parameters
    ----------
    lmax: max value of :math:`\ell` to use

    Returns
    -------
    waveform_dictionary: LALDict with all modes upto `lmax` included

    """

    mode_array = [
        [ell, emm] for ell in range(2, lmax + 1) for emm in range(-ell, ell + 1)
    ]
    return get_lal_mode_dictionary(mode_array)


def get_modes_from_lvcnr_file(
    path_to_file: str,
    Mtot: float,
    distance: float,
    srate: int,
    lmax: int = 4,
    f_low: float | None = None,
) -> tuple:
    r"""
    Get individual modes from LVCNR format file.


    Parameters
    ==========
    path_to_file: string
        Path to LVCNR file
    Mtot: float
        Total mass (in units of MSUN) to scale the waveform to
    distance: float
        Luminosity Distance (in units of MPc) to scale the waveform to
    srate: int
        Sampling rate for the waveform
    lmax: int
        Max value of :math:`\ell` to use
        (Default: 4)
    f_low: float
        Value of the low frequency to start waveform generation
        Uses value given from the LVCNR file if `None` is provided
        (Default: None)

    Returns
    =======
    mass_ratio: float
        Mass ratio derived from the LVCNR file
    spins_args: list
        List of spins derived from the LVCNR file
    eccentricity: float
        Eccentricty derived from the LVCNR file.
        Returns `None` is eccentricity is not a number.
    f_low: float
        Low Frequency derived either from the file, or provided
        in the call
    f_ref: float
        Reference Frequency derived from the file
    modes: dict of pycbc TimeSeries objects
        dict containing all the read in modes
    """

    with h5py.File(path_to_file) as h5file:
        waveform_dict = get_lal_mode_dictionary_from_lmax(lmax)

        f_low_in_file = h5file.attrs["f_lower_at_1MSUN"] / Mtot
        f_ref = f_low_in_file

        if f_low is None:
            f_low = f_low_in_file

        if h5file.attrs["Format"] < 3:
            spin_args = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(
                -1, Mtot, path_to_file
            )
        else:
            spin_args = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(
                f_low, Mtot, path_to_file
            )

        mass_args = list(
            mtotal_eta_to_mass1_mass2(Mtot * lal.MSUN_SI, h5file.attrs["eta"])
        )

        try:
            eccentricity = float(h5file.attrs["eccentricity"])
        except ValueError:
            eccentricity = None

    values_mode_array = lalsim.SimInspiralWaveformParamsLookupModeArray(waveform_dict)
    _, modes = lalsim.SimInspiralNRWaveformGetHlms(
        1 / srate,
        *mass_args,
        distance * 1e6 * lal.PC_SI,
        f_low,
        f_ref,
        *spin_args,
        path_to_file,
        values_mode_array,
    )
    mode = modes
    return_dict = dict()
    while 1 > 0:
        try:
            l, m = mode.l, mode.m
            read_mode = mode.mode.data.data
            return_dict[(l, m)] = TimeSeries(read_mode, 1 / srate)
            mode = mode.next
        except AttributeError:
            break

    return (
        mass_args[1] / mass_args[0],
        spin_args,
        eccentricity,
        f_low,
        f_ref,
        return_dict,
    )


def get_strain_from_lvcnr_file(
    path_to_file: str,
    Mtot: float,
    distance: float,
    inclination: float,
    phi_ref: float,
    srate: int,
    mode_array: list | None = None,
) -> tuple:
    """
    Get full strain from LVCNR format file.


    Parameters
    ==========
    path_to_file: string
        Path to LVCNR file
    Mtot: float
        Total mass (in units of MSUN) to scale the waveform to
    distance: float
        Luminosity Distance (in units of MPc) to scale the waveform to
    srate: int
        Sampling rate for the waveform
    mode_array: list
        list of modes to be included. `None` means all modes are included.
        (Default:None)

    Returns
    =======
    UNDER CONSTRUCTION
    """

    longAscNodes = 0
    eccentricity = 0
    meanPerAno = 0

    fixed_args = [
        distance * lal.PC_SI * 1e6,
        inclination,
        phi_ref,
        longAscNodes,
        eccentricity,
        meanPerAno,
        1 / srate,
    ]

    with h5py.File(path_to_file) as h5file:
        if mode_array is not None:
            waveform_dict = get_lal_mode_dictionary(mode_array)
        else:
            waveform_dict = lal.CreateDict()

        lalsim.SimInspiralWaveformParamsInsertNumRelData(waveform_dict, path_to_file)
        f_low = h5file.attrs["f_lower_at_1MSUN"] / Mtot

        if h5file.attrs["Format"] < 3:
            spin_args = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(
                -1, Mtot, path_to_file
            )
        else:
            spin_args = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(
                f_low, Mtot, path_to_file
            )

        mass_args = list(
            mtotal_eta_to_mass1_mass2(Mtot * lal.MSUN_SI, h5file.attrs["eta"])
        )

    hp, hc = lalsim.SimInspiralChooseTDWaveform(
        *mass_args,
        *spin_args,
        *fixed_args,
        f_low,
        f_low,
        waveform_dict,
        lalsim.GetApproximantFromString("NR_hdf5"),
    )

    hp_timeseries = TimeSeries(hp.data.data, fixed_args[-1])
    hc_timeseries = TimeSeries(hc.data.data, fixed_args[-1])

    return dict(
        hp=hp_timeseries,
        hc=hc_timeseries,
        f_low=f_low,
        mass_args=mass_args,
        spin_args=spin_args,
        fixed_args=fixed_args,
    )


def check_interp_req(
    h5_file: object = None,
    metadata: dict | None = None,
    ref_time: float | None = None,
    avail_ref_time: float | None = None,
) -> tuple:
    """Check if the required reference time is different from
    the available reference time in the NR HDF5 file or the
    simulation metadata.

    Parameters
    ----------
    h5_file : file object
                The waveform h5 file handle.
    metadata : dict, optional
                The simulation metadata.
    ref_time, avail_ref_time : float
                                The requested and available NR reference time.

    Returns
    -------
    interp : bool
             Whether interpolation across time is required.
    avail_ref_time: float or None
                    The reference time recorded in the NR HDF5 file or
                    the metadata, or None if neither records one.

    Notes
    -----
    RIT records ``relaxed-time``. For RIT's LVCNR files (format 1) the
    frame vectors hold at the first sample of the stored waveform, not
    at ``relaxed-time`` on the waveform's time axis, so the value found
    here is only bookkeeping for those files.
    """

    time_keys = ("reference_time", "ref_time", "relaxed-time", "relaxed_time")

    if avail_ref_time is None:
        # Check for ref time in h5 file
        if h5_file is not None:
            for key in time_keys:
                if key in h5_file.attrs:
                    avail_ref_time = float(h5_file.attrs[key])
                    break
            else:
                logger.info("Reference time not found in waveform h5 file.")

        if avail_ref_time is None and metadata is not None:
            # If not found, continue search in metadata.
            for key in time_keys:
                if metadata.get(key) is not None:
                    avail_ref_time = float(metadata[key])
                    break
            else:
                logger.info("Reference time not found in simulation metadata file.")

        if avail_ref_time is None:
            logger.info(
                "Reference time should be computed from "
                "the reference orbital frequency!"
            )

    if ref_time is None:
        # ref_time was not supplied; no interpolation needed.
        interp = False
    elif avail_ref_time is None:
        # A reference time was asked for and none is recorded.
        interp = True
    else:
        interp = bool(abs(avail_ref_time - float(ref_time)) >= 1e-5)

    return interp, avail_ref_time


def get_lvcnr_format(h5_file: object) -> int:
    """Return the LVCNR format number of an NR HDF5 file.

    Files without a ``Format`` attribute are treated as format 1, a
    single snapshot of the frame vectors and no time series.
    """
    if "Format" in h5_file.attrs:
        return int(h5_file.attrs["Format"])
    return 1


def read_lvcnr_group(h5_file: object, name: str) -> tuple:
    """Read the knots of an LVCNR spline group.

    LVCNR files store each time series (``amp_l2_m2``,
    ``Omega-vs-time``, ``LNhatx-vs-time`` ...) as a group holding the
    datasets ``X`` and ``Y``.

    Returns
    -------
    X, Y : numpy.ndarray
    """
    group = h5_file[name]
    return (
        np.asarray(group["X"][:], dtype=float),
        np.asarray(group["Y"][:], dtype=float),
    )


def _lal_cspline_eval(x: object, y: object, point: float, what: str) -> float:
    """Evaluate a natural cubic spline through (x, y) at ``point``.

    This is the interpolant LAL uses for the LVCNR time series
    (``gsl_interp_cspline``). Like LAL, it fails when ``x`` is not
    strictly increasing or when ``point`` lies outside the data.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(np.diff(x) <= 0):
        raise ValueError(
            f"Cannot interpolate {what}: the abscissa is not strictly increasing."
        )
    if not (x[0] <= point <= x[-1]):
        raise ValueError(
            f"Cannot interpolate {what} at {point}: outside the data range "
            f"[{x[0]}, {x[-1]}]."
        )
    return float(CubicSpline(x, y, bc_type="natural")(point))


def check_fref(h5_file: object, f_ref: float) -> float:
    """Check a reference frequency against an NR HDF5 file, as LAL does.

    Python version of ``XLALSimInspiralNRWaveformCheckFRef``.

    Parameters
    ----------
    h5_file : file object
              The waveform h5 file handle.
    f_ref : float
            The reference GW frequency in Hz, scaled to a total mass of
            1 solar mass (``f_ref * M / Msun``).

    Returns
    -------
    f_ref : float
            ``f_ref`` unchanged, or -1 when the ``Omega-vs-time`` data
            start above it, in which case the frame vectors stored as
            attributes are used.

    Raises
    ------
    ValueError
        If ``f_ref`` is below the start frequency of the waveform.
    """
    if f_ref is None or f_ref <= 0:
        return f_ref

    min_fref = float(h5_file.attrs["f_lower_at_1MSUN"])
    _, omega = read_lvcnr_group(h5_file, "Omega-vs-time")
    lowest_arr_fref = omega[0] / (lal.MTSUN_SI * lal.PI)

    if f_ref < 0.9999 * min_fref:
        raise ValueError(
            "The supplied reference frequency is lower than the start "
            "frequency of the waveform. Start frequency (scaled by total "
            f"mass) is {min_fref} and supplied f_ref (scaled by total mass) "
            f"is {f_ref}."
        )
    if f_ref < lowest_arr_fref:
        # The Omega array does not quite reach f_ref; use the attributes.
        return -1
    return f_ref


def get_ref_freq_from_ref_time(h5_file: object, ref_time: float) -> float:
    """Get the reference frequency from reference time

    Parameters
    ----------
    h5_file : file object
             The waveform h5 file handle.
    ref_time : float
               Reference time, on the time axis of the file's
               ``Omega-vs-time`` group.
    Returns
    -------
    f_ref : float
           Reference GW frequency in Hz, scaled to a total mass of
           1 solar mass. Divide by the total mass to get Hz.
    """

    time, omega = read_lvcnr_group(h5_file, "Omega-vs-time")

    ref_omega = _lal_cspline_eval(time, omega, ref_time, "Omega-vs-time")

    return ref_omega / (lal.MTSUN_SI * lal.PI)


def get_ref_time_from_ref_freq(h5_file: object, ref_freq: float) -> float:
    """Get the reference time from reference frequency

    Python version of ``XLALSimInspiralNRWaveformGetRefTimeFromRefFreq``.

    Parameters
    ----------
    h5_file : file object
             The waveform h5 file handle.
    ref_freq : float
               The reference GW frequency in Hz, scaled to a total mass
               of 1 solar mass (``f_ref * M / Msun``).
    Returns
    -------
    ref_time : float
           Reference time, on the time axis of the file's
           ``Omega-vs-time`` group.
    """

    time, omega = read_lvcnr_group(h5_file, "Omega-vs-time")

    # Omega is stored as M times the orbital angular frequency, and
    # M omega_orb = pi * M f_gw, with M f_gw = f_ref * MTSUN_SI here.
    ref_omega = ref_freq * lal.MTSUN_SI * lal.PI

    return _lal_cspline_eval(omega, time, ref_omega, "time vs Omega-vs-time")


def get_lvcnr_ref_time(h5_file: object, f_ref: float | None) -> float | None:
    """Reference time at which LAL evaluates the frame of an NR file.

    Follows ``XLALSimInspiralNRWaveformGetHplusHcross``: a reference
    frequency is honoured only for LVCNR format 2 and above, and only
    when it is positive and the ``Omega-vs-time`` data reach it.

    Parameters
    ----------
    h5_file : file object
              The waveform h5 file handle.
    f_ref : float or None
            The reference GW frequency in Hz, scaled to a total mass of
            1 solar mass.

    Returns
    -------
    ref_time : float or None
               The reference time on the file's time axis, or None if
               the frame vectors stored as attributes are to be used.
    """
    if f_ref is None or f_ref <= 0:
        return None

    if get_lvcnr_format(h5_file) < 2:
        logger.info(
            "This NR file is format 1. Only formats 2 and above support "
            "the use of reference frequency. The reference frequency is "
            "ignored and the frame at the start of the waveform is used."
        )
        return None

    f_ref = check_fref(h5_file, f_ref)
    if f_ref <= 0:
        return None

    return get_ref_time_from_ref_freq(h5_file, f_ref)


def check_nr_attrs(
    sim_metadata_object: object,
    req_attrs: list = ["LNhatx", "LNhaty", "LNhatz", "nhatx", "nhaty", "nhatz"],
) -> bool:
    """Check if the NR h5 file or a simulation metadata dictionary
        contains all the attributes required.

    Parameters
    ----------
    sim_metadata_object : h5 file object, dict
                     The NR h5py file handle or simulation metadata.
                     For an h5 file both its attributes and its groups
                     (e.g. ``Omega-vs-time``) are searched.
    req_attrs : list
               A list of attribute keys.
    Returns
    -------
    present : bool
              Whether or not all specified attributes are present.
    absent_attrs : list
                 The attributes that are absent.
    """
    if isinstance(sim_metadata_object, h5py.File):
        all_attrs = list(sim_metadata_object.attrs.keys()) + list(
            sim_metadata_object.keys()
        )

    elif isinstance(sim_metadata_object, dict):
        all_attrs = list(sim_metadata_object.keys())
    else:
        raise TypeError("Please supply an open h5py file handle or a dictionary")

    absent_attrs = []
    present = True

    for item in req_attrs:
        if item not in all_attrs:
            present = False
            absent_attrs.append(item)

    return present, absent_attrs


def get_interp_ref_values_from_h5_file(
    h5_file: object, req_ts_attrs: list, ref_time: float
) -> dict:
    """Get the interpolated reference values at a given reference time
    from the NR HDF5 File

    Each key names an LVCNR time-series group (``LNhatx-vs-time`` ...)
    holding the datasets ``X`` and ``Y``. It is interpolated with a
    natural cubic spline, as in
    ``XLALSimInspiralNRWaveformGetInterpValueFromGroupAtPoint``.

    Parameters
    ----------
    h5_file : file object
             The waveform h5 file handle.
    req_ts_attrs : list
               A list of group names.
    ref_time : float
            Reference time.
    Returns
    -------
    params : dict
             The parameter values at the reference time.
    """

    params = {}

    for key in req_ts_attrs:
        time, var = read_lvcnr_group(h5_file, key)
        params[key] = _lal_cspline_eval(time, var, ref_time, key)
    return params


def get_ref_vals(
    sim_metadata_object: object,
    req_attrs: list = ["LNhatx", "LNhaty", "LNhatz", "nhatx", "nhaty", "nhatz"],
) -> dict:
    """Get the reference values from a NR HDF5 file
    or a simulation metadata dictionary.

    Parameters
    ----------
    sim_metadata_object : h5 file object, dict
                     The NR h5py file handle or
                     the simulation metadata.
    req_attrs : list
               A list of attribute keys.
    Returns
    -------
    params : dict
             The parameter values at the reference time.
    """
    if isinstance(sim_metadata_object, h5py.File):
        source = sim_metadata_object.attrs

    elif isinstance(sim_metadata_object, dict):
        source = sim_metadata_object
    else:
        raise TypeError("Please supply an open h5py file handle or a dictionary")

    params = {}

    for key in req_attrs:
        RefVal = source[key]
        params.update({key: RefVal})
    return params


def compute_lal_source_frame_from_sxs_metadata(sim_metadata: dict) -> tuple:
    """Compute the LAL source frame vectors at the SXS reference time from
    the SXS simulation metadata.

    The vectors are built from the reference-epoch quantities that SXS
    publishes (``reference_mass1/2``, ``reference_position1/2`` and the
    ``reference_orbital_frequency`` vector), all given at
    ``reference_time`` in the simulation's inertial coordinates.  SXS v3
    strain is translated and boosted to the centre-of-mass frame but not
    rotated, so directions computed here hold in the frame of the strain
    modes as well.

    Conventions (LAL, ``XLALSimInspiralNRWaveformGetHplusHcross``):

    - ``nhat`` points **from body 2 to body 1**, ``(x_1 - x_2) / |x_1 - x_2|``.
      In SXS metadata body 1 is object A, the heavier hole, so this matches
      the LAL labelling (mass1 >= mass2) with no relabelling.
    - ``LNhat`` is the direction of the Newtonian orbital angular momentum
      ``sum_i m_i x_i x (Omega x x_i)``, positions taken relative to the
      centre of mass.  It is orthogonal to ``nhat`` by construction.

    Parameters
    ----------
    sim_metadata : dict
               The SXS simulation metadata.
    Returns
    -------
    params : dict
             The LAL source frame vectors, keys ``LNhatx``, ``LNhaty``,
             ``LNhatz``, ``nhatx``, ``nhaty``, ``nhatz``.
    """
    # Masses
    M1 = sim_metadata["reference_mass1"]
    M2 = sim_metadata["reference_mass2"]
    M = M1 + M2

    # Orbital angular frequency vector
    Omega = np.array(sim_metadata["reference_orbital_frequency"], dtype=float)

    # Positions
    Pos1 = np.array(sim_metadata["reference_position1"], dtype=float)
    Pos2 = np.array(sim_metadata["reference_position2"], dtype=float)

    # Positions relative to the centre of mass
    CoMPos = (M1 * Pos1 + M2 * Pos2) / (M)

    Pos1 = Pos1 - CoMPos
    Pos2 = Pos2 - CoMPos

    # Separation vector from body 2 to body 1 (LAL's nhat convention).
    # SXS body 1 (object A) is the heavier hole.
    DPos = Pos1 - Pos2

    # Newtonian orbital angular momentum
    P1 = M1 * np.cross(Omega, Pos1)
    P2 = M2 * np.cross(Omega, Pos2)

    Lbar = np.cross(Pos1, P1) + np.cross(Pos2, P2)

    # Orbital normal LNhat
    LNhat = Lbar / np.linalg.norm(Lbar)
    LNhatx, LNhaty, LNhatz = LNhat

    # Separation unit vector nhat
    nhat = (DPos) / np.linalg.norm(DPos)
    nhatx, nhaty, nhatz = nhat

    params = {
        "LNhatx": LNhatx,
        "LNhaty": LNhaty,
        "LNhatz": LNhatz,
        "nhatx": nhatx,
        "nhaty": nhaty,
        "nhatz": nhatz,
    }

    return params


def compute_lal_source_frame_by_interp(
    h5_file: object, req_ts_attrs: list, t_ref: float
) -> tuple:
    """
    Compute the LAL source frame vectors at a given reference time
    by interpolation of time series data.

    Parameters
    ----------
    h5_file : h5 file object
             The NR h5py file handle that contains the simulation metadata.
    t_ref : float
           The reference time.
    Returns
    -------
    params : dict
             The LAL source frame vectors at the reference time.

    """
    # For reference: attributes required for interpolation.
    # req_ts_attrs = ['LNhatx-vs-time', 'LNhaty-vs-time', 'LNhatz-vs-time', \
    #            'position1x-vs-time', 'position1y-vs-time', 'position1z-vs-time', \
    #            'position2x-vs-time', 'position2y-vs-time', 'position2z-vs-time']

    ref_params = get_interp_ref_values_from_h5_file(h5_file, req_ts_attrs, t_ref)

    # r21 vec
    r21_x = ref_params["position1x-vs-time"] - ref_params["position2x-vs-time"]
    r21_y = ref_params["position1y-vs-time"] - ref_params["position2y-vs-time"]
    r21_z = ref_params["position1z-vs-time"] - ref_params["position2z-vs-time"]

    # Position unit vector nhat
    dPos = np.array([r21_x, r21_y, r21_z])
    nhat = dPos / np.linalg.norm(dPos)
    nhatx, nhaty, nhatz = nhat

    # Orbital unit normal LNhat
    LNhatx = ref_params["LNhatx-vs-time"]
    LNhaty = ref_params["LNhaty-vs-time"]
    LNhatz = ref_params["LNhatz-vs-time"]

    LNhat = np.array([LNhatx, LNhaty, LNhatz])
    LNhat = LNhat / np.linalg.norm(LNhat)
    # LNhatx, LNhaty, LNhatz = LNhat

    # params = {"LNhat": LNhat, "nhat": nhat}
    params = {
        "LNhatx": LNhatx,
        "LNhaty": LNhaty,
        "LNhatz": LNhatz,
        "nhatx": nhatx,
        "nhaty": nhaty,
        "nhatz": nhatz,
    }

    return params


def normalize_metadata(sim_metadata: dict) -> dict:
    """Ensure that the keys of the metadata are
    as required.

    Parameters
    ----------
    sim_metadata : dict
               The NR sim_metadata.

    Returns
    -------
    norm_sim_metadata : dict
               The normalized simulation metadata
    """

    norm_sim_metadata = {}

    for key, val in sim_metadata.items():
        norm_sim_metadata.update({key.replace("relaxed-", ""): val})

    return norm_sim_metadata


def get_ref_time_from_metadata(sim_metadata: dict) -> float:
    """Get the reference time of definition of the LAL
    frame from the simulation metadata, if available.

    Parameters
    ----------
    sim_metadata : dict
                   The simulation metadata.

    Returns
    -------
    t_ref : float
           The reference time
    """

    MdataKeys = list(sim_metadata.keys())

    if "relaxed-time" in MdataKeys:
        # RIT style
        t_ref = sim_metadata["relaxed-time"]
    elif "reference_time" in MdataKeys:
        # SXS Style
        t_ref = sim_metadata["reference_time"]
    else:
        t_ref = -1

    return t_ref


def transform_spins_nr_to_lal(
    nrSpin1: object, nrSpin2: object, n_hat: object, ln_hat: object
) -> tuple:
    """Trnasform the spins of the NR simulation from the
    NR frame to the  frame.
    Parameters
    ---------
    nrSpin1, nrSpin2 : list
             A list of the components of the spins of the objects.
    n_hat, ln_hat : list
             A list of the components of the unit vectors of the objects,
             against which the components of the spins are specified.
    Returns
    -------
    S1, S2 : list
             The transformed spins in LAL frame.
    """
    nrSpin1x, nrSpin1y, nrSpin1z = nrSpin1
    nrSpin2x, nrSpin2y, nrSpin2z = nrSpin2

    n_hat_x, n_hat_y, n_hat_z = n_hat
    ln_hat_x, ln_hat_y, ln_hat_z = ln_hat

    S1x = nrSpin1x * n_hat_x + nrSpin1y * n_hat_y + nrSpin1z * n_hat_z

    S1y = (
        nrSpin1x * (-ln_hat_z * n_hat_y + ln_hat_y * n_hat_z)
        + nrSpin1y * (ln_hat_z * n_hat_x - ln_hat_x * n_hat_z)
        + nrSpin1z * (-ln_hat_y * n_hat_x + ln_hat_x * n_hat_y)
    )

    S1z = nrSpin1x * ln_hat_x + nrSpin1y * ln_hat_y + nrSpin1z * ln_hat_z

    S2x = nrSpin2x * n_hat_x + nrSpin2y * n_hat_y + nrSpin2z * n_hat_z
    S2y = (
        nrSpin2x * (-ln_hat_z * n_hat_y + ln_hat_y * n_hat_z)
        + nrSpin2y * (ln_hat_z * n_hat_x - ln_hat_x * n_hat_z)
        + nrSpin2z * (-ln_hat_y * n_hat_x + ln_hat_x * n_hat_y)
    )

    S2z = nrSpin2x * ln_hat_x + nrSpin2y * ln_hat_y + nrSpin2z * ln_hat_z

    S1 = [S1x, S1y, S1z]
    S2 = [S2x, S2y, S2z]

    return S1, S2


def _normalise_frame_vector(vec: object, name: str, tol: float = 1e-3) -> object:
    """Return ``vec`` divided by its norm.

    As in ``XLALSimInspiralNRWaveformGetRotationAnglesFromH5File``, a
    vector whose norm differs from 1 by more than ``tol`` is an error:
    the frame vectors in an NR file are meant to be unit vectors.
    """
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if abs(norm - 1) > tol:
        raise ValueError(
            f"The size of the {name} vector is {norm}, not unity. "
            "Check the frame vectors in the NR file."
        )
    return vec / norm


def get_nr_to_lal_rotation_angles(
    h5_file: object,
    sim_metadata: dict,
    inclination: float,
    phi_ref: float = 0,
    f_ref: float | None = None,
    t_ref: float | None = None,
    tol: float = 1e-6,
    total_mass: float = 1.0,
) -> tuple:
    r"""Get the angular coordinates :math:`\theta, \phi`
    and the rotation angle :math:`\alpha` from the H5 file

    Parameters
    ----------
    h5_file : file object
            The waveform h5 file handle.

    inclination : float
                  The inclination angle.
    phi_ref : float
             The orbital phase at reference time.
    f_ref : float, optional
            The reference GW frequency in Hz for a binary of total
            mass ``total_mass``, as in LAL. It is used only for LVCNR
            format 2 and above, where the frame is interpolated from the
            ``*-vs-time`` groups at the time where the orbital frequency
            reaches it; for format 1 files it is ignored, as LAL does.
    t_ref : float, optional
            The reference time on the file's time axis. Takes
            precedence over ``f_ref``.

    sim_metadata : dict
               The sim_metadata of the waveform file.
    tol : float
          The tolerance to use to allow floating point
            representation errors.
    total_mass : float, optional
            Total mass in solar masses that ``f_ref`` refers to.
            The default of 1 means ``f_ref`` is already scaled to
            1 solar mass (LAL's ``f_lower_at_1MSUN`` convention).

    Returns
    -------
    angles : dict
             The angular corrdinates Theta, Psi,  and the rotation angle Alpha.
             If available, this also contains the reference time and frequency.
             ``f_ref`` is None when the frame stored as attributes was used
             (format 1 files, or ``f_ref`` below the ``Omega-vs-time`` data).

    Notes
    -----

    Variable definitions.

    theta : Returned inclination angle of source in NR coordinates.
    psi :   Returned azimuth angle of source in NR coordinates.
    alpha: Returned polarisation angle.
    h5_file: h5py object of the NR HDF5 file.
    inclination: inclination of source in LAL source frame.
    phi_ref: Orbital reference phase.
    t_ref : Reference time. -1 or None indicates it was not found in the sim_metadata.
    f_ref: Reference frequency.

    The reference epoch is defined close to the beginning of the simulation.
    """

    # Compute the angles necessary to rotate from the intrinsic NR source frame
    # into the LAL frame. See DCC-T1600045 for details.

    # Following section IV of DCC-T1600045
    # Step 1: Define Phi = phiref
    orb_phase = phi_ref

    ##########################################
    # Step 2: Compute Zref
    # 2.1 : Check if interpolation is required in IntReq
    # 2.2 : Get/ Compute the basis vectors of the LAL
    #       frame.
    # 2.2.1 : If IntReq=yes, given the reference time, interpolate and get
    #         the required basis vectors in the LAL source frame.
    # 2.2.2 : If no, then check for the default values of the
    #         LAL frame basis vectors in the h5_file
    # 2.2.3 : If the h5_file does not contain the required default
    #         vectors, then raise an error.
    # 2.3 : Carryout vector math to get Zref.
    # 2.1: Compute LN_hat from file. LN_hat = direction of orbital ang. mom.
    # 2.2: Compute n_hat from file. n_hat = direction from object 2 to object 1

    ###########################################
    # Cases
    ###########################################
    # req_def_attrs = ["LNhatx", "LNhaty", "LNhatz", "nhatx", "nhaty", "nhatz"]

    # SXS default attributes required
    # for computing the LAL source frame.
    req_def_attrs_sxs = [
        "reference_time",
        "reference_mass1",
        "reference_mass2",
        "reference_orbital_frequency",
        "reference_position1",
        "reference_position2",
    ]

    # Check if interpolation is necessary
    # if t_ref is supplied

    interp = False

    if t_ref is None:
        if f_ref is not None and f_ref > 0:
            # As LAL: format < 2 ignores f_ref, an f_ref below the start
            # of the waveform raises, and one below the start of the
            # Omega-vs-time data falls back on the stored attributes.
            lvcnr_t_ref = get_lvcnr_ref_time(h5_file, f_ref * total_mass)
            if lvcnr_t_ref is not None:
                t_ref = lvcnr_t_ref
                interp = True
            else:
                # Report that f_ref was not used.
                f_ref = None
    else:
        interp, avail_ref_time = check_interp_req(h5_file, ref_time=t_ref)

    if interp is False:
        # Then load default values from the NR data
        # at hard coded reference time.

        # Get the available reference time for book keeping
        t_ref = get_ref_time_from_metadata(sim_metadata)

        # Check for LAL frame in simulation metadata.
        # RIT and GT qualify this.
        # Default attributes in case of no interpolation
        ref_check_def_h5, absent_attrs_h5 = check_nr_attrs(h5_file)

        if ref_check_def_h5 is False:
            # Then the LAL source frame information is not present in the H5 file.
            # Then this could be SXS or GT data. The LAL source frame need to be computed from
            # the H5 File or simulation metadata.

            # Check if LAL source frame info is present in the simulation metadata.
            ref_check_def_meta, absent_attrs_meta = check_nr_attrs(sim_metadata)

            if ref_check_def_meta is False:
                # Then this is SXS data.

                # Check for raw information in metadata to compute the LAL source frame.
                ref_check_def_meta_sxs, absent_attrs_meta_sxs = check_nr_attrs(
                    sim_metadata, req_def_attrs_sxs
                )

                if ref_check_def_meta_sxs is True:
                    # Compute the LAL source frame from simulation metadata
                    ref_params = compute_lal_source_frame_from_sxs_metadata(
                        sim_metadata
                    )
                else:
                    raise Exception(
                        "Insufficient information to compute the LAL source frame."
                        f"\n Missing information is {absent_attrs_meta_sxs}."
                    )
            else:
                # LAL source frame is present in the simulation metadata
                ref_params = get_ref_vals(sim_metadata)
        else:
            ref_params = get_ref_vals(h5_file)

    elif interp is True:
        # Experimental; This assumes all the required atributes  needed
        # to compute the LAL source frame at the given reference time
        # are present in the H5file only.

        # Attributes required for interpolation.
        req_ts_attrs = [
            "LNhatx-vs-time",
            "LNhaty-vs-time",
            "LNhatz-vs-time",
            "position1x-vs-time",
            "position1y-vs-time",
            "position1z-vs-time",
            "position2x-vs-time",
            "position2y-vs-time",
            "position2z-vs-time",
        ]

        # Check if time series data of required reference data is present
        ref_check_interp_req, absent_interp_attrs = check_nr_attrs(
            h5_file, req_ts_attrs
        )

        if ref_check_interp_req is False:
            raise KeyError(
                "Insufficient information to compute the LAL source frame at given reference time."
                f"Missing information is {absent_interp_attrs}."
            )
        else:
            ref_params = compute_lal_source_frame_by_interp(
                h5_file, req_ts_attrs, t_ref
            )

        # Warning 1
        # Implement this Warning
        # XLAL_CHECK( ref_time!=XLAL_FAILURE, XLAL_FAILURE, "Error computing reference time.
        # Try setting fRef equal to the f_low given by the NR simulation or to a value <=0 to deactivate
        # fRef for a non-precessing simulation.\n")

    # Get the LAL source frame vectors
    ln_hat_x = ref_params["LNhatx"]
    ln_hat_y = ref_params["LNhaty"]
    ln_hat_z = ref_params["LNhatz"]

    n_hat_x = ref_params["nhatx"]
    n_hat_y = ref_params["nhaty"]
    n_hat_z = ref_params["nhatz"]

    # Normalise the frame vectors as LAL does, refusing vectors that
    # are far from unit length.
    ln_hat = _normalise_frame_vector([ln_hat_x, ln_hat_y, ln_hat_z], "LN hat")
    n_hat = _normalise_frame_vector([n_hat_x, n_hat_y, n_hat_z], "n hat")
    n_hat_x, n_hat_y, n_hat_z = n_hat
    ln_hat_x, ln_hat_y, ln_hat_z = ln_hat

    # 2.3: Carryout vector math to get Zref in the lal wave frame
    corb_phase = np.cos(orb_phase)
    sorb_phase = np.sin(orb_phase)
    sinclination = np.sin(inclination)
    cinclination = np.cos(inclination)

    ln_cross_n = _normalise_frame_vector(np.cross(ln_hat, n_hat), "LN x n")
    ln_cross_n_x, ln_cross_n_y, ln_cross_n_z = ln_cross_n

    z_wave_x = sinclination * (sorb_phase * n_hat_x + corb_phase * ln_cross_n_x)
    z_wave_y = sinclination * (sorb_phase * n_hat_y + corb_phase * ln_cross_n_y)
    z_wave_z = sinclination * (sorb_phase * n_hat_z + corb_phase * ln_cross_n_z)

    z_wave_x += cinclination * ln_hat_x
    z_wave_y += cinclination * ln_hat_y
    z_wave_z += cinclination * ln_hat_z

    z_wave = _normalise_frame_vector([z_wave_x, z_wave_y, z_wave_z], "Z")
    z_wave_x, z_wave_y, z_wave_z = z_wave

    #################################################################
    # Step 3.1: Extract theta and psi from Z in the lal wave frame
    # NOTE: Theta can only run between 0 and pi, so no problem with arccos here
    theta = np.arccos(z_wave_z)

    # Degenerate if Z_wave[2] == 1. In this case just choose psi randomly,
    # the choice will be cancelled out by alpha correction (I hope!)

    # If theta is very close to the poles
    # return a random value
    if abs(z_wave_z - 1.0) < tol:
        psi = 0.5

    else:
        # psi can run between 0 and 2pi, but only one solution works for x and y */
        # Possible numerical issues if z_wave_x = sin(theta) */
        if abs(z_wave_x / np.sin(theta)) > 1.0:
            if abs(z_wave_x / np.sin(theta)) < (1 + 10 * tol):
                # LAL tol retained.
                if (z_wave_x * np.sin(theta)) < 0.0:
                    psi = np.pi

                else:
                    psi = 0.0

            else:
                logger.info(f"z_wave_x = {z_wave_x}")
                logger.info(f"sin(theta) = {np.sin(theta)}")
                raise ValueError(
                    "Z_x cannot be bigger than sin(theta). Please contact the developers."
                )

        else:
            psi = np.arccos(z_wave_x / np.sin(theta))

        y_val = np.sin(psi) * np.sin(theta)

        # If z_wave[1] is negative, flip psi so that sin(psi) goes negative
        # while preserving cos(psi) */
        if z_wave_y < 0.0:
            psi = 2 * np.pi - psi
            y_val = np.sin(psi) * np.sin(theta)

        if abs(y_val - z_wave_y) > (5e3 * tol):
            # LAL tol retained.
            logger.info(f"orb_phase = {orb_phase}")
            logger.info(
                f"y_val = {y_val}, z_wave_y = {z_wave_y}, abs(y_val - z_wave_y) = {abs(y_val - z_wave_y)}"
            )
            raise ValueError("Math consistency failure! Please contact the developers.")

    # 3.2: Compute the vectors theta_hat and psi_hat
    stheta = np.sin(theta)
    ctheta = np.cos(theta)

    spsi = np.sin(psi)
    cpsi = np.cos(psi)

    theta_hat_x = cpsi * ctheta
    theta_hat_y = spsi * ctheta
    theta_hat_z = -stheta
    theta_hat = np.array([theta_hat_x, theta_hat_y, theta_hat_z])

    psi_hat_x = -spsi
    psi_hat_y = cpsi
    psi_hat_z = 0.0
    psi_hat = np.array([psi_hat_x, psi_hat_y, psi_hat_z])

    # Step 4: Compute sin(alpha) and cos(alpha)
    # Rotation angles on the tangent plane
    # due to spin weight.

    n_dot_theta = np.dot(n_hat, theta_hat)
    ln_cross_n_dot_theta = np.dot(ln_cross_n, theta_hat)

    n_dot_psi = np.dot(n_hat, psi_hat)
    ln_cross_n_dot_psi = np.dot(ln_cross_n, psi_hat)

    salpha = corb_phase * n_dot_theta - sorb_phase * ln_cross_n_dot_theta
    calpha = corb_phase * n_dot_psi - sorb_phase * ln_cross_n_dot_psi

    if abs(calpha) > 1:
        calpha_err = abs(calpha) - 1
        if calpha_err < tol:
            # This tol could have been much smaller.
            # Just resuing the default for now.
            logger.info(
                f"Correcting the polarization angle for finite precision error {calpha_err}"
            )
            calpha = calpha / abs(calpha)
        else:
            raise ValueError(
                "Seems like something is wrong with the polarization angle. Please contact the developers!"
            )

    alpha = -np.arctan2(salpha, calpha)

    angles = {
        "theta": theta,
        "psi": psi,
        "alpha": alpha,
        "t_ref": t_ref,
        "f_ref": f_ref,
    }

    return angles
