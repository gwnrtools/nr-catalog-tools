import h5py
import lal
import lalsimulation as lalsim
from pycbc.pnutils import mtotal_eta_to_mass1_mass2
from pycbc.types import TimeSeries


def get_lal_mode_dictionary(mode_array):
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


def get_lal_mode_dictionary_from_lmax(lmax):
    r"""
    Get LALDict with modes derived from `lmax`.


    Parameters
    ----------
    lmax: max value of \ell to use

    Returns
    -------
    waveform_dictionary: LALDict with all modes upto `lmax` included

    """

    mode_array = [
        [ell, emm] for ell in range(2, lmax + 1) for emm in range(-ell, ell + 1)
    ]
    return get_lal_mode_dictionary(mode_array)


def get_modes_from_lvcnr_file(path_to_file, Mtot, distance, srate, lmax=4, f_low=None):
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
        Max value of \ell to use
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
    path_to_file, Mtot, distance, inclination, phi_ref, srate, mode_array=None
):
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


def check_interp_req(H5File, ref_time):
    """Check if the required reference time is different from
    the available reference time in the NR HDF5 file

    Parameters
    ----------
    H5File : file object
                The waveform h5 file handle.
    ref_time : float
               Reference time.

    Returns
    -------
    Interp : bool
             Whether interpolation across time is required.
    avail_ref_time: float
                    The ref_time available in the NR HDF5 file.
    """

    avail_ref_time = H5File.attrs("ref_time")

    if abs(avail_ref_time - ref_time) < 1e-5:
        return False, avail_ref_time
    else:
        return True, avail_ref_time


def get_ref_freq_from_ref_time(H5File, ref_time):
    """Get the reference frequency from reference time

    Parameters
    ----------
    H5File : file object
             The waveform h5 file handle.
    ref_time : float
               Reference time.
    Returns
    -------
    fRef : float
           Reference frequency.
    """

    time, freq = H5File.attrs["Omega-vs-time"]

    from scipy.interpolate import interp1d

    RefFreq = interp1d(time, freq, kind="cubic")[ref_time]

    return RefFreq


def get_ref_time_from_ref_freq(H5File, ref_freq):
    """Get the reference time from reference frequency

    Parameters
    ----------
    H5File : file object
             The waveform h5 file handle.
    ref_freq : float
               The reference frequency.
    Returns
    -------
    fTime : float
           Reference time.
    """
    from scipy.interpolate import interp1d

    time, freq = H5File.attrs["Omega-vs-time"]

    RefTime = interp1d(freq, time, kind="cubic")[ref_freq]

    return RefTime


def check_nr_attrs(
    MetadataObject, ReqAttrs=["LNhatx", "LNhaty", "LNhatz", "nhatx", "nhaty", "nhatz"]
):
    """Check if the NR h5 file or a metadata dictionary
        contains all the attributes required.

    Parameters
    ----------
    MetadataObject : h5 file object, dict
                     The NR h5py file handle or metadata.
    ReqAttrs : list
               A list of attribute keys.
    Returns
    -------
    Present : bool
              Whether or not all specified attributes are present.
    AbsentAttrs : list
                 The attributes that are absent.
    """
    if isinstance(MetadataObject, h5py.File):
        all_attrs = list(MetadataObject.attrs.keys())

    elif isinstance(MetadataObject, dict):
        all_attrs = list(MetadataObject.keys())
    else:
        raise TypeError("Please supply an open h5py file handle or a dictionary")

    AbsentAttrs = []
    Present = True

    for item in ReqAttrs:
        if item not in all_attrs:
            Present = False
            AbsentAttrs.append(item)

    return Present, AbsentAttrs


def get_interp_ref_values_from_h5_file(H5File, ReqTSAttrs, ref_time):
    """Get the interpolated reference values at a given reference time
    from the NR HDF5 File

    Parameters
    ----------
    H5File : file object
             The waveform h5 file handle.
    ReqTSAttrs : list
               A list of attribute keys.
    ref_time : float
            Reference time.
    Returns
    -------
    params : dict
             The parameter values at the reference time.
    """
    from scipy.interpolate import interp1d

    params = {}

    for key in ReqTSAttrs:
        time, var = H5File.attrs[key]
        RefVal = interp1d(time, var, kind="cubic")[ref_time]
        params.update({key: RefVal})
    return params


def get_ref_vals(
    MetadataObject, ReqAttrs=["LNhatx", "LNhaty", "LNhatz", "nhatx", "nhaty", "nhatz"]
):
    """Get the reference values from a NR HDF5 file
    or a metadata dictionary.

    Parameters
    ----------
    MetadataObject : h5 file object, dict
                     The NR h5py file handle or metadata.
    ReqAttrs : list
               A list of attribute keys.
    Returns
    -------
    params : dict
             The parameter values at the reference time.
    """
    if isinstance(MetadataObject, h5py.File):
        Source = MetadataObject.attrs

    elif isinstance(MetadataObject, dict):
        Source = MetadataObject
    else:
        raise TypeError("Please supply an open h5py file handle or a dictionary")

    params = {}

    # print(MetadataObject.keys())
    for key in ReqAttrs:
        RefVal = Source[key]
        params.update({key: RefVal})
    return params


def compute_lal_source_frame_from_sxs_metadata(Metadata):
    """Compute the LAL source frame vectors at the
    available reference time from the SXS metadata.

    Parameters
    ----------
    Metadata : dict
               The NR metadata.
    Returns
    -------
    params : dict
             A dictionary containing the LAL source frame
             vectors.
    """

    M1 = Metadata["reference_mass1"]
    M2 = Metadata["reference_mass2"]
    M = M1 + M2
    Omega = np.array(Metadata["reference_orbital_frequency"])

    Pos1 = np.array(Metadata["reference_position1"])
    Pos2 = np.array(Metadata["reference_position2"])

    # CoM position
    CoMPos = (M1 * Pos1 + M2 * Pos2) / (M)

    Pos1 = Pos1 - CoMPos
    Pos2 = Pos2 - CoMPos

    # This should point from
    # smaller to larger BH
    # Assumed here is 1 for smaller
    # BH
    DPos = Pos2 - Pos1

    # Oribital angular momentum, Eucledian
    P1 = M1 * np.cross(Omega, Pos1)
    P2 = M2 * np.cross(Omega, Pos2)

    Lbar = np.cross(Pos1, P1) + np.cross(Pos2, P2)

    # Orbital normal LNhat
    LNhat = Lbar / np.linalg.norm(Lbar)
    # LNhatx, LNhaty, LNhatz = LNhat

    # Position vector nhat
    nhat = (DPos) / np.linalg.norm(DPos)
    # nhatx, nhaty, nhatz = Nhat

    params = {
        "LNhatx": LNhatx,
        "LNhaty": LNhaty,
        "LNhatz": LNhatz,
        "nhatx": nhatx,
        "nhaty": nhaty,
        "nhatz": nhatz,
    }
    # params = {"LNhat": LNhat, "nhat": nhat}

    return params


def compute_lal_source_frame_by_interp(H5File, ReqTSAttrs, TRef):
    """
    Compute the LAL source frame vectors at a given reference time
    by interpolation of time series data.

    Parameters
    ----------
    H5File : h5 file object
             The NR h5py file handle that contains the metadata.
    Tref : float
           The reference time.
    Returns
    -------
    params : dict
             The LAL source frame vectors at the reference time.

    """
    # For reference: attributes required for interpolation.
    # ReqTSAttrs = ['LNhatx-vs-time', 'LNhaty-vs-time', 'LNhatz-vs-time', \
    #            'position1x-vs-time', 'position1y-vs-time', 'position1z-vs-time', \
    #            'position2x-vs-time', 'position2y-vs-time', 'position2z-vs-time']

    IntParams = GetInterpRefValuesFromH5File(H5File, ReqTSAttrs, TRef)

    # r21 vec
    r21_x = RefParams["position1x-vs-time"] - RefParams["position2x-vs-time"]
    r21_y = RefParams["position1y-vs-time"] - RefParams["position2y-vs-time"]
    r21_z = RefParams["position1z-vs-time"] - RefParams["position2z-vs-time"]

    # Position unit vector nhat
    dPos = np.array([r21_x, r21_y, r21_z])
    nhat = dPos / np.linalg.norm(dPos)
    # nhatx, nhaty, nhatz = Nhat

    # Orbital unit normal LNhat
    LNhat_x = RefParams["LNhatx-vs-time"]
    LNhat_y = RefParams["LNhaty-vs-time"]
    LNhat_z = RefParams["LNhatz-vs-time"]

    LNhat = np.array([LNhat_x, LNhat_y, LNhat_z])
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


def normalize_metadata(Metadata):
    """Ensure that the keys of the metadata are
    as required

    Parameters
    ----------
    Metadata : dict
               The NR metadata.

    Returns
    -------
    NMetadata : dict
               The normalized metadata
    """

    NMetadata = {}

    for key, val in Metadata.items():
        NMetadata.update({key.replace("relaxed-", ""): val})

    return NMetadata


def get_ref_time_from_metadata(Metadata):
    """Get the reference time of definition of the LAL
    frame from metadata, if available
    Parameters
    ----------


    Returns
    -------
    TRef : float
           The reference time
    """

    MdataKeys = list(Metadata.keys())

    if "relaxed-time" in MdataKeys:
        # RIT style
        TRef = Metadata["relaxed-time"]
    elif "reference_time" in MdataKeys:
        # SXS Style
        TRef = Metadata["reference_time"]
    else:
        TRef = -1

    return TRef


def transform_spins_nr_to_lal(nrSpin1, nrSpin2, n_hat, ln_hat):
    """Trnasform the spins of the NR simulation from the
    NR frame to the  frame.
    Parameters
    ---------
    nrSpin1, nrSpin2 : list
             A list of the components of the spins of the objects.
    nhat, ln_hat : list
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


def get_nr_to_lal_rotation_angles(
    H5File, Metadata, Inclination, PhiRef=0, FRef=None, TRef=None
):
    """Get the angular coordinates :math:`\theta, \phi`
    and the rotation angle :math:`\alpha` from the H5 file

    Parameters
    ----------
    H5File : file object
            The waveform h5 file handle.

    Inclination : float
                  The inclination angle.
    PhiRef : float
             The orbital phase at reference time.
    Fref, TRef : float, optional
                 The reference orbital frequency or time

    metadata : dict
               The metadata of the waveform file.
    Returns
    -------
    angles : dict
             The angular corrdinates Theta, Psi,  and the rotation angle Alpha.
             If available, this also contains the reference time and frequency.

    Notes
    -----

    Variable definitions.

    theta : Returned inclination angle of source in NR coordinates.
    psi :   Returned azimuth angle of source in NR coordinates.
    alpha: Returned polarisation angle.
    H5File: h5py object of the NR HDF5 file.
    inclination: Inclination of source in LAL source frame.
    phi_ref: Orbital reference phase.
    TRef : Reference time. -1 or None indicates it was not found in the metadata.
    FRef: Reference frequency.

    The reference epoch is defined close to the beginning of the simulation.
    """

    # tolerence for sanity checks
    tol = 1e-3

    # Compute the angles necessary to rotate from the intrinsic NR source frame
    # into the LAL frame. See DCC-T1600045 for details.

    # Following section IV of DCC-T1600045
    # Step 1: Define Phi = phiref
    orb_phase = PhiRef

    ##########################################
    # Step 2: Compute Zref
    # 2.1 : Check if interpolation is required in IntReq
    # 2.2 : Get/ Compute the basis vectors of the LAL
    #       frame.
    # 2.2.1 : If IntReq=yes, given the reference time, interpolate and get
    #         the required basis vectors in the LAL source frame.
    # 2.2.2 : If no, then check for the default values of the
    #         LAL frame basis vectors in the H5File
    # 2.2.3 : If the H5File does not contain the required default
    #         vectors, then raise an error.
    # 2.3 : Carryout vector math to get Zref.
    # 2.1: Compute LN_hat from file. LN_hat = direction of orbital ang. mom.
    # 2.2: Compute n_hat from file. n_hat = direction from object 2 to object 1

    ###########################################
    # Cases
    ###########################################
    # ReqDefAttrs = ["LNhatx", "LNhaty", "LNhatz", "nhatx", "nhaty", "nhatz"]
    ReqDefAttrsSXS = [
        "reference_time",
        "reference_mass1",
        "reference_mass2",
        "reference_orbital_frequency",
        "reference_position1",
        "reference_position2",
    ]

    # Check if interpolation is necessary
    # if TRef is supplied

    Interp = False

    if not TRef:
        if not FRef:
            # No interpolation required. Use available reference values.
            Interp = False

        else:
            # Try to get the reference time from orbital frequency
            try:
                TRef = GetRefTimeFromRefFreq(H5File, FRef)
                # Check if interpolation is required
                Interp, avail_ref_time = CheckInterpReq(H5File, TRef)
            except:
                print(
                    f"Could not obtain reference time from given reference frequency {FRef} \n Choosing available reference time"
                )
                Interp = False
    else:
        Interp, avail_ref_time = CheckInterpReq(H5File, TRef)

    if Interp == False:
        # Then load default values from the NR data
        # at hard coded reference time.

        # Get the available reference time for book keeping
        TRef = GetRefTimeFromMetadata(Metadata)

        # Check for LAL frame in metadata.
        # RIT and GT qualify this.
        # Default attributes in case of no interpolation
        # RefCheckInterp = CheckNRAttrs(H5File, ReqTSAttrs)
        # RefCheckDefMeta, AbsentAttrsMeta = CheckNRAttrs(Metadata, ReqDefAttrs)
        RefCheckDefH5, AbsentAttrsH5 = CheckNRAttrs(H5File)

        # RefCheckDefMeta, AbsentAttrsMeta  = CheckNRAttrs(Metadata, ReqDefAttrs)

        if RefCheckDefH5 == False:
            # Then the LAL source frame information is not present in the H5 file.
            # Then this could be SXS or GT data. The LAL source frame need to be computed from
            # the H5 File or metadata.

            # Check if LAL source frame info is present in the metadata.
            RefCheckDefMeta, AbsentAttrsMeta = CheckNRAttrs(Metadata)

            if RefCheckDefMeta == False:
                # Then this is SXS data.
                try:
                    # Compute the LAL source frame from metadata
                    RefParams = ComputeLALSourceFrameFromSXSMetadata(Metadata)
                except Exception as ex:
                    ex(
                        f"Insufficient information to compute the LAL source frame. Missing information is {AbsentAttrsH5}."
                    )
            else:
                # LAL source frame is present in the metadata
                RefParams = GetRefVals(Metadata)
        else:
            # print(RefCheckDefH5, AbsentAttrsH5)
            RefParams = GetRefVals(H5File)
            # print(RefParams)
    elif Interp == True:
        # Experimental; This assumes all the required atributes  needed
        # to compute the LAL source frame at the given reference time
        # are present in the H5file only.

        # Attributes required for interpolation.
        ReqTSAttrs = [
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
        RefCheckInterpReq, AbsentInterpAttrs = CheckNRAttrs(H5File, ReqTSAttrs)

        if RefCheckInterpReq == False:
            raise Exception(
                f"Insufficient information to compute the LAL source frame at given reference time. Missing information is {AbsentInterpAttrs}."
            )
        else:
            RefParams = ComputeLALSourceFrameByInterp(H5File, ReqTSAttrs, TRef)

        # Warning 1
        # Implement this Warning
        # XLAL_CHECK( ref_time!=XLAL_FAILURE, XLAL_FAILURE, "Error computing reference time.
        # Try setting fRef equal to the f_low given by the NR simulation or to a value <=0 to deactivate
        # fRef for a non-precessing simulation.\n")

    print(RefParams)

    # Get the LAL source frame vectors
    ln_hat_x = RefParams["LNhatx"]
    ln_hat_y = RefParams["LNhaty"]
    ln_hat_z = RefParams["LNhatz"]

    n_hat_x = RefParams["nhatx"]
    n_hat_y = RefParams["nhaty"]
    n_hat_z = RefParams["nhatz"]

    ln_hat = np.array([ln_hat_x, ln_hat_y, ln_hat_z])
    n_hat = np.array([n_hat_x, n_hat_y, n_hat_z])

    # 2.3: Carryout vector math to get Zref in the lal wave frame
    corb_phase = np.cos(orb_phase)
    sorb_phase = np.sin(orb_phase)
    sinclination = np.sin(Inclination)
    cinclination = np.cos(Inclination)

    ln_cross_n = np.cross(ln_hat, n_hat)
    ln_cross_n_x, ln_cross_n_y, ln_cross_n_z = ln_cross_n

    z_wave_x = sinclination * (sorb_phase * n_hat_x + corb_phase * ln_cross_n_x)
    z_wave_y = sinclination * (sorb_phase * n_hat_y + corb_phase * ln_cross_n_y)
    z_wave_z = sinclination * (sorb_phase * n_hat_z + corb_phase * ln_cross_n_z)

    z_wave_x += cinclination * ln_hat_x
    z_wave_y += cinclination * ln_hat_y
    z_wave_z += cinclination * ln_hat_z

    z_wave = np.array([z_wave_x, z_wave_y, z_wave_z])
    z_wave = z_wave / np.linalg.norm(z_wave)

    #################################################################
    # Step 3.1: Extract theta and psi from Z in the lal wave frame
    # NOTE: Theta can only run between 0 and pi, so no problem with arccos here
    theta = np.arccos(z_wave_z)

    # Degenerate if Z_wave[2] == 1. In this case just choose psi randomly,
    # the choice will be cancelled out by alpha correction (I hope!)

    # If theta is very close to the poles
    # return a random value
    if abs(z_wave_z - 1.0) < 0.000001:
        psi = 0.5

    else:
        # psi can run between 0 and 2pi, but only one solution works for x and y */
        # Possible numerical issues if z_wave_x = sin(theta) */
        if abs(z_wave_x / np.sin(theta)) > 1.0:
            if abs(z_wave_x / np.sin(theta)) < 1.00001:
                if (z_wave_x * np.sin(theta)) < 0.0:
                    psi = np.pi  # LAL_PI

                else:
                    psi = 0.0

            else:
                print(f"z_wave_x = {z_wave_x}")
                print(f"sin(theta) = {np.sin(theta)}")
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

        if abs(y_val - z_wave_y) > 0.005:
            print(f"orb_phase = {orb_phase}")
            print(
                f"y_val = {y_val}, z_wave_y = {z_wave_y}, abs(y_val - z_wave_y) = {abs(y_val - z_wave_y)}"
            )
            raise ValueError("Math consistency failure! Please contact the developers.")

    # 3.2: Compute the vectors theta_hat and psi_hat */
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
    n_dot_theta = np.dot(n_hat, theta_hat)
    ln_cross_n_dot_theta = np.dot(ln_cross_n, theta_hat)
    n_dot_psi = np.dot(n_hat, psi_hat)
    ln_cross_n_dot_psi = np.dot(ln_cross_n, psi_hat)

    salpha = corb_phase * n_dot_theta - sorb_phase * ln_cross_n_dot_theta
    calpha = corb_phase * n_dot_psi - sorb_phase * ln_cross_n_dot_psi

    alpha = np.arccos(calpha)

    ############################
    # Optional
    ############################

    # Step 5: Also useful to keep the source frame vectors as defined in
    # equation 16 of Harald's document.

    # x_source_hat[0] = corb_phase * n_hat_x - sorb_phase * ln_cross_n_x
    # x_source_hat[1] = corb_phase * n_hat_y - sorb_phase * ln_cross_n_y
    # x_source_hat[2] = corb_phase * n_hat_z - sorb_phase * ln_cross_n_z
    # y_source_hat[0] = sorb_phase * n_hat_x + corb_phase * ln_cross_n_x
    # y_source_hat[1] = sorb_phase * n_hat_y + corb_phase * ln_cross_n_y
    # y_source_hat[2] = sorb_phase * n_hat_z + corb_phase * ln_cross_n_z
    # z_source_hat[0] = ln_hat_x
    # z_source_hat[1] = ln_hat_y
    # z_source_hat[2] = ln_hat_z
    ##############################

    angles = {"theta": theta, "psi": psi, "alpha": alpha, "TRef": TRef, "FRef": FRef}

    return angles
