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
        lalsim.SimInspiralModeArrayActivateMode(mode_array_lal, mode[0],
                                                mode[1])
    lalsim.SimInspiralWaveformParamsInsertModeArray(waveform_dictionary,
                                                    mode_array_lal)

    return waveform_dictionary


def get_lal_mode_dictionary_from_lmax(lmax):
    """
    Get LALDict with modes derived from `lmax`.


    Parameters
    ----------
    lmax: max value of \ell to use

    Returns
    -------
    waveform_dictionary: LALDict with all modes upto `lmax` included

    """

    mode_array = [[ell, emm] for ell in range(2, lmax + 1)
                  for emm in range(-ell, ell + 1)]
    return get_lal_mode_dictionary(mode_array)


def get_modes_from_lvcnr_file(path_to_file,
                              Mtot,
                              distance,
                              srate,
                              lmax=4,
                              f_low=None):
    """
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
                -1, Mtot, path_to_file)
        else:
            spin_args = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(
                f_low, Mtot, path_to_file)

        mass_args = list(
            mtotal_eta_to_mass1_mass2(Mtot * MSUN, h5file.attrs["eta"]))

        try:
            eccentricity = float(h5file.attrs["eccentricity"])
        except ValueError:
            eccentricity = None

    values_mode_array = lalsim.SimInspiralWaveformParamsLookupModeArray(
        waveform_dict)
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


def get_strain_from_lvcnr_file(path_to_file,
                               Mtot,
                               distance,
                               srate,
                               mode_array=None):
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

    fixed_args = [distance * lal.PC_SI * 1e6, 0, 0, 0, 0, 0, 1 / srate]

    with h5py.File(path_to_file) as h5file:
        if mode_array is not None:
            waveform_dict = get_lal_mode_dictionary(mode_array)
        else:
            waveform_dict = lal.CreateDict()

        lalsim.SimInspiralWaveformParamsInsertNumRelData(
            waveform_dict, path_to_file)
        f_low = h5file.attrs["f_lower_at_1MSUN"] / Mtot

        if h5file.attrs["Format"] < 3:
            spin_args = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(
                -1, Mtot, path_to_file)
        else:
            spin_args = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(
                f_low, Mtot, path_to_file)

        mass_args = list(
            mtotal_eta_to_mass1_mass2(Mtot * lal.MSUN, h5file.attrs["eta"]))

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




##################################
# Generate time domain waveforms
##################################

def Norm(vec_x, vec_y, vec_z):
    ''' Compute the norm of a vector.

    Parameters
    ----------
    vec_x, vec_y, vec_z : float
                          The components of a three-vector

    Returns
    -------
    norm : float
           The norm of the three-vector.
    '''

    vec = np.array([vec_x, vec_y, vec_z])

    return np.sqrt(np.dot(vec, vec))

def Normalize(vec_x, vec_y, vec_z):
    ''' Normalize a vector.

    Parameters
    ----------
    vec_x, vec_y, vec_z : float
                          The components of a three-vector

    Returns
    -------
    vec_x, vec_y, vec_z : float
                          The normalized components of the three-vector.
    '''

    norm_vec = Norm(vec_x, vec_y, vec_z)

    return vec_x/norm_vec, vec_y/norm_vec, vec_z/norm_vec\

def CrossProduct(vec1_x, vec1_y, vec1_z, vec2_x, vec2_y, vec2_z):
    ''' Compute the cross product of the two input vectors

    Parameters
    ----------
    vec1_x, vec1_y, vec1_z, vec2_x, vec2_y, vec2_z  : float
                                                      The components of the two three-vectors :math:`\vec{v_1}, \vec{v_2}`

    Returns
    -------
    vec1_cross_vec2_x, vec1_cross_vec2_y, vec1_cross_vec2_z : float
                                                              The cross product :math:`\vec{v_1} \times \vec{v_2}`
    '''

    vec1_cross_vec2_x = vec1_y * vec2_z - vec1_z * vec2_y
    vec1_cross_vec2_y = vec1_z * vec2_x - vec1_x * vec2_z
    vec1_cross_vec2_z = vec1_x * vec2_y - vec1_y * vec2_x

    return vec1_cross_vec2_x, vec1_cross_vec2_y, vec1_cross_vec2_z


def CheckInterpReq(H5File, ref_time):
    ''' Check if the required reference time is different from
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
    '''

    avail_ref_time = H5File.attrs('ref_time')

    if abs(avail_ref_time-ref_time)<1e-5:
        return False, avail_ref_time
    else:
        return True, avail_ref_time


def GetRefFreqFromRefTime(H5File, ref_time):
    ''' Get the reference frequency from reference time 
    
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
    '''
    
    time, freq = H5File.attrs['Omega-vs-time']
    
    from scipy.interpolate import interp1d
    
    RefFreq = interp1d(time, freq, kind='cubic')[ref_time]
    
    return RefFreq

def GetRefTimeFromRefFreq(H5File, ref_freq):
    ''' Get the reference time from reference frequency 
    
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
    '''
    from scipy.interpolate import interp1d
    time, freq = H5File.attrs['Omega-vs-time']
    
    
    
    RefTime = interp1d(freq, time, kind='cubic')[ref_freq]
    
    return RefTime

def CheckNRAttrs(H5File, ReqAttrs):
    ''' Check if the NR file contains all the attributes
    specified.

    Parameters
    ----------
    H5File : file object
             The waveform h5 file handle.
    ReqAttrs : list
               A list of attribute keys.
    Returns
    -------
    Present : bool
              Whether or not all specified attributes are present.
    AbsentAttrs : list
                 The attributes that are absent.
    '''

    all_attrs = list(H5File.attrs.keys())

    AbsentAttrs = []
    Present=True

    for item in ReqAttrs:
        if item not in all_attrs:
            Present=False
            AbsentAttrs.append(item)

    return Present, AbsentAttrs

def GetInterpRefValuesFromH5File(H5File, ReqTSAttrs, ref_time):
    ''' Get the interpolated reference values at a given reference time
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

    Notes
    -----


    '''
    from scipy.interpolate import interp1d

    params = {}

    for key in ReqTSAttrs:

        time, var = H5File.attrs[key]
        RefVal = interp1d(time, var, kind='cubic')[ref_time]
        params.update({key : RefVal})
    return params


def GetRefValsfromH5File(H5File, ReqAttrs):
    ''' Get the reference values from the NR HDF5 file

    Parameters
    ----------
    H5File : file object
             The waveform h5 file handle.
    ReqTSAttrs : list
               A list of attribute keys.
    Returns
    -------
    params : dict
             The parameter values at the reference time.
    '''

    params = {}

    for key in ReqAttrs:
        RefVal = H5File.attrs[key]
        params.update({key : RefVal})
    return params


def GetNRToLALRotationAnglesFromH5(H5File, PhiRef, FRef, TRef):
    ''' Get the angular coordinates :math:`\theta, \phi`
    and the rotation angle :math:`\alpha` from the H5 file

    Parameters
    ----------
    H5File : file object
            The waveform h5 file handle.

    PhiRef : float
             The reference orbital phase.
    TRef : float
           The reference time

    Returns
    -------
    angles : dict
             The angular corrdinates Theta, Phi,  and the rotation angles Alpha.

    Notes
    -----
    Variable definitions.

    theta : Returned inclination angle of source in NR coordinates.
    psi :   Returned azimuth angle of source in NR coordinates.
    alpha: Returned polarisation angle.
    H5File: h5py object of the NR HDF5 file.
    inclination: Inclination of source in LAL source frame.
    phi_ref: Orbital reference phase.
    fRef: Reference frequency.
   '''

    # tolerence for sanity checks
    # Note: there are some checks 
    # which use other tol values.
    # Attempt to unify.
    tol = 1e-3

    # Compute the angles necessary to rotate from the intrinsic NR source frame
    # into the LAL frame. See DCC-T1600045 for details.

    # Following section IV of DCC-T1600045
    # Step 1: Define Phi = phiref
    orb_phase = PhiRef

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

    # Check if interpolation is necessary
    Interp, avail_ref_time = CheckInterpReq(H5File, TRef)

    # Get the reference orbital frequency from reference time
    avail_ref_freq = GetRefFreqFromRefTime(H5File, avail_ref_time)

    # Check if time series data of NR coordinate system is present
    # Attributes pertinent to interpolation
    ReqTSAttrs = ['LNhatx-vs-time', 'LNhaty-vs-time', 'LNhatz-vs-time', \
                'position1x-vs-time', 'position1y-vs-time', 'position1z-vs-time', \
                'position2x-vs-time', 'position2y-vs-time', 'position2z-vs-time']

    # Default attributes in case of no interpolation
    ReqDefAttrs = ['LNhatx', 'LNhaty', 'LNhatz', 'nhatx', 'nhaty', 'nhatz']

    RefCheckInterp = CheckNRAttrs(H5File, ReqTSAttrs)
    RefCheckDef    = CheckNRAttrs(H5File, ReqDefAttrs)


    if (Interp==True and RefCheckInterp==True):
    # If iterpolation is required and time series data is available

        ref_time = GetRefTimeFromRefFreq(H5File, FRef) # Look in here. Incorp below error msg

        # Warning 1
        # Implement this Warning
        # XLAL_CHECK( ref_time!=XLAL_FAILURE, XLAL_FAILURE, "Error computing reference time.
        # Try setting fRef equal to the f_low given by the NR simulation or to a value <=0 to deactivate
        # fRef for a non-precessing simulation.\n");

        RefParams = GetInterpRefValuesFromH5File(H5File, ReqTSAttrs, TRef)

        # r21 vec
        r21_x = RefParams['pos1x'] - RefParams['pos2x']
        r21_y = RefParams['pos1y'] - RefParams['pos2y']
        r21_z = RefParams['pos1z'] - RefParams['pos2z']


    elif RefCheckInterp==False or RefCheckDef==False:
        # If either of the data is not available
        raise ValueError(f'Cannot compute the LAL frame from the data available in the h5 file! \
        Please choose the reference time {avail_ref_time} or reference frequency {avail_ref_freq}')

    elif (Interp==False and RefCheckDef==True):
        # If iterpolation is not required and default ref data is available
        RefParams = GetRefValuesFromH5File(H5File, ReqDefAttrs)

        ln_hat_x, ln_hat_y, ln_hat_z = RefParams['LNhat']
        n_hat_x, n_hat_y, n_hat_z = RefParams['Nhat']

    # Normalize the vectors
    n_hat_x, n_hat_y, n_hat_z = Normalize(n_hat_x, n_hat_y, n_hat_z)
    ln_hat_x, ln_hat_y, ln_hat_z = Normalize(ln_hat_x, ln_hat_y, ln_hat_z)

    n_hat = np.array([n_hat_x, n_hat_y, n_hat_z])
    ln_hat = np.array([ln_hat_x, ln_hat_y, ln_hat_z])

    # 2.3: Carryout vector math to get Zref in the lal wave frame
    corb_phase = np.cos(orb_phase)
    sorb_phase = np.sin(orb_phase)
    sinclination = np.sin(inclination)
    cinclination = np.cos(inclination)

    ln_cross_n_x, ln_cross_n_y, ln_cross_n_z  = CrossProduct(ln_hat_x, ln_hat_y, ln_hat_z, n_hat_x, n_hat_y, n_hat_z)
    ln_cross_n_x, ln_cross_n_y, ln_cross_n_z = Normalize(ln_cross_n_x, ln_cross_n_y, ln_cross_n_z)
    ln_cross_n = np.array([ln_cross_n_x, ln_cross_n_y, ln_cross_n_z])

    z_wave_x = sinclination * (sorb_phase * n_hat_x + corb_phase * ln_cross_n_x)
    z_wave_y = sinclination * (sorb_phase * n_hat_y + corb_phase * ln_cross_n_y)
    z_wave_z = sinclination * (sorb_phase * n_hat_z + corb_phase * ln_cross_n_z)

    z_wave_x += cinclination * ln_hat_x
    z_wave_y += cinclination * ln_hat_y
    z_wave_z += cinclination * ln_hat_z

    z_wave_x, z_wave_y, z_wave_z = Normalize(z_wave_x, z_wave_y, z_wave_z)
    z_wave = np.array([z_wave_x, z_wave_y, z_wave_z])

    # Step 3.1: Extract theta and psi from Z in the lal wave frame
    # NOTE: Theta can only run between 0 and pi, so no problem with arccos here
    theta = acos(z_wave_z);

    # Degenerate if Z_wave[2] == 1. In this case just choose psi randomly,
    # the choice will be cancelled out by alpha correction (I hope!)

    # If theta is very close to the poles
    if(abs(z_wave_z - 1.0 ) < 0.000001):
        psi = 0.5

    else:
        # psi can run between 0 and 2pi, but only one solution works for x and y */
        # Possible numerical issues if z_wave_x = sin(theta) */
        if (abs(z_wave_x / sin(theta)) > 1.):

            if (abs(z_wave_x / sin(theta)) < 1.00001):

                if ((z_wave_x * sin(theta)) < 0.):
                    psi =  np.pi #LAL_PI;

                else:
                    psi = 0.

            else:
                print(f'z_wave_x = {z_wave_x}')
                print(f'sin(theta) = {sin(theta)}')
                raise ValueError('Z_x cannot be bigger than sin(theta). Please email the developers.')

        else:
            psi = acos(z_wave_x / sin(theta))

        y_val = sin(*psi) * sin(theta);

        # If z_wave[1] is negative, flip psi so that sin(psi) goes negative
        # while preserving cos(psi) */
        if (z_wave_y < 0.):
            psi = 2 * np.pi - psi
            y_val = sin(psi) * sin(theta)

        if (abs(y_val - z_wave_y) > 0.005):
            print(f'orb_phase = {orb_phase}')
            print(f'y_val = {y_val}, z_wave_y = {z_wave_y}, abs(y_val - z_wave_y) = {abs(y_val - z_wave_y)}')
            raise ValueError('Math consistency failure! Please contact the developers.')

    # 3.2: Compute the vectors theta_hat and psi_hat */
    stheta = sin(theta)
    ctheta = cos(theta)
    spsi = sin(psi)
    cpsi = cos(psi)

    theta_hat_x = cpsi * ctheta
    theta_hat_y = spsi * ctheta
    theta_hat_z = - stheta
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

    # x_source_hat[0] = corb_phase * n_hat_x - sorb_phase * ln_cross_n_x;
    # x_source_hat[1] = corb_phase * n_hat_y - sorb_phase * ln_cross_n_y;
    # x_source_hat[2] = corb_phase * n_hat_z - sorb_phase * ln_cross_n_z;
    # y_source_hat[0] = sorb_phase * n_hat_x + corb_phase * ln_cross_n_x;
    # y_source_hat[1] = sorb_phase * n_hat_y + corb_phase * ln_cross_n_y;
    # y_source_hat[2] = sorb_phase * n_hat_z + corb_phase * ln_cross_n_z;
    # z_source_hat[0] = ln_hat_x;
    # z_source_hat[1] = ln_hat_y;
    # z_source_hat[2] = ln_hat_z;
    ##############################


    angles = {'theta' : theta, 'psi' : psi, 'alpha' : alpha}

    return angles

##############################################################################
