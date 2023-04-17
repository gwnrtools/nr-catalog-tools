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
