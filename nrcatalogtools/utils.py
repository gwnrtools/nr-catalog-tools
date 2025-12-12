import functools
import os
import pathlib
import shutil

import lal
import requests

import sxs

if os.getenv("NR_CATALOG_CACHE"):
    nrcatalog_cache_dir = (
        pathlib.Path(os.getenv("NR_CATALOG_CACHE")).expanduser().resolve()
    )
else:
    nrcatalog_cache_dir = pathlib.Path("~/.cache/").expanduser().resolve()
nr_group_tags = {}
nr_group_tags["SXS"] = "SXS"
nr_group_tags["RIT"] = "RIT"
nr_group_tags["MAYA"] = "MAYA"
nr_group_tags["UNKNOWN"] = "UNKNOWN"

rit_catalog_info = {}
rit_catalog_info["cache_dir"] = nrcatalog_cache_dir / "RIT"
rit_catalog_info["metadata_dir"] = rit_catalog_info["cache_dir"] / "metadata/"
rit_catalog_info["data_dir"] = rit_catalog_info["cache_dir"] / "data/"
rit_catalog_info["url"] = "https://ccrgpages.rit.edu/~RITCatalog/"
rit_catalog_info["metadata_url"] = rit_catalog_info["url"] + "/Metadata/"
rit_catalog_info["data_url"] = rit_catalog_info["url"] + "/Data/"
rit_catalog_info["possible_resolutions"] = [
    100,
    120,
    88,
    84,
    118,
    130,
    140,
    144,
    160,
    200,
]
rit_catalog_info["metadata_file_fmts"] = [
    "RIT:BBH:{:04d}-n{:03d}-id{:d}_Metadata.txt",
    "RIT:eBBH:{:04d}-n{:03d}-ecc_Metadata.txt",
]
rit_catalog_info["waveform_file_fmts"] = [
    "ExtrapStrain_RIT-BBH-{:04d}-n{:03d}.h5",
    "ExtrapStrain_RIT-eBBH-{:04d}-n{:03d}.h5",
]
rit_catalog_info["psi4_file_fmts"] = [
    "ExtrapPsi4_RIT-BBH-{:04d}-n{:03d}-id{:d}.tar.gz",
    "ExtrapPsi4_RIT-eBBH-{:04d}-n{:03d}-ecc.tar.gz",
]
rit_catalog_info["max_id_val"] = 6

maya_catalog_info = {
    "cache_dir": nrcatalog_cache_dir / "MAYA",
    "data_url": "https://cgpstorage.ph.utexas.edu/",
    "metadata_url": "https://cgpstorage.ph.utexas.edu/MAYAmetadata.pkl",
}
maya_catalog_info["data_dir"] = maya_catalog_info["cache_dir"] / "data/"
maya_catalog_info["metadata_dir"] = maya_catalog_info["cache_dir"] / "metadata"

sxs_catalog_info = {
    "cache_dir": nrcatalog_cache_dir / "SXS",
    "data_url": "https://www.black-holes.org/waveforms/",
    "metadata_url": "https://www.black-holes.org/waveforms/metadata.json",
}
sxs_catalog_info["data_dir"] = sxs_catalog_info["cache_dir"] / "data/"
sxs_catalog_info["metadata_dir"] = sxs_catalog_info["cache_dir"] / "metadata"


def url_exists(link, num_retries=100):
    """Check if a given URL exists on the web.

    Args:
        link : complete web URL

    Returns:
        bool: True/False whether the URL could be found on WWW.
    """
    requests.packages.urllib3.disable_warnings()
    for n in range(num_retries):
        try:
            response = requests.head(link, verify=False)
            if response.status_code == requests.codes.ok:
                return True
            else:
                return False
        except Exception:
            continue
    return False


def download_file(url, path, progress=False, if_newer=True):
    """
    Download a file from the given URL to the specified local path.

    This function attempts to download the file at `url` and save it to `path`.
    It first tries to use the `sxs.utilities.downloads.download_file` utility (if available).
    If that fails, it falls back to using the Python `requests` package, with SSL
    verification disabled and up to 100 retry attempts.

    Args:
        url (str): The URL to download the file from.
        path (str or pathlib.Path): The destination path where the file should be saved.
        progress (bool, optional): Whether to show a progress bar if supported.
        if_newer (bool, optional): Only download if the remote file is newer than the local file.

    Returns:
        The path (as a pathlib.Path object) to the downloaded file.

    Raises:
        RuntimeError: If the file could not be downloaded due to network or server errors.
    """
    if url_exists(url):
        try:
            return sxs.utilities.downloads.download_file(
                url, path, progress=progress, if_newer=if_newer
            )
        except Exception:
            requests.packages.urllib3.disable_warnings()
            for n in range(100):
                try:
                    r = requests.get(
                        url, verify=False, stream=True, allow_redirects=True
                    )
                    break
                except Exception:
                    continue
            if r.status_code != 200:
                print(f"An error occurred when trying to access <{url}>.")
                try:
                    print(r.json())
                except Exception:
                    pass
                r.raise_for_status()
                raise RuntimeError()  # Will only happen if the response was not strictly an error
            r.raw.read = functools.partial(r.raw.read, decode_content=True)
            path = pathlib.Path(path).expanduser().resolve()
            with path.open("wb") as f:
                shutil.copyfileobj(r.raw, f)
    return path


def call_with_timeout(myfunc, args=(), kwargs={}, timeout=5):
    """
    Call a function with a time limit in a separate process.

    Executes the provided function `myfunc` with given positional (`args`) and keyword
    arguments (`kwargs`) in a separate process. If the function does not complete
    within `timeout` seconds, the process is terminated and an exception is raised.
    If the function completes within the timeout, its result is returned.

    Args:
        myfunc (callable): The function to execute.
        args (tuple, optional): Positional arguments to pass to `myfunc`. Defaults to ().
        kwargs (dict, optional): Keyword arguments to pass to `myfunc`. Defaults to {}.
        timeout (int or float, optional): Maximum allowed execution time in seconds. Defaults to 5.

    Returns:
        The result of `myfunc(*args, **kwargs)` if completed within the timeout.

    Raises:
        Exception: If the function does not complete within the specified timeout.
    """

    from multiprocessing import Process, Queue

    def funcwrapper(p, *args, **kwargs):
        """
        This thin wrapper calls the user-provided function, and puts
        its result into the multiprocessing `Queue`  so that it can be
        obtained via `Queue().get()`.
        """
        res = myfunc(*args, **kwargs)
        p.put(res)

    queue = Queue()
    task = Process(target=funcwrapper, args=(queue, *args))
    task.start()
    task.join(timeout=timeout)
    task.terminate()
    try:
        result = queue.get(timeout=0)
        return result
    except Exception:
        raise Exception("Timeout")


def time_to_physical(M):
    """
    Factor to convert time from dimensionless units to SI units

    parameters
    ----------
    M: mass of system in the units of solar mass

    Returns
    -------
    converting factor
    """

    return M * lal.MTSUN_SI


def amp_to_physical(M, D):
    """
    Factor to rescale strain to mass M and distance D convert from
    dimensionless units to SI units

    parameters
    ----------
    M: mass of the system in units of solar mass
    D: Luminosity distance in units of megaparsecs

    Returns
    -------
    Scaling factor
    """

    return lal.G_SI * M * lal.MSUN_SI / (lal.C_SI**2 * D * 1e6 * lal.PC_SI)


def amplitude_phase_frequency_from_complex_mode(hlm):
    """
    Compute amplitude, phase, and instantaneous frequency from a complex mode time series.

    Parameters
    ----------
    hlm : tuple of (real, imag) pycbc.types.TimeSeries, or a single complex pycbc.types.TimeSeries
        Either a tuple of real and imaginary parts of a mode (as PyCBC TimeSeries with matching sample_times),
        or a single complex-valued PyCBC TimeSeries.

    Returns
    -------
    amp : pycbc.types.TimeSeries
        The instantaneous amplitude as a function of time.
    phase : pycbc.types.TimeSeries
        The instantaneous phase as a function of time.
    freq : pycbc.types.TimeSeries
        The instantaneous frequency (cycles per unit time) as a function of time.
    """
    import numpy as np
    from pycbc.types import TimeSeries

    # Check if hlm is a tuple with PyCBC TimeSeries (real, imag)
    if isinstance(hlm, tuple) and len(hlm) == 2:
        re, im = hlm
        h_complex = re.numpy() + 1j * im.numpy()
        # Assume re/im have sample_times attribute if PyCBC TimeSeries
        if hasattr(re, "sample_times"):
            t = re.sample_times.numpy()
        else:
            raise AttributeError(
                "Real/Imag PyCBC TimeSeries objects must have sample_times."
            )
        delta_t = re.delta_t
    else:
        # Accept complex-valued PyCBC TimeSeries input
        if isinstance(hlm, TimeSeries) and np.iscomplexobj(hlm):
            h_complex = hlm.numpy()
            delta_t = hlm.delta_t
            re = hlm  # use for .start_time and .delta_t
        else:
            raise ValueError(
                "Input must be a tuple of PyCBC TimeSeries (re, im), or a complex-valued PyCBC TimeSeries."
            )

    # Compute amplitude
    amp_arr = np.abs(h_complex)

    amp = TimeSeries(amp_arr, delta_t=delta_t, epoch=re.start_time)

    # Compute phase
    phase_arr = np.unwrap(np.angle(h_complex))
    phase = TimeSeries(phase_arr, delta_t=delta_t, epoch=re.start_time)

    # Compute dphase/dt (frequency) as a TimeSeries (using uniform time steps)
    # (careful: np.gradient(y, dx) is d(y)/d(x) for uniform x spacing dx)
    dphase_dt_arr = np.gradient(phase_arr, re.delta_t)
    freq = TimeSeries(dphase_dt_arr / 2 / np.pi, delta_t=delta_t, epoch=re.start_time)
    return amp, phase, freq
