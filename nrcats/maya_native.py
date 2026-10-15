"""Read MAYA's native (``maya_format``) files: finite-radius Psi4 and horizon tracks.

The native files (0.4-7 GB each, ``<data_url>/maya_format/<sim>.h5``) hold
Psi4 modes at up to 16 extraction radii under ``radiative/psi4/radius=R``
and the puncture and horizon data under ``compact_object/object=i``.  Only
the parts needed are read, over HTTP with fsspec (or from a local copy), and
the small result is cached as ``<cache_dir>/<sim>_r<R>.npz``.

From the horizon tracks this module builds the reference-epoch metadata
that nrcats uses when a waveform has no LVCNR file
(``reference_time``, ``reference_mass1/2``, ``reference_position1/2``,
``reference_orbital_frequency``, ``reference_dimensionless_spin1/2``), with
body 1 the heavier hole.  :mod:`nrcats.frames` then gives the LAL frame at
that epoch with no MAYA-specific code.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default extraction radius: the largest available at or below this value,
# the choice used in the October 2026 comparison with NRSur7dq4.
DEFAULT_MAX_EXTRACTION_RADIUS = 145.0
# Rows of a compact_object dataset read per HTTP request.
TRACK_READ_BLOCK_ROWS = 100000
# Smallest spacing kept in the horizon tracks (they are written every few
# coarse steps and move on the orbital time scale, ~100 M).
TRACK_MIN_SPACING = 0.25
TRACK_COLUMNS = ("time", "x", "y", "z", "vx", "vy", "vz", "Sx", "Sy", "Sz")


def _open_native_file(url_or_path):
    """Open a native file from a local path or a URL; returns (h5_file, handles to close)."""
    import h5py

    if os.path.exists(str(url_or_path)):
        h5_file = h5py.File(url_or_path, "r")
        return h5_file, [h5_file]
    try:
        import fsspec
    except ImportError as excep:
        raise ImportError(
            "Reading MAYA native files over HTTP needs fsspec and aiohttp "
            "(pip install fsspec aiohttp), or a local copy of the file."
        ) from excep
    remote_file = fsspec.filesystem("https").open(str(url_or_path), "rb", block_size=2**22, cache_type="readahead")
    h5_file = h5py.File(remote_file, "r")
    return h5_file, [h5_file, remote_file]


def available_extraction_radii(h5_file):
    """Coordinate radii of the Psi4 extraction spheres in a native file, sorted."""
    return sorted(float(group_name.split("=")[1]) for group_name in h5_file["radiative/psi4"].keys())


def choose_extraction_radius(radii, extraction_radius=None, max_radius=DEFAULT_MAX_EXTRACTION_RADIUS):
    """``extraction_radius`` if available, else the largest radius at or below ``max_radius``."""
    if extraction_radius is not None:
        matching = [radius for radius in radii if abs(radius - extraction_radius) < 1e-6]
        if not matching:
            raise ValueError(f"No Psi4 extraction at radius {extraction_radius}; available radii: {radii}")
        return matching[0]
    below_max = [radius for radius in radii if radius <= max_radius]
    return max(below_max) if below_max else min(radii)


def _read_track(dataset):
    """Finite samples of each ``TRACK_COLUMNS`` column of one compact_object dataset.

    The datasets interleave rows written by different diagnostics, so each
    column has its own sample times.  Returns ``{column: values,
    column + "_time": times}``, sorted, without repeated times, and thinned
    to a spacing of at least ``TRACK_MIN_SPACING``.
    """
    header = [str(column) for column in dataset.attrs["header"]]
    present_columns = [column for column in TRACK_COLUMNS if column in header]
    column_indices = [header.index(column) for column in present_columns]
    blocks = []
    for start_row in range(0, dataset.shape[0], TRACK_READ_BLOCK_ROWS):
        block = dataset[start_row : min(dataset.shape[0], start_row + TRACK_READ_BLOCK_ROWS)]
        blocks.append(block[:, column_indices])
    rows = np.concatenate(blocks)
    rows = rows[np.isfinite(rows[:, 0])]
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    track = {}
    for index, column in enumerate(present_columns):
        if column == "time":
            continue
        finite = np.isfinite(rows[:, index])
        sample_times, sample_values = rows[finite, 0], rows[finite, index]
        sample_times, unique_indices = np.unique(sample_times, return_index=True)
        sample_values = sample_values[unique_indices]
        spacing_bins = np.floor(sample_times / TRACK_MIN_SPACING)
        keep = np.concatenate([[True], np.diff(spacing_bins) > 0]) if len(sample_times) else []
        track[column] = sample_values[keep]
        track[column + "_time"] = sample_times[keep]
    return track


def read_native_data(url_or_path, extraction_radius=None, ell_max=4, cache_path=None):
    """Psi4 modes at one radius, the horizon tracks and the attributes of a native file.

    Parameters
    ----------
    url_or_path : str
                  Local path or URL of the native file.
    extraction_radius : float, optional
                        Coordinate radius to read. Defaults to the largest
                        at or below ``DEFAULT_MAX_EXTRACTION_RADIUS``.
    ell_max : int, optional
              Largest ell to read. Defaults to 4.
    cache_path : str or Path, optional
                 Directory for the cached npz. Without it nothing is cached.

    Returns
    -------
    native_data : dict
                  ``time``, ``rpsi4_modes`` (``{(ell, emm): r * Psi4}``),
                  ``extraction_radius``, ``available_radii``, ``attrs``
                  (file attributes) and ``tracks``
                  (``{object_index: {column: values, column + "_time": times}}``).
    """
    cached_file = None
    if cache_path is not None and extraction_radius is not None:
        cached_file = Path(cache_path) / f"{Path(str(url_or_path)).stem}_r{extraction_radius:.2f}.npz"
        if cached_file.exists():
            return _load_cached(cached_file, ell_max)

    h5_file, handles = _open_native_file(url_or_path)
    try:
        radii = available_extraction_radii(h5_file)
        radius = choose_extraction_radius(radii, extraction_radius)
        if cache_path is not None:
            cached_file = Path(cache_path) / f"{Path(str(url_or_path)).stem}_r{radius:.2f}.npz"
            if cached_file.exists():
                return _load_cached(cached_file, ell_max)
        psi4_group = h5_file[f"radiative/psi4/radius={radius:.2f}"]
        time = np.asarray(psi4_group["time"][:], dtype=float)
        rpsi4_modes = {}
        for ell in range(2, ell_max + 1):
            for emm in range(-ell, ell + 1):
                mode_group = psi4_group[f"modes/l={ell}/m={emm}"]
                psi4_mode = np.asarray(mode_group["real"][:]) + 1j * np.asarray(mode_group["imaginary"][:])
                rpsi4_modes[(ell, emm)] = radius * psi4_mode
        attrs = {key: _to_plain(value) for key, value in h5_file.attrs.items()}
        tracks = {}
        for object_name, dataset in h5_file["compact_object"].items():
            header = [str(column) for column in dataset.attrs.get("header", [])]
            if "time" in header and "x" in header:
                tracks[int(object_name.split("=")[1])] = _read_track(dataset)
    finally:
        for handle in handles:
            handle.close()

    native_data = {
        "time": time,
        "rpsi4_modes": rpsi4_modes,
        "extraction_radius": radius,
        "available_radii": radii,
        "attrs": attrs,
        "tracks": tracks,
    }
    if cached_file is not None:
        _save_cached(cached_file, native_data)
    return native_data


def _to_plain(value):
    """h5py attribute value as a JSON-serialisable Python value."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save_cached(cached_file, native_data):
    cached_file.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"time": native_data["time"]}
    for (ell, emm), mode_data in native_data["rpsi4_modes"].items():
        arrays[f"rpsi4_l{ell}_m{emm}"] = mode_data
    for object_index, track in native_data["tracks"].items():
        for column, values in track.items():
            arrays[f"track{object_index}_{column}"] = values
    arrays["info"] = np.array(json.dumps({
        "extraction_radius": native_data["extraction_radius"],
        "available_radii": native_data["available_radii"],
        "attrs": native_data["attrs"],
    }))
    temporary_file = cached_file.with_suffix(".tmp.npz")
    np.savez(temporary_file, **arrays)
    temporary_file.replace(cached_file)


def _load_cached(cached_file, ell_max):
    with np.load(cached_file) as stored:
        info = json.loads(str(stored["info"]))
        rpsi4_modes, tracks = {}, {}
        for key in stored.files:
            if key.startswith("rpsi4_l"):
                ell_part, emm_part = key[len("rpsi4_l"):].split("_m")
                if int(ell_part) <= ell_max:
                    rpsi4_modes[(int(ell_part), int(emm_part))] = stored[key]
            elif key.startswith("track"):
                object_part, column = key[len("track"):].split("_", 1)
                tracks.setdefault(int(object_part), {})[column] = stored[key]
        return {
            "time": stored["time"],
            "rpsi4_modes": rpsi4_modes,
            "extraction_radius": info["extraction_radius"],
            "available_radii": info["available_radii"],
            "attrs": info["attrs"],
            "tracks": tracks,
        }


def _track_value_at(track, column, time):
    """Cubic-spline value of one track column at ``time``, from its finite samples."""
    from scipy.interpolate import CubicSpline

    if column not in track:
        raise KeyError(f"The horizon track has no column {column!r}.")
    column_times = track.get(column + "_time", track.get("time"))
    finite = np.isfinite(track[column])
    sample_times, sample_values = column_times[finite], track[column][finite]
    sample_times, unique_indices = np.unique(sample_times, return_index=True)
    sample_values = sample_values[unique_indices]
    if not sample_times[0] <= time <= sample_times[-1]:
        raise ValueError(
            f"t = {time} is outside the {column!r} track ({sample_times[0]} to {sample_times[-1]})."
        )
    return float(CubicSpline(sample_times, sample_values)(time))


def _track_vector_at(track, columns, time):
    return np.array([_track_value_at(track, column, time) for column in columns])


def reference_metadata_from_tracks(native_data, reference_time, waveform_time_offset=0.0):
    """Reference-epoch metadata from the horizon tracks, in the SXS form nrcats reads.

    Body 1 is the heavier hole.  Masses are the initial horizon masses of
    the file's attributes; positions, velocities and spins are interpolated
    at ``reference_time`` (spins at the end of their track if it ends
    earlier, see ``reference_spin_time``).  ``reference_orbital_frequency`` is the vector
    :math:`\\vec r \\times \\vec v / r^2` of the relative motion.  Spins are
    divided by the initial horizon mass squared.

    Parameters
    ----------
    native_data : dict
                  As returned by :func:`read_native_data`.
    reference_time : float
                     Coordinate time of the epoch in the tracks.
    waveform_time_offset : float, optional
                           Waveform time minus track time at the same
                           epoch; ``reference_time`` in the result is
                           ``reference_time + waveform_time_offset``, the
                           epoch on the waveform's time axis.

    Returns
    -------
    metadata : dict
    """
    attrs, tracks = native_data["attrs"], native_data["tracks"]
    if 0 not in tracks or 1 not in tracks:
        raise ValueError("The native file has no horizon tracks for objects 0 and 1.")
    horizon_masses = {0: float(attrs["horizon mass 0"]), 1: float(attrs["horizon mass 1"])}
    heavier_object, lighter_object = (0, 1) if horizon_masses[0] >= horizon_masses[1] else (1, 0)

    positions, velocities, spins, spin_times = {}, {}, {}, {}
    for object_index in (heavier_object, lighter_object):
        track = tracks[object_index]
        positions[object_index] = _track_vector_at(track, ("x", "y", "z"), reference_time)
        if all(column in track for column in ("vx", "vy", "vz")):
            velocities[object_index] = _track_vector_at(track, ("vx", "vy", "vz"), reference_time)
        else:
            step = 0.5
            velocities[object_index] = (
                _track_vector_at(track, ("x", "y", "z"), reference_time + step)
                - _track_vector_at(track, ("x", "y", "z"), reference_time - step)
            ) / (2 * step)
        spin_times[object_index] = _spin_sample_time(track, reference_time)
        spins[object_index] = _track_vector_at(track, ("Sx", "Sy", "Sz"), spin_times[object_index])

    relative_position = positions[heavier_object] - positions[lighter_object]
    relative_velocity = velocities[heavier_object] - velocities[lighter_object]
    orbital_frequency = np.cross(relative_position, relative_velocity) / np.dot(relative_position, relative_position)

    return {
        "reference_time": float(reference_time + waveform_time_offset),
        "reference_dynamics_time": float(reference_time),
        "reference_mass1": horizon_masses[heavier_object],
        "reference_mass2": horizon_masses[lighter_object],
        "reference_position1": positions[heavier_object].tolist(),
        "reference_position2": positions[lighter_object].tolist(),
        "reference_orbital_frequency": orbital_frequency.tolist(),
        "reference_dimensionless_spin1": (spins[heavier_object] / horizon_masses[heavier_object] ** 2).tolist(),
        "reference_dimensionless_spin2": (spins[lighter_object] / horizon_masses[lighter_object] ** 2).tolist(),
        "reference_frame_source": "MAYA native horizon tracks",
        "reference_spin_time": min(spin_times.values()),
    }


def _spin_sample_time(track, reference_time):
    """``reference_time``, or the last spin sample if the spin track ends before it.

    Older runs (e.g. the GT series) record the horizon spins only near the
    start.  As the MAYA LVCNR export does, the last available spins are used
    then; this is exact for aligned spins only, so precessing runs warn.
    """
    finite_spin_times = track.get("Sx_time", track.get("time"))[np.isfinite(track["Sx"])]
    last_spin_time = float(finite_spin_times.max())
    if reference_time <= last_spin_time:
        return reference_time
    logger.warning(
        f"The spin track ends at t = {last_spin_time}, before the reference time {reference_time}; "
        "using the spins at its end (exact only if the spins do not precess)."
    )
    return last_spin_time
