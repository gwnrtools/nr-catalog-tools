"""Source frames: putting waveform modes and metadata in the LAL frame at a reference epoch.

The LAL source frame at a reference epoch ``t_ref`` has

* ``z`` along ``L_hat``, the orbital angular momentum direction,
* ``x`` along ``n_hat``, the separation unit vector pointing from body 2 to body 1,
* ``y = z x x``,

with body 1 the heavier hole.  In this frame the NR-to-LAL polarisation angle is
exactly ``pi`` for every inclination and phase, so the observed strain is
``h+ - i hx = sum_lm h_lm  -2Y_lm(iota, pi/2 - phi_ref)``, which is what
``lalsimulation.SimInspiralNRWaveformGetHplusHcross`` computes after its own
rotation.  See ``docs/frame_conventions.md``.

A :class:`ReferenceFrame` records the triad, expressed in the simulation's own
coordinates, and the epoch it belongs to.  :func:`rotate_modes` turns modes from
the simulation frame into that frame with a single constant rotation; the
precession stays in the modes.  :func:`vector_to_frame` does the same for any
vector (spins, kick, remnant spin).

Convention (verified in the tests): ``sxs.WaveformModes.rotate(R)`` returns the
modes as seen in a frame rotated by ``R``, so a feature along ``v`` moves to
``R^-1 v R``.  To bring ``L_hat`` to ``z`` and ``n_hat`` to ``x`` the rotor is
therefore the one that takes ``(x, y, z)`` to ``(n_hat, L_hat x n_hat, L_hat)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import quaternionic

__all__ = [
    "ReferenceFrame",
    "orthonormal_triad",
    "triad_rotor",
    "rotate_modes",
    "vector_to_frame",
    "frame_from_vectors",
    "frame_from_lvcnr_attrs",
    "reference_metadata_from_lvcnr",
    "catalog_frame",
    "put_in_frame",
    "frame_of",
    "reference_metadata",
    "lal_frame_angles",
]

#: Tolerance on |L_hat| and |n_hat| before renormalising, as in LAL.
UNIT_TOLERANCE = 1e-3


def orthonormal_triad(L_hat, n_hat):
    """Return ``(n_hat, m_hat, L_hat)`` as an orthonormal right-handed triad.

    ``L_hat`` and ``n_hat`` are normalised (LAL checks they are within
    ``UNIT_TOLERANCE`` of unit length); ``n_hat`` is then made exactly
    perpendicular to ``L_hat``, and ``m_hat = L_hat x n_hat``.
    """
    L = np.asarray(L_hat, dtype=float)
    n = np.asarray(n_hat, dtype=float)
    for name, v in (("L_hat", L), ("n_hat", n)):
        norm = np.linalg.norm(v)
        if not np.isfinite(norm) or abs(norm - 1) > UNIT_TOLERANCE:
            raise ValueError(f"{name} has norm {norm}, not 1 within {UNIT_TOLERANCE}.")
    L = L / np.linalg.norm(L)
    n = n - np.dot(n, L) * L
    n = n / np.linalg.norm(n)
    m = np.cross(L, n)
    return n, m, L


def triad_rotor(L_hat, n_hat):
    """Unit quaternion taking ``(x, y, z)`` to ``(n_hat, L_hat x n_hat, L_hat)``."""
    n, m, L = orthonormal_triad(L_hat, n_hat)
    return quaternionic.array.from_rotation_matrix(np.column_stack([n, m, L]))


@dataclass
class ReferenceFrame:
    """A source frame and the epoch it belongs to.

    Attributes
    ----------
    name : str
        ``"lal"`` (LAL source frame), ``"simulation"`` (the catalog's own
        inertial frame, no rotation) or ``"waveform"`` (LAL frame built from the
        waveform-derived triad).
    method : str
        How the triad was obtained: ``"catalog"`` (published frame vectors or
        dynamics), ``"waveform"`` (radiation axis and orbital phase), or
        ``"none"`` for the simulation frame.
    L_hat, n_hat : numpy.ndarray or None
        The triad in the simulation's coordinates (orthonormalised).
    t_ref : float or None
        Reference epoch on the waveform's own time axis (M).
    f_ref_dimless : float or None
        (2,2) gravitational-wave frequency at ``t_ref`` in cycles per M
        (``M * f_GW``), when known.
    source : str
        Free-text description of where the triad came from (file, metadata keys).
    """

    name: str = "simulation"
    method: str = "none"
    L_hat: np.ndarray | None = None
    n_hat: np.ndarray | None = None
    t_ref: float | None = None
    f_ref_dimless: float | None = None
    source: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.L_hat is not None and self.n_hat is not None:
            n, _, L = orthonormal_triad(self.L_hat, self.n_hat)
            self.L_hat, self.n_hat = L, n

    @property
    def is_rotated(self) -> bool:
        return self.name != "simulation"

    @property
    def m_hat(self):
        return None if self.L_hat is None else np.cross(self.L_hat, self.n_hat)

    @property
    def rotor(self):
        """Rotor from the simulation frame to this frame (identity for ``simulation``)."""
        if not self.is_rotated:
            return quaternionic.one
        return triad_rotor(self.L_hat, self.n_hat)

    @property
    def matrix(self):
        """3x3 matrix whose rows are the frame axes in simulation coordinates."""
        if not self.is_rotated:
            return np.eye(3)
        return np.vstack([self.n_hat, self.m_hat, self.L_hat])

    def as_dict(self) -> dict:
        def _l(v):
            return None if v is None else [float(x) for x in v]

        return {
            "name": self.name,
            "method": self.method,
            "L_hat": _l(self.L_hat),
            "n_hat": _l(self.n_hat),
            "t_ref": self.t_ref,
            "f_ref_dimless": self.f_ref_dimless,
            "source": self.source,
            **self.extra,
        }


def frame_from_vectors(L_hat, n_hat, t_ref=None, f_ref_dimless=None, method="catalog", source="", name="lal"):
    """Build a LAL :class:`ReferenceFrame` from a triad given in simulation coordinates."""
    return ReferenceFrame(
        name=name, method=method, L_hat=np.asarray(L_hat, float), n_hat=np.asarray(n_hat, float),
        t_ref=t_ref, f_ref_dimless=f_ref_dimless, source=source,
    )


def frame_from_lvcnr_attrs(attrs, t_ref=None, total_mass_dimless=1.0, source="LVCNR file attributes"):
    """LAL frame from an LVCNR file's default ``LNhat``/``nhat`` attributes.

    For Format 1 files LAL ties these vectors to the start of the stored waveform
    and ignores any reference frequency; the measurements in
    ``docs/frame_conventions.md`` confirm that for RIT and MAYA.  ``t_ref`` should
    therefore be the first sample of the waveform for Format 1.  ``Omega`` (orbital
    angular frequency in units of 1/M) gives the (2,2) frequency there.
    """
    L = [float(attrs["LNhatx"]), float(attrs["LNhaty"]), float(attrs["LNhatz"])]
    n = [float(attrs["nhatx"]), float(attrs["nhaty"]), float(attrs["nhatz"])]
    f_ref = None
    if "Omega" in attrs:
        f_ref = float(attrs["Omega"]) / np.pi / total_mass_dimless
    return frame_from_vectors(L, n, t_ref=t_ref, f_ref_dimless=f_ref, source=source)


def rotate_modes(wfm, frame: ReferenceFrame):
    """Return ``wfm`` expressed in ``frame`` (a new object; the input is unchanged).

    Uses ``sxs.WaveformModes.rotate``.  nrcats metadata attached to ``wfm`` are
    carried over, and the frame is recorded in the result's metadata under
    ``"reference_frame"``.
    """
    if not frame.is_rotated:
        out = wfm.copy()
    else:
        out = wfm.rotate(frame.rotor)
    meta = getattr(wfm, "_metadata", None)
    out_meta = getattr(out, "_metadata", None)
    if isinstance(meta, dict) and isinstance(out_meta, dict):
        for k, v in meta.items():
            out_meta.setdefault(k, v)
        out_meta["reference_frame"] = frame
    return out


def vector_to_frame(v, frame: ReferenceFrame):
    """Components of a simulation-frame vector along the frame's axes."""
    return frame.matrix @ np.asarray(v, dtype=float)


def _group_final_value(h5, name):
    """Last value of an LVCNR ``*-vs-time`` group (romspline data with X and Y datasets)."""
    g = h5[name]
    return float(np.asarray(g["Y"])[-1])


def reference_metadata_from_lvcnr(h5, frame: ReferenceFrame | None = None):
    """Masses, spins and vectors from an open LVCNR file, expressed in ``frame``.

    Spins are the file's default ``spin1x..spin2z`` attributes (dimensionless, in
    the simulation frame at the file's reference epoch), projected onto the frame
    axes, which is what ``XLALSimInspiralNRWaveformGetSpinsFromHDF5File`` returns
    when no reference frequency is used.  ``frame`` defaults to the file's own
    LAL frame (:func:`frame_from_lvcnr_attrs`).  Body 1 must be the heavier hole;
    a file labelled the other way raises, since relabelling also rotates the
    orbital phase by pi and should be done explicitly.
    """
    attrs = h5.attrs
    if frame is None:
        frame = frame_from_lvcnr_attrs(attrs)
    m1, m2 = float(attrs["mass1"]), float(attrs["mass2"])
    if m1 < m2 * (1 - 1e-6):
        raise ValueError(f"Body 1 is the lighter hole in this file (mass1={m1}, mass2={m2}).")
    chi1 = np.array([float(attrs[f"spin1{c}"]) for c in "xyz"])
    chi2 = np.array([float(attrs[f"spin2{c}"]) for c in "xyz"])
    out = {
        "frame": frame,
        "mass1": m1 / (m1 + m2),
        "mass2": m2 / (m1 + m2),
        "q": m1 / m2,
        "chi1": vector_to_frame(chi1, frame),
        "chi2": vector_to_frame(chi2, frame),
        "L_hat": vector_to_frame(frame.L_hat, frame) if frame.is_rotated else None,
        "n_hat": vector_to_frame(frame.n_hat, frame) if frame.is_rotated else None,
        "t_ref": frame.t_ref,
        "f_ref_dimless": frame.f_ref_dimless,
        "format": int(attrs.get("Format", 1)),
    }
    if "eccentricity" in attrs:
        out["eccentricity"] = float(attrs["eccentricity"])
    keys = ("remnant-mass-vs-time", "remnant-spinx-vs-time", "remnant-spiny-vs-time", "remnant-spinz-vs-time")
    if all(k in h5 for k in keys):
        # LVCNR stores the remnant spin dimensionless (the SXS exporter writes the
        # horizon's chiInertial), like the component spins.
        out["remnant_mass"] = _group_final_value(h5, keys[0])
        out["remnant_chi"] = vector_to_frame(np.array([_group_final_value(h5, k) for k in keys[1:]]), frame)
    return out


FRAME_NAMES = ("lal", "simulation", "waveform")


def _lvcnr_vectors_at(h5, t_ref):
    """``L_hat``, ``n_hat`` and the spins interpolated at ``t_ref`` from a Format >= 2 file."""
    from nrcats.lvc import compute_lal_source_frame_by_interp, read_lvcnr_group

    keys = [f"LNhat{c}-vs-time" for c in "xyz"] + [f"position{i}{c}-vs-time" for i in (1, 2) for c in "xyz"]
    v = compute_lal_source_frame_by_interp(h5, keys, t_ref)
    L = np.array([v["LNhatx"], v["LNhaty"], v["LNhatz"]], dtype=float)
    n = np.array([v["nhatx"], v["nhaty"], v["nhatz"]], dtype=float)
    spins = {}
    for i in (1, 2):
        if all(f"spin{i}{c}-vs-time" in h5 for c in "xyz"):
            comps = []
            for c in "xyz":
                X, Y = read_lvcnr_group(h5, f"spin{i}{c}-vs-time")
                comps.append(float(np.interp(t_ref, X, Y)))
            spins[i] = np.array(comps)
    return L / np.linalg.norm(L), n / np.linalg.norm(n), spins


def catalog_frame(wfm, t_ref=None):
    """The LAL frame from the catalog's own frame data ("method A").

    * LVCNR file (RIT, MAYA): the default ``LNhat``/``nhat`` attributes at the
      file's reference epoch (``wfm.t_ref_nr``: the first sample for Format 1).
      A different ``t_ref`` is allowed only for Format >= 2 files, whose
      ``*-vs-time`` groups are interpolated there.
    * No LVCNR file (SXS): ``L_hat`` and ``n_hat`` from the simulation metadata
      at ``reference_time`` (single epoch).
    """
    from nrcats.lvc import compute_lal_source_frame_from_sxs_metadata, get_lvcnr_format

    import h5py

    default_t = wfm.t_ref_nr
    if wfm._has_lvcnr_file():
        with h5py.File(wfm.filepath, "r") as h5:
            fmt = get_lvcnr_format(h5)
            if t_ref is None or abs(float(t_ref) - default_t) < 1e-8:
                frame = frame_from_lvcnr_attrs(h5.attrs, t_ref=default_t,
                                               source=f"LVCNR attributes (Format {fmt}) of {wfm.filepath}")
                frame.extra["lvcnr_format"] = fmt
                return frame
            if fmt < 2:
                raise ValueError(
                    f"Format {fmt} LVCNR files describe their frame at one epoch only "
                    f"(t = {default_t}); t_ref={t_ref} cannot be used."
                )
            L, n, _ = _lvcnr_vectors_at(h5, float(t_ref))
            frame = frame_from_vectors(L, n, t_ref=float(t_ref),
                                       source=f"LVCNR *-vs-time groups (Format {fmt}) of {wfm.filepath}")
            frame.extra["lvcnr_format"] = fmt
            return frame
    md = wfm.sim_metadata
    if t_ref is not None and abs(float(t_ref) - default_t) > 1e-5:
        raise ValueError(f"The metadata frame holds at reference_time={default_t} only; t_ref={t_ref} cannot be used.")
    v = compute_lal_source_frame_from_sxs_metadata(md)
    L = [v["LNhatx"], v["LNhaty"], v["LNhatz"]]
    n = [v["nhatx"], v["nhaty"], v["nhatz"]]
    omega = np.linalg.norm(np.asarray(md.get("reference_orbital_frequency", [np.nan] * 3), dtype=float))
    return frame_from_vectors(L, n, t_ref=default_t, f_ref_dimless=omega / np.pi if np.isfinite(omega) else None,
                              source="simulation metadata at reference_time")


def put_in_frame(wfm, frame="lal", t_ref=None, **waveform_frame_kwargs):
    """Return ``wfm`` with its modes expressed in the requested frame.

    ``frame`` is ``"lal"`` (catalog frame data, :func:`catalog_frame`),
    ``"waveform"`` (the waveform-derived triad, :func:`nrcats.frame_waveform.waveform_frame`)
    or ``"simulation"`` (no rotation).  The result records the frame under
    ``wfm._metadata["reference_frame"]`` (see :func:`frame_of`).
    """
    if frame not in FRAME_NAMES:
        raise ValueError(f"frame must be one of {FRAME_NAMES}, not {frame!r}")
    if frame == "simulation":
        ref = ReferenceFrame(t_ref=None if t_ref is None else float(t_ref))
    elif frame == "lal":
        ref = catalog_frame(wfm, t_ref=t_ref)
    else:
        from nrcats.frame_waveform import waveform_frame

        ref = waveform_frame(wfm, wfm.t_ref_nr if t_ref is None else float(t_ref), **waveform_frame_kwargs)
        ref.name = "waveform"
    out = rotate_modes(wfm, ref)
    if hasattr(out, "_metadata") and isinstance(out._metadata, dict):
        out._metadata["_t_ref_nr"] = wfm.t_ref_nr if ref.t_ref is None else ref.t_ref
        if ref.is_rotated:
            # Keep the simulation-frame (2,2) phase: WaveformModes uses it to map
            # coa_phase to phi_ref identically in every frame.
            h22 = np.asarray(wfm.data[:, wfm.index(2, 2)])
            out._metadata["_sim_frame_h22_phase"] = np.unwrap(np.angle(h22))
            out._metadata["_sim_frame_h22_amp"] = np.abs(h22)
    return out


def frame_of(wfm):
    """The :class:`ReferenceFrame` the modes of ``wfm`` are in (simulation frame if unrecorded)."""
    meta = getattr(wfm, "_metadata", None)
    if isinstance(meta, dict) and isinstance(meta.get("reference_frame"), ReferenceFrame):
        return meta["reference_frame"]
    return ReferenceFrame()


def reference_metadata(wfm, frame: ReferenceFrame | None = None):
    """Masses, spins and vectors of the binary in ``frame`` at the frame's epoch.

    LVCNR files: :func:`reference_metadata_from_lvcnr`, with the spins
    interpolated from the ``spin*-vs-time`` groups when the frame's epoch is not
    the file's default (Format >= 2).  SXS: ``reference_mass1/2`` and
    ``reference_dimensionless_spin1/2`` at ``reference_time``.
    """
    import h5py

    frame = frame_of(wfm) if frame is None else frame
    if wfm._has_lvcnr_file():
        with h5py.File(wfm.filepath, "r") as h5:
            out = reference_metadata_from_lvcnr(h5, frame if frame.is_rotated else frame_from_lvcnr_attrs(h5.attrs))
            if frame.t_ref is not None and abs(frame.t_ref - wfm.t_ref_nr) > 1e-8 and int(h5.attrs.get("Format", 1)) >= 2:
                _, _, spins = _lvcnr_vectors_at(h5, frame.t_ref)
                if 1 in spins:
                    out["chi1"] = vector_to_frame(spins[1], out["frame"])
                if 2 in spins:
                    out["chi2"] = vector_to_frame(spins[2], out["frame"])
            return out
    md = wfm.sim_metadata
    if not frame.is_rotated:
        frame = catalog_frame(wfm)
    m1, m2 = float(md["reference_mass1"]), float(md["reference_mass2"])
    if m1 < m2 * (1 - 1e-6):
        raise ValueError(f"Body 1 is the lighter hole (reference_mass1={m1}, reference_mass2={m2}).")
    return {
        "frame": frame,
        "mass1": m1 / (m1 + m2),
        "mass2": m2 / (m1 + m2),
        "q": m1 / m2,
        "chi1": vector_to_frame(np.asarray(md["reference_dimensionless_spin1"], dtype=float), frame),
        "chi2": vector_to_frame(np.asarray(md["reference_dimensionless_spin2"], dtype=float), frame),
        "L_hat": vector_to_frame(frame.L_hat, frame),
        "n_hat": vector_to_frame(frame.n_hat, frame),
        "t_ref": frame.t_ref,
        "f_ref_dimless": frame.f_ref_dimless,
    }


def lal_frame_angles(inclination, phi_ref):
    """Spherical angles of the observer in the LAL frame: ``(theta, phi, alpha)``.

    In the LAL frame the observer sits at polar angle ``inclination`` and azimuth
    ``pi/2 - phi_ref``; the polarisation angle LAL's NR routine applies there is
    ``pi`` (a rotation of (h+, hx) by 2 pi, i.e. none).
    """
    return float(inclination), float(np.pi / 2 - phi_ref), float(np.pi)
