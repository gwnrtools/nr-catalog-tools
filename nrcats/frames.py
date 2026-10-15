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


def lal_frame_angles(inclination, phi_ref):
    """Spherical angles of the observer in the LAL frame: ``(theta, phi, alpha)``.

    In the LAL frame the observer sits at polar angle ``inclination`` and azimuth
    ``pi/2 - phi_ref``; the polarisation angle LAL's NR routine applies there is
    ``pi`` (a rotation of (h+, hx) by 2 pi, i.e. none).
    """
    return float(inclination), float(np.pi / 2 - phi_ref), float(np.pi)
