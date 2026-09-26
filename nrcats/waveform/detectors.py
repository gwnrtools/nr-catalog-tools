"""Observatories: one place where a name resolves to a noise curve and a band.

Every quantity in these comparisons is a noise-weighted overlap, so it depends
on the observatory twice -- through the weighting across the band, and through
the band's edges.  Measuring the same waveforms in three observatories is then
one argument changed, *provided* the mapping from a name to a curve lives in
exactly one place.  Two copies of that mapping drift, and a comparison whose
noise curve differs between two code paths is not a comparison.

Curves come from ``pycbc.psd``'s LALSimulation set, so none is hand-entered.

Measured sensitivity, ``sqrt(S_n)`` in strain per root-Hz, on the grid these
comparisons use::

                                      5 Hz     10 Hz     20 Hz    100 Hz
    aLIGOZeroDetHighPower           4.9e-22   9.7e-23   2.1e-23   3.9e-24
    CosmicExplorerP1600143          1.2e-22   7.3e-25   2.0e-25   1.6e-25
    EinsteinTelescopeP1600143       3.6e-24   1.2e-24   9.4e-25   3.9e-25
    CosmicExplorerPessimistic...    2.6e-20   2.1e-24   3.9e-25   2.5e-25
    CosmicExplorerWideband...       1.2e-22   7.4e-25   2.5e-25   2.1e-25

Two things in that table decide how the results should be read.  **Cosmic
Explorer is no more sensitive at 5 Hz than Advanced LIGO is**: its wall sits
near 10 Hz, and the pessimistic variant has effectively nothing below 10 Hz at
all.  **The Einstein Telescope is**, by a factor of thirty over CE at 5 Hz.  So
CE's advantage is breadth and depth above 10 Hz while ET's includes genuine
low-frequency reach, and any statement about which observatory a numerical
waveform's error matters most for has to say at which mass -- because the
frequency a numerical waveform *starts* at, in Hz, falls as 1/M.

The low cutoff is therefore not simply the detector's.  A numerical waveform has
finite length, so its own starting frequency bounds the band from below; at
40 Msun an SXS waveform begins near 20 Hz and reaching 5 Hz would need of order
forty times more inspiral than the catalogs publish.  :func:`band` returns the
binding constraint along with the value, so every result can say which one set
its band rather than leaving the reader to assume.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Detector", "DETECTORS", "detector", "band", "psd_for"]


@dataclass(frozen=True)
class Detector:
    """One observatory: a curve, a band, and where the curve came from."""

    name: str
    psd_name: str
    f_lower: float
    f_upper: float | None
    reference: str
    note: str = ""


DETECTORS: dict[str, Detector] = {
    "aLIGO": Detector(
        name="aLIGO",
        psd_name="aLIGOZeroDetHighPower",
        f_lower=20.0,
        f_upper=None,
        reference="Advanced LIGO design sensitivity, zero-detuned high power",
        note="the default throughout this work; every published number uses it",
    ),
    "CE": Detector(
        name="CE",
        psd_name="CosmicExplorerP1600143",
        f_lower=5.0,
        f_upper=None,
        reference="LIGO-P1600143",
        note=(
            "sensitive from ~10 Hz; at 5 Hz it is no better than Advanced LIGO, "
            "so the 5-10 Hz decade is carried but contributes almost nothing"
        ),
    ),
    "ET": Detector(
        name="ET",
        psd_name="EinsteinTelescopeP1600143",
        f_lower=5.0,
        f_upper=None,
        reference="LIGO-P1600143",
        note="the one curve here with real sensitivity below 10 Hz",
    ),
    # Carried as sensitivity checks, not headline numbers.
    "CE-pessimistic": Detector(
        name="CE-pessimistic",
        psd_name="CosmicExplorerPessimisticP1600143",
        f_lower=10.0,
        f_upper=None,
        reference="LIGO-P1600143",
        note="effectively blind below 10 Hz, hence the higher cutoff",
    ),
    "CE-wideband": Detector(
        name="CE-wideband",
        psd_name="CosmicExplorerWidebandP1600143",
        f_lower=5.0,
        f_upper=None,
        reference="LIGO-P1600143",
        note="sensitivity check on the CE tuning",
    ),
}


def detector(name: str | Detector) -> Detector:
    """Resolve an observatory name, case-insensitively, with a useful error."""
    if isinstance(name, Detector):
        return name
    key = str(name).strip()
    if key in DETECTORS:
        return DETECTORS[key]
    lowered = {k.lower(): k for k in DETECTORS}
    if key.lower() in lowered:
        return DETECTORS[lowered[key.lower()]]
    raise KeyError(
        f"unknown detector {name!r}; known: {', '.join(sorted(DETECTORS))}"
    )


def band(name: str | Detector, f_waveform: float | None = None):
    """``(f_lower, f_upper, bound_by)`` for one observatory and one waveform.

    ``f_waveform`` is the lowest frequency the waveform itself supports, as the
    campaigns already compute it from the model's starting frequency.  The band
    starts at whichever of the two constraints is higher, and ``bound_by`` says
    which -- ``"detector"``, ``"waveform"``, or ``"equal"`` -- so that a result
    can report whether it was limited by the instrument or by the length of the
    simulation.  Passing ``None`` gives the detector's own band.
    """
    det = detector(name)
    if f_waveform is None:
        return det.f_lower, det.f_upper, "detector"
    f_w = float(f_waveform)
    if f_w > det.f_lower:
        return f_w, det.f_upper, "waveform"
    if f_w < det.f_lower:
        return det.f_lower, det.f_upper, "detector"
    return det.f_lower, det.f_upper, "equal"


def psd_for(name: str | Detector, length: int, delta_f: float, f_lower=None):
    """The observatory's one-sided PSD on a given frequency grid.

    Thin wrapper over ``pycbc.psd.from_string`` so that callers never name a
    curve directly; the string lives in :data:`DETECTORS` alone.
    """
    from pycbc.psd import from_string

    det = detector(name)
    low = det.f_lower if f_lower is None else float(f_lower)
    return from_string(det.psd_name, int(length), float(delta_f), low_freq_cutoff=low)
