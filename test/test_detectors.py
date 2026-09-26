"""The observatory registry: one mapping, and a band that says what bound it."""

import numpy as np
import pytest

from nrcats.waveform import detectors as D


def test_every_registered_curve_loads_and_is_finite_in_its_band():
    """A registry entry naming a curve pycbc cannot build is worse than no entry:
    the failure would surface inside a campaign, hours in."""
    for name, det in D.DETECTORS.items():
        psd = np.asarray(D.psd_for(name, 4097, 0.25))
        f = np.arange(len(psd)) * 0.25
        # The Nyquist bin is zero for every lalsimulation curve -- pycbc's
        # convention, not a property of the registry -- so the band tested is
        # open at the top.
        inband = (f >= det.f_lower) & (f < f[-1])
        vals = psd[inband]
        assert np.all(np.isfinite(vals)), name
        assert np.all(vals > 0), name


def test_the_band_reports_which_constraint_bound_it():
    """At 40 Msun an NR waveform starts near 20 Hz, so for CE and ET the waveform
    binds and for aLIGO the two coincide -- and a result has to be able to say
    which, because a band set by the simulation's length is not a statement about
    the instrument."""
    lo, _, why = D.band("CE", 20.0)
    assert (lo, why) == (20.0, "waveform")
    lo, _, why = D.band("ET", 20.0)
    assert (lo, why) == (20.0, "waveform")
    lo, _, why = D.band("aLIGO", 20.0)
    assert (lo, why) == (20.0, "equal")
    # a long, heavy-binary waveform reaching below the instrument
    lo, _, why = D.band("CE", 3.0)
    assert (lo, why) == (5.0, "detector")
    # no waveform given: the instrument's own band
    lo, hi, why = D.band("aLIGO")
    assert (lo, why) == (20.0, "detector")


def test_resolution_is_case_insensitive_and_fails_loudly():
    assert D.detector("ce").name == "CE"
    assert D.detector("aligo").name == "aLIGO"
    assert D.detector(D.DETECTORS["ET"]).name == "ET"
    with pytest.raises(KeyError, match="unknown detector"):
        D.detector("LISA")


def test_the_measured_sensitivity_ordering_holds():
    """The docstring's table is load-bearing -- it is the reason the results say
    ET gains at low frequency and CE gains above 10 Hz -- so assert the ordering
    rather than trusting a comment.
    """
    df, flen = 0.25, 8193
    f = np.arange(flen) * df

    def at(name, freq):
        psd = np.asarray(D.psd_for(name, flen, df, f_lower=1.0))
        return float(np.sqrt(np.interp(freq, f, psd)))

    # ET is far better than CE at 5 Hz; CE is better than ET above 10 Hz
    assert at("ET", 5.0) < at("CE", 5.0) / 10.0
    assert at("CE", 20.0) < at("ET", 20.0)
    # and CE at 5 Hz is no better than aLIGO at 5 Hz, which is why that decade
    # contributes almost nothing
    assert at("CE", 5.0) > at("aLIGO", 5.0) / 10.0
