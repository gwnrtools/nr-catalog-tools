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


def test_the_three_observatories_give_different_matches():
    """The safeguard against reusing one observatory's numbers for another.

    A match is a *noise-weighted* overlap, so each observatory must re-run it;
    only the conversion to a threshold SNR is arithmetic.  The failure this test
    exists to catch is a forwarding bug -- a default argument, a switch that
    silently falls back, a psd_name that never reaches the matcher -- which would
    produce three identical columns that look like a result and pass every other
    check in the repository.

    Measured on real waveforms at 40 Msun with the seed-minimum estimator
    (catalog-comparison-paper findings 5am): aLIGO 4.988e-3, CE 5.868e-3, ET
    4.801e-3 for SXS:BBH:1868, a spread of 1.07e-3 against a within-observatory
    seed spread of 6e-8.  This test reproduces the *property* on a synthetic pair
    so it needs no network and no cache.
    """
    pytest.importorskip("pycbc")
    from test_match_consolidation import DT, _chirp_wfm, _mode_dict

    w = _chirp_wfm(n_times=2048, ell_max=2, ringdown=64.0)
    md = _mode_dict(w)
    # A perturbed copy, so the match is neither 1 nor degenerate: a small
    # frequency-dependent phase, which the three curves weight differently.
    perturbed = {}
    for k, v in md.items():
        arr = np.asarray(v)
        n = len(arr)
        ramp = np.exp(1j * 0.05 * np.linspace(0.0, 1.0, n) ** 2)
        perturbed[k] = type(v)(arr * ramp, delta_t=v.delta_t, epoch=v.start_time)

    got = {}
    for name in ("aLIGO", "CE", "ET"):
        det = D.detector(name)
        lo, hi, _ = D.band(name, 30.0)
        np.random.seed(0)
        m = w.match_sphere_averaged(
            perturbed,
            None,
            lo,
            f_upper=hi,
            psd_name=det.psd_name,
            delta_t=DT,
            total_mass=40.0,
            distance=1.0,
        )
        got[name] = 1.0 - m

    assert all(np.isfinite(v) and v > 0 for v in got.values()), got
    spread = max(got.values()) - min(got.values())
    typical = float(np.median(list(got.values())))
    # Identical columns would give spread == 0 exactly; require the observatories
    # to separate by a part in a thousand of the mismatch, far below the ~20%
    # seen on real waveforms but far above any floating-point coincidence.
    assert spread > 1e-3 * typical, (
        "the three observatories returned effectively the same mismatch; "
        f"the noise curve is probably not reaching the matcher: {got}"
    )
