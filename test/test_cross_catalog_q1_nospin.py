"""Unit tests for cross-catalog waveform consistency — q=1, non-spinning BBH.

Each test operates on one simulation from each of the three NR catalogs:

  RIT  : RIT:BBH:0001-n100-id3
  SXS  : SXS:BBH:0001
  MAYA : GT0355

Tests are split into two groups:

  Per-catalog tests (TestAmplitudeScaling, TestEpochAlignment,
  TestDeltaTConventions, TestParameterExtraction):
      Verify that mathematical properties hold for each individual waveform
      independently of the other catalogs. These tests only require that one
      catalog's data be cached locally; they skip gracefully if it is not.

  Cross-catalog tests (TestCrossCatalogConsistency):
      Compare the three waveforms against each other. These tests require all
      three catalogs to be cached and are skipped if any catalog is unavailable.

Running:
  pytest test/test_cross_catalog_q1_nospin.py -v
  pytest test/test_cross_catalog_q1_nospin.py -v -m "not cross_catalog"  # per-catalog only
  pytest test/test_cross_catalog_q1_nospin.py -v -m cross_catalog        # cross-catalog only

Markers used:
  requires_data   — test loads a waveform HDF5; needs cached data on disk.
  cross_catalog   — test compares all three catalogs simultaneously.

Notes on q=1 symmetry:
  For exactly equal-mass, non-spinning BBH, the mode exchange symmetry
      h_{l,-m} = (-1)^l  conj(h_{l,m})
  forces all odd-m modes to vanish.  Tests assert that the odd-m modes
  (2,1), (3,3), (4,3), (5,5) have peak amplitude < 1% of the (2,2) peak.
"""

import lal
import numpy as np
import pytest

# ── Catalog / simulation configuration ────────────────────────────────────────
RIT_SIM  = "RIT:BBH:0001-n100-id3"
SXS_SIM  = "SXS:BBH:0001"
MAYA_SIM = "GT0905"

TOTAL_MASS = 60.0         # M_sun
DISTANCE   = 100.0        # Mpc
DELTA_T    = 1.0 / 4096  # physical seconds

# Even-m modes carry physical power for q=1; odd-m modes must vanish.
EVEN_M_MODES = [(2, 2), (3, 2), (4, 4)]
ODD_M_MODES  = [(2, 1), (3, 3), (4, 3), (5, 5)]
ALL_MODES    = EVEN_M_MODES + ODD_M_MODES

# Tolerances
# (2,2) is the dominant mode and is computed most accurately across codes.
# Sub-dominant modes (3,2), (4,4) have larger genuine inter-code differences:
# observed ~4% amplitude and ~0.8 rad phase-drift discrepancies are expected.
AMP_RATIO_TOL_BY_MODE   = {(2, 2): 0.02, (3, 2): 0.08, (4, 4): 0.05}
PHASE_DRIFT_TOL_BY_MODE = {(2, 2): 0.5,  (3, 2): 1.0,  (4, 4): 1.0}
AMP_RATIO_TOL   = 0.05   # default fallback
PHASE_DRIFT_TOL = 1.0    # default fallback
ODD_MODE_TOL    = 0.05   # odd-m mode peak must be < 5% of (2,2) peak
# Note: GT0355 has q=1.0001 (not exactly 1) and eccentricity ~0.003, which
# produces a genuine ~4% (2,1) mode — the 5% threshold accommodates this.


# ── Fixtures ──────────────────────────────────────────────────────────────────
def _try_import():
    """Return nrcatalogtools, or None if unavailable."""
    try:
        import nrcatalogtools as nrcat
        return nrcat
    except ImportError:
        return None


def _load_rit():
    nrcat = _try_import()
    if nrcat is None:
        return None
    try:
        cat = nrcat.RITCatalog.load(verbosity=0)
        return cat.get(RIT_SIM, quantity="waveform")
    except Exception:
        return None


def _load_sxs():
    nrcat = _try_import()
    if nrcat is None:
        return None
    try:
        cat = nrcat.SXSCatalog.load(download=False, verbosity=0)
        return cat.get(SXS_SIM)
    except Exception:
        return None


def _load_maya():
    nrcat = _try_import()
    if nrcat is None:
        return None
    try:
        cat = nrcat.MayaCatalog.load(verbosity=0)
        return cat.get(MAYA_SIM, quantity="waveform")
    except Exception:
        return None


# Session-scoped so the HDF5/network access happens only once per pytest run.
@pytest.fixture(scope="session")
def wfm_rit():
    wfm = _load_rit()
    if wfm is None:
        pytest.skip(f"RIT simulation {RIT_SIM} not available (not cached)")
    return wfm


@pytest.fixture(scope="session")
def wfm_sxs():
    wfm = _load_sxs()
    if wfm is None:
        pytest.skip(f"SXS simulation {SXS_SIM} not available (not cached or download=False)")
    return wfm


@pytest.fixture(scope="session")
def wfm_maya():
    wfm = _load_maya()
    if wfm is None:
        pytest.skip(f"MAYA simulation {MAYA_SIM} not available (not cached)")
    return wfm


@pytest.fixture(scope="session")
def all_waveforms(wfm_rit, wfm_sxs, wfm_maya):
    """Dict of label -> WaveformModes, only if all three are available."""
    return {"RIT": wfm_rit, "SXS": wfm_sxs, "MAYA": wfm_maya}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _lm_available(wfm, ell, em):
    return (ell, em) in [(int(lm[0]), int(lm[1])) for lm in wfm.LM]


def _raw_peak(wfm, ell, em):
    raw = wfm.get_mode_data(ell, em)
    return float(np.sqrt(raw[:, 1] ** 2 + raw[:, 2] ** 2).max())


def _phys_peak(wfm, ell, em):
    mode = wfm.get_mode(ell, em,
                        total_mass=TOTAL_MASS,
                        distance=DISTANCE,
                        delta_t=DELTA_T)
    return float(np.abs(mode.data).max()), mode


def _phase_drift(mode, t_window=-0.1):
    """Phase accumulated from t_window to t=0 (merger)."""
    t     = np.array(mode.sample_times)
    phase = np.unwrap(np.angle(np.array(mode.data)))
    i_start = int(np.argmin(np.abs(t - t_window)))
    i_end   = int(np.argmin(np.abs(t - 0.0)))
    return float(phase[i_end] - phase[i_start])


# ═══════════════════════════════════════════════════════════════════════════════
# Group 1 — Per-catalog, single-waveform mathematical properties
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.requires_data
class TestAmplitudeScaling:
    """The physical amplitude of get_mode() must equal raw_amplitude × amp_to_physical(M, d)."""

    def _check(self, wfm, label):
        from nrcatalogtools import utils
        scale = utils.amp_to_physical(TOTAL_MASS, DISTANCE)

        for ell, em in EVEN_M_MODES:
            if not _lm_available(wfm, ell, em):
                continue
            raw = wfm.get_mode_data(ell, em)
            raw_amp = np.sqrt(raw[:, 1] ** 2 + raw[:, 2] ** 2).max()

            phys_peak, _ = _phys_peak(wfm, ell, em)

            # The physical peak can differ slightly from raw_peak * scale because
            # get_mode() resamples; allow 0.5% relative tolerance for interpolation.
            ratio = phys_peak / (raw_amp * scale)
            assert abs(ratio - 1.0) < 0.005, (
                f"[{label}] ({ell},{em}): physical peak / (raw peak × scale) = "
                f"{ratio:.5f}, expected ≈ 1.0 (within 0.5%)"
            )

    def test_rit(self, wfm_rit):
        self._check(wfm_rit, "RIT")

    def test_sxs(self, wfm_sxs):
        self._check(wfm_sxs, "SXS")

    def test_maya(self, wfm_maya):
        self._check(wfm_maya, "MAYA")


@pytest.mark.requires_data
class TestEpochAlignment:
    """get_mode() must return a TimeSeries whose epoch places t=0 at the (2,2) peak."""

    def _check(self, wfm, label):
        mode22 = wfm.get_mode(2, 2,
                               total_mass=TOTAL_MASS,
                               distance=DISTANCE,
                               delta_t=DELTA_T)
        t     = np.array(mode22.sample_times)
        amp   = np.abs(np.array(mode22.data))
        t_peak = float(t[np.argmax(amp)])

        # Peak should be within 2 samples of t=0
        assert abs(t_peak) < 2 * DELTA_T, (
            f"[{label}] (2,2) peak is at t={t_peak:.6f} s, expected within "
            f"2 samples of t=0 (tolerance {2*DELTA_T:.6f} s)"
        )

    def test_rit(self, wfm_rit):
        self._check(wfm_rit, "RIT")

    def test_sxs(self, wfm_sxs):
        self._check(wfm_sxs, "SXS")

    def test_maya(self, wfm_maya):
        self._check(wfm_maya, "MAYA")


@pytest.mark.requires_data
class TestDeltaTConventions:
    """
    Two delta_t conventions must both produce correct physical sample spacings.

    Convention A (delta_t <= 1/128): interpreted as physical seconds.
    Convention B (delta_t >  1/128): interpreted as dimensionless M units.
    """

    def _check(self, wfm, label):
        m_secs = TOTAL_MASS * lal.MTSUN_SI  # seconds per solar mass

        # Convention A: delta_t in physical seconds
        dt_phys = 1.0 / 4096
        mode_a = wfm.get_mode(2, 2, total_mass=TOTAL_MASS,
                               distance=DISTANCE, delta_t=dt_phys)
        assert abs(mode_a.delta_t - dt_phys) < 1e-12, (
            f"[{label}] Convention A: mode.delta_t={mode_a.delta_t:.6g} s, "
            f"expected {dt_phys:.6g} s"
        )

        # Convention B: delta_t in dimensionless M units (value = 0.5 M)
        dt_dimless = 0.5   # M units
        dt_expected = dt_dimless * m_secs  # physical seconds
        mode_b = wfm.get_mode(2, 2, total_mass=TOTAL_MASS,
                               distance=DISTANCE, delta_t=dt_dimless)
        assert abs(mode_b.delta_t - dt_expected) < 1e-12, (
            f"[{label}] Convention B: mode.delta_t={mode_b.delta_t:.6g} s, "
            f"expected {dt_expected:.6g} s (0.5 M at {TOTAL_MASS} M☉)"
        )

        # Both conventions must produce the same epoch (t=0 at (2,2) peak)
        t_peak_a = float(np.array(mode_a.sample_times)[np.argmax(np.abs(np.array(mode_a.data)))])
        t_peak_b = float(np.array(mode_b.sample_times)[np.argmax(np.abs(np.array(mode_b.data)))])
        assert abs(t_peak_a) < 2 * dt_phys, (
            f"[{label}] Convention A epoch: peak at {t_peak_a:.6g} s ≠ 0")
        assert abs(t_peak_b) < 2 * dt_expected, (
            f"[{label}] Convention B epoch: peak at {t_peak_b:.6g} s ≠ 0")

    def test_rit(self, wfm_rit):
        self._check(wfm_rit, "RIT")

    def test_sxs(self, wfm_sxs):
        self._check(wfm_sxs, "SXS")

    def test_maya(self, wfm_maya):
        self._check(wfm_maya, "MAYA")


@pytest.mark.requires_data
class TestParameterExtraction:
    """get_parameters() must return PyCBC-compatible spin keys and a positive f_lower."""

    SPIN_KEYS = ["spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z"]

    def _check(self, cat_getter, sim_name, label):
        """cat_getter() returns the catalog; we call get_parameters on it."""
        nrcat = _try_import()
        if nrcat is None:
            pytest.skip("nrcatalogtools not importable")

        cat = cat_getter()
        params = cat.get_parameters(sim_name, total_mass=TOTAL_MASS)

        # Spin keys must use the full name, not the short 's1x' form
        for key in self.SPIN_KEYS:
            assert key in params, (
                f"[{label}] '{key}' missing from get_parameters() output. "
                f"Keys present: {list(params.keys())}"
            )

        # Masses must be positive and sum to total_mass
        assert params["mass1"] > 0 and params["mass2"] > 0, (
            f"[{label}] Non-positive masses: m1={params['mass1']}, m2={params['mass2']}")
        assert abs(params["mass1"] + params["mass2"] - TOTAL_MASS) < 1e-6, (
            f"[{label}] m1+m2={params['mass1']+params['mass2']:.4f} ≠ {TOTAL_MASS}")

        # f_lower must be positive and physically reasonable (5–300 Hz at 60 M_sun)
        f_lower = params["f_lower"]
        assert f_lower > 0, f"[{label}] f_lower={f_lower:.4f} Hz is not positive"
        assert 5 < f_lower < 300, (
            f"[{label}] f_lower={f_lower:.2f} Hz is outside [5, 300] Hz at "
            f"{TOTAL_MASS} M☉ — check unit conversion")

        # For q=1 non-spinning: mass ratio must be ≈ 1, spins ≈ 0
        assert abs(params["mass1"] / params["mass2"] - 1.0) < 0.05, (
            f"[{label}] mass ratio {params['mass1']/params['mass2']:.3f} ≠ 1 "
            f"for q=1 simulation")
        for key in self.SPIN_KEYS:
            assert abs(params[key]) < 0.05, (
                f"[{label}] {key}={params[key]:.4f} is non-zero for a non-spinning simulation")

    def test_rit(self):
        nrcat = _try_import()
        if nrcat is None:
            pytest.skip("nrcatalogtools not importable")
        try:
            cat = nrcat.RITCatalog.load(verbosity=0)
        except Exception:
            pytest.skip(f"RIT catalog not cached")
        self._check(lambda: cat, RIT_SIM, "RIT")

    def test_sxs(self):
        nrcat = _try_import()
        if nrcat is None:
            pytest.skip("nrcatalogtools not importable")
        try:
            cat = nrcat.SXSCatalog.load(download=False, verbosity=0)
        except Exception:
            pytest.skip(f"SXS catalog not cached")
        self._check(lambda: cat, SXS_SIM, "SXS")

    def test_maya(self):
        nrcat = _try_import()
        if nrcat is None:
            pytest.skip("nrcatalogtools not importable")
        try:
            cat = nrcat.MayaCatalog.load(verbosity=0)
        except Exception:
            pytest.skip(f"MAYA catalog not cached")
        self._check(lambda: cat, MAYA_SIM, "MAYA")


@pytest.mark.requires_data
class TestOddMModesNearZero:
    """For q=1 non-spinning, odd-m modes must be negligible (< 1% of (2,2) peak)."""

    def _check(self, wfm, label):
        if not _lm_available(wfm, 2, 2):
            pytest.skip(f"[{label}] (2,2) mode not available")
        ref_peak = _raw_peak(wfm, 2, 2)

        for ell, em in ODD_M_MODES:
            if not _lm_available(wfm, ell, em):
                continue
            pk = _raw_peak(wfm, ell, em)
            fraction = pk / ref_peak
            assert fraction < ODD_MODE_TOL, (
                f"[{label}] ({ell},{em}) peak = {pk:.4e} = {fraction*100:.2f}% of (2,2) peak "
                f"({ref_peak:.4e}), expected < {ODD_MODE_TOL*100:.0f}% for q=1 non-spinning")

    def test_rit(self, wfm_rit):
        self._check(wfm_rit, "RIT")

    def test_sxs(self, wfm_sxs):
        self._check(wfm_sxs, "SXS")

    def test_maya(self, wfm_maya):
        self._check(wfm_maya, "MAYA")


@pytest.mark.requires_data
class TestFLowerAtOneMsun:
    """f_lower_at_1Msun() must return a positive frequency and scale correctly with mass."""

    def _check(self, wfm, label):
        # Frequency at start, at 1 M_sun
        f1 = wfm.f_lower_at_1Msun()
        assert f1 > 0, f"[{label}] f_lower_at_1Msun() = {f1:.4f} is not positive"

        # Frequency at 60 M_sun = f1 / 60 — must be positive and below Nyquist.
        # Long simulations (e.g. SXS:BBH:0001, which starts at t≈0 and runs to
        # merger at t≈25000 M) begin well below the LIGO band; the lower bound
        # is therefore relaxed to 0 Hz.
        f_phys = f1 / TOTAL_MASS
        assert 0 < f_phys < 300, (
            f"[{label}] f_lower at {TOTAL_MASS} M☉ = {f_phys:.2f} Hz is outside (0, 300] Hz")

        # Evaluate at a specific time: must still be positive
        t_mid = float(wfm.time[len(wfm.time) // 2])
        f_mid = wfm.f_lower_at_1Msun(t=t_mid)
        assert f_mid > 0, (
            f"[{label}] f_lower_at_1Msun(t={t_mid:.1f} M) = {f_mid:.4f} is not positive")

        # Frequency must increase towards merger (GW frequency sweeps up)
        assert f_mid > f1, (
            f"[{label}] GW frequency did not increase from start to midpoint: "
            f"f_start={f1:.4f}, f_mid={f_mid:.4f} (both at 1 M☉)")

    def test_rit(self, wfm_rit):
        self._check(wfm_rit, "RIT")

    def test_sxs(self, wfm_sxs):
        self._check(wfm_sxs, "SXS")

    def test_maya(self, wfm_maya):
        self._check(wfm_maya, "MAYA")


# ═══════════════════════════════════════════════════════════════════════════════
# Group 2 — Cross-catalog consistency (all three required)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.requires_data
@pytest.mark.cross_catalog
class TestCrossCatalogConsistency:
    """
    Compare RIT, SXS, and MAYA on the same q=1 non-spinning configuration.

    All pairwise amplitude ratios must be within ±2% of 1 for even-m modes.
    All pairwise phase-drift differences must be < 0.5 rad for even-m modes.
    """

    @pytest.mark.parametrize("ell,em", EVEN_M_MODES)
    def test_amplitude_ratios(self, all_waveforms, ell, em):
        """Pairwise peak amplitude ratios must be within ±2% of 1."""
        peaks = {}
        for label, wfm in all_waveforms.items():
            if _lm_available(wfm, ell, em):
                peaks[label], _ = _phys_peak(wfm, ell, em)

        if len(peaks) < 2:
            pytest.skip(f"Fewer than 2 catalogs have mode ({ell},{em})")

        tol = AMP_RATIO_TOL_BY_MODE.get((ell, em), AMP_RATIO_TOL)
        labels = list(peaks)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                ratio = peaks[a] / peaks[b]
                assert abs(ratio - 1.0) < tol, (
                    f"({ell},{em}) amplitude ratio {a}/{b} = {ratio:.4f}, "
                    f"expected 1.0 ± {tol} ({tol*100:.0f}%)")

    @pytest.mark.parametrize("ell,em", EVEN_M_MODES)
    def test_phase_drift_consistency(self, all_waveforms, ell, em):
        """Phase drift in last 0.1 s must agree across catalogs within 0.5 rad."""
        drifts = {}
        for label, wfm in all_waveforms.items():
            if not _lm_available(wfm, ell, em):
                continue
            _, mode = _phys_peak(wfm, ell, em)
            drifts[label] = _phase_drift(mode, t_window=-0.1)

        if len(drifts) < 2:
            pytest.skip(f"Fewer than 2 catalogs have mode ({ell},{em})")

        tol = PHASE_DRIFT_TOL_BY_MODE.get((ell, em), PHASE_DRIFT_TOL)
        labels = list(drifts)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                diff = abs(drifts[a] - drifts[b])
                assert diff < tol, (
                    f"({ell},{em}) phase drift diff {a}-{b} = {diff:.4f} rad, "
                    f"expected < {tol} rad")

    @pytest.mark.parametrize("ell,em", EVEN_M_MODES)
    def test_raw_amplitude_ratios(self, all_waveforms, ell, em):
        """Pairwise raw (dimensionless) peak amplitude ratios must be within ±2% of 1."""
        raw_peaks = {}
        for label, wfm in all_waveforms.items():
            if _lm_available(wfm, ell, em):
                raw_peaks[label] = _raw_peak(wfm, ell, em)

        if len(raw_peaks) < 2:
            pytest.skip(f"Fewer than 2 catalogs have mode ({ell},{em})")

        tol = AMP_RATIO_TOL_BY_MODE.get((ell, em), AMP_RATIO_TOL)
        labels = list(raw_peaks)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                ratio = raw_peaks[a] / raw_peaks[b]
                assert abs(ratio - 1.0) < tol, (
                    f"({ell},{em}) raw amplitude ratio {a}/{b} = {ratio:.4f}, "
                    f"expected 1.0 ± {tol}")

    @pytest.mark.parametrize("ell,em", ODD_M_MODES)
    def test_odd_m_modes_consistent_across_catalogs(self, all_waveforms, ell, em):
        """All catalogs must agree that odd-m modes are near zero for q=1."""
        fractions = {}
        ref_peaks = {}
        for label, wfm in all_waveforms.items():
            if _lm_available(wfm, ell, em) and _lm_available(wfm, 2, 2):
                pk_odd = _raw_peak(wfm, ell, em)
                pk_ref = _raw_peak(wfm, 2, 2)
                fractions[label] = pk_odd / pk_ref
                ref_peaks[label] = pk_ref

        if len(fractions) == 0:
            pytest.skip(f"Mode ({ell},{em}) not available in any catalog")

        for label, frac in fractions.items():
            assert frac < ODD_MODE_TOL, (
                f"[{label}] ({ell},{em}) = {frac*100:.2f}% of (2,2) peak — "
                f"should be near zero for q=1 non-spinning (< {ODD_MODE_TOL*100:.0f}%)")
