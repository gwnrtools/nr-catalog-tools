# Tutorial: Cross-Catalog Comparison

This tutorial shows how to find the "same" physical system in the RIT and SXS catalogs,
load waveforms from both, align them in time and phase, and compute a noise-weighted
sky-averaged mismatch — the standard metric for quantifying NR waveform accuracy.

## Prerequisites

```bash
pip install nr-catalog-tools pycbc
```

---

## Background

Different NR codes (SpEC for SXS, LazEv for RIT) use independent formulations of
Einstein's equations and different numerical schemes.  Even for nominally identical
binary parameters, the waveforms differ due to:

- **Initial data mismatch**: eccentricity and spin directions are set differently
  in each code's initial-data solver, so identical $q$, $\chi_1$, $\chi_2$ in the
  input do not produce identical physical binaries.
- **Source-frame ambiguity**: each code defines the $z$-axis of its waveform frame
  differently (e.g. initial orbital angular momentum vs. grid axes), introducing an
  overall rotation between the two multipole decompositions.
- **BMS supertranslation frame**: codes differ in how they extrapolate the waveform
  to null infinity, leaving a direction-dependent time offset across the sphere.

The mismatch computed here accounts for all three by maximizing the noise-weighted
inner product over time shift $t_c$, coalescence phase $\phi_c$, and source-frame
rotation $R \in SO(3)$.

---

## 1. Load both catalogs

```python
import nrcatalogtools as nrcat

ritcat = nrcat.RITCatalog.load(verbosity=0)
sxscat = nrcat.SXSCatalog.load(download=False, verbosity=0)

print(f"RIT:  {len(ritcat.simulations_list)} simulations")
print(f"SXS:  {len(sxscat.simulations_list)} simulations")
```

---

## 2. Find matching simulations

We look for equal-mass, non-spinning (q ≈ 1, |χ| < 0.05) simulations in both catalogs.

```python
import numpy as np

# --- RIT ---
df_rit = ritcat.simulations_dataframe
rit_q1 = df_rit[
    (df_rit["relaxed-mass-ratio-1-over-2"].astype(float) - 1.0).abs() < 0.02
].index.tolist()
print(f"RIT q≈1 simulations: {len(rit_q1)}")

# --- SXS ---
df_sxs = sxscat.simulations_dataframe
sxs_q1 = df_sxs[
    (df_sxs["reference_mass_ratio"].astype(float) - 1.0).abs() < 0.02
].index.tolist()
print(f"SXS q≈1 simulations: {len(sxs_q1)}")

# Pick one pair for illustration
rit_sim = "RIT:BBH:0001-n100-id3"   # q=1, χ₁=χ₂=0
sxs_sim = "SXS:BBH:0001"            # q=1, χ₁=χ₂=0
```

Both `RIT:BBH:0001` and `SXS:BBH:0001` are equal-mass, zero-spin binary merger
simulations — the canonical testbed for cross-code comparison.

---

## 3. Load the waveforms

```python
print("Loading RIT waveform...")
rit_wfm = ritcat.get(rit_sim)        # downloads ~50 MB HDF5 on first call

print("Loading SXS waveform...")
sxs_wfm = sxscat.get(sxs_sim)       # streams from Zenodo via sxs package

print(f"RIT modes: {rit_wfm.LM}")
print(f"SXS modes: {sxs_wfm.LM}")
```

> **Note:** `sxscat.get()` requires an internet connection for the first load.
> The `sxs` package caches the waveform under `~/.cache/sxs/`.  Pass
> `download=False` to use the cache only (raises an error if not cached).
> If you hit a 404 from `data.black-holes.org`, upgrade the `sxs` package:
> `pip install --upgrade sxs`.

---

## 4. Choose physical parameters

```python
M      = 60.0      # total mass  [M_sun]
d      = 100.0     # distance    [Mpc]
f_low  = 20.0      # low-frequency cutoff [Hz]
delta_t = 1./4096  # sampling interval [s]
```

The mismatch depends on the total mass because the signal is shifted in frequency.
At $M = 60\,M_\odot$ the $(2,2)$ mode sweeps from ~20 Hz to ~500 Hz — well within
the aLIGO sensitive band.

---

## 5. Build the PSD

```python
from pycbc.psd import aLIGOZeroDetHighPower

flen   = int(1.0 / (delta_t * f_low)) + 1   # length for 1/f_low seconds
delta_f = 1.0 / (flen * delta_t)

psd = aLIGOZeroDetHighPower(flen, delta_f=delta_f, low_freq_cutoff=f_low)
```

---

## 6. Compute the sky-averaged mismatch

```python
mismatch = rit_wfm.match_sphere_averaged(
    sxs_wfm,
    psd=psd,
    f_lower=f_low,
    delta_t=delta_t,
    total_mass=M,
    distance=d,
)
print(f"Sky-averaged mismatch (RIT vs SXS): {mismatch:.2e}")
```

`match_sphere_averaged()` maximizes the noise-weighted overlap over:

- $t_c$ (time shift, found via FFT)
- $\phi_c$ (coalescence phase, found analytically)
- $(\alpha, \beta) \in S^2$ (sky orientation = source-frame rotation, found by
  grid search over polar angles)

The returned value is $\mathcal{M} = 1 - \max \mathcal{O}$, where
$\mathcal{O} \in [0, 1]$.

---

## 7. Interpret the result

Typical values for well-matched, high-resolution simulations:

| Mismatch range | Interpretation |
|----------------|----------------|
| $< 10^{-3}$ | Excellent agreement — waveforms consistent within numerical truncation error |
| $10^{-3}$ – $10^{-2}$ | Good — small residual due to eccentricity or spin differences |
| $> 10^{-2}$ | Significant systematic difference — check parameter matching |

For $q=1$, $\chi=0$ at $M=60\,M_\odot$ you should expect $\mathcal{M} \sim 10^{-3}$
to $10^{-4}$ between RIT and SXS, consistent with the NR truncation error budget
reported in the NRAR comparison study.

---

## 8. Compare the (2,2) modes visually

```python
import matplotlib.pyplot as plt

mode_rit = rit_wfm.get_mode(2, 2, total_mass=M, distance=d, delta_t_seconds=delta_t)
mode_sxs = sxs_wfm.get_mode(2, 2, total_mass=M, distance=d, delta_t_seconds=delta_t)

t_rit = mode_rit.sample_times
t_sxs = mode_sxs.sample_times

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_rit, mode_rit.real(), label="RIT:BBH:0001", lw=1.2, color="steelblue")
ax.plot(t_sxs, mode_sxs.real(), label="SXS:BBH:0001", lw=1.2, color="darkorange",
        ls="--")
ax.set_xlabel("Time [s]  ($t=0$ at $(2,2)$ peak)", fontsize=12)
ax.set_ylabel(r"Re[$h_{22}$]  (strain)", fontsize=12)
ax.set_title(f"RIT vs SXS  —  $M = {M}\\,M_\\odot$, $d = {d}\\,$Mpc")
ax.legend(fontsize=11)
ax.set_xlim(-0.3, 0.05)   # last 0.3 s before merger
plt.tight_layout()
plt.savefig("rit_vs_sxs_mode22.png", dpi=150)
plt.show()
```

---

## 9. Extended: BMS-maximized mismatch (optional)

For precessing or high-mass-ratio systems, source-frame differences may include
BMS supertranslations — direction-dependent retarded-time offsets at null infinity.
`match_sphere_averaged_bms_maximized()` extends the optimization to include
supertranslation coefficients $\alpha_{jk}$ up to angular order `j_max`:

```python
# Requires the `scri` package: pip install scri
mismatch_bms = rit_wfm.match_sphere_averaged_bms_maximized(
    sxs_wfm,
    psd=psd,
    f_lower=f_low,
    delta_t=delta_t,
    total_mass=M,
    distance=d,
    j_max=2,   # optimize ℓ=0,1,2 supertranslation modes
)
print(f"BMS-maximized mismatch: {mismatch_bms:.2e}")
```

The BMS-maximized mismatch is always ≤ the SO(3)-only mismatch.  A large reduction
indicates a significant supertranslation frame difference between the two waveforms.

---

## Summary

| Step | Method |
|------|--------|
| Load catalogs | `RITCatalog.load()`, `SXSCatalog.load()` |
| Browse metadata | `cat.simulations_dataframe` |
| Load waveforms | `cat.get(sim_name)` |
| Build PSD | `pycbc.psd.aLIGOZeroDetHighPower(...)` |
| SO(3)-maximized mismatch | `wfm_a.match_sphere_averaged(wfm_b, psd, f_lower, delta_t, total_mass, distance)` |
| BMS-maximized mismatch | `wfm_a.match_sphere_averaged_bms_maximized(..., j_max=2)` |

See [goal.md](../goal.md) for the full scientific derivation of the mismatch formalism,
including the BMS supertranslation mode-mixing formula.
