# Tutorial: Loading and Plotting a Waveform

This tutorial walks through loading a binary black-hole merger waveform from the RIT catalog,
inspecting its metadata, extracting physically-scaled strain modes, and plotting the
time-domain polarizations.

## Prerequisites

```bash
pip install nr-catalog-tools matplotlib
```

---

## 1. Load the RIT catalog

```python
import nrcatalogtools as nrcat

cat = nrcat.RITCatalog.load()
print(f"Loaded {len(cat.simulations_list)} RIT simulations")
# Loaded 1879 RIT simulations
```

`RITCatalog.load()` reads the aggregated metadata CSV from your local cache
(`~/.cache/RIT/metadata/metadata.csv`).  On first run it scrapes ~1900 per-simulation
metadata files from `ccrgpages.rit.edu`, which takes about 30–60 s.  Subsequent calls
are instant.

---

## 2. Browse the simulation list

```python
df = cat.simulations_dataframe
print(df.index[:5].tolist())
# ['RIT:BBH:0001-n100-id3', 'RIT:BBH:0002-n100-id0', ...]

# Filter: equal-mass, non-spinning (q ≈ 1, |χ| < 0.05)
q_col = "relaxed-mass-ratio-1-over-2"
chi_cols = ["relaxed-chi1z", "relaxed-chi2z"]

mask_q   = (df[q_col].astype(float) - 1.0).abs() < 0.02
mask_chi = (df[chi_cols].astype(float).abs() < 0.05).all(axis=1)
mask = mask_q & mask_chi
q1_nospin = df[mask].index.tolist()
print(f"Found {len(q1_nospin)} q≈1 non-spinning simulations")
```

---

## 3. Inspect simulation metadata

```python
sim_name = "RIT:BBH:0001-n100-id3"
meta = cat.get_metadata(sim_name)

print(f"Mass ratio:    {meta['relaxed-mass-ratio-1-over-2']}")
print(f"χ₁z:           {meta['relaxed-chi1z']}")
print(f"χ₂z:           {meta['relaxed-chi2z']}")
print(f"Eccentricity:  {meta['eccentricity']}")
print(f"f_start (22):  {meta['freq-start-22']}")   # dimensionless M·Hz
```

`get_metadata()` returns the raw per-simulation dict with catalog-native key names
(hyphens for RIT).  To get a PyCBC-compatible parameter dict in physical units, use
`get_parameters()` instead:

```python
params = cat.get_parameters(sim_name, total_mass=60.0)
# {'mass1': 30.0, 'mass2': 30.0, 'spin1x': 0.0, ...,
#  'f_lower': 23.4, 'approximant': 'NR_hdf5'}
print(params)
```

---

## 4. Load the waveform

```python
wfm = cat.get(sim_name)
print(f"Mode pairs available: {wfm.LM}")
# [(2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (3, -3), ...]
print(f"Time array length: {len(wfm.time)} samples")
print(f"Time range: [{wfm.time[0]:.1f}, {wfm.time[-1]:.1f}] M")
```

On first call `cat.get()` downloads the HDF5 waveform file from the RIT server
(`~50 MB`) into `~/.cache/RIT/data/`.  Subsequent calls use the cached file.

---

## 5. Extract a physically-scaled $(2,2)$ mode

```python
TOTAL_MASS = 60.0    # M_sun
DISTANCE   = 100.0   # Mpc
DELTA_T    = 1./4096 # seconds  (4096 Hz sampling rate)

mode22 = wfm.get_mode(2, 2,
                      total_mass=TOTAL_MASS,
                      distance=DISTANCE,
                      delta_t_seconds=DELTA_T)

# mode22 is a complex pycbc.types.TimeSeries
# .real()  ≈  h_plus  (in strain units)
# .imag()  ≈  h_cross (sign convention: +h_cross, not −h_cross)
print(f"Duration:  {mode22.duration:.3f} s")
print(f"Peak amplitude: {abs(mode22).max():.3e}")
```

`get_mode()` scales the amplitude by $G M_\text{tot} / (c^2 \cdot d_\text{Mpc} \cdot \text{Mpc})$
and sets $t=0$ at the peak of the $(2,2)$ mode.

---

## 6. Compute polarizations

For a sky-averaged strain summed over all modes, use `get_td_waveform()`:

```python
pols = wfm.get_td_waveform(
    total_mass=TOTAL_MASS,
    distance=DISTANCE,
    inclination=0.0,    # face-on: strongest signal
    coa_phase=0.0,
    delta_t_seconds=DELTA_T,
)

hp = pols.real()       # h_plus
hc = -1 * pols.imag()  # h_cross  (negate because of package convention)
```

> **Polarization sign convention**  
> `get_td_waveform()` returns `conjugate(h)`, so `.real()` = $h_+$ and `.imag()` = $+h_\times$.
> Multiply `.imag()` by $-1$ to match the LAL sign convention where $h_\times > 0$ for
> a face-on binary early in the inspiral.

---

## 7. Plot

```python
import matplotlib.pyplot as plt

t = hp.sample_times   # seconds, t=0 at (2,2) peak

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

axes[0].plot(t, hp, color="steelblue")
axes[0].set_ylabel(r"$h_+$", fontsize=13)
axes[0].axvline(0, color="gray", lw=0.8, ls="--", label="merger")
axes[0].legend(fontsize=10)

axes[1].plot(t, hc, color="darkorange")
axes[1].set_ylabel(r"$h_\times$", fontsize=13)
axes[1].set_xlabel("Time [s]", fontsize=13)
axes[1].axvline(0, color="gray", lw=0.8, ls="--")

fig.suptitle(
    f"RIT:BBH:0001-n100-id3  —  $M={TOTAL_MASS}\\,M_\\odot$, $d={DISTANCE}\\,$Mpc",
    fontsize=12,
)
plt.tight_layout()
plt.savefig("rit_bbh_0001_waveform.png", dpi=150)
plt.show()
```

---

## 8. Trim to the relaxation epoch

NR waveforms contain junk radiation in the first ~100–200 M (due to the initial
data not being exactly on the constraint manifold).  Use `trim_to_relaxation_time()`
to discard it:

```python
wfm_clean = wfm.trim_to_relaxation_time(total_mass=TOTAL_MASS)

# Get the GW frequency at the relaxation epoch (call on the original WaveformModes)
f_start = wfm.f_lower_at_relaxation(total_mass=TOTAL_MASS)
print(f"f_lower at relaxation: {f_start:.1f} Hz")
```

---

## Summary

| Step | Method |
|------|--------|
| Load catalog | `RITCatalog.load()` |
| Browse simulations | `cat.simulations_dataframe` |
| Inspect metadata | `cat.get_metadata(sim_name)` |
| PyCBC-compatible params | `cat.get_parameters(sim_name, total_mass=M)` |
| Load waveform | `cat.get(sim_name)` |
| Physical mode | `wfm.get_mode(ell, m, total_mass, distance, delta_t)` |
| Polarizations | `wfm.get_td_waveform(total_mass, distance, inclination, coa_phase, delta_t)` |
| Trim junk | `wfm.trim_to_relaxation_time(total_mass)` |
| Starting frequency | `wfm.f_lower_at_relaxation(total_mass)` — call on original `WaveformModes` |

The same workflow works for SXS and MAYA — just replace `RITCatalog` with
`SXSCatalog` or `MayaCatalog` and use the appropriate simulation name.
