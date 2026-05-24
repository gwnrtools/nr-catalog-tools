# `nr-catalog-tools` — Architectural Overview

## What it Does

This package provides a unified Python interface to three public NR binary-black-hole waveform
catalogs — SXS (SpEC), RIT (LazEv), and MAYA/GT (MayaKranc) — for the purpose of cross-catalog
waveform comparison. The scientific goal is to quantify NR catalog accuracy by computing
noise-weighted mismatches between waveforms, maximized over frame ambiguities: SO(3) rotations,
time/phase shifts, and BMS supertranslations.

---

## Class Hierarchy

```
sxs.Catalog
  └── CatalogBase (catalog.py)        ← abstract interface + shared get()/get_parameters()
        ├── RITCatalog  (rit.py)       ← web-scraped .txt metadata; HDF5 / tar.gz data
        ├── SXSCatalog  (sxs.py)       ← delegates to sxs package (Zenodo-backed)
        └── MayaCatalog (maya.py)      ← pickle metadata; HDF5 data via mayawaves

sxs.WaveformModes (ndarray subclass)
  └── WaveformModes (waveform.py)      ← adds physical unit scaling, frame rotation, matching
```

---

## Key Design Decisions

1. **`CatalogBase.get()` is the central dispatch** ([catalog.py:62](../nrcatalogtools/catalog.py#L62)). It handles download-on-demand for RIT/MAYA, then calls `WaveformModes.load_from_h5()`. `SXSCatalog` overrides `get()` entirely because SXS data goes through `sxs.load()` / Zenodo, not local HDF5.

2. **Lazy path resolution for SXS** ([sxs.py](../nrcatalogtools/sxs.py)). All path columns are stub empty strings at catalog-load time. Resolving real paths for all ~2000 SXS simulations would require ~2000 `sxs.load()` calls. Actual file access is deferred to `get()`.

3. **`_filepath` as per-instance attribute** ([waveform.py](../nrcatalogtools/waveform.py)). Extracted from `w_attributes` before passing to the `sxs.WaveformModes` parent `__new__`, preventing class-level sharing where loading a second simulation would silently overwrite the first object's path.

4. **`sxs` memoryview → numpy wrapping** ([waveform.py](../nrcatalogtools/waveform.py)). `sxs.WaveformModes.data` may return a memoryview (not a writable numpy array). All arithmetic wraps it with `np.array(..., dtype=complex)`.

5. **`delta_t` dual convention** ([waveform.py](../nrcatalogtools/waveform.py)). Values `> 1/128` are dimensionless M units (NR native); `≤ 1/128` are physical seconds. The returned `TimeSeries.delta_t` is always in seconds.

---

## Data Flows

### RIT load path

```
RITCatalog.load()
  → RITCatalogHelper.read_metadata_df_from_disk()   # ~/.cache/RIT/metadata/metadata.csv
  → [scrape web if missing]
  → RITCatalog.get(sim_name)
  → RITCatalogHelper.download_waveform_data()        # ExtrapStrain_RIT-BBH-XXXX-nYYY.h5
  → WaveformModes.load_from_h5()
      reads amp_l{l}_m{m}/X,Y + phase_l{l}_m{m}/X,Y
      interpolates all modes onto common uniform grid
      returns complex (n_times, n_modes) array
```

### SXS load path

```
SXSCatalog.load()
  → sxs.load("catalog", download=None)
  → SXSCatalog.get(sim_name)
  → sxs.load(sim_name, auto_supersede=True)          # Zenodo-backed
  → sim_obj.strain                                   # sxs.WaveformModes
  → WaveformModes(raw_obj.data, raw_obj.time, ...)  # thin wrapper
```

### MAYA load path

```
MayaCatalog.load()
  → download MAYAmetadata.pkl → catalog.zip          # ~/.cache/MAYA/
  → parse pickle → DataFrame → simulations dict
  → MayaCatalog.get(sim_name)
  → download GT{ID}.h5
  → WaveformModes.load_from_h5()
```

---

## `WaveformModes` Core Methods

| Method | What it does |
|---|---|
| `load_from_h5()` | HDF5 → complex `(n_times, n_modes)`; interpolates amp+phase onto uniform grid |
| `load_from_targz()` | ASCII `.asc/.dat/.txt` in tar.gz → same output |
| `get_mode(l, m, M, D, dt)` | Single mode in physical units; epoch set at (2,2) peak |
| `f_lower_at_1Msun(t)` | Instantaneous GW freq (Hz @ 1 M☉) from (2,2); divide by M for physical |
| `get_td_waveform(M, D, iota, phi, dt)` | Sky-averaged h₊+ih× summed over all modes |
| `trim_to_relaxation_time(M)` | (2,2) mode starting at relaxation epoch |
| `rotated(R)` | Wigner D-matrix rotation of all modes (inherited + overridden) |
| `match_sphere_averaged(other, psd, f_lower)` | Mismatch minimized over t_c, φ_c, R∈SO(3) via Nelder-Mead |
| `match_sphere_averaged_bms_maximized(...)` | Same + BMS supertranslation optimization via spin-weighted Gaunt coefficients (`scri`) |

---

## Metadata Normalization ([metadata.py](../nrcatalogtools/metadata.py))

Catalog-specific keys are normalized to PyCBC-compatible output in `get_source_parameters_from_metadata()`:

| Catalog sentinel key | Input keys | Output keys |
|---|---|---|
| `relaxed_mass1` (RIT) | `relaxed_chi1x/y/z`, `freq_start_22` | `spin1x/y/z`, `f_lower` |
| `GTID` (MAYA) | `a1x/y/z`, `omega_orbital` | `spin1x/y/z`, `f_lower` |
| _(else)_ (SXS) | `reference_dimensionless_spin1/2`, `reference_orbital_frequency` | `spin1x/y/z`, `f_lower` |

**RIT quirk**: raw metadata text files use hyphens (`relaxed-chi1z`); `parse_metadata_txt()` in
`RITCatalogHelper` converts to underscores when building the DataFrame.

