# Catalog Reference

This page documents the three supported NR catalog backends and their metadata conventions.
All three expose the same high-level interface defined in
[`CatalogBase`](../nrcatalogtools/catalog.py).

---

## Common Interface (`CatalogBase`)

Every catalog class inherits from `CatalogBase`, which provides:

| Method / Property | Description |
|---|---|
| `load(download=None, verbosity=0)` | Class method. Load catalog metadata from cache or web. |
| `get(sim_name, quantity='waveform')` | Return a `WaveformModes` for the named simulation. Downloads data if not cached. `quantity` can be `'waveform'` or `'psi4'`. |
| `get_metadata(sim_name)` | Raw metadata `dict` for one simulation. |
| `get_parameters(sim_name, total_mass=1.0)` | PyCBC-compatible parameter dict: masses, spins, `f_lower`. |
| `simulations_list` | `list` of all simulation names in the catalog. |
| `simulations_dataframe` | Pandas `DataFrame` indexed by simulation name. |
| `waveform_filepath_from_simname(sim_name)` | Local cache path to the waveform HDF5 file. |
| `download_waveform_data(sim_name)` | Explicitly download waveform data to local cache. |
| `psi4_filepath_from_simname(sim_name)` | Local cache path to psi4 data (RIT only). |
| `download_psi4_data(sim_name)` | Explicitly download psi4 data (RIT only). |

`get()` implements download-on-demand: it checks for a local cache file, downloads if absent or
empty, then returns a `WaveformModes` object. See [waveform.md](waveform.md) for the full API.

---

## RIT Catalog

**Class:** `nrcatalogtools.RITCatalog`  
**Source:** [nrcatalogtools/rit.py](../nrcatalogtools/rit.py)  
**Data host:** `https://ccrg.rit.edu/content/data/rit-waveform-catalog`

### Loading

```python
import nrcatalogtools as nrcat

ritcat = nrcat.RITCatalog.load(verbosity=0)

# Access the simulations DataFrame
df = ritcat.simulations_dataframe
print(df.index)
# Index(['RIT:BBH:0001-n100-id3', 'RIT:BBH:0002-n100-id0', ...])

# Load a waveform
wfm = ritcat.get("RIT:BBH:0003-n100-id0", quantity="waveform")

# Load psi4 data
psi4 = ritcat.get("RIT:BBH:0003-n100-id0", quantity="psi4")
```

### Simulation naming

- Non-eccentric BBH: `RIT:BBH:{id:04d}-n{res}-id{init}` (e.g. `RIT:BBH:0001-n100-id3`)
- Eccentric BBH: `RIT:eBBH:{id:04d}-n{res}-ecc` (e.g. `RIT:eBBH:1843-n100-ecc`)

### Metadata keys (DataFrame columns)

RIT metadata is stored with **hyphens** in the raw `.txt` files and the DataFrame.
`get_source_parameters_from_metadata()` normalizes them to underscores internally.

| Parameter | Column name |
|-----------|-------------|
| Mass ratio $m_1/m_2$ | `relaxed-mass-ratio-1-over-2` |
| Spin $\chi_1$ x,y,z | `relaxed-chi1x`, `relaxed-chi1y`, `relaxed-chi1z` |
| Spin $\chi_2$ x,y,z | `relaxed-chi2x`, `relaxed-chi2y`, `relaxed-chi2z` |
| Eccentricity | `eccentricity` |
| Relaxation time (dimless M) | `relaxed-time` |
| Initial GW frequency (dimless) | `freq-start-22` |

### File formats

- **Waveform:** HDF5 files, e.g. `ExtrapStrain_RIT-BBH-0001-n100.h5`.
  Datasets: `amp_l{ell}_m{em}/X` (time), `amp_l{ell}_m{em}/Y` (amplitude),
  `phase_l{ell}_m{em}/X` (time), `phase_l{ell}_m{em}/Y` (phase).
- **Psi4:** Tar-gzip archives, e.g. `ExtrapPsi4_RIT-BBH-0001-n100-id3.tar.gz`,
  containing ASCII `(time, real, imag)` text files per mode.

### Filter example

```python
df = ritcat.simulations_dataframe
# Equal-mass, non-spinning
q1_ns = df[
    (df["relaxed-mass-ratio-1-over-2"].astype(float) - 1.0).abs() < 0.02
].index.tolist()
```

---

## SXS Catalog

**Class:** `nrcatalogtools.SXSCatalog`  
**Source:** [nrcatalogtools/sxs.py](../nrcatalogtools/sxs.py)  
**Data host:** Zenodo (via the `sxs` package ≥ 2025.0.0)

### Loading

```python
import nrcatalogtools as nrcat

# download=False uses only locally cached data
sxscat = nrcat.SXSCatalog.load(download=False, verbosity=0)

df = sxscat.simulations_dataframe
wfm = sxscat.get("SXS:BBH:0001")
```

### Notes on path resolution

`SXSCatalog` uses **lazy path resolution**: path columns in `simulations_dataframe` are empty
strings at catalog-load time. Resolving real on-disk paths for all ~2000 SXS simulations would
require ~2000 calls to `sxs.load()` (each potentially triggering a Zenodo download). Paths are
resolved on demand inside `get()` via the `sxs` package's caching infrastructure.

The call chain inside `get()` is:
```
sxs.load(sim_name, auto_supersede=True)  →  sim_obj.strain  →  WaveformModes(...)
```

### Metadata keys (DataFrame columns)

| Parameter | Column name |
|-----------|-------------|
| Mass ratio | `reference_mass_ratio` |
| Spin $\chi_1$ vector | `reference_dimensionless_spin1` (3-element list) |
| Spin $\chi_2$ vector | `reference_dimensionless_spin2` (3-element list) |
| Spin $\chi_1$ magnitude | `reference_chi1_mag` |
| Spin $\chi_2$ magnitude | `reference_chi2_mag` |
| Orbital frequency (3-vector) | `reference_orbital_frequency` |
| Reference time (dimless M) | `reference_time` |
| Relaxation time (dimless M) | `relaxation_time` |
| Eccentricity | `reference_eccentricity` |

### Filter example

```python
df = sxscat.simulations_dataframe
q1_ns = df[
    (df["reference_mass_ratio"].astype(float) - 1.0).abs() < 0.02
].index.tolist()
```

---

## MAYA / GT Catalog

**Class:** `nrcatalogtools.MayaCatalog`  
**Source:** [nrcatalogtools/maya.py](../nrcatalogtools/maya.py)  
**Data host:** `https://cgpstorage.ph.utexas.edu/`

### Loading

```python
import nrcatalogtools as nrcat

mayacat = nrcat.MayaCatalog.load(verbosity=0)

df = mayacat.simulations_dataframe
wfm = mayacat.get("GT0001", quantity="waveform")
```

> **Note:** MAYA does not provide psi4 data. Calling `get(..., quantity="psi4")` raises
> `NotImplementedError`.

### Metadata keys (DataFrame columns)

| Parameter | Column name |
|-----------|-------------|
| Mass ratio $m_1/m_2$ | `q` |
| Spin $\chi_1$ x,y,z | `a1x`, `a1y`, `a1z` |
| Spin $\chi_2$ x,y,z | `a2x`, `a2y`, `a2z` |
| Eccentricity | `eccentricity` |
| Orbital angular frequency $M\Omega$ | `omega_orbital` |
| GT simulation ID | `GTID` |

### File formats

- **Metadata:** Downloaded as `MAYAmetadata.pkl` (pickle), stored locally as
  `~/.cache/MAYA/catalog.zip` (bzip2-compressed).
- **Waveform:** HDF5 files in LVCNR format (`GT{ID}.h5`), loaded via the `mayawaves` package.

### Filter example

```python
df = mayacat.simulations_dataframe
q1_ns = df[
    (df["q"].astype(float) - 1.0).abs() < 0.02
].index.tolist()
```

---

## Metadata Normalization

[`metadata.py`](../nrcatalogtools/metadata.py) converts catalog-specific metadata into a
PyCBC-compatible parameter dict via `get_source_parameters_from_metadata(metadata, total_mass)`.
The catalog is detected by sentinel keys:

| Sentinel key present | Catalog | Parameters read |
|---|---|---|
| `relaxed_mass1` | RIT | `relaxed_mass_ratio_1_over_2`, `relaxed_chi1x/y/z`, `relaxed_chi2x/y/z`, `freq_start_22` |
| `GTID` | MAYA | `q`, `a1x/y/z`, `a2x/y/z`, `omega_orbital` |
| _(neither)_ | SXS | `reference_mass_ratio`, `reference_dimensionless_spin1/2`, `reference_orbital_frequency` |

All three paths produce the same output keys:

```python
{
    "mass1": float,     # M_sun (scaled by total_mass)
    "mass2": float,
    "spin1x": float, "spin1y": float, "spin1z": float,
    "spin2x": float, "spin2y": float, "spin2z": float,
    "f_lower": float,   # Hz; or -1 if unknown
}
```

Initial GW frequency is converted to physical Hz as:

```
f_lower [Hz] = f_dimless / (total_mass [M_sun] × lal.MTSUN_SI)
```

For SXS, the magnitude of `reference_orbital_frequency` (a 3-vector) gives $M\Omega_{orb}$,
and $f_{GW} = M\Omega_{orb} / \pi$.
