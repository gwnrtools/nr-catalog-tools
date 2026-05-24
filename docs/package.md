# nr-catalog-tools: Package Overview

## 1. Scientific Purpose

`nr-catalog-tools` provides a **unified Python interface** to three publicly available numerical-relativity (NR) binary black-hole (BBH) waveform catalogs:

| Catalog | Code | Naming convention |
|---------|------|-------------------|
| **SXS** | SpEC (Spectral Einstein Code) | `SXS:BBH:0001`, `SXS:BBH:0002`, … |
| **RIT** | LazEv | `RIT:BBH:0001-n100-id3`, `RIT:eBBH:1843-n100-ecc`, … |
| **MAYA/GT** | MayaKranc | `GT0001`, `GT0355`, … |

The scientific goal (documented in [goal.md](goal.md)) is to enable large-scale cross-catalog comparison of BBH waveforms via noise-weighted inner products (matches), maximized over source-frame ambiguities — rotations $R \in SO(3)$, time translations $t_c$, phase offsets $\phi_c$, and BMS supertranslations. The package is the data-loading and preprocessing layer for this analysis pipeline.

---

## 2. Module Structure

```
nrcatalogtools/
├── __init__.py        # Public exports: MayaCatalog, RITCatalog, SXSCatalog,
│                      #   WaveformModes, apply_wigner_rotation_to_mode_dict
├── catalog.py         # Abstract base class CatalogABC + CatalogBase
├── rit.py             # RITCatalog + RITCatalogHelper
├── sxs.py             # SXSCatalog
├── maya.py            # MayaCatalog
├── waveform.py        # WaveformModes (core waveform object)
├── metadata.py        # get_source_parameters_from_metadata()
├── lvc.py             # Frame rotation helpers (check_interp_req,
│                      #   get_nr_to_lal_rotation_angles, get_ref_vals)
└── utils.py           # Cache paths, download helpers, unit-conversion factors
```

---

## 3. Class Hierarchy

```
sxs.Catalog  (from the sxs package)
    └── CatalogBase  (nrcatalogtools/catalog.py)
            ├── RITCatalog   (nrcatalogtools/rit.py)
            ├── SXSCatalog   (nrcatalogtools/sxs.py)
            └── MayaCatalog  (nrcatalogtools/maya.py)

sxs.WaveformModes  (from the sxs package)
    └── WaveformModes  (nrcatalogtools/waveform.py)
```

`CatalogBase` inherits from `sxs.Catalog`, which stores all simulation metadata in `self._dict["simulations"]` — a plain `dict` keyed by simulation name, where each value is another `dict` of metadata fields.

`WaveformModes` inherits from `sxs.WaveformModes`, which is itself an ndarray-like object storing complex mode data with shape `(n_times, n_modes)`.

---

## 4. Catalog Classes

### 4.1 `CatalogBase` (`catalog.py`)

The shared interface that all three catalog classes implement. Key methods:

| Method | Description |
|--------|-------------|
| `load(download=None, ...)` | Class method. Load catalog metadata (from cache or web). Returns catalog instance. |
| `get(sim_name, quantity='waveform')` | Load waveform or psi4 data for one simulation. Returns `WaveformModes`. |
| `get_metadata(sim_name)` | Return simulation metadata as a `dict`. |
| `get_parameters(sim_name, total_mass=1.0)` | Return PyCBC-compatible intrinsic parameter dict (masses, spins, `f_lower`). |
| `simulations_list` | List of all simulation names. |
| `simulations_dataframe` | Pandas DataFrame indexed by simulation name. |
| `waveform_filepath_from_simname(sim_name)` | Local cache path to the waveform HDF5 file. |
| `download_waveform_data(sim_name)` | Download waveform data to local cache. |
| `psi4_filepath_from_simname(sim_name)` | Local cache path to psi4 data. |
| `download_psi4_data(sim_name)` | Download psi4 data to local cache. |

The `get()` method in `CatalogBase` handles download-on-demand logic: it checks whether the local file exists, downloads if needed, then calls `WaveformModes.load_from_h5()`. **SXSCatalog overrides `get()`** entirely because SXS data access goes through the `sxs` package's own download and caching infrastructure, not through local HDF5 files.

### 4.2 `RITCatalog` (`rit.py`)

**Metadata source:** Scraped from `https://ccrgpages.rit.edu/~RITCatalog/Metadata/`. Text files (`.txt`) with `key = value` lines. Cached as `~/.cache/RIT/metadata/metadata.csv`.

**Waveform files:** HDF5 files, e.g. `ExtrapStrain_RIT-BBH-0001-n100.h5`, hosted at `https://ccrgpages.rit.edu/~RITCatalog/Data/`. The HDF5 format uses datasets named `amp_l{ell}_m{em}` and `phase_l{ell}_m{em}`, each with sub-datasets `X` (time) and `Y` (amplitude or phase).

**Psi4 files:** Tar-gzip archives, e.g. `ExtrapPsi4_RIT-BBH-0001-n100-id3.tar.gz`. Loaded via `WaveformModes.load_from_targz()`.

**Loading pattern:**
```python
ritcat = nrcat.RITCatalog.load(verbosity=0)   # loads metadata.csv from cache
wfm = ritcat.get("RIT:BBH:0001-n100-id3", quantity="waveform")
```

**Key metadata column names** (DataFrame columns use hyphens):

| Parameter | Column name |
|-----------|-------------|
| Mass ratio $m_1/m_2$ | `relaxed-mass-ratio-1-over-2` |
| Spin $\chi_1$ (x component) | `relaxed-chi1x` |
| Spin $\chi_1$ (y component) | `relaxed-chi1y` |
| Spin $\chi_1$ (z component) | `relaxed-chi1z` |
| Spin $\chi_2$ (x,y,z) | `relaxed-chi2x`, `relaxed-chi2y`, `relaxed-chi2z` |
| Eccentricity | `eccentricity` |
| Relaxation time (dimless M) | `relaxed-time` |
| Initial GW frequency (dimless) | `freq-start-22` |

**Note:** The `metadata.py` parser normalises hyphenated keys to underscored keys (e.g. `relaxed_mass_ratio_1_over_2`) when building the PyCBC parameter dict.

**Helper class:** `RITCatalogHelper` handles all web-crawling, file naming, caching, and download logic. The naming conventions for RIT files are encoded in `utils.rit_catalog_info`.

### 4.3 `SXSCatalog` (`sxs.py`)

**Metadata source:** Loaded via `sxs.load("catalog", download=None)` (sxs package ≥ 2025.0.0), which fetches from the Simulations directory on Zenodo.

**Waveform files:** Managed entirely by the `sxs` package (Zenodo-backed). Accessed via `sxs.load(sim_name, auto_supersede=True)` which returns a `Simulation` object; `.strain` gives the `sxs.WaveformModes`.

**Loading pattern:**
```python
sxscat = nrcat.SXSCatalog.load(download=False, verbosity=0)
wfm = sxscat.get("SXS:BBH:0001")     # uses sxs.load() internally
```

**Key metadata column names** (underscores):

| Parameter | Column name |
|-----------|-------------|
| Mass ratio | `reference_mass_ratio` |
| Spin $\chi_1$ vector | `reference_dimensionless_spin1` (3-element list) |
| Spin $\chi_2$ vector | `reference_dimensionless_spin2` (3-element list) |
| Spin $\chi_1$ magnitude | `reference_chi1_mag` |
| Spin $\chi_2$ magnitude | `reference_chi2_mag` |
| Orbital frequency | `reference_orbital_frequency` (3-vector) |
| Reference time (dimless) | `reference_time` |
| Relaxation time (dimless) | `relaxation_time` |
| Eccentricity | `reference_eccentricity` |

**Important design note:** `_add_paths_to_metadata()` in `SXSCatalog` uses **lazy stub strings** (empty strings) for all path columns (`waveform_data_location`, `metadata_location`, etc.) because resolving real paths requires calling `sxs.load(sim_name)` per simulation, which would trigger ~2000 downloads. Paths are resolved on demand via the `sxs` package when `get()` is called.

**API difference:** `SXSCatalog.get()` does not call `WaveformModes.load_from_h5()`; it calls `sxs.load(sim_name, auto_supersede=True)` to get the simulation object, then accesses `.strain` to get the `sxs.WaveformModes`, and wraps it into a `nrcatalogtools.WaveformModes`.

### 4.4 `MayaCatalog` (`maya.py`)

**Metadata source:** A pickle file (`MAYAmetadata.pkl`) downloaded from `https://cgpstorage.ph.utexas.edu/`. Cached locally at `~/.cache/MAYA/catalog.zip`.

**Waveform files:** Loaded via the `mayawaves` package (`maya_coalescence.Coalescence`).

**Loading pattern:**
```python
mayacat = nrcat.MayaCatalog.load(verbosity=0)
wfm = mayacat.get("GT0001", quantity="waveform")
```

**Key metadata column names:**

| Parameter | Column name |
|-----------|-------------|
| Mass ratio $m_1/m_2$ | `q` |
| Spin $\chi_1$ (x,y,z) | `a1x`, `a1y`, `a1z` |
| Spin $\chi_2$ (x,y,z) | `a2x`, `a2y`, `a2z` |
| Eccentricity | `eccentricity` |
| Orbital frequency $M\Omega$ | `Momega` |
| GT simulation ID | `GTID` |

---

## 5. `WaveformModes` (`waveform.py`)

The central data object. Inherits from `sxs.WaveformModes`, which in turn is an ndarray subclass. The internal data array has shape `(n_times, n_modes)` with complex dtype.

### 5.1 Construction

Three construction paths:

1. **From HDF5** (RIT/MAYA waveform files):
   ```python
   WaveformModes.load_from_h5(filepath, metadata=metadata_dict)
   ```
   Reads `amp_l{ell}_m{em}` + `phase_l{ell}_m{em}` datasets, interpolates all modes onto a common uniform time grid, builds complex data array.

2. **From tar.gz** (RIT psi4 files):
   ```python
   WaveformModes.load_from_targz(filepath, metadata=metadata_dict)
   ```
   Reads ASCII `.asc`/`.dat`/`.txt` files inside the archive, each containing `(time, real, imag)` columns.

3. **From sxs.WaveformModes** (SXS catalog):
   ```python
   WaveformModes(raw_obj.data, raw_obj.time, sim_metadata=..., **meta)
   ```
   Wraps the sxs native object directly; used in `SXSCatalog.get()`.

### 5.2 Key Properties

| Property | Description |
|----------|-------------|
| `time` | 1-D array of dimensionless NR times (in units of $GM/c^3$) |
| `data` | Complex 2-D array, shape `(n_times, n_modes)` |
| `LM` | List of `[ell, m]` pairs for each column in `data` |
| `ell_min`, `ell_max` | Minimum and maximum $\ell$ values present |
| `filepath` | Path to the source data file |
| `sim_metadata` | The raw metadata dict (catalog-specific keys) |
| `metadata` | Alias for `sim_metadata` |
| `peak_time_22` | Dimensionless time of peak `|(2,2)| mode amplitude` (cached) |
| `label` | LaTeX string summarizing $q$, $\chi_1$, $\chi_2$ (catalog-agnostic) |
| `label_nolatex` | Plain-text version of the same label |

### 5.3 Core Methods

#### `get_mode_data(ell, em)`
Returns the raw NR data for mode `(ell, em)` as a 2-column real array `[:, 0]` = real part, `[:, 1]` = imaginary part, in dimensionless units. Uses `sxs.WaveformModes.index(ell, em)` to locate the column.

#### `get_mode(ell, em, total_mass, distance, delta_t, to_pycbc=True)`
The primary method for retrieving a physically-scaled mode. Returns a PyCBC `TimeSeries` (complex) with:
- **Amplitude** scaled by $G M_\mathrm{tot} / (c^2 \cdot d)$ where $d$ is in Mpc
- **Time** resampled to `delta_t` (see convention below)
- **Epoch** set so that $t = 0$ coincides with the peak of the $(2,2)$ mode

**`delta_t` convention:**
- `delta_t > 1/128` → interpreted as **dimensionless M units** (NR native; e.g. `0.5` means $0.5\,GM/c^3$)
- `delta_t ≤ 1/128` → interpreted as **physical seconds** (e.g. `1/4096` for detector-band data)

The returned `TimeSeries.delta_t` is **always in physical seconds**.

#### `f_lower_at_1Msun(t=None)`
Returns the instantaneous GW frequency of the (2,2) mode, in Hz, **scaled to a 1 M☉ system**. The argument `t` must be in **dimensionless M units** (same grid as `self.time`). To get physical Hz:
```python
f_hz = wfm.f_lower_at_1Msun(t=t_dimless) / total_mass_msun
```
If `t=None`, returns the frequency at the start of the waveform.

#### `get_td_waveform(total_mass, distance, inclination, coa_phase, delta_t)`
Returns the sky-averaged strain $h_+ - ih_\times$ evaluated at the given inclination and coalescence phase, summed over all modes. Returns a complex PyCBC `TimeSeries`; `.real()` gives $h_+$, `.imag()` gives $h_\times$.

**Polarization convention:** The function returns `conjugate(h)` to align with LAL conventions, meaning `real() = h_+` and `imag() = +h_\times` (note: LAL defines `imag() = -h_\times`; the conjugation effectively flips the sign).

#### `trim_to_relaxation_time(total_mass)` (convenience)
Returns a slice of the waveform starting from the relaxation epoch. Reads `t_relax` from metadata (tries keys `'relaxed-time'`, `'relaxation_time'`, `'reference_time'`).

#### `f_lower_at_relaxation(total_mass)` (convenience)
Returns the GW frequency (Hz) at the relaxation time for the given total mass.

#### `match_sphere_averaged(other, psd, f_lower, delta_t, ...)`
Compute the sky-averaged match between two `WaveformModes` objects. Optimizes over time shift, coalescence phase, and source-frame rotation $R \in SO(3)$ (parameterized by Euler angles $\alpha, \beta$).

#### `match_sphere_averaged_bms_maximized(...)`
Extended version of the above that additionally optimizes over BMS supertranslations $\alpha(\theta,\phi) = \sum_{jk} \alpha_{jk} Y_{jk}$ up to `l_max_alpha`. Uses spin-weighted Gaunt coefficients from the `scri` package to compute mode mixing.

### 5.4 Rotation Methods (inherited from `sxs.WaveformModes`)

`WaveformModes.rotated(R)` — applies a Wigner D-matrix rotation. The standalone function `apply_wigner_rotation_to_mode_dict(mode_dict, R, ell_max)` applies the same transformation to a `dict` of PyCBC `TimeSeries` (as returned by `gwsurrogate` or `pycbc.waveform.get_td_waveform_modes`).

---

## 6. Unit Conventions

All internally stored NR data is in **geometrized, mass-scaled dimensionless units**:

| Quantity | Dimensionless unit |
|----------|--------------------|
| Time | $GM_\mathrm{tot}/c^3$ |
| Amplitude ($r\,h_{\ell m}$) | $GM_\mathrm{tot}/c^2$ (i.e. $h_{\ell m}^{NR} = r\,c^2\,h_{\ell m} / (G M_\mathrm{tot})$) |

**Conversion factors** (from `utils.py`):

```python
# Physical time step for 1 sample at NR time step of 1 M
m_secs = total_mass * lal.MTSUN_SI   # utils.time_to_physical(total_mass)

# Strain amplitude scaling factor
amp_scale = lal.G_SI * total_mass * lal.MSUN_SI / (lal.C_SI**2 * distance_mpc * 1e6 * lal.PC_SI)
# = utils.amp_to_physical(total_mass, distance_mpc)
```

**Cross-catalog validation (q=1, chi=0, M=60 M☉, d=100 Mpc):**
All three catalogs agree to within ~0.75% on the peak amplitude of the $(2,2)$ mode and ~0.5% on the accumulated orbital phase in the last 0.1 s before merger, confirming consistent scaling across catalogs.

---

## 7. Metadata Normalization (`metadata.py`)

`get_source_parameters_from_metadata(metadata, total_mass)` converts catalog-specific metadata into a PyCBC-compatible parameter dict. The catalog is identified by the presence of sentinel keys:

| Key present | Catalog | |
|------------|---------|--|
| `relaxed_mass1` | RIT | reads `relaxed_mass_ratio_1_over_2`, `relaxed_chi1x`, …, `freq_start_22` |
| `GTID` | MAYA/GT | reads `q`, `a1x`/`a1y`/`a1z`, `a2x`/`a2y`/`a2z`, `Momega` |
| _(none of above)_ | SXS | reads `reference_mass_ratio`, `reference_dimensionless_spin1/2`, `reference_orbital_frequency` |

**Spin key output:** All three paths write spin components as `spin1x`, `spin1y`, `spin1z`, `spin2x`, `spin2y`, `spin2z` — compatible with PyCBC's `get_td_waveform_modes()`.

**Initial frequency:** Converted to physical Hz via:
```
f_lower [Hz] = f_dimless / (total_mass [M☉] × lal.MTSUN_SI)
```
For SXS, the magnitude of `reference_orbital_frequency` (a 3-vector) is used as $M\Omega$; then $f_{GW} = M\Omega / \pi$.

---

## 8. Cache Layout

Controlled by the `NR_CATALOG_CACHE` environment variable (defaults to `~/.cache/`):

```
~/.cache/
├── RIT/
│   ├── metadata/
│   │   ├── metadata.csv          # aggregated DataFrame of all RIT sim metadata
│   │   └── RIT:BBH:0001-n100-id3_Metadata.txt   # individual files
│   └── data/
│       ├── ExtrapStrain_RIT-BBH-0001-n100.h5
│       └── ExtrapPsi4_RIT-BBH-0001-n100-id3.tar.gz
├── MAYA/
│   ├── metadata/
│   └── data/
│       └── catalog.zip           # zipped MAYAmetadata.pkl
└── SXS/
    └── (managed entirely by the sxs package; typically ~/.cache/sxs/)
```

---

## 9. Key Dependencies

| Package | Role |
|---------|------|
| `sxs` (≥ 2025.0.0) | SXS catalog access, base classes (`sxs.Catalog`, `sxs.WaveformModes`) |
| `mayawaves` | MAYA coalescence loading |
| `pycbc` | `TimeSeries`, `match()`, `get_td_waveform_modes()`, `pnutils` |
| `lal` / `lalsimulation` | Physical constants (`MTSUN_SI`, `MSUN_SI`, `G_SI`, `C_SI`, `PC_SI`) |
| `h5py` | HDF5 file reading (RIT waveform files) |
| `quaternionic` | Quaternion representation of $SO(3)$ rotations |
| `spherical` | Wigner D-matrix computation |
| `scipy` | `InterpolatedUnivariateSpline` for mode resampling |
| `scri` | Spin-weighted Gaunt coefficients (optional; needed for BMS optimization) |
| `gwsurrogate` | Surrogate model evaluation (used in analysis scripts, not the package itself) |

---

## 10. Common Usage Patterns

### Load all three catalogs
```python
import nrcatalogtools as nrcat

ritcat  = nrcat.RITCatalog.load(verbosity=0)
sxscat  = nrcat.SXSCatalog.load(download=False, verbosity=0)
mayacat = nrcat.MayaCatalog.load(verbosity=0)
```

### Find q=1, non-spinning simulations
```python
df_rit  = ritcat.simulations_dataframe
df_sxs  = sxscat.simulations_dataframe
df_maya = mayacat.simulations_dataframe

# RIT (hyphenated column names)
rit_q1 = df_rit[(df_rit["relaxed-mass-ratio-1-over-2"].astype(float) - 1.0).abs() < 0.02].index.tolist()

# SXS (underscore column names)
sxs_q1 = df_sxs[(df_sxs["reference_mass_ratio"].astype(float) - 1.0).abs() < 0.02].index.tolist()

# MAYA
maya_q1 = df_maya[(df_maya["q"].astype(float) - 1.0).abs() < 0.02].index.tolist()
```

### Load a waveform and get the (2,2) mode in physical units
```python
TOTAL_MASS = 60.0   # M_sun
DISTANCE   = 100.0  # Mpc
DELTA_T    = 1.0 / 4096  # seconds

# RIT / MAYA
wfm = ritcat.get("RIT:BBH:0001-n100-id3", quantity="waveform")
mode22 = wfm.get_mode(2, 2, total_mass=TOTAL_MASS, distance=DISTANCE, delta_t=DELTA_T)
# mode22 is a complex PyCBC TimeSeries; epoch set so peak is at t=0
# mode22.real() ≈ h_plus;  mode22.imag() ≈ h_cross

# SXS (uses sxs.load internally)
wfm_sxs = sxscat.get("SXS:BBH:0001")
mode22_sxs = wfm_sxs.get_mode(2, 2, total_mass=TOTAL_MASS, distance=DISTANCE, delta_t=DELTA_T)
```

### Get PyCBC-compatible source parameters
```python
params = ritcat.get_parameters("RIT:BBH:0001-n100-id3", total_mass=60.0)
# Returns: {'mass1': 30.0, 'mass2': 30.0, 'spin1x': 0.0, ..., 'f_lower': 23.4}
# Compatible with pycbc.waveform.get_td_waveform_modes(**params)
```

### Get the starting GW frequency
```python
# Frequency at the start of the NR data
f_start = wfm.f_lower_at_1Msun() / TOTAL_MASS     # Hz

# Frequency at the relaxation time
t_relax_dimless = wfm.time[0] + metadata["relaxed-time"]   # both in dimensionless M
f_relax = wfm.f_lower_at_1Msun(t=t_relax_dimless) / TOTAL_MASS   # Hz
```

---

## 11. Known Design Decisions and Gotchas

### SXS path resolution is lazy
`SXSCatalog._add_paths_to_metadata()` sets all path columns to empty strings at catalog-load time, because resolving real on-disk paths requires calling `sxs.load(sim_name)` for every simulation which would trigger ~2000 network requests. Actual file access happens inside `SXSCatalog.get()` through the `sxs` package's own caching.

### `WaveformModes._filepath` is a per-instance attribute
`_filepath` is extracted from `w_attributes` inside `__new__` before passing to the parent `sxs.WaveformModes` constructor. This prevents class-level attribute sharing where loading a second simulation would silently overwrite the first object's `_filepath`.

### sxs memoryview → numpy wrapping
`sxs.WaveformModes.data` may return a memoryview rather than a writable numpy array. All arithmetic operations (especially in-place `*=`) must wrap the result with `np.array(..., dtype=complex)` first. The relevant locations are: `get_mode()` (mode_data, h_mode_complex) and `peak_time_22` (mode22_data).

### `sxs` API version
The package requires `sxs ≥ 2025.0.0`. The older API `sxs.load("SXS:BBH:0001/rhOverM")` with catalog metadata from `data.black-holes.org` is no longer functional. The current API is:
```python
sim = sxs.load("SXS:BBH:0001", auto_supersede=True)
strain = sim.strain   # sxs.WaveformModes
```

### Metadata key naming across catalogs
RIT metadata uses **hyphens** in raw text files (`relaxed-chi1z`) but the internal DataFrame retains them as-is. When calling `get_source_parameters_from_metadata()`, these are accessed with underscores (`relaxed_chi1z`) — the parser maps hyphen → underscore during `parse_metadata_txt()`. MAYA uses short names (`a1x`, `q`). SXS uses long underscored names (`reference_dimensionless_spin1`).

### `check_interp_req` call signature (`lvc.py`)
The function signature is `check_interp_req(h5_file=None, metadata=None, ref_time=None, ...)`. Call sites must pass `ref_time` as a **keyword argument**:
```python
interp, avail_ref_time = check_interp_req(h5_file, ref_time=t_ref)
```
Passing `t_ref` as a positional second argument binds it to `metadata`, causing the interpolation check to never fire.

### Polarization convention
`get_td_waveform()` returns `conjugate(h)` so that `.real()` gives $h_+$. This differs from LAL's convention where the imaginary part equals $-h_\times$. In this package, `.imag()` gives $+h_\times$.
