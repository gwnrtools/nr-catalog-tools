# Catalog Reference

This page documents the three supported NR catalog backends and their metadata conventions.
All three expose an **identical high-level interface** defined in
[`CatalogBase`](api/catalog.md), so that LVK analysis pipelines, waveform
modeling workflows, and cross-catalog comparison studies can switch between catalogs —
or sweep across all three — without any catalog-specific code paths.

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
**Source:** [api/rit.md](api/rit.md)  
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
**Source:** [api/sxs.md](api/sxs.md)  
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
**Source:** [api/maya.md](api/maya.md)  
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

[`metadata.py`](api/metadata.md) converts catalog-specific metadata into a
PyCBC-compatible parameter dict via `get_source_parameters_from_metadata(metadata, total_mass)`.
The catalog is detected by sentinel keys:

| Sentinel key present | Catalog |
|---|---|
| `relaxed_mass1` | RIT |
| `GTID` | MAYA |
| _(neither of the above)_ | SXS |

All three paths produce the same output dict:

```python
{
    "mass1": float,     # M_sun (scaled by total_mass)
    "mass2": float,
    "spin1x": float, "spin1y": float, "spin1z": float,
    "spin2x": float, "spin2y": float, "spin2z": float,
    "f_lower": float,   # Hz; or -1 if unknown
}
```

Initial GW frequency conversion:
- **RIT / MAYA**: `f_lower [Hz] = f_dimless / (total_mass [M_sun] × lal.MTSUN_SI)`
- **SXS**: magnitude of `reference_orbital_frequency` (3-vector) gives $M\Omega_{orb}$; $f_{GW} = M\Omega_{orb} / \pi / (M_\text{tot} \times \text{MTSUN\_SI})$

---

## Exhaustive Metadata Key Mapping

The table below maps every physical quantity to its key name in each catalog's raw metadata,
the normalized key used internally by `nrcatalogtools`, and the corresponding PyCBC parameter
name. Units are **dimensionless code units** (total mass $M=1$, $G=c=1$) unless noted.

### Identification

| Physical quantity | RIT (raw `.txt`, hyphens) | SXS (`snake_case`) | MAYA (`snake_case`) | PyCBC / nrcatalogtools output |
|---|---|---|---|---|
| Catalog simulation ID | `catalog-tag` + `resolution-tag` + `id-tag` → e.g. `RIT:BBH:0001-n100-id3` | `alternative_names` → e.g. `SXS:BBH:0001` | `GTID` → e.g. `GT0001` | `simulation_name` (index key) |
| Internal run name | `run-name` | `simulation_name` | `GT_Tag` | — |
| Object types | `system-type` (`Aligned`, `Precessing`, `Nonspinning`) | `object_types` (`BHBH`, `NSNS`, `BHNS`) | — | — |
| Bibtex citation keys | `simulation-bibtex-keys`, `code-bibtex-keys` | `simulation_bibtex_keys`, `code_bibtex_keys` | — | — |

### Masses

| Physical quantity | RIT (raw, hyphens) | SXS | MAYA | PyCBC output |
|---|---|---|---|---|
| Primary mass (physical epoch) | `relaxed-mass1` | `reference_mass1` | `m1` | `mass1` (M☉, scaled by `total_mass`) |
| Secondary mass (physical epoch) | `relaxed-mass2` | `reference_mass2` | `m2` | `mass2` (M☉) |
| Total mass (physical epoch) | `relaxed-total-mass` | `reference_mass1` + `reference_mass2` | `m1` + `m2` | `mtotal` (M☉) |
| Mass ratio $m_1/m_2 \ge 1$ | `relaxed-mass-ratio-1-over-2` | `reference_mass_ratio` | `q` | — (derived: `mass1/mass2`) |
| Symmetric mass ratio $\eta$ | — (derived) | — (derived) | `eta` | `eta` |
| Irreducible mass, primary | — | — | `m1_irr` | — |
| Irreducible mass, secondary | — | — | `m2_irr` | — |
| Initial bare mass, primary | `initial-mass1` | `initial_mass1` | — | — |
| Initial bare mass, secondary | `initial-mass2` | `initial_mass2` | — | — |
| Remnant / final mass | `final-mass` | `remnant_mass` | — | — |

### Spins

| Physical quantity | RIT (raw, hyphens) | SXS | MAYA | PyCBC output |
|---|---|---|---|---|
| Primary spin x-component $\chi_{1x}$ | `relaxed-chi1x` | `reference_dimensionless_spin1[0]` | `a1x` | `spin1x` |
| Primary spin y-component $\chi_{1y}$ | `relaxed-chi1y` | `reference_dimensionless_spin1[1]` | `a1y` | `spin1y` |
| Primary spin z-component $\chi_{1z}$ | `relaxed-chi1z` | `reference_dimensionless_spin1[2]` | `a1z` | `spin1z` |
| Secondary spin x-component $\chi_{2x}$ | `relaxed-chi2x` | `reference_dimensionless_spin2[0]` | `a2x` | `spin2x` |
| Secondary spin y-component $\chi_{2y}$ | `relaxed-chi2y` | `reference_dimensionless_spin2[1]` | `a2y` | `spin2y` |
| Secondary spin z-component $\chi_{2z}$ | `relaxed-chi2z` | `reference_dimensionless_spin2[2]` | `a2z` | `spin2z` |
| Primary spin magnitude $\|\chi_1\|$ | derived from components | `reference_chi1_mag` | derived | — |
| Secondary spin magnitude $\|\chi_2\|$ | derived from components | `reference_chi2_mag` | derived | — |
| Effective spin $\chi_\text{eff}$ | derived | `reference_chi_eff` | derived | — |
| Primary in-plane spin $\chi_{1\perp}$ | derived | `reference_chi1_perp` | derived | — |
| Secondary in-plane spin $\chi_{2\perp}$ | derived | `reference_chi2_perp` | derived | — |
| Initial spin x, primary | `initial-bh-chi1x` | `initial_dimensionless_spin1[0]` | — | — |
| Initial spin y, primary | `initial-bh-chi1y` | `initial_dimensionless_spin1[1]` | — | — |
| Initial spin z, primary | `initial-bh-chi1z` | `initial_dimensionless_spin1[2]` | — | — |
| Initial spin x, secondary | `initial-bh-chi2x` | `initial_dimensionless_spin2[0]` | — | — |
| Initial spin y, secondary | `initial-bh-chi2y` | `initial_dimensionless_spin2[1]` | — | — |
| Initial spin z, secondary | `initial-bh-chi2z` | `initial_dimensionless_spin2[2]` | — | — |
| Remnant spin magnitude | `final-chi` | `remnant_dimensionless_spin` (3-vector) | — | — |

### Orbital Dynamics and Frequencies

| Physical quantity | RIT (raw, hyphens) | SXS | MAYA | PyCBC output |
|---|---|---|---|---|
| Physical epoch time (code units M) | `relaxed-time` | `reference_time` | — | — |
| Relaxation / junk-transient end time | `relaxed-time` | `relaxation_time` | — | — |
| Initial orbital frequency $M\Omega_0$ | — | `initial_orbital_frequency` (scalar) | `omega_orbital` | — |
| Reference orbital frequency (3-vector) | — | `reference_orbital_frequency` ([Ωx, Ωy, Ωz]) | — | — |
| Starting GW (2,2) frequency (code units) | `freq-start-22` | derived: $\|\mathbf{\Omega}\|/\pi$ | derived: `omega_orbital`/π | `f_lower` (Hz, after unit conversion) |
| Starting GW freq at 1 M☉ (Hz) | `freq-start-22-Hz-1Msun` | derived | `f_lower_at_1MSUN` | `f_lower` = value / `total_mass` |
| Initial separation | `initial-separation` | `initial_separation` | `separation` | — |
| Reference separation | — | `reference_separation` | — | — |
| Orbital eccentricity | `eccentricity` | `reference_eccentricity` | `eccentricity` | `eccentricity` |
| Mean anomaly | — | `reference_mean_anomaly` | `mean_anomaly` | `mean_per_ano` |
| Eccentricity measurement method | `eccentricity-measurement-method` | — | — | — |
| Unit orbital angular momentum $\hat{L}$ | `relaxed-LNhatx/y/z` | derived from `reference_orbital_frequency` | — | — |
| Unit separation vector $\hat{n}$ | `relaxed-nhatx/y/z` | `reference_position1/2` (difference) | — | — |
| Number of GW (2,2) cycles | `number-of-cycles-22` | `number_of_orbits` × 2 | — | — |
| Number of orbits | `number-of-orbits` | `number_of_orbits` | — | — |
| Merger / common horizon time | — | `common_horizon_time` | `merge_time` | — |
| Peak (2,2) orbital frequency at merger | `peak-omega-22` | — | — | — |
| Peak (2,2) amplitude | `peak-ampl-22` | — | — | — |

### Initial ADM Conserved Quantities

| Physical quantity | RIT (raw, hyphens) | SXS | MAYA | PyCBC output |
|---|---|---|---|---|
| Initial ADM energy | `initial-ADM-energy` | `initial_ADM_energy` | — | — |
| Initial ADM angular momentum (magnitude) | `initial-orbital-angular-momentum` | `initial_ADM_angular_momentum` (3-vector) | — | — |
| Initial ADM linear momentum | — | `initial_ADM_linear_momentum` (3-vector) | — | — |
| Initial position, primary | — | `initial_position1` ([x,y,z]) | — | — |
| Initial position, secondary | — | `initial_position2` ([x,y,z]) | — | — |

### Remnant Properties

| Physical quantity | RIT (raw, hyphens) | SXS | MAYA | PyCBC output |
|---|---|---|---|---|
| Remnant mass | `final-mass` | `remnant_mass` | — | — |
| Remnant spin magnitude | `final-chi` | magnitude of `remnant_dimensionless_spin` | — | — |
| Remnant spin vector | — | `remnant_dimensionless_spin` ([χx, χy, χz]) | — | — |
| Remnant recoil / kick velocity | `final-kick` (km/s) | `remnant_velocity` ([vx, vy, vz] in units of c) | — | — |
| Peak GW luminosity | `peak-luminosity-ergs-per-sec` | — | — | — |

### Numerical Method

| Physical quantity | RIT (raw, hyphens) | SXS | MAYA | PyCBC output |
|---|---|---|---|---|
| Evolution code | `code` (`LazEv`) | `code_bibtex_keys` | MayaKranc (implicit) | — |
| Formulation | `evolution-system` (`BSSN`) | — | — | — |
| Grid resolution tag | `resolution-tag` (e.g. `n100`) | `simulation_name` Lev suffix | — | — |
| Initial data type | `initial-data-type` | `initial_data_type` | — | — |
| Finite difference order | `fd-order` | — | — | — |
| CFL factor | `cfl` | — | — | — |

### Cache / Path Bookkeeping (added by nrcatalogtools)

| Key | RIT | SXS | MAYA |
|---|---|---|---|
| Waveform data URL | `waveform_data_link` | `waveform_data_link` (empty stub) | `waveform_data_link` |
| Waveform local path | `waveform_data_location` | `waveform_data_location` (empty stub) | `waveform_data_location` |
| Psi4 data URL | `psi4_data_link` | `psi4_data_link` (empty stub) | — (not available) |
| Psi4 local path | `psi4_data_location` | `psi4_data_location` (empty stub) | — (not available) |
| Metadata URL | `metadata_link` | `metadata_link` (empty stub) | `metadata_link` |
| Metadata local path | `metadata_location` | `metadata_location` (empty stub) | `metadata_location` |

> **SXS path stubs**: All SXS path/URL columns are empty strings at catalog-load time.
> They are resolved on demand inside `get()` via `sxs.load(sim_name)` to avoid triggering
> ~2000 Zenodo downloads at startup.

### PyCBC Waveform Parameter Reference

The following PyCBC parameter names are accepted by `pycbc.waveform.get_td_waveform()`,
`get_fd_waveform()`, and `get_td_waveform_modes()`. The `get_parameters()` method on all
catalog classes returns a dict with the starred (★) names populated.

| PyCBC parameter | Units | Description | Populated by `get_parameters()`? |
|---|---|---|---|
| `mass1` ★ | M☉ | Primary (larger) object mass | Yes |
| `mass2` ★ | M☉ | Secondary (smaller) object mass | Yes |
| `spin1x` ★ | — | Primary spin x-component $\chi_{1x} = S_{1x}/m_1^2$ | Yes |
| `spin1y` ★ | — | Primary spin y-component | Yes |
| `spin1z` ★ | — | Primary spin z-component | Yes |
| `spin2x` ★ | — | Secondary spin x-component | Yes |
| `spin2y` ★ | — | Secondary spin y-component | Yes |
| `spin2z` ★ | — | Secondary spin z-component | Yes |
| `f_lower` ★ | Hz | Starting GW frequency | Yes (or -1 if unavailable in metadata) |
| `f_ref` | Hz | Reference frequency for spin definitions | No |
| `f_final` | Hz | Maximum frequency | No |
| `delta_t` | s | Time sample spacing (e.g. `1./4096`) | No |
| `delta_f` | Hz | Frequency bin spacing | No |
| `distance` | Mpc | Luminosity distance | No |
| `inclination` | rad | Angle between $\hat{L}$ and line of sight | No |
| `coa_phase` | rad | Orbital phase at peak amplitude | No |
| `tc` | GPS s | Coalescence time | No |
| `ra` | rad | Right ascension | No |
| `dec` | rad | Declination | No |
| `polarization` | rad | GW polarization angle | No |
| `eccentricity` | — | Orbital eccentricity | No |
| `mean_per_ano` | rad | Mean anomaly at periastron | No |
| `long_asc_nodes` | rad | Longitude of ascending nodes | No |
| `mchirp` | M☉ | Chirp mass (derived from `mass1`, `mass2`) | No |
| `mtotal` | M☉ | Total mass (derived) | No |
| `eta` | — | Symmetric mass ratio (derived) | No |
| `lambda1` | — | Dimensionless tidal deformability of object 1 (BNS/BHNS) | No |
| `lambda2` | — | Dimensionless tidal deformability of object 2 | No |
| `approximant` | str | Waveform model name (e.g. `IMRPhenomXPHM`, `NR_hdf5`) | No |
| `numrel_data` | path | Path to NR HDF5 file; required when `approximant='NR_hdf5'` | No |
| `mode_array` | list | List of `[l,m]` mode pairs to include | No |
| `phase_order` | int | PN phase order (−1 = highest available) | No |
| `amplitude_order` | int | PN amplitude order | No |
| `spin_order` | int | PN spin-correction order | No |
| `tidal_order` | int | PN tidal-correction order | No |
