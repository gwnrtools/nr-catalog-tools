# `WaveformModes` API Reference

**Class:** `nrcatalogtools.WaveformModes`  
**Source:** [nrcatalogtools/waveform.py](../nrcatalogtools/waveform.py)  
**Inherits from:** `sxs.WaveformModes` (which is an ndarray subclass)

`WaveformModes` is the central data object returned by all catalog `get()` calls. It stores
complex gravitational-wave strain multipole modes $h_{\ell m}(t)$ in dimensionless NR units,
and provides methods to convert to physical units, extract individual modes, sum to polarizations,
apply frame rotations, and compute mismatches.

---

## Construction

Three construction paths are available. In normal usage you will not call these directly —
use the catalog `get()` method instead.

### From HDF5 (RIT / MAYA waveform files)

```python
WaveformModes.load_from_h5(file_path, metadata=metadata_dict, verbosity=0)
```

Reads `amp_l{ell}_m{em}/X,Y` and `phase_l{ell}_m{em}/X,Y` datasets. Interpolates all modes
onto a common uniform time grid. Returns a `WaveformModes` with shape `(n_times, n_modes)`.

### From tar.gz (RIT psi4 files)

```python
WaveformModes.load_from_targz(file_path, metadata=metadata_dict, verbosity=0)
```

Reads ASCII `.asc` / `.dat` / `.txt` files (columns: `time`, `real`, `imag`) from inside the
archive. Missing modes are filled with zeros. Interpolates onto a uniform grid.

### Wrapping an `sxs.WaveformModes` (SXS catalog)

```python
WaveformModes(raw_sxs_obj.data, raw_sxs_obj.time, sim_metadata=metadata_dict, **meta)
```

Used internally by `SXSCatalog.get()`. The `sxs.WaveformModes.data` property may return a
memoryview; all arithmetic wraps it with `np.array(..., dtype=complex)`.

---

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `time` | `np.ndarray` (1-D) | Dimensionless NR times in units of $GM_\text{tot}/c^3$ |
| `data` | `np.ndarray` (2-D, complex) | Shape `(n_times, n_modes)` |
| `LM` | `np.ndarray` (2-D, int) | `[[ell, m], ...]` for each column in `data` |
| `ell_min`, `ell_max` | `int` | Minimum and maximum $\ell$ present |
| `n_modes` | `int` | Number of mode columns |
| `filepath` | `str` | Path to the source data file |
| `sim_metadata` | `dict` | Raw catalog-specific metadata |
| `metadata` | `dict` | Alias for `sim_metadata` |
| `peak_time_22` | `float` | Dimensionless time of peak $|(2,2)|$ amplitude (cached) |
| `label` | `str` | LaTeX summary string: $q$, $\chi_1$, $\chi_2$ |
| `label_nolatex` | `str` | Plain-text version of the same label |

---

## Core Methods

### `get_mode_data(ell, em)`

Returns the raw NR data column for mode `(ell, em)` as a real 2-column array
(`[:, 0]` = real part, `[:, 1]` = imaginary part) in dimensionless units.

```python
raw = wfm.get_mode_data(2, 2)   # shape (n_times, 2)
```

---

### `get_mode(ell, em, total_mass=None, distance=None, delta_t=None, to_pycbc=True)`

The primary method for retrieving a physically-scaled mode.

```python
mode22 = wfm.get_mode(2, 2,
                      total_mass=60.0,   # M_sun
                      distance=100.0,    # Mpc
                      delta_t=1./4096)   # seconds (or M units, see below)
```

**Returns:** A complex PyCBC `TimeSeries` (if `to_pycbc=True`) or a numpy array.

- Amplitude is scaled by $G M_\text{tot} / (c^2 \cdot d_\text{Mpc} \cdot \text{Mpc})$
- Time is resampled to `delta_t`
- Epoch is set so that $t = 0$ coincides with the peak of the $(2,2)$ mode

#### `delta_t` convention

| Value | Interpretation |
|-------|---------------|
| `delta_t > 1/128` | **Dimensionless M units** (NR native, e.g. `0.5` means $0.5\,GM/c^3$) |
| `delta_t ≤ 1/128` | **Physical seconds** (e.g. `1/4096` for detector-band data) |

The returned `TimeSeries.delta_t` is **always in physical seconds** regardless of which
convention was used for input.

---

### `f_lower_at_1Msun(t=None)`

Returns the instantaneous GW frequency of the $(2,2)$ mode in Hz, **normalized to a
1 M☉ system**. `t` must be given in dimensionless M units (same as `self.time`).

```python
# Frequency at the start of the waveform (for any total mass M)
f_start_hz = wfm.f_lower_at_1Msun() / total_mass_msun

# Frequency at a specific NR time
f_hz = wfm.f_lower_at_1Msun(t=wfm.time[0]) / total_mass_msun
```

If `t=None`, returns the frequency at `wfm.time[0]`.

> **Phase convention note:** Returns `abs(f)` to handle the SXS sign convention where
> $h \propto e^{-i\Phi}$ (decreasing phase) versus RIT/MAYA where $h \propto e^{+i\Phi}$.

---

### `get_td_waveform(total_mass, distance, inclination, coa_phase, delta_t, lal_convention=False)`

Sums modes with spin-weight $-2$ spherical harmonics to produce the sky-direction strain:

$$H = h_+ + i h_\times = \sum_{\ell,m} {}^{-2}Y_{\ell m}(\iota, \phi_c) \, h_{\ell m}(t)$$

```python
pols = wfm.get_td_waveform(
    total_mass=40.,     # M_sun
    distance=100.,      # Mpc
    inclination=0.2,    # radians
    coa_phase=0.3,      # radians
    delta_t=1./4096,    # seconds
)
hp = pols.real()
hc = -1 * pols.imag()
```

**Polarization convention:** Returns `conjugate(h)` so that:
- `.real()` gives $h_+$
- `.imag()` gives $+h_\times$

Note this differs from LAL convention where `imag() = -h_\times`. Set `lal_convention=True`
to get LAL-compatible output.

---

### `trim_to_relaxation_time(total_mass)`

Returns a slice of the waveform starting from the relaxation epoch. Reads the relaxation
time from metadata (tries keys `'relaxed-time'`, `'relaxation_time'`, `'reference_time'`).

```python
wfm_trimmed = wfm.trim_to_relaxation_time(total_mass=60.0)
```

---

### `f_lower_at_relaxation(total_mass)`

Convenience method. Returns the GW frequency (Hz) at the relaxation time for the given
total mass.

```python
f_relax = wfm.f_lower_at_relaxation(total_mass=60.0)   # Hz
```

---

## Frame Rotation

### `rotated(R)`

Applies a Wigner D-matrix rotation to all modes:

$$h^{\text{rot}}_{\ell m}(t) = \sum_{m'} h_{\ell m'}(t) \, D^\ell_{m' m}(R)$$

`R` is a `quaternionic.array` representing the rotation. Uses the `spherical` package for
Wigner D-matrix computation.

```python
import quaternionic
R = quaternionic.array.from_euler_angles(alpha, beta, gamma)
wfm_rotated = wfm.rotated(R)
```

### `apply_wigner_rotation_to_mode_dict(mode_dict, R, ell_max)`

Standalone function (also importable from `nrcatalogtools`) for rotating a `dict` of PyCBC
`TimeSeries` objects as returned by `gwsurrogate` or `pycbc.waveform.get_td_waveform_modes()`:

```python
from nrcatalogtools import apply_wigner_rotation_to_mode_dict

rotated_modes = apply_wigner_rotation_to_mode_dict(
    mode_dict,      # {(ell, m): TimeSeries, ...}
    R,              # quaternionic rotation
    ell_max=4,
)
```

---

## Mismatch Methods

### `match_sphere_averaged(other, psd, f_lower, delta_t, ...)`

Compute the sky-averaged mismatch between `self` and `other`:

$$\mathcal{M} = 1 - \max_{t_c,\, \phi_c,\, R \in SO(3)} \frac{\langle h_1 | h_2 \rangle}{\sqrt{\langle h_1 | h_1 \rangle \langle h_2 | h_2 \rangle}}$$

Optimization is via Nelder-Mead over $(t_c, \phi_c, \alpha, \beta)$ where $(\alpha, \beta)$ are
Euler angles parameterizing $R$.

```python
mismatch = wfm_a.match_sphere_averaged(
    wfm_b,
    psd=my_psd,
    f_lower=20.0,     # Hz
    delta_t=1./4096,
)
```

### `match_sphere_averaged_bms_maximized(other, psd, f_lower, j_max, ...)`

Extended version that additionally optimizes over BMS supertranslation coefficients
$\alpha_{jk}$ up to degree `j_max`:

$$h'_{\ell m}(u) = h_{\ell m}(u) - \sum_{j,k,p,q} \alpha_{jk} \, \mathcal{G}^{\ell m}_{jk,pq} \, \dot{h}_{pq}(u)$$

Requires the `scri` package for spin-weighted Gaunt coefficients.

```python
mismatch = wfm_a.match_sphere_averaged_bms_maximized(
    wfm_b,
    psd=my_psd,
    f_lower=20.0,
    j_max=2,          # include ell=1 spatial translations + ell=2 proper supertranslations
)
```

---

## Unit Conventions

All data stored in `WaveformModes` is in **geometrized, mass-scaled dimensionless units**:

| Quantity | Dimensionless unit |
|----------|--------------------|
| Time | $GM_\text{tot}/c^3$ |
| Amplitude ($r \, h_{\ell m}$) | $GM_\text{tot}/c^2$ |

Physical conversion factors (from [`utils.py`](../nrcatalogtools/utils.py)):

```python
import lal

# Time: 1 NR M unit → seconds
m_secs = total_mass_msun * lal.MTSUN_SI

# Amplitude: dimensionless → strain at distance d_Mpc
amp_scale = (lal.G_SI * total_mass_msun * lal.MSUN_SI
             / (lal.C_SI**2 * d_Mpc * 1e6 * lal.PC_SI))
```
