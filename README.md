[![Code coverage](https://gwnrtools.github.io/nr-catalog-tools/cov_badge.svg)](https://gwnrtools.github.io/nr-catalog-tools/)

# nr-catalog-tools

A unified Python interface to public numerical-relativity (NR) binary black-hole waveform
catalogs, designed for cross-catalog waveform comparison and downstream gravitational-wave
data analysis.

**Supported catalogs:**

| Catalog | Code | Example simulation name |
|---------|------|------------------------|
| [SXS](https://data.black-holes.org/waveforms/catalog.html) | SpEC | `SXS:BBH:0001` |
| [RIT](https://ccrg.rit.edu/content/data/rit-waveform-catalog) | LazEv | `RIT:BBH:0001-n100-id3` |
| [MAYA / GT](https://einstein.gatech.edu/catalog/) | MayaKranc | `GT0001` |

---

## Installation

```bash
pip install nrcatalogtools
```

Dependencies: `sxs >= 2025.0.0`, `pycbc`, `lal`, `h5py`, `quaternionic`, `spherical`, `scipy`.
See [docs/index.md](docs/index.md#dependencies) for the full list.

---

## Quick Start

### Load a catalog

```python
import nrcatalogtools as nrcat

ritcat  = nrcat.RITCatalog.load()
sxscat  = nrcat.SXSCatalog.load(download=False)
mayacat = nrcat.MayaCatalog.load()
```

### Browse simulations

```python
print(ritcat.simulations_dataframe.index)
# Index(['RIT:BBH:0001-n100-id3', 'RIT:BBH:0002-n100-id0', ...], length=1879)
```

### Load a waveform

```python
wfm = ritcat.get("RIT:BBH:0003-n100-id0")
print(wfm.LM)     # available (ell, m) mode pairs
```

### Extract a single mode in physical units

```python
mode22 = wfm.get_mode(2, 2,
                      total_mass=60.0,   # M_sun
                      distance=100.0,    # Mpc
                      delta_t=1./4096)   # seconds
```

### Get h₊ and h✕ polarizations

```python
pols = wfm.get_td_waveform(total_mass=40., distance=100.,
                            inclination=0.2, coa_phase=0.3)
hp, hc = pols.real(), -1 * pols.imag()
```

```python
import matplotlib.pyplot as plt
plt.plot(hp.sample_times, hp, label='h+')
plt.plot(hc.sample_times, hc, label='hx')
plt.legend(); plt.show()
```

![RIT-BBH-0003](test/validation_data/RIT-BBH-0003-n100-id0_m40_d100_inc0p2_coaph0p3.png)

### Get PyCBC-compatible source parameters

```python
params = ritcat.get_parameters("RIT:BBH:0001-n100-id3", total_mass=60.0)
# {'mass1': 30.0, 'mass2': 30.0, 'spin1x': 0.0, ..., 'f_lower': 23.4}
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/index.md](docs/index.md) | Full documentation hub and dependency list |
| [docs/catalogs.md](docs/catalogs.md) | Per-catalog reference: SXS, RIT, MAYA |
| [docs/waveform.md](docs/waveform.md) | `WaveformModes` API reference |
| [docs/architecture.md](docs/architecture.md) | Architectural overview and design decisions |
| [docs/package.md](docs/package.md) | Detailed package internals |
| [docs/goal.md](docs/goal.md) | Scientific motivation and mismatch formalism |

---

## License

See [LICENSE](LICENSE).
