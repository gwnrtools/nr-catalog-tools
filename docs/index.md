# `nr-catalog-tools` — Documentation

## Contents

| Document | Description |
|----------|-------------|
| [Scientific Goal](goal.md) | Motivation, source-frame ambiguity, BMS supertranslations, mismatch formalism |
| [Catalog Reference](catalogs.md) | Per-catalog loading, metadata keys, file formats, cache layout |
| [WaveformModes API](waveform.md) | Full API reference for the central waveform object |
| [Architecture](architecture.md) | Class hierarchy, data flows, key design decisions |
| [Package Internals](package.md) | Detailed module descriptions, unit conventions, usage patterns, gotchas |

---

## Scientific Purpose

`nr-catalog-tools` provides a **unified Python interface** to three publicly available NR BBH
waveform catalogs for cross-catalog comparison. The scientific goal is to quantify NR catalog
accuracy by computing noise-weighted mismatches between waveforms from different codes,
maximized over source-frame ambiguities:

- Spatial rotations $R \in SO(3)$ (Wigner D-matrix mode mixing)
- Time translations $t_c$ and coalescence phase offsets $\phi_c$
- BMS supertranslations $\alpha(\theta, \phi)$ (direction-dependent retarded-time shifts)

See [goal.md](goal.md) for the full derivation.

---

## Module Structure

```
nrcatalogtools/
├── __init__.py     # Public API: MayaCatalog, RITCatalog, SXSCatalog,
│                   #   WaveformModes, apply_wigner_rotation_to_mode_dict
├── catalog.py      # Abstract base CatalogABC + shared CatalogBase
├── rit.py          # RITCatalog + RITCatalogHelper
├── sxs.py          # SXSCatalog
├── maya.py         # MayaCatalog
├── waveform.py     # WaveformModes
├── metadata.py     # get_source_parameters_from_metadata()
├── lvc.py          # Frame-rotation helpers
└── utils.py        # Cache paths, download helpers, unit conversions
```

---

## Installation

```bash
pip install nrcatalogtools
```

### Dependencies

| Package | Version | Role |
|---------|---------|------|
| `sxs` | ≥ 2025.0.0 | SXS catalog access; base classes `sxs.Catalog`, `sxs.WaveformModes` |
| `pycbc` | any | `TimeSeries`, `match()`, `get_td_waveform_modes()`, `pnutils` |
| `lal` / `lalsimulation` | any | Physical constants (`MTSUN_SI`, `MSUN_SI`, `G_SI`, `C_SI`, `PC_SI`) |
| `h5py` | any | HDF5 reading (RIT waveform files) |
| `quaternionic` | any | Quaternion SO(3) rotation representation |
| `spherical` | any | Wigner D-matrix computation |
| `scipy` | any | `InterpolatedUnivariateSpline` for mode resampling |
| `mayawaves` | any | MAYA coalescence loading (optional; needed for MAYA catalog) |
| `scri` | any | Spin-weighted Gaunt coefficients (optional; needed for BMS optimization) |
| `gwsurrogate` | any | Surrogate model evaluation (optional; used in analysis scripts) |

---

## Quick Start

```python
import nrcatalogtools as nrcat

# Load catalogs
ritcat  = nrcat.RITCatalog.load()
sxscat  = nrcat.SXSCatalog.load(download=False)
mayacat = nrcat.MayaCatalog.load()

# Load a waveform
wfm = ritcat.get("RIT:BBH:0003-n100-id0")

# Physical-unit (2,2) mode
mode22 = wfm.get_mode(2, 2, total_mass=60.0, distance=100.0, delta_t=1./4096)

# Polarizations
pols = wfm.get_td_waveform(total_mass=40., distance=100., inclination=0.2, coa_phase=0.3)
hp, hc = pols.real(), -1 * pols.imag()

# PyCBC-compatible source parameters
params = ritcat.get_parameters("RIT:BBH:0001-n100-id3", total_mass=60.0)
```

---

## Cache Layout

Controlled by the `NR_CATALOG_CACHE` environment variable (default: `~/.cache/`):

```
~/.cache/
├── RIT/
│   ├── metadata/
│   │   ├── metadata.csv                           # aggregated DataFrame
│   │   └── RIT:BBH:0001-n100-id3_Metadata.txt    # per-simulation files
│   └── data/
│       ├── ExtrapStrain_RIT-BBH-0001-n100.h5
│       └── ExtrapPsi4_RIT-BBH-0001-n100-id3.tar.gz
├── MAYA/
│   ├── metadata/
│   └── data/
│       └── catalog.zip                            # zipped MAYAmetadata.pkl
└── SXS/
    └── (managed by the sxs package; typically ~/.cache/sxs/)
```
