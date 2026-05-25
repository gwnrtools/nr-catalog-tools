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

## Purpose and Scope

`nr-catalog-tools` provides a **stable, unified Python interface** to three publicly available
NR BBH waveform catalogs, designed to serve a broad community of gravitational-wave researchers.

### LIGO-Virgo-KAGRA data analysis

The package is designed to be a reliable upstream dependency for LVK analysis pipelines.
It provides:

- PyCBC-compatible waveform time series and source parameter dicts directly from any catalog
- Consistent physical unit conventions (masses in M☉, distances in Mpc, strain amplitude
  scaling, time epoch at (2,2) peak) across all three backends
- A stable API that abstracts away catalog-specific file formats, metadata schemas, and
  download mechanisms, so injection studies, template bank construction, and parameter
  estimation workflows are not sensitive to which NR catalog is used

### Waveform modeling

The package provides the data-loading and preprocessing layer needed when calibrating or
validating analytical waveform models (EOB, phenomenological, surrogate) against NR:

- Frame alignment tools: SO(3) rotation via Wigner D-matrices, time/phase alignment
- `get_parameters()` returns PyCBC-compatible intrinsic parameter dicts ready to pass
  directly to `pycbc.waveform.get_td_waveform_modes()` or surrogate model evaluators
- `apply_wigner_rotation_to_mode_dict()` rotates surrogate mode dicts into the NR catalog
  frame for direct mode-by-mode comparison

### Cross-catalog comparison

The package additionally supports rigorous NR accuracy studies:

- Noise-weighted mismatch minimized over $t_c$, $\phi_c$, and $R \in SO(3)$
- Extended mismatch optimization over BMS supertranslations $\alpha(\theta,\phi)$
  (direction-dependent retarded-time shifts at null infinity)

See [goal.md](goal.md) for the full scientific derivation.

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
