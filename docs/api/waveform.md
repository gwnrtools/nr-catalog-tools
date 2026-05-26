# `nrcatalogtools.waveform`

The `waveform` sub-package contains the central `WaveformModes` object and its supporting
utilities for loading, rotating, and matching NR waveforms.

> **Conceptual guide**: See [WaveformModes Guide](../waveform.md) for worked examples,
> unit conventions, and a description of the `delta_t` dual convention.

---

## WaveformModes

::: nrcatalogtools.waveform.modes.WaveformModes
    options:
      members:
        - load_from_h5
        - load_from_targz
        - filepath
        - sim_metadata
        - metadata
        - peak_time_22
        - label
        - label_nolatex
        - get_mode_data
        - get_mode
        - get_polarizations
        - get_td_waveform
        - f_lower_at_1Msun
        - f_lower_at_relaxation
        - trim_to_relaxation_time
        - get_parameters

---

## Loaders

::: nrcatalogtools.waveform.loaders
    options:
      members:
        - load_from_h5
        - load_from_targz

---

## Matching and rotation

::: nrcatalogtools.waveform.matching
    options:
      members:
        - apply_wigner_rotation_to_mode_dict
        - interpolate_in_amp_phase

---

## Constants

::: nrcatalogtools.waveform.units
    options:
      members:
        - ELL_MIN
        - ELL_MAX
        - _modal_dt
