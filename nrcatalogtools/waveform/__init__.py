"""nrcatalogtools.waveform sub-package.

Re-exports the complete public API that was previously in the monolithic
``waveform.py`` module so that all existing import paths continue to work
without modification.
"""

from nrcatalogtools.waveform.modes import WaveformModes  # noqa: F401
from nrcatalogtools.waveform.matching import (  # noqa: F401
    apply_wigner_rotation_to_mode_dict,
    interpolate_in_amp_phase,
)
from nrcatalogtools.waveform.units import ELL_MIN, ELL_MAX, _modal_dt  # noqa: F401

__all__ = [
    "WaveformModes",
    "apply_wigner_rotation_to_mode_dict",
    "interpolate_in_amp_phase",
    "ELL_MIN",
    "ELL_MAX",
    "_modal_dt",
]
