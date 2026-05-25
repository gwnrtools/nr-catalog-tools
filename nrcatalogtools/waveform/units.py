"""Waveform-level constants and time-step helper."""

import numpy as np
from scipy.stats import mode as stat_mode

ELL_MIN, ELL_MAX = 2, 10


def _modal_dt(time_array):
    """Return the most common timestep in *time_array*.

    Calls ``scipy.stats.mode`` with ``keepdims=False`` so the result is
    correct on scipy 1.10 and the changed 1.11+ API alike (the default
    for ``keepdims`` changed, and the ``[0][0]`` indexing broke).
    """
    return float(stat_mode(np.diff(time_array), keepdims=False).mode)
