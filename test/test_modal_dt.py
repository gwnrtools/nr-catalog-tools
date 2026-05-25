"""Tests for _modal_dt — the modal (most-common) timestep utility."""

import numpy as np

from nrcatalogtools.waveform.units import _modal_dt


def test_modal_dt_uniform_grid():
    """Uniform time array: _modal_dt returns the exact step."""
    t = np.arange(0, 100, 0.1)
    assert abs(_modal_dt(t) - 0.1) < 1e-10


def test_modal_dt_nonuniform_grid_returns_mode():
    """Non-uniform array: _modal_dt returns the step that appears most often."""
    # 90 steps of 0.1 followed by 3 steps of 0.2; mode must be 0.1
    t = np.concatenate([np.arange(0, 9.0, 0.1), [9.2, 9.4, 9.6]])
    assert abs(_modal_dt(t) - 0.1) < 1e-10


def test_modal_dt_larger_step_is_modal():
    """When the majority of steps are large the large step is returned."""
    # 9 steps of 10.0, one outlier step of 1.0
    t = np.concatenate([np.arange(0, 90, 10.0), [91.0]])
    assert abs(_modal_dt(t) - 10.0) < 1e-10
