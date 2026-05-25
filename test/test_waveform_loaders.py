"""Tests for WaveformModes.load_from_targz — zero-padding and _present_modes."""

import io
import tarfile
import warnings

import numpy as np
import pytest

from nrcatalogtools.waveform import WaveformModes


def _make_targz(path, basename, modes, times):
    """Write a minimal ``.tar.gz`` containing one ASCII file per mode.

    File format: three columns — time, real-part, imaginary-part.
    Only the modes listed in *modes* are written; all others will be
    zero-padded by ``load_from_targz``.
    """
    targz_path = path / f"{basename}.tar.gz"
    amp = np.ones_like(times)
    phase = np.zeros_like(times)
    rows = np.column_stack([times, amp, phase])
    content_template = "\n".join(
        f"{r[0]:.6f} {r[1]:.6f} {r[2]:.6f}" for r in rows
    ).encode()

    with tarfile.open(targz_path, "w:gz") as tar:
        for ell, em in modes:
            filename = f"{basename}_l{ell}_m{em}.dat"
            content = content_template  # same data for every mode
            buf = io.BytesIO(content)
            info = tarfile.TarInfo(name=filename)
            info.size = len(content)
            tar.addfile(info, buf)

    return targz_path


def test_load_from_targz_warns_for_missing_modes(tmp_path):
    """UserWarning is issued when modes are absent from the archive."""
    basename = "ExtrapPsi4_RIT-BBH-0001-n100"
    times = np.linspace(0, 100, 101)
    # Only (2,2) and (2,-2) present; all other modes will be zero-padded.
    present = [(2, 2), (2, -2)]
    targz = _make_targz(tmp_path, basename, present, times)

    with pytest.warns(UserWarning, match="zero-padded"):
        wfm = WaveformModes.load_from_targz(str(targz))

    assert hasattr(wfm, "_present_modes")
    assert (2, 2) in wfm._present_modes
    assert (2, -2) in wfm._present_modes
    # A padded mode must NOT appear in _present_modes
    assert (3, 0) not in wfm._present_modes


def test_load_from_targz_present_modes_excludes_zero_padded(tmp_path):
    """_present_modes contains only modes that were physically in the archive."""
    basename = "ExtrapPsi4_RIT-BBH-0002-n100"
    times = np.linspace(0, 50, 51)
    ell2_modes = [(2, m) for m in range(-2, 3)]  # 5 modes
    targz = _make_targz(tmp_path, basename, ell2_modes, times)

    with pytest.warns(UserWarning):
        wfm = WaveformModes.load_from_targz(str(targz))

    assert wfm._present_modes == set(ell2_modes)
    # All ell >= 3 modes were zero-padded; none should appear in _present_modes.
    for ell in range(3, 11):
        for em in range(-ell, ell + 1):
            assert (ell, em) not in wfm._present_modes


def test_load_from_targz_no_warning_when_all_modes_present(tmp_path):
    """No UserWarning when every mode from ell=2 to ell_max is in the archive."""
    from nrcatalogtools.waveform.units import ELL_MIN, ELL_MAX

    basename = "ExtrapPsi4_RIT-BBH-0003-n100"
    times = np.linspace(0, 50, 51)
    all_modes = [
        (ell, em) for ell in range(ELL_MIN, ELL_MAX + 1) for em in range(-ell, ell + 1)
    ]
    targz = _make_targz(tmp_path, basename, all_modes, times)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        wfm = WaveformModes.load_from_targz(str(targz))

    assert set(wfm._present_modes) == set(all_modes)
