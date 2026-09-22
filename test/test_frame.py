"""nrcats.waveform.frame: undo a recorded BMS correction and strip memory.

Synthetic waveforms throughout -- no network, no cache -- built with the same
helper the match tests use.  The properties pinned here are the ones that, if
violated, would silently produce a wrong frame rather than an error: the
negated pair really inverts a scri transform; memory removal really inverts
sxs's memory addition; the rebuilt object keeps its mode order and metadata;
and a waveform with nothing recorded is returned untouched, not guessed at.
"""

import numpy as np
import pytest

pytest.importorskip("scri")

from nrcats.waveform import frame  # noqa: E402
from test_match_consolidation import _chirp_wfm  # noqa: E402


def _by_lm(w):
    d = np.asarray(w.data)
    return {tuple(int(v) for v in lm): d[:, i] for i, lm in enumerate(w.LM)}


def _rel(a, b, keys):
    num = np.sqrt(sum(np.linalg.norm(a[k] - b[k]) ** 2 for k in keys))
    den = np.sqrt(sum(np.linalg.norm(a[k]) ** 2 for k in keys))
    return num / den


def _on_common_grid(w_ref, w):
    """Interpolate ``w``'s modes onto ``w_ref``'s time grid, away from the ends."""
    t_ref = np.asarray(w_ref.t)
    t = np.asarray(w.t)
    lo, hi = max(t_ref[0], t[0]), min(t_ref[-1], t[-1])
    sel = (t_ref > lo + 0.05 * (hi - lo)) & (t_ref < hi - 0.05 * (hi - lo))
    tt = t_ref[sel]
    A = {k: v[sel] for k, v in _by_lm(w_ref).items()}
    B = {
        k: np.interp(tt, t, v.real) + 1j * np.interp(tt, t, v.imag)
        for k, v in _by_lm(w).items()
    }
    return A, B


def test_rebuild_preserves_mode_order_and_metadata():
    w = _chirp_wfm(n_times=1024, ell_max=3)
    t, modes = frame._modes_by_lm(w)
    w2 = frame._rebuild(w, t, modes)
    assert [tuple(map(int, x)) for x in w2.LM] == [tuple(map(int, x)) for x in w.LM]
    assert np.allclose(np.asarray(w2.data), np.asarray(w.data))
    assert w2.ell_min == w.ell_min and w2.ell_max == w.ell_max


def test_rebuild_refuses_a_scrambled_layout():
    w = _chirp_wfm(n_times=512, ell_max=2)
    t, modes = frame._modes_by_lm(w)
    # drop a mode from the middle of an ell block: the constructor derives a
    # full block from ell_min/ell_max, so the data no longer matches it
    modes.pop((2, 0))
    with pytest.raises(RuntimeError, match="mode layout mismatch"):
        frame._rebuild(w, t, modes)


def test_undo_com_correction_inverts_a_forward_transform():
    """The property the module exists for: transform by (st, bv), undo, get
    the original back -- to well inside the change the transform made."""
    w = _chirp_wfm(n_times=2048, ell_max=4)
    t, modes = frame._modes_by_lm(w)
    st = np.array([0.02, -0.01, 0.015])  # M, the scale of the catalog's corrections
    bv = np.array([3e-5, -5e-5, 8e-5])  # c
    fwd = frame._to_scri(t, modes, 4).transform(
        space_translation=st, boost_velocity=bv, ell_max=4
    )
    w_fwd = frame._rebuild(
        w, np.asarray(fwd.t), {k: fwd.data[:, fwd.index(*k)] for k in modes}
    )
    w_back = frame.undo_com_correction(w_fwd, st, bv)

    keys = list(modes)
    A, F = _on_common_grid(w, w_fwd)
    A2, B = _on_common_grid(w, w_back)
    moved = _rel(A, F, keys)
    residual = _rel(A2, B, keys)
    assert moved > 1e-3, "the forward transform did nothing; the test is vacuous"
    assert residual < 0.05 * moved
    assert residual < 1e-3


def test_strip_memory_inverts_sxs_add_memory():
    from sxs.waveforms import memory as M

    w = _chirp_wfm(n_times=2048, ell_max=4)
    t0 = float(np.asarray(w.t)[len(w.t) // 8])
    with_mem = M.add_memory(w, integration_start_time=t0)
    w_mem = frame._rebuild(
        w,
        np.asarray(with_mem.t),
        {k: v for k, v in _by_lm(with_mem).items() if k[0] >= 2},
    )
    stripped = frame.strip_memory(w_mem, t0)
    keys = [k for k in _by_lm(w)]
    added = _rel(_by_lm(w), _by_lm(w_mem), keys)
    residual = _rel(_by_lm(w), _by_lm(stripped), keys)
    assert added > 1e-4, "add_memory did nothing; the test is vacuous"
    assert residual < 0.05 * added
    assert [tuple(map(int, x)) for x in stripped.LM] == [
        tuple(map(int, x)) for x in w.LM
    ]


def test_harmonize_with_nothing_recorded_returns_the_input():
    w = _chirp_wfm(n_times=512, ell_max=2)
    out, rec = frame.harmonize(w, sidecar={"modifications": {}})
    assert out is w
    assert not rec.com_undone and not rec.memory_stripped
    assert any("nothing" in n for n in rec.notes)


def test_harmonize_as_is_is_a_no_op_with_an_empty_record():
    w = _chirp_wfm(n_times=512, ell_max=2)
    out, rec = frame.harmonize(
        w, sim_id="SXS:BBH:0000", undo_com=False, remove_memory=False
    )
    assert out is w
    assert rec.as_dict()["notes"] == []


def test_harmonize_applies_only_what_the_sidecar_records():
    w = _chirp_wfm(n_times=1024, ell_max=3)
    sidecar = {
        "modifications": {
            "transform": {
                "space_translation": [0.01, 0.0, 0.0],
                "boost_velocity": [0.0, 0.0, 2e-5],
            }
        }
    }
    out, rec = frame.harmonize(w, sidecar=sidecar)
    assert rec.com_undone and not rec.memory_stripped
    assert rec.space_translation == [0.01, 0.0, 0.0]
    assert any("add_memory" in n for n in rec.notes)
    assert out is not w
