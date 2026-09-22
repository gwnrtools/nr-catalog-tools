"""Put a catalog waveform into a stated asymptotic frame, and say what was done.

The three catalogs publish strain in three different BMS frames, and SXS v3
records what it did to each file in a JSON sidecar: a centre-of-mass
``space_translation`` and ``boost_velocity`` (a BMS transformation), and
``add_memory`` (the electric null memory, integrated from the relaxation
time).  RIT and MAYA do neither.  This module undoes those operations with
the same code that applied them -- ``scri`` for the transformation, ``sxs``
for the memory -- and returns a record of what was applied, so a comparison
can state which frame each side was in.  Nothing here fits anything: every
parameter comes from the sidecar or the caller.

Two facts that shape the implementation, both measured on ``SXS:BBH:0078``
(catalog-comparison-paper findings 5aa and the round-trip experiments behind
it):

* ``scri``'s transform by ``(space_translation, boost_velocity)`` followed
  by the negated pair returns the modes to 1e-5 in L2 over ell <= 8, 3e-7 on
  (2,2).  The negated pair is the inverse to the precision that matters,
  because the corrections are ~1e-2 M and ~1e-4 c and the cross term is
  second order.
* ``sxs.WaveformModes.remove_memory`` returns an object with ``ell_min = 0``
  and a different mode layout.  Index by ``(ell, m)``, never by position: a
  positional comparison reports a 140% change in (2,2) that is not there.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = [
    "FrameRecord",
    "read_sidecar",
    "undo_com_correction",
    "strip_memory",
    "harmonize",
]


@dataclass
class FrameRecord:
    """What was done to a waveform, for the manifest."""

    sim_id: str | None = None
    sidecar: str | None = None
    com_undone: bool = False
    space_translation: list[float] | None = None
    boost_velocity: list[float] | None = None
    memory_stripped: bool = False
    memory_integration_start_time: float | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# the sidecar
# --------------------------------------------------------------------------


def _highest_lev_key(files) -> str | None:
    keys = [k for k in files if k.endswith("Strain_N2.json")]
    if not keys:
        return None
    if "Strain_N2.json" in keys:
        return "Strain_N2.json"
    levs = []
    for k in keys:
        try:
            levs.append((int(k.split(":")[0].replace("Lev", "")), k))
        except ValueError:
            continue
    return sorted(levs)[-1][1] if levs else keys[0]


def read_sidecar(sim_id: str, download: bool = False) -> dict:
    """The ``Strain_N2.json`` sidecar of ``sim_id``'s highest Lev, as a dict.

    Loads the catalog index with ``download=False``: resolving the latest
    release through api.github.com once per process is what exhausted the
    unauthenticated limit on a job array.  The sidecar itself comes from the
    data host, so ``download=True`` is safe for a genuine cache miss.
    """
    import sxs

    sims = sxs.load("simulations", download=False)
    key = _highest_lev_key(sims[sim_id].files)
    if key is None:
        raise KeyError(f"{sim_id}: no Strain_N2.json in its file list")
    try:
        return dict(sxs.load(f"{sim_id}/{key}", download=False))
    except Exception:  # noqa: BLE001 -- cache miss; the file host is not GitHub
        if not download:
            raise
        return dict(sxs.load(f"{sim_id}/{key}", download=True))


# --------------------------------------------------------------------------
# conversions.  scri and sxs each want their own object; the pipeline wants
# nrcats.waveform.WaveformModes back, indexed the way it started.
# --------------------------------------------------------------------------


def _modes_by_lm(w) -> tuple[np.ndarray, dict]:
    t = np.asarray(w.t, dtype=float)
    data = np.asarray(w.data)
    return t, {tuple(int(v) for v in lm): data[:, i] for i, lm in enumerate(w.LM)}


def _to_scri(t, modes: dict, ell_max: int):
    import scri

    ell_min = 2
    n = (ell_max + 1) ** 2 - ell_min**2
    data = np.zeros((len(t), n), dtype=complex)
    for (ell, m), v in modes.items():
        if ell_min <= ell <= ell_max:
            data[:, ell * ell - ell_min * ell_min + (ell + m)] = v
    return scri.WaveformModes(
        t=np.ascontiguousarray(t),
        data=np.ascontiguousarray(data),
        ell_min=ell_min,
        ell_max=ell_max,
        frameType=scri.Inertial,
        dataType=scri.h,
        r_is_scaled_out=True,
        m_is_scaled_out=True,
    )


def _rebuild(template, t, modes: dict):
    """A new nrcats WaveformModes on time grid ``t`` with ``modes`` (a
    (ell, m) -> series dict), carrying ``template``'s metadata and mode order.

    Mirrors how ``SXSCatalog.get`` constructs the object: data, time, the
    simulation metadata, and the sxs-level ``_metadata`` minus the keys the
    constructor supplies itself.
    """
    from nrcats.waveform.modes import WaveformModes

    lm = [tuple(int(v) for v in x) for x in template.LM]
    keep = [k for k in lm if k in modes]
    data = np.column_stack([modes[k] for k in keep])
    meta = dict(getattr(template, "_metadata", {}) or {})
    for k in ("time", "metadata", "waveform_data_location"):
        meta.pop(k, None)
    meta["waveform_data_location"] = ""
    meta["ell_min"] = min(k[0] for k in keep)
    meta["ell_max"] = max(k[0] for k in keep)
    sim_meta = None
    try:
        sim_meta = template.sim_metadata
    except Exception:  # noqa: BLE001 -- RIT/MAYA objects carry it differently
        sim_meta = getattr(template, "metadata", None)
    w = WaveformModes(data, np.asarray(t), metadata=sim_meta, **meta)
    # The constructor derives LM from ell_min/ell_max; the data columns above
    # are ordered to match template.LM restricted to `keep`.  Assert rather
    # than trust it.
    got = [tuple(int(v) for v in x) for x in w.LM]
    if got != keep:
        raise RuntimeError(
            f"mode layout mismatch after rebuild: expected {keep[:6]}..., got {got[:6]}..."
        )
    return w


# --------------------------------------------------------------------------
# the two operations
# --------------------------------------------------------------------------


def undo_com_correction(
    w, space_translation, boost_velocity, ell_max: int | None = None
):
    """Apply the inverse of a recorded ``(space_translation, boost_velocity)``.

    The inverse is the negated pair; see the module docstring for the
    round-trip measurement.  A BMS transformation resamples retarded time, so
    the result is on a new time grid, which is why this returns a whole
    object rather than editing in place.
    """
    t, modes = _modes_by_lm(w)
    lmax = ell_max or max(k[0] for k in modes)
    s = _to_scri(t, modes, lmax)
    out = s.transform(
        space_translation=-np.asarray(space_translation, dtype=float),
        boost_velocity=-np.asarray(boost_velocity, dtype=float),
        ell_max=lmax,
    )
    new = {}
    for ell, m in modes:
        if 2 <= ell <= lmax:
            new[(ell, m)] = np.ascontiguousarray(out.data[:, out.index(ell, m)])
    return _rebuild(w, np.asarray(out.t), new)


def strip_memory(w, integration_start_time: float):
    """Remove the electric null memory ``sxs`` added, using the start time it
    recorded.  ``sxs`` documents this as returning the waveform "as close as
    possible to before the memory correction was applied" when given the
    relaxation time, which is what the sidecar records; measured on
    SXS:BBH:0078 the (2,0) step goes from 0.182 to 0.0005 of max|h_22|."""
    out = w.remove_memory(float(integration_start_time))
    t, modes = _modes_by_lm(out)
    return _rebuild(w, t, {k: v for k, v in modes.items() if k[0] >= 2})


def harmonize(
    w,
    sim_id: str | None = None,
    sidecar: dict | None = None,
    undo_com: bool = True,
    remove_memory: bool = True,
    download: bool = False,
):
    """Undo what the catalog did to ``w``, per its sidecar, and say so.

    Returns ``(waveform, FrameRecord)``.  With neither operation requested it
    returns ``w`` itself and an empty record, so a pipeline can run the
    "as-is" treatment through the same call and record that too.  A waveform
    with no sidecar (RIT, MAYA) gets an empty record and a note: nothing is
    known to undo, and nothing is guessed.
    """
    rec = FrameRecord(sim_id=sim_id)
    if sidecar is None and sim_id is not None and (undo_com or remove_memory):
        try:
            sidecar = read_sidecar(sim_id, download=download)
            rec.sidecar = f"{sim_id}/Strain_N2.json"
        except Exception as exc:  # noqa: BLE001 -- recorded, not fatal
            rec.notes.append(f"no sidecar: {type(exc).__name__}: {exc}")
            return w, rec
    mods = (sidecar or {}).get("modifications", {}) or {}
    tr = mods.get("transform", {}) or {}
    mem = mods.get("add_memory")

    if remove_memory:
        if mem and "integration_start_time" in mem:
            w = strip_memory(w, mem["integration_start_time"])
            rec.memory_stripped = True
            rec.memory_integration_start_time = float(mem["integration_start_time"])
        else:
            rec.notes.append("no add_memory in sidecar; nothing stripped")
    if undo_com:
        st, bv = tr.get("space_translation"), tr.get("boost_velocity")
        if st is not None and bv is not None:
            w = undo_com_correction(w, st, bv)
            rec.com_undone = True
            rec.space_translation = [float(x) for x in st]
            rec.boost_velocity = [float(x) for x in bv]
        else:
            rec.notes.append("no transform in sidecar; nothing undone")
    return w, rec
