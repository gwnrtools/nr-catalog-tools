---
title: Frame Conventions (design note)
nav_order: 8
---

# Source-frame conventions: design note
{: .no_toc }

**Status:** proposal for discussion, October 2026. Nothing here is implemented
yet except where marked. The measurements quoted were made on 2026-10-14 and are
reproducible with the checks described in [Validation](#validation).

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Why this note exists

Every catalog publishes its modes in the inertial frame of its own simulation,
and every catalog describes the binary's orientation differently: different
body labels, different reference epochs, different amounts of orientation data,
and in one case a different overall sign of the strain. A comparison that
maximizes over an SO(3) rotation and a phase hides all of this, which is why the
cross-catalog paper did not need to resolve it. A user who asks `nrcats` for
`h+` and `hx` at a given inclination and phase, and compares them with a LAL
waveform model at the parameters `nrcats` returns, cannot hide it: today that
comparison fails for some simulations by up to a full polarization rotation.

This note fixes one convention, records what each catalog provides, and proposes
how `nrcats` maps each catalog onto the convention.

## The contract

For every simulation, `nrcats` exposes the waveform and its parameters in the
**LAL source frame at a reference epoch `t_ref`**, and the two describe the same
binary:

- `get_td_waveform(inclination, coa_phase, ...)` and
  `get_parameters(...)` (masses, spins, `f_ref`) must agree with a LAL waveform
  model evaluated at those parameters **with time alignment only**: no phase,
  polarization or SO(3) maximization.
- An SO(3) or phase optimization (`match_sphere_averaged` and friends) stays
  available as an optional refinement on top of this. It is never needed to undo
  a convention difference, and the rotation it finds is reported as a diagnostic:
  once the convention is right it should be small.

### The LAL source frame

At `t_ref`:

| Axis | Definition |
|---|---|
| `z` | `L̂`, the direction of the orbital angular momentum |
| `x` | `n̂`, the unit separation vector pointing **from body 2 to body 1** |
| `y` | `z × x` |

- **Body 1 is the heavier hole**, so `q = m1/m2 ≥ 1`. Spins are carried with
  their bodies when labels are reordered.
- **Strain convention:** `h+ − i h× = Σ h_lm ₋₂Y_lm`, as in
  `XLALSimInspiralNRWaveformGetHplusHcross`.
- **Phase:** `φ_ref` is the orbital phase of body 1 in this frame at `t_ref`. In
  this frame the NR-to-LAL polarization angle is `α = π` exactly for every
  inclination and phase (checked numerically to 4e-16), so the polarizations
  reduce to evaluating `Σ h_lm ₋₂Y_lm(ι, π/2 − φ_ref)`.
- **Spins, kick and remnant vectors** are expressed in this frame at `t_ref`.

### Modes are loaded in this frame by default

`catalog.get(sim)` returns the modes **already rotated into the LAL source frame
at `t_ref`**: one constant rotation built from the triad, applied once at load
time (with the strain sign correction for SXS). The precession stays in the
modes; only the constant orientation of the simulation's coordinates is removed.

- A user override returns the modes in another frame:
  - `frame="simulation"`: the catalog's own inertial frame, unrotated (today's
    behaviour);
  - `frame="waveform"`: the LAL frame at `t_ref` built from the waveform-derived
    triad (method B below).
- The frame the modes are in is recorded on the object (`wfm.frame`), and every
  method that needs orientation reads it from there. Code that reads the frame
  vectors from the LVCNR file must not be applied to rotated modes, or the
  rotation is applied twice.
- Per-mode quantities of precessing simulations depend on the frame, so analyses
  that compare modes in each simulation's own frame (the cross-catalog paper's
  pipeline, for example) must pass `frame="simulation"` to reproduce their
  results.

### Choice of `t_ref`

`t_ref` is fixed by the data, not by the user, unless the catalog provides the
orbital dynamics as a function of time:

- Where orientation and spins are published at one epoch only, `t_ref` is that
  epoch, and asking for a different `f_ref` is an error (or, opt-in, a PN-evolved
  estimate with a warning).
- Where time series exist, any `t_ref` after the junk radiation can be chosen.

The spins are the reason: the orientation at another time can be read from the
waveform, but the spins cannot.

## What each catalog provides

Measured on the files listed in [Validation](#validation).

| | SXS | RIT | MAYA |
|---|---|---|---|
| Body 1 in the raw metadata | heavier | **not mass-ordered**; lighter in most runs | heavier |
| Body 1 in the LVCNR file | heavier (sxs exporter) | **heavier**: RIT relabels the bodies and swaps the spins in its LVCNR files | heavier |
| Strain sign relative to LAL models | **−h** | +h | +h |
| Orientation data | horizon positions and spins vs time (`Horizons.h5`); LVCNR export (format 3) carries `nhat`, `LNhat`, `Omega`, spins vs time | `relaxed-LNhat`, `relaxed-nhat`, `relaxed-chi1/2` at `relaxed-time` only; LVCNR files are format 1 with the same vectors and no time series | format 2/3 files (MAYA-numbered runs): `LNhat`, `Omega`, positions, spins vs time. Format 1 files (older GT runs): one snapshot |
| Epoch of the single-snapshot vectors | (any time) | the **first sample of the stored waveform**. `Omega` equals the published `freq-start-22`, and the orbital phase at the first sample is within 0.16 rad of `nhat` | the **first sample of the stored waveform**, within 5 M for three runs, 20 M for GT0587 |
| `nhat` direction in the LVCNR file | body 2 to body 1 | body 2 to body 1 (after relabelling) | body 2 to body 1 |

Notes:

- **SXS strain sign.** SXS's own LVCNR export copies the modes unchanged, and
  run through LAL's NR routine they agree with both NRSur7dq4 (overlap
  0.996–0.999) and IMRPhenomXPHM (0.93–0.96) only after `h → −h`. RIT and MAYA
  agree with each other and with LAL without the flip. SXS records this in each
  strain file's JSON sidecar: its `spec_version_hist` lists "Overall sign change
  in complex strain h." Whether files from before that change carry the opposite
  sign is still to check, so the correction should key on that record rather
  than be applied unconditionally.
- **Extrapolation and CoM correction.** SXS: polynomial extrapolation in 1/r
  (N2 default; N3, N4 available), translation and boost to the centre-of-mass
  frame, memory added; all recorded in the sidecar. RIT: perturbative
  extrapolation of Ψ4 (Nakano et al. 2015), no CoM correction. MAYA: finite
  extraction radius (radius not stated in the LVCNR files), "Center of mass drift
  corrected". None of these rotate the frame, but MAYA's time axis is retarded
  time at a finite radius, which matters for the dynamics-to-waveform time
  mapping.
- **MAYA official LVCNR files, measured.** Each matches mayawaves'
  CoM-corrected, finite-radius strain to 1e-8 to 1e-6 in every mode, at the
  radius mayawaves picks from the grid (200 M for GT0757, 145 M for MAYA0970,
  100 M for GT0479), and none is extrapolated. The native files
  (`maya_format/`) hold raw-frame Ψ4 at up to 16 finite radii plus the
  trajectories. Extrapolating the native Ψ4 (waveformtools, Nakano et al. 2015)
  lowers MAYA's median (2,2) mismatch with NRSur7dq4 from 1.6e-3 to 1.3e-4 over
  88 quasi-circular runs and removes its growth with waveform length, so MAYA
  waveforms in the LAL frame should be built from extrapolated native Ψ4, not
  from the LVCNR files. MAYA1016–1021 are under-resolved (outer boundary
  204.8 M, wave-zone grid spacing 4x coarser) and should be flagged.
- **MAYA files: the modes are rotated by π about z relative to the dynamics.**
  In every MAYA run checked (MAYA1060, MAYA1070, GT0001, GT0560, and the
  aligned-spin MAYA0937 and MAYA0970), `h_lm(file) = (-1)^m h_lm(dynamics)`.
  Here "dynamics" means the positions, `LNhat`, `nhat` and spins stored in the
  same file. The evidence:
  - In MAYA1060 and MAYA1070 the file's `LNhat-vs-time` agrees with `r x v`
    from its own position tracks to 0.35°. It is 9–59° from the waveform's
    radiation axis (the dominant eigenvector of `<L_(a L_b)>`), and 1–4° from it
    after `(Lx, Ly) -> (-Lx, -Ly)`.
  - That change of L does not distinguish a π rotation about z from a
    reflection through the orbital plane. The phase of the odd-m modes relative
    to `nhat` does: `ψ33 − ψ22` is −90° under the rotation and +90° under the
    reflection, where NRSur7dq4 and SXS (horizons plus strain) give −93° to −95°.
  - NRSur7dq4, maximised over SO(3) and time, matches the rotation 3–4 times
    better than the reflection (MAYA1060 inspiral: 4.1e-3 against 1.6e-2).
  - The flip is already in MAYA's native Ψ4 modes, so the mayawaves export did
    not introduce it.
  - The dynamics agree with the TwoPunctures initial data in the parfile, so it
    is the waveform that is rotated.
  - The waveform-derived frame of nrcats (`frame_consistency` against the
    catalog frame) is π off in phase as stored. After `(-1)^m` it agrees to
    1.4–4° and 0.05 rad (MAYA1060, MAYA1070, GT0001). On RIT the same check
    agrees as stored (2.4° and 5.6°) and fails after `(-1)^m`, so the check
    itself is sound.

  The likely cause is a `(-1)^m` (azimuth) convention in the `Ylm_Decomp`
  thorn; that source has not been inspected. For aligned-spin runs the flip is
  only a shift of π in orbital phase, which explains why nobody noticed it.
  LALSimulation NR injections from MAYA files use the stored modes with the
  stored vectors, so they carry the same flip.

  **nrcats** rotates the modes back by default:
  `MayaCatalog.get(..., correct_azimuth=True)` applies `h_lm -> (-1)^m h_lm`
  before any frame rotation and records `maya_azimuth_corrected` in the
  waveform's metadata. `correct_azimuth=False` returns the modes as stored,
  which reproduces LALSimulation.

  The measurement scripts are in `nrcmp-2026-10/handedness/` and
  `nrcmp-2026-10/handedness_check/` in the analysis workspace.
- **RIT `relaxed-time`** is not a time on the waveform's axis (the axis is
  shifted so the peak is near zero) and is not needed: the vectors hold at the
  first sample. Its value is 30 M for 813 of 1,945 simulations, which is inside
  the junk radiation; only the time series would let us move away from it, and
  RIT has none.
- **Time coordinates.** Horizon (coordinate) time and waveform (retarded) time
  are different coordinates. SXS's exporter treats them as aligned, and that
  choice validates (orbital phase −0.02 rad at the reference epoch of a second
  SXS run). Each backend must state its mapping.
- **SXS strain frame.** v3 strain is translated and boosted to the
  centre-of-mass frame; this does not rotate the frame, and the boosts are of
  order 1e-5, so vectors from the horizon data carry over.

## Two orientation methods, both for every simulation

| | Method A: catalog frame | Method B: waveform-derived |
|---|---|---|
| `L̂` | orbital angular momentum from the dynamics | dominant ℓ = 2 radiation axis, sign from the sense of rotation |
| `n̂` | from the positions | orbital phase from `h22` in that frame; the π ambiguity is resolved by the odd-m modes |
| Available | SXS and MAYA format 2/3 at any time; RIT and MAYA format 1 at the first sample | any time, every catalog |
| Spins | catalog | catalog (B gives no spins) |

`nrcats/reference_frame.py` already implements both (`frame_method="A"` and
`"B"`) for the surrogate comparison; this proposal wires it into the waveform API.

**Default:** A wherever it exists. It is what LAL's NR routine uses, so `nrcats`
agrees with NR injections built from the same files, and the paper measured it as
the better input for waveform models (better on 73–84% of SXS runs). B is the
cross-check everywhere and the fallback where A cannot reach.

**Comparing A and B.** They need not agree exactly even when both are right:

- the radiation axis and `L̂` differ physically for precessing binaries (under
  1° at the epoch for SXS and RIT in the paper);
- the relation between `h22` phase and orbital phase carries mass- and
  spin-dependent corrections of order 0.1 rad;
- for `q ≈ 1` the odd-m modes cannot resolve the π ambiguity in `n̂` (a true
  symmetry only when the spins are symmetric too).

Warnings should therefore sit above that floor, for example 2° in `L̂` and
0.3 rad in orbital phase, tuned on SXS where A is trusted. Run over a whole
catalog, the A–B differences give a per-simulation table of frame quality.

Two lessons from building waveform-derived estimates (October 2026):

- **Strip SXS's memory before method B.** SXS strain carries a constant (2,0)
  memory offset, which tilts a radiation axis built from the ℓ = 2 modes (the
  cross-catalog paper measured this). `sxs.waveforms.memory.remove_memory` with
  the sidecar's `integration_start_time` undoes it.
- **A reference computed on a model must be rotated to the NR orientation.**
  Any waveform-derived vector (an axis, a drift, a bias calibrated on
  NRSur7dq4) is a vector in the frame and turns with the binary's orbital phase.
  The first CoM-drift zero point was subtracted unrotated and made RIT's
  waveforms worse; rotated, it improves them.

## Centre-of-mass drift

A drift of the frame origin, alpha(u) = a0 + b u, leaks the dominant (2,2) into
modes where it does not belong. To first order
`dh = -(alpha . n) dh/du`, which is linear in the six numbers (a0, b), so they
follow from one least-squares solve that removes equatorial asymmetry and
content away from m times the orbital frequency, over the inspiral. The
estimator recovers injected translations and boosts to about 2% for equal and
unequal masses. For unequal masses its in-plane output carries a bias of
1–2e-5 in b from the physical odd-m modes, removed by running it on NRSur7dq4 at
the same parameters, rotated to the NR orientation and shifted to the NR time
origin. CoM correction is defined at null infinity, so it belongs after
extrapolation; finite-radius waveforms carry content that the estimator would
mistake for drift. Measured effect on per-mode mismatch with NRSur7dq4: RIT (2,1)
and (3,3) improve by 10–12% in the median (226 runs); extrapolated MAYA by
25–40% (88 runs). The prototype is `com_drift.py` in the October 2026 scratch
work; it belongs in waveformtools.

## Implementation plan

Staged, each stage with its own tests, on branch `frame-conventions`:

1. **Correctness fixes.** LVCNR path: reference-time sign, the f_ref / t_ref
   interpolation through the `*-vs-time` groups as LAL does it, the
   `relaxed-time` key, normalisation of `L̂`, `n̂` and `L̂ × n̂`. SXS path: the
   strain sign keyed on the sidecar, the frame from SXS metadata with `n̂` from
   body 2 to body 1, and `get_td_waveform` working without an LVCNR file.
2. **Frame infrastructure.** A `ReferenceFrame` object, modes rotated into the
   LAL frame at load (`frame="lal"` default, `"simulation"` override),
   `wfm.frame`, `get_reference_metadata()`, and `get_parameters` /
   `get_td_waveform` reading the same frame.
3. **Method B and `frame_consistency()`**, from `reference_frame.py`, with
   memory stripping for SXS and orientation-aware comparison.
4. **Tests and documentation.** LAL equivalence on tilted frames, per-catalog
   phase-tracking checks, the full suite.

## Proposed API

```python
# Loading: modes in the LAL frame at t_ref by default
wfm = catalog.get(sim_name)                          # frame="lal" (default), method A
wfm = catalog.get(sim_name, frame="simulation")      # catalog's own inertial frame
wfm = catalog.get(sim_name, frame="waveform")        # LAL frame from method B
wfm = catalog.get(sim_name, t_ref=...)               # only where time series exist

wfm.frame          # the frame the modes are in: name, method, t_ref, f_ref, and
                   # the rotation from the simulation frame

# Metadata in the t_ref frame
md = catalog.get_reference_metadata(sim_name, frame="lal", t_ref=None)
md.t_ref, md.f_ref, md.mass1, md.mass2               # body 1 heavier
md.chi1, md.chi2                                     # spins at t_ref, in the frame
md.L_hat, md.n_hat                                   # (0, 0, 1) and (1, 0, 0) in the LAL frame
md.J                                                 # total angular momentum, where available
md.remnant_mass, md.remnant_chi, md.kick             # remnant vectors rotated into the frame
md.frame                                             # same object as wfm.frame

# Polarizations and LAL-model parameters, both from the same frame
h = wfm.get_td_waveform(total_mass=..., distance=..., inclination=..., coa_phase=...)
params = catalog.get_parameters(sim_name, total_mass=...)   # built from md

wfm.frame_consistency()   # A vs B: angle between L_hat, orbital-phase offset, at t_ref
```

- `get_reference_metadata` rotates every vector the catalog provides (spins,
  `L̂`, `n̂`, remnant spin, kick, and positions or velocities where they exist)
  into the requested frame at `t_ref`, and relabels the bodies so that body 1 is
  the heavier. With `frame="simulation"` it returns the same quantities in the
  simulation's own coordinates, so the two can be compared.
- `get_td_waveform`, `get_parameters` and `get_reference_metadata` all read the
  waveform's `frame`, so they cannot describe different binaries.
- A request for a `t_ref` the catalog cannot support (single-epoch catalogs)
  raises, unless the user opts in to PN-evolved spins.

**Implemented on branch `frame-conventions` (October 2026):** `catalog.get(sim,
frame="lal" | "waveform" | "simulation", t_ref=None)` for RIT, MAYA and SXS
(default `"lal"`); `nrcats.frames.frame_of(wfm)`; `get_reference_metadata(sim,
frame=..., t_ref=...)`; `nrcats.frame_waveform.waveform_frame` and
`frame_consistency`. `get_parameters(sim, total_mass, frame="lal")` takes the
spins from `get_reference_metadata` in the same frame as `get`, adds `f_ref`, and
records the frame in `params["frame"]`; `frame=None` keeps the catalog metadata
as before, and a run without the frame data falls back to them with a logged
warning. Spins at another epoch of a Format >= 2 file come from the
`spin*-vs-time` groups through LAL's cubic spline and match
`SimInspiralNRWaveformGetSpinsFromHDF5File` at `fRef = 40 Hz`, 60 Msun to 1e-5
on MAYA1060. Modes rotated after loading (`rotate`, `rotated`) are marked as in
an unknown frame, and `get_angles` and `put_in_frame` refuse them. SXS strain is
stripped of its added memory before a waveform-derived frame is built. PN-evolved
spins for other epochs are not implemented (a different `t_ref` raises for
single-epoch data). The backward-incompatible changes are listed in
`CHANGELOG.md`.

## Known defects this design resolves

| Where | Problem |
|---|---|
| `nrcats/lvc.py`, `get_nr_to_lal_rotation_angles` | `alpha = arccos(calpha)` dropped the sign of α, rotating the polarizations of tilted-frame simulations by up to 4α. **Fixed** on branch `pol_sign` (`alpha = -arctan2(salpha, calpha)`); verified against LAL in 80 of 80 cases. |
| `nrcats/waveform/modes.py`, `_compute_reference_time` | The orbital phase decreases with time, so the derivative is negative, and the search for `+Omega` lands in post-merger noise (GT0001: 81.6 M, after the peak at 1.7 M). This changes the `coa_phase` → `φ_ref` mapping. **Fixed** (stage 1): the reference epoch comes from the data (Format 1: first sample; Format ≥ 2: where `Omega-vs-time` reaches `Omega`; else a recorded reference time; else `|d arg h22/dt|/2` before the peak). |
| `nrcats/lvc.py`, `get_ref_time_from_ref_freq`, `get_ref_freq_from_ref_time`, `get_interp_ref_values_from_h5_file` | `interp1d(...)[x]` raises `TypeError`, and `Omega-vs-time` is read as an attribute where LVCNR stores a group. `f_ref` is silently ignored and `t_ref` raises, so the format 2/3 time series are never used. **Fixed** (stage 1): the `*-vs-time` groups are read and interpolated with LAL's cubic spline, `f_ref` is converted and checked as LAL does (ignored for Format < 2); agrees with `SimInspiralNRWaveformGetHplusHcross` at `fRef > 0` on MAYA1060 to residual phase 2e-7 rad. |
| `nrcats/lvc.py`, `check_interp_req` | Looks for `relaxed_time`; RIT's key is `relaxed-time`. **Fixed** (stage 1): read, but deliberately not used as the epoch (not a time on the waveform's axis). |
| `nrcats/lvc.py`, `compute_lal_source_frame_from_sxs_metadata` | `n̂ = Pos2 − Pos1` points from body 1 to body 2, the reverse of LAL's convention. **Fixed** (stage 1): `n̂ = Pos1 − Pos2`. |
| `nrcats/waveform/modes.py`, `get_angles` | Opens `self.filepath` as an LVCNR file, so `get_td_waveform` fails for every SXS waveform. **Fixed** (stage 1): without an LVCNR file the reference time and frame come from the SXS metadata; agrees with LAL on the sxs LVCNR export of SXS:BBH:0792 to Re(overlap) ≥ 0.997, time alignment only. |
| SXS backend | No strain sign correction. **Fixed** (stage 1): `SXSCatalog.get` flips the sign when the sidecar records it (`correct_strain_sign=True`), recorded as `strain_sign_flipped` in the waveform metadata. |
| `test/test_gt_pol.py` | Compares with phase maximization and uses a file with `L̂ = ẑ` (where α = π exactly), so it cannot see frame errors; it also writes `f_lower_at_1MSUN = f_lower / total_mass` (wrong units) into the committed validation file. |
| `README.md` | `hc = -1 * pols.imag()` is wrong: the default output already matches LAL with `hc = pols.imag()`. The `lal_convention=True` flag returns the conjugate, which is not LAL's convention. |

## Validation

Three checks, each usable as a regression test:

1. **LAL equivalence.** `get_td_waveform` against
   `lalsimulation.SimInspiralNRWaveformGetHplusHcross` on the same LVCNR file,
   time alignment only, over a grid of inclinations and phases, including a file
   with a tilted frame. Requires the SEOBNRv4ROM data file for format 1 files.
2. **Phase tracking (data only).** Calibrate the `h22`/`h33` phase constants on
   an SXS export at its reference epoch; rotate each file's modes into its own
   triad and follow the orbital phase to find where `nhat` holds. Needs no
   waveform model. Validated on a second SXS export (−0.02 rad).
3. **Model comparison.** NRSur7dq4 through LAL at the LAL-frame parameters,
   time alignment only. Useful as a secondary check, but sensitive to the model's
   own `fRef`/`φ_ref` semantics: a few-percent change in `fRef` moves the fitted
   phase offset substantially. Treat offsets from this check with care.

Files used for the measurements above: SXS:BBH:0792 and SXS:BBH:0809 (exported
with `sxs.utilities.lvcnr.convert_simulation`); RIT:BBH:0282-n100-id3,
RIT:BBH:0168-n140-id1; MAYA GT0001, GT0587, GT0900, MAYA1060, MAYA1070.

## Open questions

1. Whether SXS files without the "Overall sign change" entry in
   `spec_version_hist` need the opposite correction (check a v2-era file).
2. GT0587's 20 M offset, and the poor model fit for MAYA1070 at any phase.
3. Whether `sxstools` and the sxs LVCNR exporter make the same choices for the
   SXS frame and time coordinate.
4. Thresholds for the A–B comparison, tuned on SXS.
5. Whether to offer PN-evolved spins for a user-chosen `f_ref` on single-epoch
   catalogs, or refuse.
