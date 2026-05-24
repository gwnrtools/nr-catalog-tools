#!/usr/bin/env python
"""
Compare equal-mass, non-spinning BBH waveforms across SXS, RIT, and MAYA catalogs.

For each catalog we pick the highest-resolution q=1, chi=0 simulation, load the
(2,2) mode, scale to physical units (M=60 Msun, d=100 Mpc), and compare:
  1. Peak amplitude
  2. Amplitude waveform shape (normalized)
  3. Phase alignment
"""
import sys
sys.path.insert(0, "/home/prayush/src/nr-catalog-tools")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nrcatalogtools as nrcat

# Physical parameters
TOTAL_MASS = 60.0   # Msun
DISTANCE   = 100.0  # Mpc
DELTA_T    = 1.0 / 4096  # seconds
ELL, EM    = 2, 2

# ── Load catalogs ─────────────────────────────────────────────────────────────
print("Loading RIT catalog...")
ritcat = nrcat.RITCatalog.load(verbosity=0)
print("  Done.")

print("Loading SXS catalog...")
sxscat = nrcat.SXSCatalog.load(download=False, verbosity=0)
print("  Done.")

print("Loading MAYA catalog...")
mayacat = nrcat.MayaCatalog.load(verbosity=0)
print("  Done.\n")

# ── Inspect columns available ─────────────────────────────────────────────────
print("=== Column inspection ===")
df_rit  = ritcat.simulations_dataframe
df_sxs  = sxscat.simulations_dataframe
df_maya = mayacat.simulations_dataframe

print(f"RIT  columns (relevant): {[c for c in df_rit.columns if any(k in c.lower() for k in ['mass','spin','chi','ecc','q'])]}")
print(f"SXS  columns (relevant): {[c for c in df_sxs.columns if any(k in c.lower() for k in ['mass','spin','chi','ecc','q','ref'])][:20]}")
print(f"MAYA columns (relevant): {[c for c in df_maya.columns if any(k in c.lower() for k in ['mass','spin','chi','ecc','q','a1','a2'])]}")
print()

# ── Find q=1 non-spinning sims ────────────────────────────────────────────────
def find_q1_nospin_rit(df):
    # RIT DataFrame columns use hyphens: 'relaxed-mass-ratio-1-over-2'
    mask = (df["relaxed-mass-ratio-1-over-2"].astype(float) - 1.0).abs() < 0.02
    for col in ["relaxed-chi1x","relaxed-chi1y","relaxed-chi1z",
                "relaxed-chi2x","relaxed-chi2y","relaxed-chi2z"]:
        if col in df.columns:
            mask &= df[col].astype(float).abs() < 0.02
    if "eccentricity" in df.columns:
        mask &= df["eccentricity"].astype(float) < 0.01
    return df[mask].index.tolist()

def find_q1_nospin_sxs(df):
    # SXS columns use underscores: 'reference_mass_ratio', 'reference_chi1_mag'
    col_q = None
    for c in ["reference_mass_ratio","initial_mass_ratio","mass_ratio"]:
        if c in df.columns:
            col_q = c; break
    if col_q is None:
        print("  [SXS] No mass-ratio column found")
        return []
    mask = (df[col_q].astype(float) - 1.0).abs() < 0.02
    for col in ["reference_chi1_mag","reference_chi2_mag",
                "reference_chi1_perp","reference_chi2_perp"]:
        if col in df.columns:
            try: mask &= df[col].astype(float).abs() < 0.05
            except: pass
    if "reference_eccentricity" in df.columns:
        try: mask &= df["reference_eccentricity"].astype(float) < 0.01
        except: pass
    return df[mask].index.tolist()

def find_q1_nospin_maya(df):
    # MAYA columns: 'q', 'a1x', 'a1y', 'a2x', 'a2y'
    col_q = None
    for c in ["q","mass_ratio"]:
        if c in df.columns:
            col_q = c; break
    if col_q is None:
        print("  [MAYA] No mass-ratio column found")
        return []
    mask = (df[col_q].astype(float) - 1.0).abs() < 0.02
    for col in ["a1x","a1y","a2x","a2y"]:
        if col in df.columns:
            try: mask &= df[col].astype(float).abs() < 0.02
            except: pass
    return df[mask].index.tolist()

print("=== Finding q=1, non-spinning simulations ===")
rit_sims  = find_q1_nospin_rit(df_rit)
sxs_sims  = find_q1_nospin_sxs(df_sxs)
maya_sims = find_q1_nospin_maya(df_maya)

print(f"  RIT  ({len(rit_sims)}): {rit_sims[:5]}")
print(f"  SXS  ({len(sxs_sims)}): {sxs_sims[:5]}")
print(f"  MAYA ({len(maya_sims)}): {maya_sims[:5]}")
print()

# ── Load raw NR mode data directly from WaveformModes ────────────────────────
def load_mode22_raw(cat, sim_name, label, is_sxs=False):
    """Load (2,2) mode as WaveformModes; get raw data before unit scaling."""
    print(f"  [{label}] Loading {sim_name} ...")
    try:
        if is_sxs:
            wfm = cat.get(sim_name)  # SXS uses its own get() via sxs.load
        else:
            wfm = cat.get(sim_name, quantity="waveform")
        print(f"    WaveformModes loaded. LM modes: {len(wfm.LM)}, "
              f"time: [{wfm.time[0]:.1f}, {wfm.time[-1]:.1f}] M")
        # Inspect raw (dimensionless) amplitude of (2,2) mode
        raw_22 = wfm.get_mode_data(ELL, EM)
        raw_amp = np.sqrt(raw_22[:,1]**2 + raw_22[:,2]**2)
        print(f"    Raw (dimless) peak |h_22|: {raw_amp.max():.6f}")
        return wfm
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return None

print("=== Loading WaveformModes objects ===")
wfm_rit  = load_mode22_raw(ritcat,  rit_sims[0],  "RIT")  if rit_sims  else None
wfm_sxs  = load_mode22_raw(sxscat,  sxs_sims[0],  "SXS",  is_sxs=True) if sxs_sims  else None
wfm_maya = load_mode22_raw(mayacat, maya_sims[0], "MAYA") if maya_sims else None
print()

# ── Check raw dimensionless amplitudes ────────────────────────────────────────
print("=== Raw dimensionless peak amplitudes of (2,2) mode ===")
print("(Before any mass/distance scaling; for q=1, chi=0 these should all be ~0.3-0.5)")
raw_peaks = {}
for label, wfm in [("RIT", wfm_rit), ("SXS", wfm_sxs), ("MAYA", wfm_maya)]:
    if wfm is not None:
        raw = wfm.get_mode_data(ELL, EM)
        amp = np.sqrt(raw[:,1]**2 + raw[:,2]**2)
        pk = amp.max()
        raw_peaks[label] = pk
        print(f"  [{label}]  peak = {pk:.6f}")

print("\n  Ratios of raw peaks:")
keys = list(raw_peaks)
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        a, b = keys[i], keys[j]
        ratio = raw_peaks[a] / raw_peaks[b]
        print(f"    {a}/{b} = {ratio:.4f}  (log2 = {np.log2(ratio):+.3f})")
print()

# ── Get physical modes ────────────────────────────────────────────────────────
print("=== Loading physical (2,2) modes (M=60 Msun, d=100 Mpc, dt=1/4096 s) ===")
def get_phys_mode(wfm, label):
    if wfm is None:
        return None
    try:
        mode = wfm.get_mode(ELL, EM,
                            total_mass=TOTAL_MASS,
                            distance=DISTANCE,
                            delta_t=DELTA_T)
        pk = np.abs(mode.data).max()
        print(f"  [{label}]  peak |h_22| = {pk:.4e}, "
              f"dt={mode.delta_t:.6g} s, len={len(mode)}, "
              f"t_start={float(mode.start_time):.4f} s")
        return mode
    except Exception as e:
        print(f"  [{label}]  FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return None

mode_rit  = get_phys_mode(wfm_rit,  "RIT")
mode_sxs  = get_phys_mode(wfm_sxs,  "SXS")
mode_maya = get_phys_mode(wfm_maya, "MAYA")
print()

# ── Amplitude comparison ──────────────────────────────────────────────────────
print("=== Physical peak amplitude comparison ===")
phys_peaks = {}
for label, mode in [("RIT", mode_rit), ("SXS", mode_sxs), ("MAYA", mode_maya)]:
    if mode is not None:
        phys_peaks[label] = np.abs(mode.data).max()

keys = list(phys_peaks)
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        a, b = keys[i], keys[j]
        ratio = phys_peaks[a] / phys_peaks[b]
        print(f"  {a}/{b} = {ratio:.4f}  (log2 = {np.log2(ratio):+.3f})")

print("\n  Testing candidate scaling correction factors (if ratio != 1):")
scales = [1, 2, 0.5, np.sqrt(2), 1/np.sqrt(2), np.sqrt(3), 1/np.sqrt(3), 4, 0.25, np.pi/2]
if "RIT" in phys_peaks and "SXS" in phys_peaks:
    print(f"  (Adjusting for RIT/SXS = {phys_peaks['RIT']/phys_peaks['SXS']:.4f})")
    for sf in scales:
        residual = phys_peaks["RIT"] / (phys_peaks["SXS"] * sf)
        flag = " <-- MATCH" if abs(residual - 1.0) < 0.02 else ""
        print(f"    RIT / (SXS × {sf:.4f}) = {residual:.4f}{flag}")
print()

# ── Phase comparison ──────────────────────────────────────────────────────────
print("=== Phase analysis ===")
def get_phase_info(mode, label):
    if mode is None:
        return None, None, None
    t = np.array(mode.sample_times)
    data = np.array(mode.data)
    amp = np.abs(data)
    phase = np.unwrap(np.angle(data))
    # Phase at merger (t=0)
    idx0 = np.argmin(np.abs(t))
    ph_at_0 = phase[idx0]
    print(f"  [{label}]  phi(t=0) = {ph_at_0:.4f} rad ({np.degrees(ph_at_0):.2f} deg),  "
          f"peak amp at t = {t[np.argmax(amp)]:.5f} s")
    return t, amp, phase

t_rit,  amp_rit,  ph_rit  = get_phase_info(mode_rit,  "RIT")
t_sxs,  amp_sxs,  ph_sxs  = get_phase_info(mode_sxs,  "SXS")
t_maya, amp_maya, ph_maya = get_phase_info(mode_maya, "MAYA")
print()

# ── Phase drift over last few cycles ─────────────────────────────────────────
print("=== Phase drift 0.1 s before merger ===")
def phase_drift(t, phase, t_window=-0.1):
    if t is None:
        return None
    idx_start = np.argmin(np.abs(t - t_window))
    idx_end   = np.argmin(np.abs(t - 0.0))
    return phase[idx_end] - phase[idx_start]

for label, t, ph in [("RIT", t_rit, ph_rit), ("SXS", t_sxs, ph_sxs), ("MAYA", t_maya, ph_maya)]:
    drift = phase_drift(t, ph)
    if drift is not None:
        print(f"  [{label}]  phi[-0.1s..0] = {drift:.4f} rad ({np.degrees(drift):.2f} deg)")
print()

# ── Figures ───────────────────────────────────────────────────────────────────
print("=== Generating plots ===")
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)
ax_amp, ax_norm, ax_phase = axes

colors = {"RIT": "C0", "SXS": "C1", "MAYA": "C2"}
lss    = {"RIT": "-",  "SXS": "--",  "MAYA": "-."}

for label, t, amp, ph in [
    ("RIT",  t_rit,  amp_rit,  ph_rit),
    ("SXS",  t_sxs,  amp_sxs,  ph_sxs),
    ("MAYA", t_maya, amp_maya, ph_maya),
]:
    if t is None:
        continue
    c, ls = colors[label], lss[label]
    idx0 = np.argmin(np.abs(t))
    ax_amp.plot(t, amp, color=c, ls=ls, lw=1.0, label=label)
    ax_norm.plot(t, amp / amp.max(), color=c, ls=ls, lw=1.0, label=label)
    ax_phase.plot(t, ph - ph[idx0], color=c, ls=ls, lw=1.0, label=label)

for ax in axes:
    ax.set_xlim(-0.5, 0.05)
    ax.axvline(0, color="k", ls=":", lw=0.7, alpha=0.6)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

ax_amp.set_ylabel(r"$|h_{22}|$ (strain)", fontsize=11)
ax_amp.set_title(f"(2,2) mode amplitude  [M={TOTAL_MASS} M☉, d={DISTANCE} Mpc]", fontsize=12)
ax_norm.set_ylabel(r"$|h_{22}|$ / peak", fontsize=11)
ax_norm.set_title("Normalized amplitude shape", fontsize=12)
ax_phase.set_ylabel(r"$\phi_{22} - \phi_{22}(t=0)$ [rad]", fontsize=11)
ax_phase.set_title("Phase aligned at t=0 (merger)", fontsize=12)
ax_phase.set_xlabel("Time [s]", fontsize=11)

plt.suptitle("q=1, non-spinning BBH: SXS vs RIT vs MAYA  (2,2) mode", fontsize=13, y=1.01)
plt.tight_layout()

import os
outfile = "/home/prayush/src/nr-catalog-tools/notebooks/figs/compare_q1_nospin.png"
os.makedirs(os.path.dirname(outfile), exist_ok=True)
plt.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"  Saved: {outfile}")
plt.close()

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"{'Catalog':6}  {'Sim':32}  {'Raw peak':10}  {'Phys peak':12}  {'len':7}")
for label, sim, wfm, mode in [
    ("RIT",  rit_sims[0]  if rit_sims  else None, wfm_rit,  mode_rit),
    ("SXS",  sxs_sims[0]  if sxs_sims  else None, wfm_sxs,  mode_sxs),
    ("MAYA", maya_sims[0] if maya_sims else None, wfm_maya, mode_maya),
]:
    if wfm is not None and mode is not None:
        raw = wfm.get_mode_data(ELL, EM)
        rp = np.sqrt(raw[:,1]**2 + raw[:,2]**2).max()
        pp = np.abs(mode.data).max()
        print(f"{label:6}  {str(sim):32}  {rp:10.5f}  {pp:12.4e}  {len(mode):7d}")
    else:
        print(f"{label:6}  {'N/A':32}  {'---':10}  {'---':12}  {'---':7}")
