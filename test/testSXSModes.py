""" Test SXS waveform load against waveformtools.

It was noted that the RAW modes data loaded from H5 files
were exactly equal. Howeverm after resampling to an axis,
there can be differences due to a differencein the 
time axis and interpolation. This script demonstrates the 
level of precision that can be expected from comparison
of waveforms that requires interpolation and resampling.
"""

import os
import sys

import h5py
import numpy as np

cwd = os.getcwd()
from pathlib import Path

home = Path.home()

libpath = f"{cwd}/../"

if libpath not in sys.path:
    sys.path.append(libpath)

import unittest
from pathlib import Path

import matplotlib.pyplot as plt
from nrcatalogtools.sxs import SXSCatalog
# from pycbc.waveform.utils import coalign_waveforms
from pycbc import pnutils
from pycbc.filter.matchedfilter import match
# pycbc
# from pycbc.waveform import td_approximants
from pycbc.types.timeseries import TimeSeries
# waveformtools
from waveformtools.waveforms import modes_array
from waveformtools.waveformtools import (interp_resam_wfs, message,
                                         xtract_camp_phase)

# unittest helper funcs
from helper import *

home = str(Path.home())

# import matplotlib.pyplot as plt

######################################
# Simulation properties
######################################

# Simulation name
sim_name = "SXS:BBH:0001"

message(f"Simulation {sim_name}")

######################################
# Waveform comparison function
######################################


def GetModesToCompare(ell, emm):
    """Get modes time-series to compare from
    two different methods.

    Parameters
    ----------
    ell, emm : int
                Mode numbers

    Returns
    -------
    waveforms : dict
                The waveform polarizations
    """

    ####################
    # Prepare modes
    ####################

    errs = {}

    aerrs = []
    perrs = []

    wf1_tlm = sxsw.get_mode(ell, emm)[:, 0] - mtime

    if not (wf1_tlm - taxis1 == 0).all():
        message('Difference in axis', wf1_tlm - taxis1)
        raise ValueError("Time axis is different across modes!")

    # nrcat
    wf1_plm = sxsw.get_mode(ell, emm)[:, 1]
    wf1_xlm = sxsw.get_mode(ell, emm)[:, 2]

    wf1_lm = wf1_plm + 1j * wf1_xlm
    wf2_lm = wf2.mode(ell, emm)

    # Interpolate in amp, phase
    wf1_Alm, wf1_Plm = xtract_camp_phase(wf1_lm.real, wf1_lm.imag)
    wf2_Alm, wf2_Plm = xtract_camp_phase(wf2_lm.real, wf2_lm.imag)

    wf1_rAlm = interp_resam_wfs(wf1_Alm, taxis1, taxis, resam_kind="cubic")
    wf2_rAlm = interp_resam_wfs(wf2_Alm, taxis2, taxis, resam_kind="cubic")

    wf1_rPlm = interp_resam_wfs(wf1_Plm, taxis1, taxis, resam_kind="cubic")
    wf2_rPlm = interp_resam_wfs(wf2_Plm, taxis2, taxis, resam_kind="cubic")

    # Get max locs
    # maxloc1 = np.argmax(wf1_rAlm)
    # maxloc2 = np.argmax(wf2_rAlm)

    # if maxloc1!=maxloc2:
    # message('Time axes are not centered! Recentering...')
    # from waveformtools.waveformtools import roll
    # wf1_rAlm = roll(wf1_rAlm, maxloc2-maxloc1)
    # wf1_rPlm = roll(wf1_rPlm, maxloc2-maxloc1)
    # raise ValueError('Time axes are not centered!')

    # Ensure increasing phase
    if np.mean(np.diff(wf1_rPlm)) < 0:
        wf1_rPlm = -wf1_rPlm

    if np.mean(np.diff(wf2_rPlm)) < 0:
        wf2_rPlm = -wf2_rPlm

    # return the waveforms
    wf1_rlm = wf1_rAlm * np.exp(1j * wf1_rPlm)
    wf2_rlm = wf2_rAlm * np.exp(1j * wf2_rPlm)

    wf1_rlm = wf1_rlm / np.linalg.norm(wf1_rlm)
    wf2_rlm = wf2_rlm / np.linalg.norm(wf2_rlm)

    wf1_rlm_p = TimeSeries(wf1_rlm.real, delta_t=delta_t)
    wf1_rlm_x = TimeSeries(wf1_rlm.imag, delta_t=delta_t)
    wf2_rlm_p = TimeSeries(wf2_rlm.real, delta_t=delta_t)
    wf2_rlm_x = TimeSeries(wf2_rlm.imag, delta_t=delta_t)

    # dphase = wf1_rPlm - wf2_rPlm

    # aerrs.append(res_amp)
    # perrs.append(res_phase)

    # errs.update({f'l{ell}m{emm}' : [res_amp, res_phase]})

    # fig, ax = plt.subplots()
    # ax.set_yscale('log')

    # ax.plot(taxis, wf1_rlm_p, label='nrcat')
    # ax.plot(taxis, wf2_rlm_p, label='wftools', linestyle='--')
    # ax.set_title(f'waveforms l{ell}m{emm}')
    # plt.grid()
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.set_yscale('log')

    # ax.plot(taxis, np.absolute(dAmp_frac))
    # ax.set_title(f'Amp diff l{ell}m{emm}')
    # plt.grid()
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.set_yscale('log')
    # ax.plot(taxis, np.absolute(dphase))
    # ax.set_title(f'Phase diff l{ell}m{emm}')
    # plt.grid()
    # plt.show()

    waveforms = {
        "wf1p": wf1_rlm_p,
        "wf1x": wf1_rlm_x,
        "wf2p": wf2_rlm_p,
        "wf2x": wf2_rlm_x,
    }

    return waveforms


#######################################################################

#################################
# Generate waveforms
#################################

#######################################
# Fetch waveform using nr-catalog-tools
#######################################

# sc = sxs.Catalog.load(download=True)
# rc = RITCatalog.load(verbosity=5, download=True)
message("Loading SXS waveform through nrcatalogtools...")
sxs1 = SXSCatalog.load(download=True)
# mc = MayaCatalog.load(verbosity=5)
# mwf = mc.get(sim_name)
sxsw = sxs1.get(sim_name)

######################################
# Load thru nrcat tools
######################################

wf1_t22 = sxsw.get_mode(2, 2)[:, 0]
wf1_p22 = sxsw.get_mode(2, 2)[:, 1]
wf1_x22 = sxsw.get_mode(2, 2)[:, 2]

wf1_22 = wf1_p22 + 1j * wf1_x22

# Find the maxloc
message("Finding Amax")
tfine = np.arange(wf1_t22[0], wf1_t22[-1], 0.001)

wf1_f22 = interp_resam_wfs(wf1_22, wf1_t22, tfine, resam_kind="cubic")

# Recenter the axis of td waveform about max amp
mloc = np.argmax(np.absolute(wf1_f22))
mtime = tfine[mloc]
message("Amax found at", mtime)

#############################
# Load via waveformtools
##############################
message("Loading SXS waveform through waveformtools...")
fdir = f"{home}/.cache/sxs/SXS:BBH:0001v3/Lev5/"
fname = "rhOverM_Asymptotic_GeometricUnits_CoM.h5"

wf2 = modes_array(label="sxs_001", spin_weight=-2)
wf2.file_name = fname
wf2.data_dir = fdir
_, wf2nl = wf2.load_modes(
    ftype="SpEC",
    var_type="strain",
    ell_max="auto",
    resam_type="auto",
    extrap_order=2,
    debug=True,
)
wf2.get_metadata()

wf2_22 = wf2.mode(2, 2)

##############################
# Construct common time axis
##############################

taxis1 = wf1_t22 - mtime
taxis2 = wf2.time_axis

t1 = max(taxis1[0], taxis2[0])
t2 = min(taxis1[-1], taxis2[-1])

taxis = np.arange(t1, t2, wf2.delta_t())
# ell_max = 2 #wf2.ell_max
delta_t = wf2.delta_t()
######################################################


class TestSXSModes(unittest.TestCase):
    """Test loading of SXS waveforms"""

    def test_modes(self):
        """Test the SXS loading of waveforms against
        that loading using waveformtools. Tested are RMS errors, maximum deviation and mismatches
        """

        modes = [(2, 2), (3, 3), (4, 2)]
        # modes = [(2, 2)]

        for item in modes:
            ell, emm = item

            message("\n--------------------------")
            message(f"Mode l{ell}m{emm}")
            message("--------------------------")

            waveforms = GetModesToCompare(ell, emm)

            wf1_p = waveforms["wf1p"]
            wf1_x = waveforms["wf1x"]
            wf2_p = waveforms["wf2p"]
            wf2_x = waveforms["wf2x"]

            wf1_nrcat = wf1_p + 1j * wf1_x
            wf2_wftools = wf2_p + 1j * wf2_x

            # L2 errors
            Res_p, Amin_p, Amax_p = RMSerrs(np.array(wf1_p), np.array(wf2_p))
            Res_x, Amin_x, Amax_x = RMSerrs(np.array(wf1_x), np.array(wf2_x))

            # Amin_p/=A1max
            # Amin
            # Match
            match_p, shift_p = match(wf1_p, wf2_p)
            match_x, shift_x = match(wf1_x, wf2_x)

            mismatch_p = 100 * (1 - match_p)
            mismatch_x = 100 * (1 - match_x)

            max_mismatch = max(mismatch_p, mismatch_x)

            message(f"Mismatch is {max_mismatch}")

            prec = 2
            # RMS error should be less than 0.01 x Amax(wf1)
            self.assertAlmostEqual(
                Res_p,
                0,
                prec,
                f"The RMS error between the + components of the waveforms must be atmost 1e-{prec} times Max amplitude of the normalized waveform",
            )
            self.assertAlmostEqual(
                Res_x,
                0,
                prec,
                f"The RMS error between the x components of the waveforms must be almost 1e-{prec} times Max amplitude of the normalized waveform",
            )

            prec = 0
            # Max relative point-wise deviation w.r.t Amax(wf1) should be less than 1 (100)%
            self.assertAlmostEqual(
                np.absolute(Amin_p),
                0,
                prec,
                f"The maximum lower deviation between the + components of the waveforms must be almost 1e-{prec}%",
            )
            self.assertAlmostEqual(
                np.absolute(Amax_x),
                0,
                prec,
                f"The maximum upper deviation between the x components of the waveforms must be almost 1e-{prec}%",
            )

            self.assertAlmostEqual(
                np.absolute(Amax_p),
                0,
                prec,
                f"The maximum upper deviation between the + components of the waveforms must be almost 1e-{prec}",
            )
            self.assertAlmostEqual(
                np.absolute(Amax_x),
                0,
                prec,
                f"The maximum upper deviation between the x components of the waveforms must be almost 1e-{prec}",
            )

            prec = 1
            # Mismatch should be less than 1e-3 or 1e-1 %
            self.assertAlmostEqual(
                mismatch_p,
                0,
                prec,
                f"The mismatch between the + components of the waveforms must be atmost 1e-{prec}%",
            )
            self.assertAlmostEqual(
                mismatch_x,
                0,
                prec,
                f"The mismatch between the x components of the waveforms must be atmost 1e-{prec}%",
            )

            # message(wf1_nrcat)
            prec = 2
            # Full arrays must agree pointwiswe
            np.testing.assert_almost_equal(
                np.array(wf1_nrcat),
                np.array(wf2_wftools),
                prec,
                f"The arrays must equal atleast upto {prec} decimals",
            )

    if __name__ == "__main__":
        unittest.main(argv=["first-arg-is-ignored"], exit=False, verbosity=3)
