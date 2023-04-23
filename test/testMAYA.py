''' Test MAYA waveform load against LALSimulation '''



import numpy as np
import sys
import os
import h5py

# nr-catalog-tools
cwd = os.getcwd()

libpath = f'{cwd}/../'

if libpath not in sys.path:
    sys.path.append(libpath)

from nrcatalogtools.maya import MayaCatalog
from nrcatalogtools.utils import maya_catalog_info

#import matplotlib.pyplot as plt

# unittest funcs
from helper import *
import unittest

# pycbc
from pycbc.waveform import td_approximants
from pycbc.types.timeseries import TimeSeries
from pycbc.filter.matchedfilter import match
from pycbc.waveform.utils import coalign_waveforms
from pycbc import pnutils




######################################
# Simulation properties
######################################

# Simulation name
sim_name = 'GT0001'

# Parameters
M = 40
D = 1000
inc = np.pi/6
coa_phase = np.pi/4
delta_t = 1./2048

# Extrinsic parameters:
f_lower = 20
f_lower_at_1MSUN = f_lower/M


#######################################
# Fetch waveform using nr-catalog-tools
#######################################

#sc = sxs.Catalog.load(download=True)
#rc = RITCatalog.load(verbosity=5, download=True)
message('Loading waveform through nrcatalogtools...')
mc = MayaCatalog.load(verbosity=5)

mwf = mc.get(sim_name)

hpc = mwf.get_td_waveform(total_mass=M, distance=D, inclination=inc,
                    coa_phase=coa_phase, delta_t=delta_t
                    )
hpc_pycbc = hpc # mwf.to_pycbc(hpc)
hp1, hx1 = hpc_pycbc.real(), hpc_pycbc.imag()

#plt.plot(hp1.sample_times, hp1)
#plt.plot(hx1.sample_times, hx1)
#plt.grid()
#plt.show()



#########################################
# Fetch waveform using LALSuite
#########################################


apx = 'SEOBNRv4_ROM'
if apx not in td_approximants():
    raise AttributeError(f'Approximant {apx} not found! Please check `LAL_DATA_PATH`')








fdir = maya_catalog_info['cache_dir']
file = f'{fdir}/data/{sim_name}.h5'


#file = f'/home/vaishakprasad/{sim_name}.h5'
# Check by changing spin by small amount
#dspins2x = np.linspace(-0.1, 0.1, 1000)
try:
    f.close()
except:
    pass

f = h5py.File(file, 'a')
if 'f_lower_at_1MSUN' not in list(f.attrs.keys()):
    message('Attribute `f_lower_at_1MSUN` not in h5 file. Adding the attribute...' )
    f.attrs['f_lower_at_1MSUN'] = f_lower_at_1MSUN
    f.close()
else:
    f.close()

f = h5py.File(file, 'r')
#print(f.attrs.keys())
params = {}

params['f_lower'] = f_lower
params['mtotal'] = M#150.0
params['inclination'] = inc#0.0
params['distance'] = D#100.0

# Metadata parameters:
params['eta'] = f.attrs['eta']

params['mass1'] = pnutils.mtotal_eta_to_mass1_mass2(params['mtotal'], params['eta'])[0]
params['mass2'] = pnutils.mtotal_eta_to_mass1_mass2(params['mtotal'], params['eta'])[1]

# BH1 spins
params['spin1x'] = f.attrs['spin1x']
params['spin1y'] = f.attrs['spin1y']
params['spin1z'] = f.attrs['spin1z']


# BH2 spins
params['spin2x'] = f.attrs['spin2x']
params['spin2y'] = f.attrs['spin2y']
params['spin2z'] = f.attrs['spin2z']


# Spin unit vectors
params['nhat'] = [f.attrs['nhatx'], f.attrs['nhaty'], f.attrs['nhatz']]
params['lnhat'] = [f.attrs['LNhatx'], f.attrs['LNhaty'], f.attrs['LNhatz']]


# Check for coa_phase, else use the phase from nr cat load.
try:
    params['coa_phase'] = f.attrs['coa_phase']
except:
    message(f'Cannot find the attribute `coa_phase` in the file. Setting to {coa_phase}')
    #raise AttributeError('Cannot find the attribute `coa_phase` in the file')
    params['coa_phase']=coa_phase

# Transform spins to LAL frame from NR frame

# NR frame
s1 = [params['spin1x'], params['spin1y'], params['spin1z']]
s2 = [params['spin2x'], params['spin2y'], params['spin2z']]

# LAL frame
S1, S2 = TransformSpinsNRtoLAL(s1, s2, params['nhat'], params['lnhat'])

from pycbc.waveform import get_td_waveform


message('Loading waveform through LAL...')
hp2, hx2 = get_td_waveform(approximant='NR_hdf5',
                         numrel_data=file,
                         mass1=params['mass1'],
                         mass2=params['mass2'],
                         spin1x=S1[0],
                         spin1y=S1[1],
                         spin1z=S1[2],
                         spin2x=S2[0],
                         spin2y=S2[1],
                         spin2z=S2[2],
                         delta_t=delta_t,
                         f_lower=f_lower,
                         inclination=params['inclination'],
                         coa_phase=params['coa_phase'],
                         distance=params['distance'])

t = np.array(range( len(hp2) ) )*delta_t

#plt.plot(t, hp2, color=[0,0.7071,1])
#plt.plot(t, hx2, color=[0.1,0,0])
#plt.show()
f.close()


###################
# Initiate tests
##################



hp2_ts = TimeSeries(hp2, delta_t=delta_t)
hx2_ts = TimeSeries(hx2, delta_t=delta_t)

mp, sp = match(hp1, hp2_ts)
mx, sx = match(hx1, hx2_ts)

wf1_p, wf2_p = coalign_waveforms(hp1, hp2_ts)
wf1_x, wf2_x = coalign_waveforms(hx1, hx2_ts)

# Normalize the arrays
wf1 = np.array(wf1_p) + 1j*np.array(wf1_x)
wf2 = np.array(wf2_p) + 1j*np.array(wf2_x)

n1 = np.sqrt(np.dot(wf1, np.conjugate(wf1)))
n2 = np.sqrt(np.dot(wf2, np.conjugate(wf2)))

wf1 = wf1/n1
wf2 = wf2/n2

wf1_p = wf1.real
wf1_x = wf1.imag

wf2_p = wf2.real
wf2_x = wf2.imag

wf1_p = TimeSeries(wf1_p, delta_t)
wf1_x = TimeSeries(wf1_x, delta_t)
wf2_p = TimeSeries(wf2_p, delta_t)
wf2_x = TimeSeries(wf2_x, delta_t)



class TestMaya(unittest.TestCase):
    ''' Test loading of MAYA waveforms '''
    
    def test_waveforms(self):
        ''' Test the MAYA loading of waveforms against 
        that loading using lalsimulation. Tested are RMS errors, maximum deviation and mismatches'''
        
        # L2 errors
        Res_p, Amin_p, Amax_p = RMSerrs(np.array(wf1_p), np.array(wf2_p))
        Res_x, Amin_x, Amax_x = RMSerrs(np.array(wf1_x), np.array(wf2_x))
        
        #Amin_p/=A1max
        #Amin
        # Match
        match_p, shift_p = match(wf1_p, wf2_p)
        match_x, shift_x = match(wf1_x, wf2_x)

        mismatch_p = 100*(1-match_p)
        mismatch_x = 100*(1-match_x)
        
        prec = 1
        # RMS error should be less than 0.1 x Amax(wf1)
        self.assertAlmostEqual(Res_p, 0, prec, f"The RMS error between the + components of the waveforms must be almost 0")
        self.assertAlmostEqual(Res_x, 0, prec, f"The RMS error between the x components of the waveforms must be almost 0")
        
        prec = 0
        # Max relative point-wise deviation w.r.t Amax(wf1) should be less than 100%
        self.assertAlmostEqual(np.absolute(Amin_p), 0, prec, f"The maximum lower deviation between the + components of the waveforms must be almost 0")
        self.assertAlmostEqual(np.absolute(Amin_p), 0, prec, f"The maximum lower deviation between the x components of the waveforms must be almost 0")
        
        self.assertAlmostEqual(np.absolute(Amax_p), 0, prec, f"The maximum upper deviation between the + components of the waveforms must be almost 0")
        self.assertAlmostEqual(np.absolute(Amax_p), 0, prec, f"The maximum upper deviation between the x components of the waveforms must be almost 0")
        
        prec = 0
        # Mismatch should be less than 1%
        self.assertAlmostEqual(mismatch_p, 0, prec, f"The mismatch between the + components of the waveforms must be almost 0")
        self.assertAlmostEqual(mismatch_p, 0, prec, f"The mismatch between the x components of the waveforms must be almost 0")

        prec=1
        # Full array
        np.testing.assert_almost_equal(wf1, wf2, prec)
        
if __name__ == '__main__':
    unittest.main()
