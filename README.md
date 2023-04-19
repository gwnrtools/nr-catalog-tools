# Interface to Numerical Relativity Catalogs

The `nrcatalogtools` python package provides a unified
high-level interface to multiple catalogs of data products
from Numerical Relativity simulations of compact-object
binary mergers. At the moment of writing, different
research groups have separate formats of data and/or
tools to interface with them. This package will be a
convenience layer atop those for downstream research
applications. 

We currently support the following catalogs:
- [Simulating eXtreme Spacetimes Waveforms Catalog](https://data.black-holes.org/waveforms/catalog.html)
- [Georgia Tech Binary Black Hole Simulations](https://einstein.gatech.edu/catalog/)
- [RIT Waveform Catalog](https://ccrg.rit.edu/content/data/rit-waveform-catalog)

# Usage
```
>>> from nrcatalogtools import RITCatalog
>>> rcatalog = RITCatalog.load()
>>> print(rcatalog.simulations_dataframe.index)
Index(['RIT:BBH:0001-n100-id3', 'RIT:BBH:0002-n100-id0',
       'RIT:BBH:0003-n100-id0', 'RIT:BBH:0004-n100-id0',
       'RIT:BBH:0005-n100-id0', 'RIT:BBH:0006-n100-id3',
       'RIT:BBH:0007-n100-id0', 'RIT:BBH:0008-n100-id0',
       'RIT:BBH:0009-n100-id0', 'RIT:BBH:0010-n100-id0',
       ...
       'RIT:BBH:1914-n144-id1', 'RIT:BBH:1915-n144-id1',
       'RIT:BBH:1916-n100-id1', 'RIT:BBH:1917-n100-id1',
       'RIT:BBH:1918-n100-id1', 'RIT:BBH:1919-n100-id1',
       'RIT:BBH:1920-n100-id1', 'RIT:BBH:1921-n100-id1',
       'RIT:BBH:1922-n100-id1', 'RIT:BBH:1923-n100-id1'],
      dtype='object', length=1879)
```

Now, if one needs a particular simulation, they can do:
```
>>> rwf = rcatalog.get('RIT:BBH:0003-n100-id0')
```
To check which modes are available for this simulation:
```
>>> print(rwf.LM)
[[ 2 -2]
 [ 2 -1]
 [ 2  0]
 [ 2  1]
 [ 2  2]
 [ 3 -3]
 [ 3 -2]
 [ 3 -1]
 [ 3  0]
 [ 3  1]
 [ 3  2]
 [ 3  3]
 [ 4 -4]
 [ 4 -3]
 [ 4 -2]
 [ 4 -1]
 [ 4  0]
 [ 4  1]
 [ 4  2]
 [ 4  3]
 [ 4  4]]
```
To extract a single mode from this:
```
>>> rwf.get_mode(2, 2)
array([[-1.18175000e+03,  8.41055081e-02,  6.60652456e-04],
       [-1.18150000e+03,  8.41034759e-02, -7.94687302e-04],
       [-1.18125000e+03,  8.40763695e-02, -2.25019642e-03],
       ...,
       [ 3.61000000e+02,  2.56889323e-12, -3.97799029e-25],
       [ 3.61250000e+02,  1.30444912e-12, -1.64922275e-25],
       [ 3.61500000e+02,  0.00000000e+00,  0.00000000e+00]])
```
To get polarizations for the same simulation:
```
>>> pols = rwf.get_td_waveform(total_mass = 40, # solar masses
                               distance = 100., # Megaparsecs
                               inclination = 0.2, # radians
                               coa_phase = 0.3) # radians
>>> hp, hc = pols.real(), -1 * pols.imag()
```
which can subsequently be plotted easily:
```
>>> import matplotlib.pyplot as plt
>>> plt.plot(hp.sample_times, hp, label='+')
>>> plt.plot(hc.sample_times, hc, label='x')
>>> plt.legend()
>>> plt.show()
```
which should give the following figure:

![RIT-BBH-0003](https://github.com/gwnrtools/nr-catalog-tools/blob/master/test/validation_data/RIT-BBH-0003-n100-id0_m40_d100_inc0p2_coaph0p3.png)




