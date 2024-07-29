#!/bin/env python

from nrcatalogtools.maya import MayaCatalog

gc = MayaCatalog.load(verbosity=5, download=True)

for s in gc.simulations:
    gc.download_waveform_data(s)
