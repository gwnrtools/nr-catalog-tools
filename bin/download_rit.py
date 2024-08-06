#!/bin/env python

from nrcatalogtools.rit import RITCatalog

rc = RITCatalog.load(verbosity=5, download=True)

for s in rc.simulations:
    rc.download_waveform_data(s)
