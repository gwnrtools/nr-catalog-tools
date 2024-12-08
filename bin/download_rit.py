#!/bin/env python

from nrcatalogtools.rit import RITCatalog

rc = RITCatalog.load(verbosity=5, download=True)

rc.download_data_for_catalog(which_data="psi4")
