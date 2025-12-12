#!/bin/env python
"""
Script: download_rit.py

Description:
------------
This script is designed to download waveform data for all simulations in the RIT NR catalog
using the `nrcatalogtools` Python library.

Workflow:
---------
1. Loads the RIT NR catalog with high verbosity and ensures that the catalog metadata is downloaded.
2. Downloads all available waveform data files for the simulations in the catalog.

Intended Use:
-------------
- Run this script in an environment where `nrcatalogtools` is installed and configured.
- The downloaded waveform files can be used for further numerical relativity analysis.

Notes:
------
- The download is performed at high verbosity (`verbosity=5`) for detailed progress and troubleshooting.
- The catalog handler can be configured to download different data products;
  by default, this script downloads the "waveform" data (not "psi4").
"""

import nrcatalogtools as nrcat

print(
    f"""Data will be downloaded in {nrcat.utils.rit_catalog_info['data_dir']}.
    Change `nrcat.utils.rit_catalog_info['data_dir']` to alter location"""
)

rc = nrcat.RITCatalog.load(verbosity=5, download=True)
rc.download_data_for_catalog(which_data="waveform")  # psi4
