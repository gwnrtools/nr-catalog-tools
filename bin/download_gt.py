#!/bin/env python
"""
Script: download_gt.py

This script is designed to download gravitational waveform data for simulations
contained within the MAYA NR catalog. It attempts to retrieve and store waveform
files using two separate catalog toolkits: `mayawaves` and `nrcatalogtools`.

Workflow:
----------
1. The script initializes two catalog handlers:
   - `mayawavescatalog` using mayawaves' `Catalog` class.
   - `mayacatalog` using nrcatalogtools' `MayaCatalog`.

2. It uses `mayawaves` tools to bulk download as many simulation waveforms as possible
   to a user-defined cache directory with LVC NR formatting.

3. For any simulations whose data failed to download in the previous step, the script
   attempts to fetch the remaining missing waveform files using the `nrcatalogtools` library.

Intended Use:
-------------
- This script is typically run in a user environment where the `mayawaves` and
  `nrcatalogtools` Python libraries are available and properly configured.
- The resulting files can be used for gravitational waveform analysis.

Notes:
------
- Ensure that you have the appropriate permissions and storage in the target directory.
- Update the `save_wf_path` variable as needed for your system.
- Verbosity is set high for both catalogs for easier troubleshooting.
"""

import os
from mayawaves.utils.catalogutils import Catalog as MWCatalog
import nrcatalogtools as nrcat

print(
    f"""Data will be downloaded in {nrcat.utils.maya_catalog_info['data_dir']}.
    Change `nrcat.utils.maya_catalog_info['data_dir']` to alter location"""
)

mayacatalog = nrcat.MayaCatalog.load(verbosity=3)
mayawavescatalog = MWCatalog()

# First, attempt to download as many simulations as we can using tools
# within `mayawaves`
mayawavescatalog.download_waveforms(
    mayawavescatalog.simulations,
    save_wf_path=nrcat.utils.maya_catalog_info['data_dir'],
    lvcnr_format=True,
)

# Second, check for all simulations that failed to download
for f in mayawavescatalog.simulations:
    if not os.path.exists(os.path.join(mayacatalog.waveform_data_dir, f + ".h5")):
        print(f"\n >>> Did not find data for {f}")
        mayacatalog.download_waveform_data(f, maya_format=False)
