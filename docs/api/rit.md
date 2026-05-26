# `nrcatalogtools.rit`

RIT catalog interface.

`RITCatalog` exposes the standard `CatalogBase` interface.  All scraping, caching, and
file-naming logic is delegated to `RITCatalogHelper`, which can also be instantiated
independently for lower-level access.

---

## RITCatalog

::: nrcatalogtools.rit.RITCatalog
    options:
      members:
        - load
        - reload
        - simulations_dataframe
        - files
        - metadata_filename_from_simname
        - metadata_filepath_from_simname
        - metadata_url_from_simname
        - waveform_filename_from_simname
        - waveform_filepath_from_simname
        - waveform_url_from_simname
        - download_waveform_data
        - psi4_filename_from_simname
        - psi4_filepath_from_simname
        - psi4_url_from_simname
        - download_psi4_data
        - refresh_metadata_df_on_disk
        - download_data_for_catalog
        - write_metadata_df_to_disk

---

## RITCatalogHelper

::: nrcatalogtools.rit.RITCatalogHelper
    options:
      members:
        - metadata_filenames
        - metadata_filename_from_simname
        - metadata_filename_from_cache
        - psi4_filename_from_simname
        - psi4_filename_from_cache
        - waveform_filename_from_simname
        - waveform_filename_from_cache
        - simname_from_metadata_filename
        - simname_from_cache
        - simnames
        - simtags
        - parse_metadata_txt
        - metadata_from_link
        - metadata_from_file
        - metadata_from_cache
        - download_metadata
        - download_metadata_for_catalog
        - write_metadata_df_to_disk
        - refresh_metadata_df_on_disk
        - read_metadata_df_from_disk
        - download_waveform_data
        - download_psi4_data
        - download_data_for_catalog
        - fetch_waveform_data_from_cache
        - sim_info_from_metadata_filename
