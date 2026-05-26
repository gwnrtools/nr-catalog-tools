# `nrcatalogtools.catalog`

Abstract base classes for NR waveform catalogs.

`CatalogABC` defines the pure interface (filename/filepath/URL/download contract) that
every catalog back-end must implement.  `CatalogBase` provides the shared `get()`,
`get_metadata()`, `get_parameters()`, and `set_attribute_in_waveform_data_file()`
implementations that all three catalog back-ends inherit.

---

::: nrcatalogtools.catalog.CatalogABC
    options:
      members:
        - waveform_filename_from_simname
        - waveform_filepath_from_simname
        - waveform_url_from_simname
        - download_waveform_data
        - psi4_filename_from_simname
        - psi4_filepath_from_simname
        - psi4_url_from_simname
        - download_psi4_data
        - metadata_filename_from_simname
        - metadata_filepath_from_simname
        - metadata_url_from_simname

---

::: nrcatalogtools.catalog.CatalogBase
    options:
      members:
        - simulations_list
        - get
        - get_metadata
        - get_parameters
        - set_attribute_in_waveform_data_file
