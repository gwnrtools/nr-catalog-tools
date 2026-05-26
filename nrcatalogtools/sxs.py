"""SXS catalog interface.

Wraps the ``sxs`` package (≥ 2025.0.0) to provide access to the
Simulating eXtreme Spacetimes (SXS) catalog of numerical-relativity BBH
waveforms stored on Zenodo.

Key design decisions
--------------------
* **Singleton pattern** – ``load()`` stores its result in ``_sxs_catalog_singleton``
  so that repeated calls don't re-parse the ~2000-entry catalog JSON.  Pass
  ``download=True`` or call ``reload()`` to force a fresh download.

* **Lazy path resolution** – individual simulation file paths are resolved on
  demand via ``waveform_filepath_from_simname()`` rather than upfront (which
  would require ~2000 ``sxs.load()`` calls at catalog-load time).  The
  ``_add_paths_to_metadata()`` helper seeds stub empty strings so downstream
  code that checks for key presence never sees a ``KeyError``.

* **Auto-supersede** – ``get()`` passes ``auto_supersede=True`` to
  ``sxs.load()`` so deprecated simulation IDs are resolved transparently.

Public classes
--------------
SXSCatalog
    Registered under the tag ``"SXS"`` in the catalog plugin registry.
"""

from __future__ import annotations

import os
from pathlib import Path
import sxs
from nrcatalogtools import catalog, waveform
from nrcatalogtools.registry import register_catalog

# Module-level singleton — same stale-result fix as RITCatalog.
_sxs_catalog_singleton = None


@register_catalog("SXS")
class SXSCatalog(catalog.CatalogBase):
    CATALOG_TYPE = "SXS"

    def __init__(
        self, catalog: dict | None = None, verbosity: int = 0, **kwargs
    ) -> None:
        if catalog is not None:
            super().__init__(catalog, **kwargs)
        else:
            obj = type(self).load(verbosity=verbosity, **kwargs)
            super().__init__(obj._dict)
        self._verbosity = verbosity
        self._add_paths_to_metadata()

    @classmethod
    def load(
        cls, download: bool | None = None, verbosity: int = 0, **kwargs
    ) -> SXSCatalog:
        """Load the SXS catalog.

        This is a wrapper around ``sxs.load``.  The result is cached in a
        module-level singleton; pass ``download=True`` or call
        ``SXSCatalog.reload()`` to force a fresh download.

        Parameters
        ----------
        download : {None, bool}, optional
            If False, this function will look for the catalog in the cache and
            raise an error if it is not found.  If True, this function will
            download the catalog and raise an error if the download fails.
            If None (the default), it will try to download the file, warn but
            fall back to the cache if that fails, and only raise an error if
            the catalog is not found in the cache.
        """
        global _sxs_catalog_singleton
        if _sxs_catalog_singleton is not None and download is not True:
            return _sxs_catalog_singleton

        sxs_catalog = sxs.load("catalog", download=download, **kwargs)
        # sxs.Catalog is not subscriptable; pass the underlying dict
        _sxs_catalog_singleton = cls(catalog=sxs_catalog._dict, verbosity=verbosity)
        return _sxs_catalog_singleton

    @classmethod
    def reload(cls, **kwargs) -> SXSCatalog:
        """Force a fresh download and replace the cached singleton.

        Equivalent to ``SXSCatalog.load(download=True, **kwargs)``.
        """
        global _sxs_catalog_singleton
        _sxs_catalog_singleton = None
        return cls.load(download=True, **kwargs)

    def waveform_filename_from_simname(self, sim_name):
        """Return the bare filename for the strain waveform file.

        Resolves the path on demand by calling ``sxs.load(sim_name,
        download=False)`` and extracting ``sim.strain_path[0]``.

        Args:
            sim_name (str): SXS simulation name, e.g. ``"SXS:BBH:0001"``.

        Returns:
            str: Filename component only (no directory).
        """
        return os.path.basename(self.waveform_filepath_from_simname(sim_name))

    def waveform_filepath_from_simname(self, sim_name):
        """Return the absolute local path for the strain waveform file.

        Path resolution is lazy: this calls ``sxs.load(sim_name,
        download=False)`` each time it is invoked.  The file may not yet
        exist locally if it has never been downloaded.

        Args:
            sim_name (str): SXS simulation name, e.g. ``"SXS:BBH:0001"``.

        Returns:
            str: POSIX-style absolute path string.
        """
        sim = sxs.load(sim_name, download=False)
        # The strain_path property returns a tuple (file_name, group)
        waveform_filename = sim.strain_path[0]
        file_path = Path(sim.__file__) / waveform_filename
        if not os.path.exists(file_path):
            if self._verbosity > 2:
                print(
                    f"WARNING: Could not resolve path for {sim_name}"
                    f"..best calculated path = {file_path}"
                )
        return file_path.as_posix()

    def metadata_filename_from_simname(self, sim_name):
        """Return the bare filename for the per-simulation metadata file.

        Args:
            sim_name (str): SXS simulation name, e.g. ``"SXS:BBH:0001"``.

        Returns:
            str: Filename component only (no directory).
        """
        return os.path.basename(self.metadata_filepath_from_simname(sim_name))

    def metadata_filepath_from_simname(self, sim_name):
        """Return the absolute local path for the per-simulation metadata file.

        Args:
            sim_name (str): SXS simulation name, e.g. ``"SXS:BBH:0001"``.

        Returns:
            str: POSIX-style absolute path string.
        """
        sim = sxs.load(sim_name, download=False)
        metadata_filename = sim.metadata_path
        file_path = Path(sim.__file__) / metadata_filename
        if not os.path.exists(file_path):
            if self._verbosity > 2:
                print(
                    f"WARNING: Could not resolve path for {sim_name}"
                    f"..best calculated path = {file_path}"
                )
        return file_path.as_posix()

    def psi4_filename_from_simname(self, sim_name):
        """Return the bare filename for the psi4 data file.

        Args:
            sim_name (str): SXS simulation name, e.g. ``"SXS:BBH:0001"``.

        Returns:
            str: Filename component only (no directory).
        """
        return os.path.basename(self.psi4_filepath_from_simname(sim_name))

    def psi4_filepath_from_simname(self, sim_name):
        """Return the absolute local path for the psi4 data file.

        Args:
            sim_name (str): SXS simulation name, e.g. ``"SXS:BBH:0001"``.

        Returns:
            str: POSIX-style absolute path string.
        """
        sim = sxs.load(sim_name, download=False)
        # The psi4_path property returns a tuple (file_name, group)
        psi4_filename = sim.psi4_path[0]
        file_path = Path(sim.__file__) / psi4_filename
        if not os.path.exists(file_path):
            if self._verbosity > 2:
                print(
                    f"WARNING: Could not resolve path for {sim_name}"
                    f"..best calculated path = {file_path}"
                )
        return file_path.as_posix()

    def psi4_url_from_simname(self, sim_name):
        """Return the remote download URL for the psi4 data file.

        Args:
            sim_name (str): SXS simulation name, e.g. ``"SXS:BBH:0001"``.

        Returns:
            str: Full HTTP(S) URL from the SXS file manifest.
        """
        sim = sxs.load(sim_name, download=False)
        psi4_filename = sim.psi4_path[0]
        return sim.files.get(psi4_filename)["link"]

    def download_psi4_data(self, sim_name):
        """Download the psi4 data for *sim_name* via the ``sxs`` package.

        Accesses ``sim.psi4`` to trigger the download; the file is cached
        in the ``sxs`` package's own cache directory.

        Args:
            sim_name (str): SXS simulation name, e.g. ``"SXS:BBH:0001"``.
        """
        sim = sxs.load(sim_name, download=True)
        # Accessing the psi4 property will trigger the download
        _ = sim.psi4

    def get(
        self, sim_name: str, extrapolation_order: int = 2, download: bool = True
    ) -> waveform.WaveformModes:
        """Load the strain waveform for *sim_name* and return a WaveformModes object.

        Overrides ``CatalogBase.get()`` because SXS waveform loading goes
        entirely through the ``sxs`` package rather than via direct HDF5 reads.

        Args:
            sim_name (str): SXS simulation name, e.g. ``"SXS:BBH:0001"``.
                Deprecated IDs are resolved automatically via
                ``auto_supersede=True``.
            extrapolation_order (int): Waveform extrapolation order used as a
                fallback when ``sim_obj.strain`` is unavailable.  Defaults to 2.
            download (bool): Whether to download the waveform if not cached.
                Defaults to True.

        Returns:
            nrcatalogtools.waveform.WaveformModes: Waveform object with the
            catalog metadata attached.
        """
        # sxs >= 2024 uses a Simulation_v3 object; access .strain for WaveformModes.
        # auto_supersede=True resolves deprecated simulation IDs automatically.
        sim_obj = sxs.load(sim_name, download=download, auto_supersede=True)
        # Prefer the highest available extrapolation order
        try:
            raw_obj = sim_obj.strain
        except Exception:
            # Fallback for older sxs versions: load rhOverM directly
            raw_obj = sxs.load(
                f"{sim_name}/rhOverM",
                extrapolation_order=extrapolation_order,
                download=download,
            )

        # Get the sim metadata (from our catalog dict, keyed by sim_name)
        sim_metadata = self.get_metadata(sim_name)

        # Build a WaveformModes object; strip keys the parent class doesn't accept
        meta = dict(raw_obj._metadata)
        meta.update({"waveform_data_location": ""})
        meta.pop("metadata", None)  # avoid nested dict conflict
        meta.pop("time", None)
        return waveform.WaveformModes(
            raw_obj.data, raw_obj.time, sim_metadata=sim_metadata, **meta
        )

    def download_waveform_data(self, sim_name):
        """Download the strain waveform data for *sim_name* via ``sxs.load()``.

        First fetches the simulation metadata JSON, then downloads the
        ``rhOverM`` strain file.  Both are handled by the ``sxs`` package's
        download and caching machinery.

        Args:
            sim_name (str): SXS simulation name, e.g. ``"SXS:BBH:0001"``.

        Returns:
            sxs.WaveformModes: Raw waveform object returned by ``sxs.load()``.
        """
        _ = sxs.load(f"{sim_name}/metadata.json")
        return sxs.load(f"{sim_name}/rhOverM")

    def waveform_url_from_simname(self, sim_name):
        """Not implemented for SXS; downloads are managed by ``sxs.load()``.

        Args:
            sim_name (str): SXS simulation name.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("This shouldn't be called.")

    def metadata_url_from_simname(self, sim_name):
        """Not implemented for SXS; use ``sxs.load()`` instead.

        Args:
            sim_name (str): SXS simulation name.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Direct URL access not supported for SXS; use sxs.load()."
        )

    def _add_paths_to_metadata(self):
        """Seed per-simulation metadata with empty stub path/URL columns.

        Resolving actual file paths for every SXS simulation would require
        calling ``sxs.load(sim_name)`` ~2000 times at catalog-load time,
        which is prohibitively slow and would trigger unwanted downloads.
        Instead, this method sets empty string placeholders for the six
        path/link columns so that downstream code checking for key presence
        never sees a ``KeyError``.  Actual paths are resolved lazily in
        ``waveform_filepath_from_simname()`` and ``metadata_filepath_from_simname()``.
        """
        metadata_dict = self._dict["simulations"]
        if not metadata_dict:
            return
        existing_cols = list(next(iter(metadata_dict.values())).keys())
        stub_cols = {
            "metadata_link": "",
            "metadata_location": "",
            "waveform_data_link": "",
            "waveform_data_location": "",
            "psi4_data_link": "",
            "psi4_data_location": "",
        }
        missing = [c for c in stub_cols if c not in existing_cols]
        if missing:
            for sim_meta in metadata_dict.values():
                for col in missing:
                    sim_meta[col] = stub_cols[col]
