import functools
import os
from pathlib import Path

import sxs
from nrcatalogtools import catalog, waveform


class SXSCatalog(catalog.CatalogBase):
    def __init__(self, catalog=None, verbosity=0, **kwargs) -> None:
        if catalog is not None:
            super().__init__(catalog, **kwargs)
        else:
            obj = type(self).load(verbosity=verbosity, **kwargs)
            super().__init__(obj)
        self._verbosity = verbosity
        self._add_paths_to_metadata()

    @classmethod
    @functools.lru_cache()
    def load(cls, download=None, verbosity=0, **kwargs):
        """Load the SXS catalog

        This is a wrapper around `sxs.load`.

        Parameters
        ----------
        download : {None, bool}, optional
            If False, this function will look for the catalog in the cache and
            raise an error if it is not found.  If True, this function will download
            the catalog and raise an error if the download fails.  If None (the
            default), it will try to download the file, warn but fall back to the cache
            if that fails, and only raise an error if the catalog is not found in the
            cache.
        """
        sxs_catalog = sxs.load("catalog", download=download, **kwargs)
        # sxs.Catalog is not subscriptable; pass the underlying dict
        return cls(catalog=sxs_catalog._dict, verbosity=verbosity)

    def waveform_filename_from_simname(self, sim_name):
        return os.path.basename(self.waveform_filepath_from_simname(sim_name))

    def waveform_filepath_from_simname(self, sim_name):
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
        return os.path.basename(self.metadata_filepath_from_simname(sim_name))

    def metadata_filepath_from_simname(self, sim_name):
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
        return os.path.basename(self.psi4_filepath_from_simname(sim_name))

    def psi4_filepath_from_simname(self, sim_name):
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
        sim = sxs.load(sim_name, download=False)
        psi4_filename = sim.psi4_path[0]
        return sim.files.get(psi4_filename)["link"]

    def download_psi4_data(self, sim_name):
        sim = sxs.load(sim_name, download=True)
        # Accessing the psi4 property will trigger the download
        _ = sim.psi4


    def get(self, sim_name, extrapolation_order=2, download=True):
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
        _ = sxs.load(f"{sim_name}/metadata.json")
        return sxs.load(f"{sim_name}/rhOverM")

    def waveform_url_from_simname(self, sim_name):
        raise NotImplementedError("This shouldn't be called.")

    def metadata_url_from_simname(self, sim_name):
        raise NotImplementedError("Direct URL access not supported for SXS; use sxs.load().")

    def _add_paths_to_metadata(self):
        # For the SXS catalog, resolving per-simulation paths requires calling
        # sxs.load(sim_name) for every simulation, which is prohibitively slow
        # (and triggers downloads) when iterating over the full catalog.
        # We leave path resolution lazy — paths are resolved on demand in
        # waveform_filepath_from_simname / metadata_filepath_from_simname.
        # Just set empty placeholder strings so downstream code that checks for
        # key presence doesn't break.
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
