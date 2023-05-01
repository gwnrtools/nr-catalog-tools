import os
import sxs
from . import (catalog, waveform)


class SXSCatalog(catalog.CatalogBase):
    def __init__(self, catalog=None, verbosity=0, **kwargs) -> None:
        super().__init__(catalog, **kwargs)
        self._verbosity = verbosity

    def waveform_filename_from_simname(self, sim_name):
        return os.path.basename(self.waveform_filepath_from_simname(sim_name))

    def waveform_filepath_from_simname(self, sim_name):
        poss_files = self.select_files(f"{sim_name}/Lev/rhOverM")
        file_path = sxs.sxs_directory("cache") / poss_files[list(
            poss_files.keys())[0]]['truepath']
        if os.path.exists(file_path):
            return file_path.as_posix()
        raise RuntimeError(f"Could not resolve path for {sim_name}"
                           f"..best calculated path = {file_path}")

    def metadata_filename_from_simname(self, sim_name):
        return os.path.basename(self.metadata_filepath_from_simname(sim_name))

    def metadata_filepath_from_simname(self, sim_name):
        poss_files = self.select_files(f"{sim_name}/Lev/metadata.json")
        file_path = sxs.sxs_directory("cache") / poss_files[list(
            poss_files.keys())[0]]['truepath']
        if os.path.exists(file_path):
            return file_path.as_posix()
        raise RuntimeError(f"Could not resolve path for {sim_name}"
                           f"..best calculated path = {file_path}")

    def get(self, sim_name, extrapolation_order=2):
        extrap_key = f'Extrapolated_N{extrapolation_order}.dir'
        raw_obj = sxs.load(f"{sim_name}/Lev/rhOverM")
        raw_obj = raw_obj.get(extrap_key)
        return waveform.WaveformModes(raw_obj.data, **raw_obj._metadata)

    def get_metadata(self, sim_name):
        return sxs.load(f"{sim_name}/Lev/metadata.json")

    def download_waveform_data(self, sim_name):
        raise NotImplementedError("This shouldn't be called.")

    def waveform_url_from_simname(self, sim_name):
        raise NotImplementedError("This shouldn't be called.")
