import os
import sxs
from . import catalog, waveform


class SXSCatalog(catalog.CatalogBase):
    def __init__(self, catalog=None, verbosity=0, **kwargs) -> None:
        super().__init__(catalog, **kwargs)
        self._verbosity = verbosity
        self._add_paths_to_metadata()

    def waveform_filename_from_simname(self, sim_name):
        return os.path.basename(self.waveform_filepath_from_simname(sim_name))

    def waveform_filepath_from_simname(self, sim_name):
        poss_files = self.select_files(f"{sim_name}/Lev/rhOverM")
        file_path = (
            sxs.sxs_directory("cache")
            / poss_files[list(poss_files.keys())[0]]["truepath"]
        )
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
        poss_files = self.select_files(f"{sim_name}/Lev/metadata.json")
        file_path = (
            sxs.sxs_directory("cache")
            / poss_files[list(poss_files.keys())[0]]["truepath"]
        )
        if not os.path.exists(file_path):
            if self._verbosity > 2:
                print(
                    f"WARNING: Could not resolve path for {sim_name}"
                    f"..best calculated path = {file_path}"
                )
        return file_path.as_posix()

    def get(self, sim_name, extrapolation_order=2):
        extrap_key = f"Extrapolated_N{extrapolation_order}.dir"
        raw_obj = sxs.load(f"{sim_name}/Lev/rhOverM")
        raw_obj = raw_obj.get(extrap_key)
        return waveform.WaveformModes(raw_obj.data, **raw_obj._metadata)

    def download_waveform_data(self, sim_name):
        _ = sxs.load(f"{sim_name}/Lev/metadata.json")
        return sxs.load(f"{sim_name}/Lev/rhOverM")

    def waveform_url_from_simname(self, sim_name):
        raise NotImplementedError("This shouldn't be called.")

    def _add_paths_to_metadata(self):
        metadata_dict = self._dict["simulations"]
        existing_cols = list(metadata_dict[list(
            metadata_dict.keys())[0]].keys())
        new_cols = [
            'metadata_link', 'metadata_location', 'waveform_data_link',
            'waveform_data_location'
        ]

        if any([col not in existing_cols for col in new_cols]):
            for sim_name in metadata_dict:
                if 'metadata_location' not in existing_cols:
                    metadata_dict[sim_name][
                        'metadata_location'] = self.metadata_filepath_from_simname(
                            sim_name)
                if 'metadata_link' not in existing_cols:
                    metadata_dict[sim_name]['metadata_link'] = ""
                if 'waveform_data_link' not in existing_cols:
                    metadata_dict[sim_name]['waveform_data_link'] = ""
                if 'waveform_data_location' not in existing_cols:
                    metadata_dict[sim_name][
                        'waveform_data_location'] = self.waveform_filepath_from_simname(
                            sim_name)
