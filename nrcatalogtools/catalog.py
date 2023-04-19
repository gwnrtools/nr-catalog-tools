from abc import (ABC, abstractmethod)


class CatalogABC(ABC):
    @abstractmethod
    def waveform_filename_from_simname(self, sim_name):
        raise NotImplementedError()

    @abstractmethod
    def waveform_filepath_from_simname(self, sim_name):
        raise NotImplementedError()


import os
import sxs
from . import waveform


class CatalogBase(CatalogABC, sxs.Catalog):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get(self, sim_name):
        if sim_name not in self.simulations_dataframe[
                'simulation_name'].to_list():
            raise IOError(f"Simulation {sim_name} not found in catalog."
                          f"Please check that it exists")
        filepath = self.waveform_filepath_from_simname(sim_name)
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            if self._verbosity > 1:
                print(f"..As data does not exist in cache:"
                      f"  (in {filepath}),\n"
                      f"..we will now download it from"
                      " {}".format(self.waveform_url_from_simname(sim_name)))
            self.download_waveform_data(sim_name)
        metadata = self.get_metadata(sim_name)
        if type(metadata) is not dict and hasattr(metadata, "to_dict"):
            metadata = metadata.to_dict()
        return waveform.WaveformModes.load_from_h5(filepath, metadata=metadata)

    def get_metadata(self, sim_name):
        df = self.simulations_dataframe
        if sim_name not in df['simulation_name'].to_list():
            raise IOError(f"Simulation {sim_name} not found in catalog."
                          f"Please check that it exists")
        return df.loc[sim_name]
