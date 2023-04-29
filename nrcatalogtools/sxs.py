import sxs
from . import catalog, waveform


class SXSCatalog(catalog.CatalogBase):
    def __init__(self, catalog=None, verbosity=0, **kwargs) -> None:
        super().__init__(catalog, **kwargs)
        self._verbosity = verbosity

    def waveform_filename_from_simname(self, sim_name):
        raise NotImplementedError("COMING SOON!")

    def waveform_filepath_from_simname(self, sim_name):
        raise NotImplementedError("COMING SOON!")

    def get(self, sim_name, extrapolation_order=2):
        extrap_key = f"Extrapolated_N{extrapolation_order}.dir"
        raw_obj = sxs.load(f"{sim_name}/Lev/rhOverM")
        raw_obj = raw_obj.get(extrap_key)
        return waveform.WaveformModes(raw_obj.data, **raw_obj._metadata)

    def get_metadata(self, sim_name):
        return sxs.load(f"{sim_name}/Lev/metadata.json")
