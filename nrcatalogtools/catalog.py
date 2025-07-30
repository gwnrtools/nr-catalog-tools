import os
from abc import ABC, abstractmethod
from sxs import Catalog as sxs_Catalog
from nrcatalogtools import waveform
from nrcatalogtools import metadata as md


class CatalogABC(ABC):
    @abstractmethod
    def waveform_filename_from_simname(self, sim_name):
        raise NotImplementedError()

    @abstractmethod
    def waveform_filepath_from_simname(self, sim_name):
        raise NotImplementedError()

    @abstractmethod
    def waveform_url_from_simname(self, sim_name):
        raise NotImplementedError()

    @abstractmethod
    def download_waveform_data(self, sim_name):
        raise NotImplementedError()

    @abstractmethod
    def psi4_filename_from_simname(self, sim_name):
        raise NotImplementedError()

    @abstractmethod
    def psi4_filepath_from_simname(self, sim_name):
        raise NotImplementedError()

    @abstractmethod
    def psi4_url_from_simname(self, sim_name):
        raise NotImplementedError()

    @abstractmethod
    def download_psi4_data(self, sim_name):
        raise NotImplementedError()

    @abstractmethod
    def metadata_filename_from_simname(self, sim_name):
        raise NotImplementedError()

    @abstractmethod
    def metadata_filepath_from_simname(self, sim_name):
        raise NotImplementedError()

    @abstractmethod
    def metadata_url_from_simname(self, sim_name):
        raise NotImplementedError()


class CatalogBase(CatalogABC, sxs_Catalog):
    def __init__(self, *args, **kwargs) -> None:
        sxs_Catalog.__init__(self, *args, **kwargs)

    @property
    def simulations_list(self):
        return list(self.simulations)

    def get(self, sim_name, **kwargs):
        """Retrieve specific quantities for one simulation

        Args:
            sim_name (str): Name of simulation in catalog
            quantity (str): Name of quantity to fetch.
                            Options: {waveform, psi4}

        Raises:
            IOError: If `sim_name` not found in the catalog
            IOError: If `quantity` is not one of the options above

        Returns:
            nrcatalogtools.waveform.WaveformModes: Waveform modes
        """
        if sim_name not in self.simulations_dataframe.index.to_list():
            raise IOError(
                f"Simulation {sim_name} not found in catalog."
                f"Please check that it exists"
            )
        metadata = self.get_metadata(sim_name)
        if type(metadata) is not dict and hasattr(metadata, "to_dict"):
            metadata = metadata.to_dict()
        elif isinstance(metadata, dict):
            metadata = dict(metadata.items())

        if quantity.lower() == "waveform":
            filepath = self.waveform_filepath_from_simname(sim_name)
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                if self._verbosity > 1:
                    print(
                        f"..As data does not exist in cache:"
                        f"  (in {filepath}),\n"
                        f"..we will now download it from"
                        " {}".format(self.waveform_url_from_simname(sim_name))
                    )
                self.download_waveform_data(sim_name)
            return waveform.WaveformModes.load_from_h5(filepath, metadata=metadata)
        elif quantity.lower() == "psi4":
            filepath = self.psi4_filepath_from_simname(sim_name)
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                if self._verbosity > 1:
                    print(
                        f"..As data does not exist in cache:"
                        f"  (in {filepath}),\n"
                        f"..we will now download it from"
                        " {}".format(self.psi4_url_from_simname(sim_name))
                    )
                self.download_psi4_data(sim_name)
            try:
                return waveform.WaveformModes.load_from_h5(filepath, metadata=metadata)
            except OSError:
                return waveform.WaveformModes.load_from_targz(
                    filepath, metadata=metadata
                )
        else:
            raise IOError(
                f"Cannot provide quantity: {quantity}. Only supported options are [waveform, psi4]"
            )

    def get_metadata(self, sim_name):
        """Get Metadata for this simulation

        Args:
            sim_name (str): Name of simulation in catalog

        Raises:
            IOError: If `sim_name` is not found in the catalog

        Returns:
            `sxs.metadata.metadata.Metadata`: Metadata as dictionary
        """
        sim_dict = self.simulations
        if sim_name not in list(sim_dict.keys()):
            raise IOError(
                f"Simulation {sim_name} not found in catalog."
                f"Please check that it exists"
            )
        return sim_dict[sim_name]

    def set_attribute_in_waveform_data_file(self, sim_name, attr_name, attr_value):
        """Set attributes in the HDF5 file holding waveform data for a given
        simulation

        Args:
            sim_name (str): Name/Tag of the simulation
            attr_name (str): Name of the attribute to set
            attr_value (any/serializable): Value of the attribute
        """
        import h5py

        file_path = self.waveform_filepath_from_simname(sim_name)
        with h5py.File(file_path, "a") as fp:
            if attr_name not in fp.attrs:
                fp.attrs[attr_name] = attr_value

    def get_parameters(self, sim_name, total_mass=1.0):
        """Return the initial physical parameters for the simulation. Only for
        quasicircular simulations are supported, orbital eccentricity is ignored

        Args:
            total_mass (float, optional): Total Mass of Binary (solar masses).
                Defaults to 1.0.

        Returns:
            dict: Initial binary parameters with names compatible with PyCBC.
        """
        metadata = self.get_metadata(sim_name)
        return md.get_source_parameters_from_metadata(metadata, total_mass=total_mass)
