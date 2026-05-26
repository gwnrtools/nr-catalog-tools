"""Abstract base classes and shared implementation for NR waveform catalogs.

Classes
-------
CatalogABC
    Pure abstract interface that every catalog must implement.  Declares the
    filename/filepath/URL/download contract for waveform, psi4, and metadata
    data products.

CatalogBase
    Concrete base class that combines ``CatalogABC`` with a plain-dict
    simulation registry and provides the shared ``get()``, ``get_metadata()``,
    ``get_parameters()``, and ``set_attribute_in_waveform_data_file()``
    implementations used by all three catalog back-ends (RIT, SXS, MAYA).

    Subclasses must set ``CATALOG_TYPE`` (e.g. ``"RIT"``) and implement all
    abstract methods declared in ``CatalogABC``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import sxs

from nrcatalogtools import waveform
from nrcatalogtools import metadata as md


class CatalogABC(ABC):
    """Pure abstract interface that every catalog back-end must implement.

    Subclasses supply the catalog-specific logic for resolving file names,
    local paths, remote URLs, and triggering downloads for the three data
    products: *waveform strain* (HDF5), *psi4* (HDF5 or tar.gz), and
    *per-simulation metadata* (text or JSON).

    All methods take a ``sim_name`` string as their first positional
    argument.  The naming convention for ``sim_name`` is catalog-specific
    (e.g. ``"RIT:BBH:0001-n100-id3"``, ``"SXS:BBH:0001"``,
    ``"GT0001"``).
    """

    @abstractmethod
    def waveform_filename_from_simname(self, sim_name: str) -> str:
        """Return the bare filename (no directory) for the waveform HDF5 file.

        Args:
            sim_name (str): Simulation name tag.

        Returns:
            str: Filename, e.g. ``"ExtrapStrain_RIT-BBH-0001-n100.h5"``.
        """
        raise NotImplementedError()

    @abstractmethod
    def waveform_filepath_from_simname(self, sim_name: str) -> str:
        """Return the absolute local path for the waveform HDF5 file.

        The file may not yet exist on disk if it has not been downloaded.

        Args:
            sim_name (str): Simulation name tag.

        Returns:
            str: Absolute path under the catalog cache directory.
        """
        raise NotImplementedError()

    @abstractmethod
    def waveform_url_from_simname(self, sim_name: str) -> str:
        """Return the remote URL for the waveform HDF5 file.

        Args:
            sim_name (str): Simulation name tag.

        Returns:
            str: Full HTTP(S) URL.
        """
        raise NotImplementedError()

    @abstractmethod
    def download_waveform_data(self, sim_name: str) -> None:
        """Download the waveform data file for *sim_name* into the local cache.

        Args:
            sim_name (str): Simulation name tag.
        """
        raise NotImplementedError()

    @abstractmethod
    def psi4_filename_from_simname(self, sim_name: str) -> str:
        """Return the bare filename for the psi4 data file.

        Args:
            sim_name (str): Simulation name tag.

        Returns:
            str: Filename, e.g. ``"ExtrapPsi4_RIT-BBH-0001-n100-id3.tar.gz"``.

        Raises:
            NotImplementedError: If the catalog does not distribute psi4 data.
        """
        raise NotImplementedError()

    @abstractmethod
    def psi4_filepath_from_simname(self, sim_name: str) -> str:
        """Return the absolute local path for the psi4 data file.

        Args:
            sim_name (str): Simulation name tag.

        Returns:
            str: Absolute path under the catalog cache directory.

        Raises:
            NotImplementedError: If the catalog does not distribute psi4 data.
        """
        raise NotImplementedError()

    @abstractmethod
    def psi4_url_from_simname(self, sim_name: str) -> str:
        """Return the remote URL for the psi4 data file.

        Args:
            sim_name (str): Simulation name tag.

        Returns:
            str: Full HTTP(S) URL.

        Raises:
            NotImplementedError: If the catalog does not distribute psi4 data.
        """
        raise NotImplementedError()

    @abstractmethod
    def download_psi4_data(self, sim_name: str) -> None:
        """Download the psi4 data file for *sim_name* into the local cache.

        Args:
            sim_name (str): Simulation name tag.

        Raises:
            NotImplementedError: If the catalog does not distribute psi4 data.
        """
        raise NotImplementedError()

    @abstractmethod
    def metadata_filename_from_simname(self, sim_name: str) -> str:
        """Return the bare filename for the per-simulation metadata file.

        Args:
            sim_name (str): Simulation name tag.

        Returns:
            str: Filename, e.g. ``"RIT:BBH:0001-n100-id3_Metadata.txt"``.
        """
        raise NotImplementedError()

    @abstractmethod
    def metadata_filepath_from_simname(self, sim_name: str) -> str:
        """Return the absolute local path for the per-simulation metadata file.

        Args:
            sim_name (str): Simulation name tag.

        Returns:
            str: Absolute path under the catalog cache directory.
        """
        raise NotImplementedError()

    @abstractmethod
    def metadata_url_from_simname(self, sim_name: str) -> str:
        """Return the remote URL for the per-simulation metadata file.

        Args:
            sim_name (str): Simulation name tag.

        Returns:
            str: Full HTTP(S) URL.
        """
        raise NotImplementedError()


class CatalogBase(CatalogABC):
    """Shared implementation base for all NR catalog back-ends.

    Owns ``self._simulations: dict[str, dict]`` (simulation name →
    metadata dict) directly, with no dependency on ``sxs.Catalog``.
    Provides the default ``get()``, ``get_metadata()``,
    ``get_parameters()``, ``to_sxs()``, and
    ``set_attribute_in_waveform_data_file()`` implementations shared
    by ``RITCatalog``, ``SXSCatalog``, and ``MayaCatalog``.

    Subclasses must:
    - Set ``CATALOG_TYPE`` to ``"RIT"``, ``"SXS"``, or ``"MAYA"``.
    - Implement all ``@abstractmethod`` methods from ``CatalogABC``
      (filename, filepath, URL, and download helpers for waveform,
      psi4, and metadata products).
    - Override ``get()`` if catalog-specific download logic is needed
      (``SXSCatalog`` overrides because SXS data is managed by the
      ``sxs`` package, not by local HDF5 files).
    """

    # Subclasses set this to "RIT", "SXS", or "MAYA".  It is injected into
    # every metadata dict returned by get_metadata() so that downstream code
    # can dispatch on catalog_type without fragile sentinel-key detection.
    CATALOG_TYPE = None

    def __init__(self, simulations: dict, **kwargs) -> None:
        """Store the simulation metadata dict.

        Args:
            simulations (dict): Mapping of simulation name → metadata dict.
            **kwargs: Accepted for subclass compatibility; not forwarded.
        """
        self._simulations = simulations

    @property
    def simulations(self) -> dict:
        """Mapping of simulation name → metadata dict for all simulations.

        Returns:
            dict[str, dict]: The full simulation registry.
        """
        return self._simulations

    @property
    def simulations_list(self) -> list:
        """List of all simulation name tags in the catalog.

        Returns:
            list[str]: Simulation names in insertion order.
        """
        return list(self._simulations)

    def to_sxs(self) -> "sxs.Simulations":
        """Return an ``sxs.Simulations`` view of this catalog's metadata.

        For ``SXSCatalog``, the live ``sxs.Simulations`` object (fully
        populated with ``.dataframe``, ``.tag``, etc.) is returned via an
        override.  For RIT and MAYA catalogs this constructs an
        ``sxs.Simulations`` object from the local metadata dict; sxs-specific
        columns in ``.dataframe`` will be NaN because RIT/MAYA keys do not
        match the SXS schema.

        Returns:
            sxs.Simulations: An sxs-native catalog object.
        """
        import sxs

        return sxs.Simulations(self._simulations)

    def save(self, file: str) -> None:
        """Save this catalog's metadata to a JSON file.

        Args:
            file (str): Path to the output JSON file.
        """
        import json

        with open(file, "w") as f:
            json.dump(dict(self._simulations), f, indent=4)

    def get(
        self, sim_name: str, quantity: str = "waveform", **kwargs
    ) -> waveform.WaveformModes:
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
        if sim_name not in self._simulations:
            raise IOError(
                f"Simulation {sim_name} not found in catalog."
                f"Please check that it exists"
            )
        metadata = self.get_metadata(sim_name)
        if type(metadata) is not dict and hasattr(metadata, "to_dict"):
            metadata = metadata.to_dict()

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

    def get_metadata(self, sim_name: str) -> dict:
        """Get Metadata for this simulation

        Args:
            sim_name (str): Name of simulation in catalog

        Raises:
            IOError: If `sim_name` is not found in the catalog

        Returns:
            dict: Metadata dictionary.  Always contains a ``"catalog_type"``
            key (value: ``"RIT"``, ``"SXS"``, or ``"MAYA"``) so that
            downstream code can dispatch without fragile sentinel-key checks.
        """
        if sim_name not in self._simulations:
            raise IOError(
                f"Simulation {sim_name} not found in catalog."
                f"Please check that it exists"
            )
        metadata = self._simulations[sim_name]
        # Inject catalog_type once; idempotent on repeated calls.
        # sxs.Metadata is an OrderedDict subclass so __setitem__ works directly;
        # RIT and MAYA use plain dicts — no conversion needed in either case.
        if self.CATALOG_TYPE is not None:
            metadata["catalog_type"] = self.CATALOG_TYPE
        return metadata

    def set_attribute_in_waveform_data_file(
        self, sim_name: str, attr_name: str, attr_value: Any
    ) -> None:
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

    def get_parameters(self, sim_name: str, total_mass: float = 1.0) -> dict:
        """Return the initial physical parameters for the simulation. Only for
        quasicircular simulations are supported, orbital eccentricity is ignored

        Args:
            total_mass (float, optional): Total Mass of Binary (solar masses).
                Defaults to 1.0.

        Returns:
            dict: Initial binary parameters with names compatible with PyCBC.
        """
        metadata = self.get_metadata(sim_name)
        params = md.get_source_parameters_from_metadata(metadata, total_mass=total_mass)
        # If frequency metadata is absent or zero, compute f_lower from the
        # waveform data itself via f_lower_at_1Msun().
        if params.get("f_lower", -1) <= 0:
            try:
                wfm = self.get(sim_name, quantity="waveform")
                f1 = wfm.f_lower_at_1Msun()
                params["f_lower"] = f1 / total_mass
            except Exception:
                pass
        return params
