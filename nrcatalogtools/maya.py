import collections
import functools
import os
import zipfile
from mayawaves.coalescence import Coalescence
from mayawaves.utils.postprocessingutils import export_to_lvcnr_catalog
import pandas as pd
from nrcatalogtools import catalog, utils


class MayaCatalog(catalog.CatalogBase):
    def __init__(self, catalog=None, use_cache=True, verbosity=0, **kwargs) -> None:
        if catalog is not None:
            super().__init__(catalog)
        else:
            obj = type(self).load(verbosity=verbosity, **kwargs)
            super().__init__(obj._dict)
        self._verbosity = verbosity
        self._dict["catalog_file_description"] = "scraped from website"
        self._dict["modified"] = {}
        self._dict["records"] = {}

        # Other info
        self.num_of_sims = 0
        self.cache_dir = utils.maya_catalog_info["cache_dir"]
        self.use_cache = use_cache

        self.metadata = pd.DataFrame.from_dict(catalog)
        self.metadata_url = utils.maya_catalog_info["metadata_url"]
        self.metadata_dir = utils.maya_catalog_info["metadata_dir"]

        self.waveform_data = {}
        self.waveform_data_url = utils.maya_catalog_info["data_url"]
        self.waveform_data_dir = utils.maya_catalog_info["data_dir"]

        self._add_paths_to_metadata()

        internal_dirs = [self.cache_dir, self.metadata_dir, self.waveform_data_dir]
        for d in internal_dirs:
            d.mkdir(parents=True, exist_ok=True)

    def clear_cache(self):
        cache_path = utils.maya_catalog_info["cache_dir"] / "catalog.zip"
        if cache_path.exists():
            os.remove(cache_path)

    @classmethod
    @functools.lru_cache()
    def load(cls, download=None, verbosity=0):
        progress = True
        utils.maya_catalog_info["cache_dir"].mkdir(parents=True, exist_ok=True)
        metadata_url = utils.maya_catalog_info["metadata_url"]
        cache_path = utils.maya_catalog_info["cache_dir"] / "catalog.zip"
        if cache_path.exists():
            if_newer = cache_path
        else:
            if_newer = False

        if download or download is None:
            # 1. Download the full pickle file (zipped in flight, but auto-decompressed on arrival)
            # 2. Zip to a temporary file (using bzip2, which is better than the in-flight compression)
            # 3. Replace the original catalog.zip with the temporary zip file
            # 4. Remove the full pickle file
            # 5. Make sure the temporary zip file is gone too
            temp_pkl = cache_path.with_suffix(".temp.pkl")
            temp_zip = cache_path.with_suffix(".temp.zip")
            try:
                try:
                    utils.download_file(
                        metadata_url, temp_pkl, progress=progress, if_newer=if_newer
                    )
                except Exception as e:
                    if download:
                        raise RuntimeError(
                            f"Failed to download '{metadata_url}'; try setting `download=False`"
                        ) from e
                    download_failed = e  # We'll try the cache
                else:
                    download_failed = False
                    if temp_pkl.exists():
                        with zipfile.ZipFile(
                            temp_zip, "w", compression=zipfile.ZIP_BZIP2
                        ) as catalog_zip:
                            catalog_zip.write(temp_pkl, arcname="catalog.pkl")
                        temp_zip.replace(cache_path)
            finally:
                # The `missing_ok` argument to `unlink` would be much nicer, but was added in python 3.8
                try:
                    temp_pkl.unlink()
                except FileNotFoundError:
                    pass
                try:
                    temp_zip.unlink()
                except FileNotFoundError:
                    pass

        if not cache_path.exists():
            if download_failed:
                raise ValueError(
                    f"Catalog not found in '{cache_path}' and download failed"
                ) from download_failed
            elif (
                download is False
            ):  # Test if it literally *is* False, rather than just casts to False
                raise ValueError(
                    f"The catalog was not found in '{cache_path}', and downloading was turned off"
                )
            else:
                raise ValueError(
                    f"Catalog not found in '{cache_path}' for unknown reasons"
                )

        try:
            with zipfile.ZipFile(cache_path, "r") as catalog_zip:
                try:
                    with catalog_zip.open("catalog.pkl") as catalog_pickle:
                        try:
                            catalog_df = pd.read_pickle(catalog_pickle)
                        except Exception as e:
                            raise ValueError(
                                f"Failed to parse 'catalog.json' in '{cache_path}'"
                            ) from e
                except Exception as e:
                    raise ValueError(
                        f"Failed to open 'catalog.pkl' in '{cache_path}'"
                    ) from e
        except Exception as e:
            raise ValueError(f"Failed to open '{cache_path}' as a ZIP file") from e

        # Fill in the catalog object
        catalog_dict = {}
        catalog_dict["GTID"] = [s.strip() for s in list(catalog_df.index)]

        for col_name in catalog_df.columns:
            column = list(catalog_df[col_name])
            if "name" in col_name:
                catalog_dict["GT_Tag"] = [s.strip() for s in column]
            else:
                catalog_dict[col_name.strip()] = [
                    float(s.strip().replace("-", "NAN")) if type(s) is str else float(s)
                    for s in column
                ]
        catalog_df = pd.DataFrame(catalog_dict)
        catalog = {}
        simulations = {}
        for idx, row in catalog_df.iterrows():
            name = row["GTID"]
            metadata_dict = row.to_dict()
            simulations[name] = metadata_dict
        catalog["simulations"] = simulations
        return cls(catalog=catalog, verbosity=verbosity)

    def _add_paths_to_metadata(self):
        metadata_dict = self._dict["simulations"]
        existing_cols = list(metadata_dict[list(metadata_dict.keys())[0]].keys())
        new_cols = [
            "metadata_link",
            "metadata_location",
            "waveform_data_link",
            "waveform_data_location",
        ]

        if any([col not in existing_cols for col in new_cols]):
            for sim_name in metadata_dict:
                if "metadata_location" not in existing_cols:
                    metadata_dict[sim_name][
                        "metadata_location"
                    ] = self.metadata_filepath_from_simname(sim_name)
                if "metadata_link" not in existing_cols:
                    metadata_dict[sim_name]["metadata_link"] = self.metadata_url
                if "waveform_data_link" not in existing_cols:
                    metadata_dict[sim_name]["waveform_data_link"] = (
                        self.waveform_data_url + "/" + f"{sim_name}.h5"
                    )
                if "waveform_data_location" not in existing_cols:
                    metadata_dict[sim_name][
                        "waveform_data_location"
                    ] = self.waveform_filepath_from_simname(sim_name)

    @property
    @functools.lru_cache()
    def simulations_dataframe(self):
        df = pd.DataFrame(self.simulations).transpose()
        df.rename(columns={"GTID": "simulation_name"}, inplace=True)
        return df

    @property
    @functools.lru_cache()
    def files(self):
        """Map of all file names to the corresponding file info"""
        file_infos = {}
        for _, row in self.simulations_dataframe.iterrows():
            waveform_data_location = row["waveform_data_location"]
            path_str = os.path.basename(waveform_data_location)
            if os.path.exists(waveform_data_location):
                file_size = os.path.getsize(waveform_data_location)
            else:
                file_size = 0
            file_info = {
                "checksum": None,
                "filename": os.path.basename(waveform_data_location),
                "filesize": file_size,
                "download": row["waveform_data_link"],
            }
            file_infos[path_str] = file_info

        unique_files = collections.defaultdict(list)
        for k, v in file_infos.items():
            unique_files[f"{v['checksum']}{v['filesize']}"].append(k)

        original_paths = {k: min(v) for k, v in unique_files.items()}

        for v in file_infos.values():
            v["truepath"] = original_paths[f"{v['checksum']}{v['filesize']}"]

        return file_infos

    def waveform_filename_from_simname(self, sim_name):
        return sim_name + ".h5"

    def waveform_filepath_from_simname(self, sim_name):
        file_path = self.waveform_data_dir / self.waveform_filename_from_simname(
            sim_name
        )
        if not os.path.exists(file_path):
            if self._verbosity > 2:
                print(
                    f"WARNING: Could not resolve path for {sim_name}"
                    f"..best calculated path = {file_path}"
                )
        return file_path.as_posix()

    def waveform_url_from_simname(self, sim_name, maya_format=False):
        if maya_format:
            format = "maya_format"
        else:
            format = "lvcnr_format"
        return f"{self.waveform_data_url}/{format}/{self.waveform_filename_from_simname(sim_name)}"

    def metadata_filename_from_simname(self, sim_name):
        return os.path.basename(self.metadata_filepath_from_simname(sim_name))

    def metadata_filepath_from_simname(self, sim_name, ext="txt"):
        return str(self.metadata_dir / f"{sim_name}.{ext}")

    def download_waveform_data(self, sim_name, maya_format=True, use_cache=None):
        if use_cache is None:
            use_cache = self.use_cache
        file_name = self.waveform_filename_from_simname(sim_name)
        file_path_web = self.waveform_url_from_simname(
            sim_name, maya_format=maya_format
        )
        local_file_path = self.waveform_data_dir / file_name
        if (
            use_cache
            and os.path.exists(local_file_path)
            and os.path.getsize(local_file_path) > 0
        ):
            if self._verbosity > 2:
                print("...can read from cache: {}".format(str(local_file_path)))
            pass
        elif os.path.exists(local_file_path) and os.path.getsize(local_file_path) > 0:
            pass
        else:
            if self._verbosity > 2:
                print("...writing to cache: {}".format(str(local_file_path)))
            if utils.url_exists(file_path_web):
                if self._verbosity > 2:
                    print("...downloading {}".format(file_path_web))
                utils.download_file(file_path_web, local_file_path)
                if maya_format:
                    if self._verbosity > 2:
                        print("...exporting to LVCNR catalog format")
                    export_to_lvcnr_catalog(
                        Coalescence(local_file_path),
                        self.waveform_data_dir,
                        name=sim_name + "_LVCNR",
                        NR_group="UT Austin",
                        NR_code="MAYA",
                        bibtex_keys="Jani:2016wkt",
                        contact_email="email@email.com",
                        center_of_mass_correction=True,
                    )
                    if self._verbosity > 2:
                        print("...removing maya format file")
                    os.remove(local_file_path)
                    if self._verbosity > 2:
                        print("...renaming LVCNR format file in the cache")
                    os.rename(
                        self.waveform_data_dir / (sim_name + "_LVCNR.h5"),
                        local_file_path,
                    )
            else:
                if self._verbosity > 2:
                    print(
                        "... ... but couldnt find link: {}".format(str(file_path_web))
                    )
