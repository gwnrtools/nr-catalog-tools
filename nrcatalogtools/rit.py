import os
import glob
import functools
from tqdm import tqdm
import pandas as pd
import requests

from . import (catalog, utils)


class RITCatalog(catalog.CatalogBase):
    def __init__(self,
                 catalog=None,
                 helper=None,
                 verbosity=0,
                 **kwargs) -> None:
        if catalog is not None:
            super().__init__(catalog)
        else:
            type(self).load(verbosity=verbosity, **kwargs)
        self._helper = helper
        self._verbosity = verbosity
        self._dict["catalog_file_description"] = "scraped from website"
        self._dict["modified"] = {}
        self._dict["records"] = {}

    @classmethod
    @functools.lru_cache()
    def load(cls,
             download=None,
             num_sims_to_crawl=2000,
             acceptable_scraping_fraction=0.7,
             verbosity=0):
        helper = RITCatalogHelper(use_cache=True, verbosity=verbosity)
        if verbosity > 2:
            print("..Going to read catalog file from cache.")
        catalog_df = helper.read_metadata_df_from_disk()
        if len(catalog_df) == 0:
            if verbosity > 2:
                print(
                    "..Catalog file not found on disk. Going to refresh from cache."
                )
            catalog_df = helper.refresh_metadata_df_on_disk(
                num_sims_to_crawl=num_sims_to_crawl)
        elif len(
                catalog_df) < acceptable_scraping_fraction * num_sims_to_crawl:
            if verbosity > 2:
                print(
                    """..Catalog file on disk is likely incomplete with only {} sims.
                    ...Going to refresh from cache.
                    """.format(len(catalog_df)))
            catalog_df = helper.refresh_metadata_df_on_disk(
                num_sims_to_crawl=num_sims_to_crawl)

        if len(catalog_df) < acceptable_scraping_fraction * num_sims_to_crawl:
            if verbosity > 2:
                print("""..Refreshing catalog file from cache did not work.
...Falling back to downloading metadata for the full 
...catalog. This will take some time.
                    """)
            if download:
                catalog_df = helper.fetch_metadata_for_catalog(
                    num_sims_to_crawl=num_sims_to_crawl)
            else:
                raise ValueError(
                    "Catalog not found in {}. Please set `download=True`".
                    format(helper.metadata_dir))
        # Get the catalog from helper object
        catalog = {}
        simulations = {}
        for idx, row in catalog_df.iterrows():
            name = row['simulation_name']
            metadata_dict = row.to_dict()
            simulations[name] = metadata_dict
        catalog["simulations"] = simulations
        return cls(catalog=catalog, helper=helper, verbosity=verbosity)

    @property
    @functools.lru_cache()
    def simulations_dataframe(self):
        df = self._helper.metadata
        for col_name in list(df.columns):
            if 'Unnamed' in col_name:
                df = df.drop(columns=[col_name])
                break
        self._helper.metadata = df
        df = df.set_index('simulation_name')
        df.index.names = [None]
        df['simulation_name'] = df.index.to_list()
        return df

    @property
    @functools.lru_cache()
    def files(self):
        """Map of all file names to the corresponding file info"""
        file_infos = {}
        for _, row in self.simulations_dataframe.iterrows():
            waveform_data_location = row['waveform_data_location']
            path_str = os.path.basename(waveform_data_location)
            if os.path.exists(waveform_data_location):
                file_size = os.path.getsize(waveform_data_location)
            else:
                file_size = 0
            file_info = {
                "checksum": None,
                "filename": os.path.basename(waveform_data_location),
                "filesize": file_size,
                "download": row['waveform_data_link']
            }
            file_infos[path_str] = file_info

        import collections
        unique_files = collections.defaultdict(list)
        for k, v in file_infos.items():
            unique_files[f"{v['checksum']}{v['filesize']}"].append(k)

        original_paths = {k: min(v) for k, v in unique_files.items()}

        for v in file_infos.values():
            v["truepath"] = original_paths[f"{v['checksum']}{v['filesize']}"]

        return file_infos

    def waveform_filename_from_simname(self, sim_name):
        return self._helper.waveform_filename_from_simname(sim_name)

    def waveform_filepath_from_simname(self, sim_name):
        return self._helper.waveform_data_dir / self.waveform_filename_from_simname(
            sim_name)

    def waveform_url_from_simname(self, sim_name):
        return self._helper.waveform_data_url + "/" + self.waveform_filename_from_simname(
            sim_name)

    def download_waveform_data(self, sim_name, use_cache=None):
        raise self._helper.download_waveform_data(sim_name,
                                                  use_cache=use_cache)


class RITCatalogHelper(object):
    def __init__(self, catalog=None, use_cache=True, verbosity=0) -> None:
        self.verbosity = verbosity
        self.catalog_url = utils.rit_catalog_info["url"]
        self.use_cache = use_cache
        self.cache_dir = utils.rit_catalog_info["cache_dir"]

        self.num_of_sims = 0

        self.metadata = pd.DataFrame.from_dict({})
        self.metadata_url = utils.rit_catalog_info["metadata_url"]
        self.metadata_file_fmts = utils.rit_catalog_info["metadata_file_fmts"]
        self.metadata_dir = utils.rit_catalog_info["metadata_dir"]

        self.waveform_data = {}
        self.waveform_data_url = utils.rit_catalog_info["data_url"]
        self.waveform_file_fmts = utils.rit_catalog_info["waveform_file_fmts"]
        self.data_dir = utils.rit_catalog_info["data_dir"]
        self.waveform_data_dir = utils.rit_catalog_info["data_dir"]

        self.possible_res = utils.rit_catalog_info["possible_resolutions"]
        self.max_id_val = utils.rit_catalog_info["max_id_val"]

        internal_dirs = [
            self.cache_dir, self.metadata_dir, self.waveform_data_dir
        ]
        for d in internal_dirs:
            d.mkdir(parents=True, exist_ok=True)

    def sim_info_from_metadata_filename(self, file_name):
        '''
        Input:
        ------
        file_name: name (not path) of metadata file as hosted on the web
        
        Output:
        -------
        - simulation number
        - resolution as indicated with an integer
        - ID value (only for non-eccentric simulations)
        '''
        sim_number = int(file_name.split('-')[0][-4:])
        res_number = int(file_name.split('-')[1][1:])
        try:
            id_val = int(file_name.split('-')[2].split('_')[0][2:])
        except:
            id_val = -1
        return (sim_number, res_number, id_val)

    def simname_from_metadata_filename(self, filename):
        '''
        Input:
        ------
        - filename: name (not path) of metadata file as hosted on the web
        
        Output:
        -------
        - Simulation Name Tag (Class uses this tag for internal indexing)
        '''
        return filename.split('_Meta')[0]

    def metadata_filename_from_simname(self, sim_name):
        '''
        We assume the sim names are either of the format:
        (1) RIT:eBBH:1109-n100-ecc
        (2) RIT:BBH:1109-n100-id1
        '''
        txt = sim_name.split(':')[-1]
        idx = int(txt[:4])
        res = int(txt.split('-')[1][1:])
        if 'eBBH' not in sim_name:
            # If this works, its a quasicircular sim
            id_val = int(txt[-1])
            return self.metadata_file_fmts[0].format(idx, res, id_val)
        else:
            return self.metadata_file_fmts[1].format(idx, res)

    def metadata_filename_from_cache(self, idx):
        possible_sim_tags = self.simtags(idx)
        for sim_tag in possible_sim_tags:
            mf = self.metadata_dir / sim_tag
            poss_files = glob.glob(str(mf) + "*")
            if len(poss_files) == 0:
                if self.verbosity > 4:
                    print(
                        "...found no files matching {}".format(str(mf) + "*"))
                continue
            file_name = poss_files[0]
        return file_name

    def waveform_filename_from_simname(self, sim_name):
        '''
        ExtrapStrain_RIT-BBH-0005-n100.h5 --> 
        ExtrapStrain_RIT-eBBH-1843-n100.h5
        RIT:eBBH:1843-n100-ecc_Metadata.txt
        '''
        txt = sim_name.split(':')[-1]
        idx = int(txt[:4])
        res = int(txt.split('-')[1][1:])
        try:
            # If this works, its a quasicircular sim
            id_val = int(txt[-1])
            mf = self.metadata_file_fmts[0].format(idx, res, id_val)
        except:
            mf = self.metadata_file_fmts[1].format(idx, res)
        parts = mf.split(':')
        return "ExtrapStrain_" + parts[0] + "-" + parts[1] + "-" + parts[
            2].split('_')[0].split('-')[0] + "-" + parts[2].split(
                '_')[0].split('-')[1] + '.h5'

    def waveform_filename_from_cache(self, idx):
        mf = self.metadata_filename_from_cache(idx)
        sim_name = self.simname_from_metadata_filename(mf)
        return self.waveform_filename_from_simname(sim_name)

    def metadata_filenames(self, idx, res, id_val):
        return [
            self.metadata_file_fmts[0].format(idx, res, id_val),
            self.metadata_file_fmts[1].format(idx, res)
        ]

    def simname_from_cache(self, idx):
        possible_sim_tags = self.simtags(idx)
        for sim_tag in possible_sim_tags:
            mf = self.metadata_dir / sim_tag
            poss_files = glob.glob(str(mf) + "*")
            if len(poss_files) == 0:
                if self.verbosity > 4:
                    print(
                        "...found no files matching {}".format(str(mf) + "*"))
                continue
            file_path = poss_files[0]  # glob gives full paths
            file_name = os.path.basename(file_path)
            return self.simname_from_metadata_filename(file_name)
        return ''

    def simnames(self, idx, res, id_val):
        return [
            self.simname_from_metadata_filename(mf)
            for mf in self.metadata_filenames(idx, res, id_val)
        ]

    def simtags(self, idx):
        return [
            self.metadata_file_fmts[0].split('-')[0].format(idx),
            self.metadata_file_fmts[1].split('-')[0].format(idx)
        ]

    def parse_metadata_txt(self, raw):
        next = [s for s in raw if len(s) > 0 and s[0].isalpha()]
        opts = {}
        for s in next:
            kv = s.split('=')
            try:
                opts[kv[0].strip()] = float(kv[1].strip())
            except:
                opts[kv[0].strip()] = str(kv[1].strip())
        return next, opts

    def metadata_from_link(self, link, save_to=None):
        if save_to is not None:
            utils.download_file(link, save_to, progress=True)
            return self.metadata_from_file(save_to)
        else:
            requests.packages.urllib3.disable_warnings()
            for n in range(100):
                try:
                    response = requests.get(link, verify=False)
                    break
                except:
                    continue
            return self.parse_metadata_txt(
                response.content.decode().split('\n'))

    def metadata_from_file(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        return self.parse_metadata_txt(lines)

    def metadata_from_cache(self, idx):
        possible_sim_tags = self.simtags(idx)
        for sim_tag in possible_sim_tags:
            mf = self.metadata_dir / sim_tag
            poss_files = glob.glob(str(mf) + "*")
            if len(poss_files) == 0:
                if self.verbosity > 4:
                    print(
                        "...found no files matching {}".format(str(mf) + "*"))
                continue
            file_path = poss_files[0]  # glob gives full paths
            file_name = os.path.basename(file_path)
            file_path_web = self.metadata_url + '/' + file_name
            wf_file_name = self.waveform_filename_from_cache(idx)
            wf_file_path_web = self.waveform_data_url + '/' + wf_file_name
            _, metadata_dict = self.metadata_from_file(file_path)
            if len(metadata_dict) > 0:
                metadata_dict['simulation_name'] = [
                    self.simname_from_metadata_filename(file_name)
                ]
                metadata_dict['metadata_link'] = [file_path_web]
                metadata_dict['metadata_location'] = [file_path]
                metadata_dict['waveform_data_link'] = [wf_file_path_web]
                metadata_dict['waveform_data_location'] = [
                    str(self.waveform_data_dir /
                        self.waveform_filename_from_simname(
                            metadata_dict['simulation_name'][0]))
                ]
                return pd.DataFrame.from_dict(metadata_dict)
        return pd.DataFrame({})

    def fetch_metadata(self, idx, res, id_val=-1):
        import pandas as pd
        possible_file_names = [
            self.metadata_file_fmts[0].format(idx, res, id_val),
            self.metadata_file_fmts[1].format(idx, res)
        ]
        metadata_txt, metadata_dict = "", {}

        for file_name in possible_file_names:
            if self.verbosity > 2:
                print("...beginning search for {}".format(file_name))
            file_path_web = self.metadata_url + '/' + file_name
            mf = self.metadata_dir / file_name

            if self.use_cache:
                if os.path.exists(mf) and os.path.getsize(mf) > 0:
                    if self.verbosity > 2:
                        print("...reading from cache: {}".format(str(mf)))
                    metadata_txt, metadata_dict = self.metadata_from_file(mf)

            if len(metadata_dict) == 0:
                if utils.url_exists(file_path_web):
                    if self.verbosity > 2:
                        print("...found {}".format(file_path_web))
                    metadata_txt, metadata_dict = self.metadata_from_link(
                        file_path_web, save_to=mf)
                else:
                    if self.verbosity > 3:
                        print("...tried and failed to find {}".format(
                            file_path_web))

            if len(metadata_dict) > 0:
                # Convert to DataFrame and break loop
                metadata_dict['simulation_name'] = [
                    self.simname_from_metadata_filename(file_name)
                ]
                metadata_dict['metadata_link'] = [file_path_web]
                metadata_dict['metadata_location'] = [mf]
                metadata_dict['waveform_data_location'] = [
                    str(self.waveform_data_dir /
                        self.waveform_filename_from_simname(
                            metadata_dict['simulation_name'][0]))
                ]
                break

        sim = pd.DataFrame.from_dict(metadata_dict)
        return sim

    def fetch_metadata_for_catalog(self,
                                   num_sims_to_crawl=2000,
                                   possible_res=[],
                                   max_id_in_name=-1):
        '''
        We crawl the webdirectory where RIT metadata usually lives,
        and try to read metadata for as many simulations as we can
        '''
        if len(possible_res) == 0:
            possible_res = self.possible_res
        if max_id_in_name <= 0:
            max_id_in_name = self.max_id_val
        import pandas as pd
        sims = pd.DataFrame({})

        if self.use_cache:
            metadata_df_fpath = self.metadata_dir / "metadata.csv"
            if os.path.exists(metadata_df_fpath
                              ) and os.path.getsize(metadata_df_fpath) > 0:
                print("Opening file {}".format(metadata_df_fpath))
                self.metadata = pd.read_csv(metadata_df_fpath)
                if len(self.metadata) >= (num_sims_to_crawl - 1):
                    # return self.metadata
                    return self.metadata.iloc[:num_sims_to_crawl - 1]
                else:
                    sims = self.metadata
        if self.verbosity > 2:
            print("Found metadata for {} sims".format(len(sims)))

        for idx in tqdm(range(1, 1 + num_sims_to_crawl)):
            found = False
            possible_sim_tags = self.simtags(idx)

            if self.verbosity > 3:
                print("\nHunting for sim with idx: {}".format(idx))

            # First, check if metadata present as file on disk
            if not found and self.use_cache:
                if self.verbosity > 3:
                    print("checking for metadata file on disk")
                sim_data = self.metadata_from_cache(idx)
                if len(sim_data) > 0:
                    found = True
                    if self.verbosity > 3:
                        print("...metadata found on disk for {}".format(idx))

            # Second, check if metadata present already in DataFrame
            if len(sims) > 0 and not found:
                print("Checking existing dataframe")
                for _, row in sims.iterrows():
                    name = row['simulation_name']
                    for sim_tag in possible_sim_tags:
                        if sim_tag in name:
                            found = True
                            f_idx, res, id_val = self.sim_info_from_metadata_filename(
                                name)
                            assert f_idx == idx, """Index found for sim from metadata is not the same as we were searching for ({} vs {}).""".format(
                                f_idx, idx)
                            if self.verbosity > 3:
                                print("...metadata found in DF for {}, {}, {}".
                                      format(idx, res, id_val))
                            sim_data = pd.DataFrame.from_dict(row.to_dict(),
                                                              index=[0])
                            break

            # If not already present, fetch metadata the hard way
            if not found:
                for res in possible_res:
                    for id_val in range(max_id_in_name):
                        # If not already present, fetch metadata
                        sim_data = self.fetch_metadata(idx, res, id_val)
                        if len(sim_data) > 0:
                            found = True
                            if self.verbosity > 3:
                                print(
                                    "...metadata txt file found for {}, {}, {}"
                                    .format(idx, res, id_val))
                            break
                        else:
                            if self.verbosity > 3:
                                print("...metadata not found for {}, {}, {}".
                                      format(idx, res, id_val))
                    # just need to find one resolution, so exit loop if its been found
                    if found:
                        break
            if found:
                sims = pd.concat([sims, sim_data])
            else:
                if self.verbosity > 3:
                    print("...metadata for {} NOT FOUND.".format(
                        possible_sim_tags))

            self.metadata = sims
            if self.use_cache:
                self.write_metadata_df_to_disk()

        self.num_of_sims = len(sims)
        return self.metadata

    def write_metadata_df_to_disk(self):
        metadata_df_fpath = self.metadata_dir / "metadata.csv"
        with open(metadata_df_fpath, "w+") as f:
            try:
                self.metadata.to_csv(f)
            except:
                self.metadata.reset_index(drop=True, inplace=True)
                self.metadata.to_csv(f)

    def refresh_metadata_df_on_disk(self, num_sims_to_crawl=2000):
        sims = []
        for idx in tqdm(range(1, 1 + num_sims_to_crawl)):
            sim_data = self.metadata_from_cache(idx)
            if len(sims) == 0:
                sims = sim_data
            else:
                sims = pd.concat([sims, sim_data])
        sims.reset_index(drop=True, inplace=True)
        metadata_df_fpath = self.metadata_dir / "metadata.csv"
        with open(metadata_df_fpath, "w") as f:
            sims.to_csv(f)
        self.metadata = sims  # set this member
        return self.metadata

    def read_metadata_df_from_disk(self):
        metadata_df_fpath = self.metadata_dir / "metadata.csv"
        if os.path.exists(
                metadata_df_fpath) and os.path.getsize(metadata_df_fpath) > 0:
            self.metadata = pd.read_csv(metadata_df_fpath)
        else:
            self.metadata = pd.DataFrame([])
        return self.metadata

    def download_waveform_data(self, sim_name, use_cache=None):
        '''
        Possible file formats:
        (1) https://ccrgpages.rit.edu/~RITCatalog/Data/ExtrapStrain_RIT-BBH-0193-n100.h5
        (2) https://ccrgpages.rit.edu/~RITCatalog/Data/ExtrapStrain_RIT-eBBH-1911-n100.h5
        '''
        if use_cache is None:
            use_cache = self.use_cache
        file_name = self.waveform_filename_from_simname(sim_name)
        file_path_web = self.waveform_data_url + "/" + file_name
        local_file_path = self.waveform_data_dir / file_name
        if use_cache and os.path.exists(
                local_file_path) and os.path.getsize(local_file_path) > 0:
            if self.verbosity > 2:
                print("...can read from cache: {}".format(
                    str(local_file_path)))
            pass
        elif os.path.exists(
                local_file_path) and os.path.getsize(local_file_path) > 0:
            pass
        else:
            if self.verbosity > 2:
                print("...writing to cache: {}".format(str(local_file_path)))
            if utils.url_exists(file_path_web):
                if self.verbosity > 2:
                    print("...downloading {}".format(file_path_web))
                # wget.download(str(file_path_web), str(local_file_path))
                import subprocess
                subprocess.call([
                    'wget', '--no-check-certificate',
                    str(file_path_web), '-O',
                    str(local_file_path)
                ])
            else:
                if self.verbosity > 2:
                    print("... ... but couldnt find link: {}".format(
                        str(file_path_web)))

    def fetch_waveform_data_from_cache(self, idx):
        wf = self.waveform_filename_from_cache(idx)
        wf_local_path = self.waveform_data_dir / wf
        raise NotImplementedError()

    def download_waveform_data_for_catalog(self,
                                           num_sims_to_crawl=100,
                                           possible_res=[],
                                           max_id_in_name=-1,
                                           use_cache=None):
        '''
        We crawl the webdirectory where RIT waveform data usually lives,
        and try to read waveform data for as many simulations as we can
        '''
        if len(possible_res) == 0:
            possible_res = self.possible_res
        if max_id_in_name <= 0:
            max_id_in_name = self.max_id_val
        if use_cache is None:
            use_cache = self.use_cache

        try:
            x = os.popen('/bin/ls {}/*.txt | wc -l'.format(
                str(self.metadata_dir)))
            num_metadata_txt_files = int(x.read().strip())
            x = os.popen('/bin/cat {}/metadata.csv | wc -l'.format(
                str(self.metadata_dir)))
            num_metadata_df = int(x.read().strip())
        except:
            # dummy values to force refresh below
            num_metadata_txt_files, num_metadata_df = 10, 0

        if num_metadata_df - 1 < num_metadata_txt_files:
            metadata = self.refresh_metadata_df_on_disk()
        else:
            metadata = self.read_metadata_df_from_disk()
        sims = {}

        for idx, sim_name in tqdm(enumerate(metadata['simulation_name'])):
            if idx + 1 > num_sims_to_crawl:
                break
            file_name = self.waveform_filename_from_simname(sim_name)
            local_file_path = self.waveform_data_dir / file_name
            self.download_waveform_data(sim_name, use_cache=use_cache)
            sims[sim_name] = local_file_path

        return sims
