# Copyright (C) 2023 Aditya Vijaykumar, Md Arif, Prayush Kumar
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline


class simulation():
    """Explore catalog of nr simulations.
    catalog: Name of the catalog. Example: "MAYA"
    catalog_metadata_dir: Path to directory where metadata of catalogs are
    stored in CSV format.
    metadata_files_dict: Dictionary of metadata file names.
    """
    def __init__(self,
                 catalog,
                 catalog_metadata_dir="../metadata/",
                 metadata_files_dict={
                     "MAYA": "catalog-table-maya.csv",
                     "RIT": "catalog-table-rit.csv"
                 },
                 data_dir=""):
        self.catalog = catalog
        self.data_dir = data_dir
        self.catalog_metadata_dir = catalog_metadata_dir
        if self.catalog not in ["MAYA", "RIT"]:
            raise Exception("Catalog must be one of ['MAYA', 'RIT']")
        self.metadata_file = os.path.join(self.catalog_metadata_dir,
                                          metadata_files_dict[self.catalog])
        if os.path.exists(self.metadata_file) is False:
            raise Exception(f"...metadata file {self.metadata_file} "
                            "does not exist!")
        self.catalog_tag_keys = {"MAYA": "GTID", "RIT": "catalog-tag"}
        self.mass_ratio_keys = {
            "MAYA": "q",
            "RIT": "relaxed-mass-ratio-1-over-2"
        }
        self.chi_keys = {"MAYA": "a", "RIT": "relaxed-chi"}
        self.freq_keys = {"MAYA": "Momega"}
        self.metadata = self.get_sim_dataframe()

    def get_sim_dataframe(self):
        return pd.read_csv(self.metadata_file, sep=",")

    def get_sim_ids(self):
        df = self.get_sim_dataframe()
        return df[self.catalog_tag_keys[self.catalog]].values

    def get_params_for_sim_id(self, sim_id):
        if self.catalog == "RIT":
            sim_id = self.catalog + ":" + sim_id
        if sim_id not in self.get_sim_ids():
            raise Exception(f"...simulation id {sim_id} is not found in the"
                            f" catalog metadata in {self.metadata_file}.")
        df = self.get_sim_dataframe()
        df_params = df.loc[df[self.catalog_tag_keys[self.catalog]] == sim_id]
        q = df_params[f"{self.mass_ratio_keys[self.catalog]}"].values[0]
        spin = self.chi_keys[self.catalog]
        components = ["x", "y", "z"]
        chi1 = [df_params[f"{spin}1{comp}"].values[0] for comp in components]
        chi2 = [df_params[f"{spin}2{comp}"].values[0] for comp in components]
        ref_omega = df_params[self.freq_keys[self.catalog]].values[0]
        params = [q, chi1, chi2, ref_omega]
        return params

    def get_sim_resolutions(self, sim_id):
        if self.catalog != "RIT":
            print("We only support resolution tag finding for RIT")
            return [None]
        return list(self.metadata[self.metadata['catalog-tag'] == sim_id]
                    ['resolution-tag'])

    def get_sim_datafile(self, sim_id, res_tag=None):
        """
    sim_id: str, simulation id. For RIT it is like BBH-0100. For MAYA, it is
    like GT0100.
    res_tag: str, required for RIT. Example n100. Default is None.

    returns:
    file_name: Path to HDF5 waveform file for given simulation
        """
        if self.catalog == "MAYA":
            file_name = os.path.join(self.data_dir, f"{sim_id}.h5")
        elif self.catalog == "RIT":
            if res_tag is None:
                raise Exception("For RIT catalog a res_tag is required.")
            file_regex = os.path.join(
                self.data_dir, "ExtrapStrain_RIT-{0}-{1}.h5".format(
                    '*' + sim_id.split(':')[-1], res_tag))
            file_name = glob.glob(file_regex, recursive=True)
            if len(file_name) > 0:
                file_name = file_name[0]
            else:
                raise Exception(
                    "DATA not found for {}\n ..using REGEX {}".format(
                        sim_id, file_regex))
        else:
            raise Exception("Catalog must be one of ['MAYA','RIT']")
        return os.path.abspath(file_name)

    def get_sim_params(self, sim_id):
        if self.catalog == "RIT" and self.catalog not in sim_id:
            sim_id = self.catalog + ":" + sim_id
        if sim_id not in self.get_sim_ids():
            raise Exception(f"...simulation id {sim_id} is not found in the"
                            f" catalog metadata in {self.metadata_file}.")
        df = self.metadata
        df_params = df.loc[df[self.catalog_tag_keys[self.catalog]] == sim_id]
        q = df_params[f"{self.mass_ratio_keys[self.catalog]}"].values[0]
        spin = self.chi_keys[self.catalog]
        components = ["x", "y", "z"]
        chi1 = [df_params[f"{spin}1{comp}"].values[0] for comp in components]
        chi2 = [df_params[f"{spin}2{comp}"].values[0] for comp in components]
        ref_omega = df_params[self.freq_keys[
            self.
            catalog]].values[0] if self.catalog in self.freq_keys else -1.0
        return {
            'q': q,
            'spin1x': chi1[0],
            'spin1y': chi1[1],
            'spin1z': chi1[2],
            'spin2x': chi2[0],
            'spin2y': chi2[1],
            'spin2z': chi2[2],
            'ref_omega': ref_omega
        }


class waveform():
    def __init__(self,
                 catalog,
                 sim_id,
                 res_tag=None,
                 delta_t=1,
                 modes=[[2, 2]],
                 M_fed=50,
                 dMpc=1,
                 theta=0,
                 phi=0,
                 data_dir="/home1/aditya.vijaykumar/nr-simulations/",
                 physical_units=False,
                 k=3):
        self.catalog = catalog
        self.sim_id = sim_id
        self.res_tag = res_tag
        self.delta_t = delta_t
        self.modes = modes
        self.M_fed = M_fed
        self.dMpc = dMpc
        self.theta = theta
        self.phi = phi
        self.physical_units = physical_units
        self.k = k
        self.data_dir = data_dir
        sim = simulation(self.catalog)
        self.q, self.chi1, self.chi2, self.omega_ref = sim.get_params_for_sim_id(
            self.sim_id)
        self.f_ref = self.omega_ref / np.pi

    def get_td_waveform(self):
        hp, hc = td_maya_or_rit(self.catalog, self.sim_id, self.res_tag,
                                self.modes, self.delta_t, self.M_fed,
                                self.dMpc, self.data_dir, self.physical_units,
                                self.k, self.theta, self.phi)
        self.times = hp.sample_times
        return hp, hc

    def get_td_waveform_surrogate(self, tmax=100):
        times = self.times[self.times < tmax]
        hp, hc = td_surrogate(times,
                              q=self.q,
                              chi1=self.chi1,
                              chi2=self.chi2,
                              modes=self.modes,
                              M_fed=self.M_fed,
                              dMpc=self.dMpc,
                              f_ref=self.f_ref,
                              physical_units=self.physical_units,
                              theta=self.theta,
                              phi=self.phi)
        return hp, hc

    def get_mismatch(self):
        print("To be implemented")

    def get_mode(self, mode=[2, 2]):
        times, hlm = mode_maya_or_rit(self.catalog, self.sim_id, self.res_tag,
                                      mode, self.delta_t, self.M_fed,
                                      self.dMpc, self.data_dir,
                                      self.physical_units, self.k)
        return times, hlm

    def get_mode_surrogate(self, mode=[2, 2], tmax=100):
        times = self.times[self.times <= tmax]
        times, hlm = mode_surrogate(times,
                                    q=self.q,
                                    chi1=self.chi1,
                                    chi2=self.chi2,
                                    mode=mode,
                                    M_fed=self.M_fed,
                                    dMpc=self.dMpc,
                                    f_ref=self.f_ref,
                                    physical_units=self.physical_units)
        return times, hlm

    def get_L2_norm(self):
        sum_num = 0
        norm = 0
        for mode in self.modes:
            times_maya, hlm_maya = self.get_mode(mode=mode)
            times_sur, hlm_sur = self.get_mode_surrogate(mode=mode)
            phase_maya = np.unwrap(np.angle(hlm_maya))
            phase_maya = phase_maya - phase_maya[0]
            phase_sur = np.unwrap(np.angle(hlm_sur))
            phase_sur = phase_sur - phase_sur[0]
            amp_maya = np.abs(hlm_maya)
            amp_maya_interp = InterpolatedUnivariateSpline(
                times_maya, amp_maya)
            phase_maya_interp = InterpolatedUnivariateSpline(
                times_maya, phase_maya)
            amp_sur = np.abs(hlm_sur)
            aligned_strain_maya = (amp_maya_interp(times_sur) *
                                   np.exp(1j * phase_maya_interp(times_sur)))
            aligned_strain_sur = amp_sur * np.exp(1j * phase_sur)

            diff = np.sum(
                np.abs(aligned_strain_maya - aligned_strain_sur)**2 *
                self.delta_t)
            sum_num += diff
            norm += np.sum(np.abs(aligned_strain_maya)**2 * self.delta_t)
        return 0.5 * (sum_num / norm)

    def plot(self, figsize=(12, 12)):
        hp, hc = self.get_td_waveform()
        hp_sur, hc_sur = self.get_td_waveform_surrogate()
        strain = hp - 1j * hc
        strain_sur = hp_sur - 1j * hc_sur
        amp = np.abs(strain)
        amp_sur = np.abs(strain_sur)
        phase = np.unwrap(np.angle(strain))
        phase_sur = np.unwrap(np.angle(strain_sur))

        fig, ax = plt.subplots(ncols=1, nrows=3, figsize=figsize)
        ax[0].plot(hp.sample_times,
                   amp,
                   label=rf"amp {self.catalog}:{self.sim_id}")
        ax[0].plot(hp_sur.sample_times,
                   amp_sur,
                   label=r"amp surrogate",
                   ls="--")
        ax[0].legend()

        ax[1].plot(hp.sample_times,
                   phase - phase[0],
                   label=f"phase {self.catalog}:{self.sim_id}")
        ax[1].plot(hp_sur.sample_times,
                   phase_sur - phase_sur[0],
                   label="phase surrogate",
                   ls="--")
        ax[1].legend()

        phase_maya = phase[hp.sample_times <= hp_sur.sample_times[-1]]
        phase_maya = phase_maya - phase_maya[0]
        phase_sur = phase_sur - phase_sur[0]
        ax[2].plot(hp_sur.sample_times,
                   phase_maya - phase_sur,
                   label=r"$\Delta \phi$")
        ax[2].legend()
        return fig, ax
