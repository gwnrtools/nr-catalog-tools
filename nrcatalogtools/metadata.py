import numpy as np

import lal
from pycbc.pnutils import mtotal_eta_to_mass1_mass2

# ---------------------------------------------------------------------------
# Cross-catalog metadata key mappings
# ---------------------------------------------------------------------------
# Each dict maps a canonical physical quantity name to the corresponding
# metadata key in that catalog's raw dict.  A value of None means the
# quantity is not directly stored as a scalar key (e.g. it is a component
# of a vector, or must be derived).
#
# Canonical quantity names follow the pattern used in the "Physical quantity"
# column of docs/catalogs.md.
#
# Usage example — translate one simulation's metadata to another catalog's
# key convention:
#
#   from nrcatalogtools.metadata import RIT_KEYS, SXS_KEYS, MAYA_KEYS
#   q_rit  = metadata_rit[RIT_KEYS["mass_ratio"]]
#   q_sxs  = metadata_sxs[SXS_KEYS["mass_ratio"]]
#   q_maya = metadata_maya[MAYA_KEYS["mass_ratio"]]
#
# For quantities whose catalog key is None, see the docstring of each dict
# for the derivation.
# ---------------------------------------------------------------------------

RIT_KEYS = {
    # --- Identification ---
    "simulation_id": "catalog-tag",  # e.g. "RIT:BBH:0001"
    "resolution_tag": "resolution-tag",  # e.g. "n100"
    "id_tag": "id-tag",  # e.g. "id3" or "ecc"
    "run_name": "run-name",
    "spin_config": "system-type",  # "Aligned", "Precessing", "Nonspinning"
    # --- Masses (physical / relaxed epoch, code units M=1) ---
    "mass1": "relaxed-mass1",
    "mass2": "relaxed-mass2",
    "total_mass": "relaxed-total-mass",
    "mass_ratio": "relaxed-mass-ratio-1-over-2",  # m1/m2 >= 1
    "mass1_initial": "initial-mass1",
    "mass2_initial": "initial-mass2",
    "mass_final": "final-mass",
    # --- Spins (physical / relaxed epoch, dimensionless chi = S/m^2) ---
    "spin1x": "relaxed-chi1x",
    "spin1y": "relaxed-chi1y",
    "spin1z": "relaxed-chi1z",
    "spin2x": "relaxed-chi2x",
    "spin2y": "relaxed-chi2y",
    "spin2z": "relaxed-chi2z",
    "spin1x_initial": "initial-bh-chi1x",
    "spin1y_initial": "initial-bh-chi1y",
    "spin1z_initial": "initial-bh-chi1z",
    "spin2x_initial": "initial-bh-chi2x",
    "spin2y_initial": "initial-bh-chi2y",
    "spin2z_initial": "initial-bh-chi2z",
    "spin_final_magnitude": "final-chi",
    # --- Reference / relaxation epoch ---
    "reference_time": "relaxed-time",
    "relaxation_time": "relaxed-time",  # same key in RIT
    # --- Orbital dynamics ---
    "separation_initial": "initial-separation",
    "eccentricity": "eccentricity",
    "n_orbits": "number-of-orbits",
    "n_cycles_22": "number-of-cycles-22",
    # --- Frequencies (code units: M*Omega or M*f) ---
    "freq_start_22": "freq-start-22",  # M * f_GW^(2,2) at waveform start
    "freq_start_22_hz_1msun": "freq-start-22-Hz-1Msun",  # Hz at M_tot = 1 M_sun
    # orbital angular momentum direction at relaxed epoch
    "LNhat_x": "relaxed-LNhatx",
    "LNhat_y": "relaxed-LNhaty",
    "LNhat_z": "relaxed-LNhatz",
    # separation unit vector at relaxed epoch
    "nhat_x": "relaxed-nhatx",
    "nhat_y": "relaxed-nhaty",
    "nhat_z": "relaxed-nhatz",
    # --- Remnant / merger ---
    "peak_omega_22": "peak-omega-22",
    "peak_amplitude_22": "peak-ampl-22",
    "remnant_kick_kmps": "final-kick",  # km/s
    # --- ADM quantities ---
    "adm_energy_initial": "initial-ADM-energy",
    "adm_Lmag_initial": "initial-orbital-angular-momentum",
    # --- Numerical method ---
    "fd_order": "fd-order",
    "cfl": "cfl",
    "evolution_system": "evolution-system",
    "initial_data_type": "initial-data-type",
}
"""Mapping from canonical quantity names to RIT metadata keys (hyphenated).

In the `simulations_dataframe` these keys appear as-is (with hyphens).
`get_source_parameters_from_metadata()` accesses them with underscores after
`parse_metadata_txt()` converts hyphens to underscores during DataFrame
construction.
"""

SXS_KEYS = {
    # --- Identification ---
    "simulation_id": "alternative_names",  # e.g. "SXS:BBH:0001"
    "run_name": "simulation_name",  # internal directory path
    "object_types": "object_types",  # "BHBH", "NSNS", "BHNS"
    # --- Masses (reference epoch, code units) ---
    "mass1": "reference_mass1",
    "mass2": "reference_mass2",
    "mass_ratio": "reference_mass_ratio",  # m1/m2 >= 1
    "mass1_initial": "initial_mass1",
    "mass2_initial": "initial_mass2",
    "mass_final": "remnant_mass",
    # --- Spins (reference epoch, dimensionless chi = S/m^2, 3-vectors) ---
    # Note: SXS stores spins as 3-element lists; index [0]=x, [1]=y, [2]=z
    "spin1_vector": "reference_dimensionless_spin1",
    "spin2_vector": "reference_dimensionless_spin2",
    "spin1x": None,  # reference_dimensionless_spin1[0]
    "spin1y": None,  # reference_dimensionless_spin1[1]
    "spin1z": None,  # reference_dimensionless_spin1[2]
    "spin2x": None,  # reference_dimensionless_spin2[0]
    "spin2y": None,  # reference_dimensionless_spin2[1]
    "spin2z": None,  # reference_dimensionless_spin2[2]
    "spin1_magnitude": "reference_chi1_mag",
    "spin2_magnitude": "reference_chi2_mag",
    "chi_eff": "reference_chi_eff",
    "spin1_perp": "reference_chi1_perp",
    "spin2_perp": "reference_chi2_perp",
    "spin1_vector_initial": "initial_dimensionless_spin1",
    "spin2_vector_initial": "initial_dimensionless_spin2",
    "remnant_spin_vector": "remnant_dimensionless_spin",
    # --- Reference / relaxation epoch ---
    "reference_time": "reference_time",
    "relaxation_time": "relaxation_time",
    # --- Orbital dynamics ---
    # orbital_frequency is a 3-vector; magnitude = M*Omega_orb;
    # f_GW^(2,2) = |Omega| / pi  (in code units)
    "orbital_frequency_vector": "reference_orbital_frequency",
    "separation_initial": "initial_separation",
    "separation_reference": "reference_separation",
    "eccentricity": "reference_eccentricity",
    "mean_anomaly": "reference_mean_anomaly",
    "n_orbits": "number_of_orbits",
    "merger_time": "common_horizon_time",
    # --- Remnant ---
    "remnant_kick_vector": "remnant_velocity",  # [vx,vy,vz] in units of c
    # --- ADM quantities ---
    "adm_energy_initial": "initial_ADM_energy",
    "adm_angular_momentum_vector": "initial_ADM_angular_momentum",
    "adm_linear_momentum_vector": "initial_ADM_linear_momentum",
    "position1_initial": "initial_position1",
    "position2_initial": "initial_position2",
    # --- Center-of-mass correction ---
    "com_parameters": "com_parameters",
    # --- Numerical method ---
    "initial_data_type": "initial_data_type",
}
"""Mapping from canonical quantity names to SXS metadata keys (snake_case).

Spin components are stored as 3-element lists under ``spin1_vector`` /
``spin2_vector``; the individual ``spin1x`` etc. entries are ``None`` to
signal that they must be accessed by index::

    spin1 = metadata[SXS_KEYS["spin1_vector"]]   # [chi_x, chi_y, chi_z]
    chi1x = spin1[0]
"""

MAYA_KEYS = {
    # --- Identification ---
    "simulation_id": "GTID",  # e.g. "GT0001"
    "run_name": "GT_Tag",
    # --- Masses (code units, total mass ~ 1) ---
    "mass1": "m1",
    "mass2": "m2",
    "mass1_irreducible": "m1_irr",
    "mass2_irreducible": "m2_irr",
    "mass_ratio": "q",  # m1/m2 >= 1
    "eta": "eta",  # symmetric mass ratio
    # --- Spins (dimensionless chi = S/m^2) ---
    "spin1x": "a1x",
    "spin1y": "a1y",
    "spin1z": "a1z",
    "spin2x": "a2x",
    "spin2y": "a2y",
    "spin2z": "a2z",
    # --- Orbital dynamics ---
    "separation_initial": "separation",
    "eccentricity": "eccentricity",
    "mean_anomaly": "mean_anomaly",
    "merger_time": "merge_time",
    # --- Frequencies ---
    # omega_orbital is M*Omega_orb (code units);
    # f_GW^(2,2) = omega_orbital / pi  (code units)
    "orbital_frequency": "omega_orbital",
    "freq_start_22_hz_1msun": "f_lower_at_1MSUN",  # Hz at M_tot = 1 M_sun
    # --- File sizes ---
    "maya_file_size_gb": "maya file size (GB)",
    "lvcnr_file_size_gb": "lvcnr file size (GB)",
}
"""Mapping from canonical quantity names to MAYA/GT metadata keys.

Note that MAYA does not record a dedicated relaxation time, initial ADM
quantities, or a separate reference epoch distinct from the simulation start.
"""

# ---------------------------------------------------------------------------
# Unified cross-catalog lookup: canonical name → {catalog: key}
# ---------------------------------------------------------------------------

CANONICAL_TO_CATALOG = {
    canonical: {
        "RIT": RIT_KEYS.get(canonical),
        "SXS": SXS_KEYS.get(canonical),
        "MAYA": MAYA_KEYS.get(canonical),
    }
    for canonical in sorted(set(RIT_KEYS) | set(SXS_KEYS) | set(MAYA_KEYS))
}
"""Dict mapping each canonical quantity name to its key in every catalog.

Example::

    >>> from nrcatalogtools.metadata import CANONICAL_TO_CATALOG
    >>> CANONICAL_TO_CATALOG["mass_ratio"]
    {'RIT': 'relaxed-mass-ratio-1-over-2', 'SXS': 'reference_mass_ratio', 'MAYA': 'q'}
    >>> CANONICAL_TO_CATALOG["spin1x"]
    {'RIT': 'relaxed-chi1x', 'SXS': None, 'MAYA': 'a1x'}

A value of ``None`` means the quantity is not stored as a scalar key in that
catalog (see the per-catalog dict docstring for the access pattern).
"""

# ---------------------------------------------------------------------------
# PyCBC output parameter names (as produced by get_source_parameters_from_metadata)
# ---------------------------------------------------------------------------

PYCBC_KEYS = {
    "mass1": "mass1",  # M_sun
    "mass2": "mass2",  # M_sun
    "spin1x": "spin1x",
    "spin1y": "spin1y",
    "spin1z": "spin1z",
    "spin2x": "spin2x",
    "spin2y": "spin2y",
    "spin2z": "spin2z",
    "f_lower": "f_lower",  # Hz; -1 if unavailable
}
"""PyCBC-compatible parameter names output by ``get_source_parameters_from_metadata()``.

These are the keys accepted by ``pycbc.waveform.get_td_waveform_modes()`` and
related functions.  All catalog-specific keys are normalised to these names by
``get_source_parameters_from_metadata()``.
"""

CANONICAL_TO_PYCBC = {
    "mass1": "mass1",
    "mass2": "mass2",
    "spin1x": "spin1x",
    "spin1y": "spin1y",
    "spin1z": "spin1z",
    "spin2x": "spin2x",
    "spin2y": "spin2y",
    "spin2z": "spin2z",
    "freq_start_22": "f_lower",
    "orbital_frequency": "f_lower",  # after unit conversion
    "orbital_frequency_vector": "f_lower",
}
"""Maps canonical quantity names to their PyCBC output parameter name.

Quantities absent from this dict are not directly exposed as PyCBC parameters
(e.g. remnant properties, ADM quantities, numerical method flags).
"""


def get_source_parameters_from_metadata(metadata, total_mass=1.0):
    """Return the initial physical parameters for the simulation. Only for
    quasicircular simulations are supported, orbital eccentricity is ignored

    Args:
        total_mass (float, optional): Total Mass of Binary (solar masses).
            Defaults to 1.0.

    Returns:
        dict: Initial binary parameters with names compatible with PyCBC.
    """
    parameters = dict()
    if "relaxed_mass1" in metadata:
        # RIT Catalog
        q = metadata["relaxed_mass_ratio_1_over_2"]
        eta = min(q / (1 + q) ** 2, 0.25)
        m1, m2 = mtotal_eta_to_mass1_mass2(total_mass, eta)
        s1x = metadata["relaxed_chi1x"]
        s1y = metadata["relaxed_chi1y"]
        s1z = metadata["relaxed_chi1z"]
        if np.isnan(s1x):
            s1x = 0
        if np.isnan(s1y):
            s1y = 0
        if np.isnan(s1z):
            s1z = 0
        s2x = metadata["relaxed_chi2x"]
        s2y = metadata["relaxed_chi2y"]
        s2z = metadata["relaxed_chi2z"]
        if np.isnan(s2x):
            s2x = 0
        if np.isnan(s2y):
            s2y = 0
        if np.isnan(s2z):
            s2z = 0
        parameters.update(
            mass1=m1,
            mass2=m2,
            spin1x=s1x,
            spin1y=s1y,
            spin1z=s1z,
            spin2x=s2x,
            spin2y=s2y,
            spin2z=s2z,
        )
        # Now gather initial frequency information
        freq22 = metadata["freq_start_22"]
        if not np.isnan(freq22) and float(freq22) > 0:
            parameters.update(f_lower=float(freq22) / (total_mass * lal.MTSUN_SI))
        else:
            parameters.update(f_lower=-1)
    elif "GTID" in metadata:
        # GT / MAYA CAtalog
        q = metadata["q"]
        eta = min(q / (1 + q) ** 2, 0.25)
        m1, m2 = mtotal_eta_to_mass1_mass2(total_mass, eta)
        parameters.update(mass1=m1, mass2=m2)
        for suffix in ["1x", "1y", "1z", "2x", "2y", "2z"]:
            parameters["spin" + suffix] = metadata["a" + suffix]
        # MAYA metadata uses 'omega_orbital' (orbital angular frequency in M units).
        # GW frequency = omega_orbital / pi (since f_gw = 2 * f_orbital = omega_orbital/pi).
        omega = metadata.get("omega_orbital", None)
        if omega is not None and not np.isnan(float(omega)) and float(omega) > 0:
            parameters.update(
                f_lower=float(omega) / np.pi / (total_mass * lal.MTSUN_SI)
            )
        elif "f_lower_at_1MSUN" in metadata and not np.isnan(
            float(metadata["f_lower_at_1MSUN"])
        ):
            parameters.update(f_lower=float(metadata["f_lower_at_1MSUN"]) / total_mass)
        else:
            parameters.update(f_lower=-1)
    else:
        # SXS Catalog — always use the reference_time epoch (canonical SXS epoch).
        # reference_time may differ from relaxation_time; when they differ,
        # the reference epoch is chosen by SXS to coincide with a given GW
        # frequency, making it the more physically meaningful choice.
        if metadata["relaxation_time"] != metadata["reference_time"]:
            import warnings

            warnings.warn(
                "SXS simulation has relaxation_time != reference_time. "
                "Using reference_time values (spins, mass ratio) as the canonical epoch.",
                UserWarning,
            )
        q = metadata["reference_mass_ratio"]
        eta = min(q / (1 + q) ** 2, 0.25)
        m1, m2 = mtotal_eta_to_mass1_mass2(total_mass, eta)
        spin1 = metadata["reference_dimensionless_spin1"]
        spin2 = metadata["reference_dimensionless_spin2"]
        parameters.update(
            mass1=m1,
            mass2=m2,
            spin1x=spin1[0],
            spin1y=spin1[1],
            spin1z=spin1[2],
            spin2x=spin2[0],
            spin2y=spin2[1],
            spin2z=spin2[2],
        )

        Momega = (np.array(metadata["reference_orbital_frequency"]) ** 2).sum() ** 0.5
        if not np.isnan(Momega):
            parameters.update(f_lower=Momega / np.pi / (total_mass * lal.MTSUN_SI))
        else:
            parameters.update(f_lower=-1)

    return parameters
