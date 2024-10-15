import numpy as np

import lal
from pycbc.pnutils import mtotal_eta_to_mass1_mass2


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
        # Now father initial frequency information
        if not np.isnan(metadata["freq_start_22"]):
            parameters.update(
                f_lower=float(metadata["freq_start_22"]) / (total_mass * lal.MTSUN_SI)
            )
        else:
            parameters.update(f_lower=-1)
    elif "GTID" in metadata:
        # GT / MAYA CAtalog
        q = metadata["q"]
        eta = min(q / (1 + q) ** 2, 0.25)
        m1, m2 = mtotal_eta_to_mass1_mass2(total_mass, eta)
        parameters.update(mass1=m1, mass2=m2)
        for suffix in ["1x", "1y", "1z", "2x", "2y", "2z"]:
            parameters["s" + suffix] = metadata["a" + suffix]
        if not np.isnan(metadata["Momega"]):
            parameters.update(
                f_lower=float(metadata["Momega"]) / np.pi / (total_mass * lal.MTSUN_SI)
            )
        else:
            parameters.update(f_lower=-1)
    else:
        # SXS Catalog
        if metadata["relaxation_time"] == metadata["reference_time"]:
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
        else:
            raise IOError(
                """`relaxation_time` is not the same as `reference_time`
for this SXS simulation. Its not clear what to do in such a situation."""
            )

        Momega = (np.array(metadata["reference_orbital_frequency"]) ** 2).sum() ** 0.5
        if not np.isnan(Momega):
            parameters.update(f_lower=Momega / np.pi / (total_mass * lal.MTSUN_SI))
        else:
            parameters.update(f_lower=-1)

    return parameters
