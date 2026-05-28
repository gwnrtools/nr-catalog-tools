#!/usr/bin/env python3
"""Execute classification and organization of numerical relativity catalogs (SXS, RIT, MAYA).

This script uses the ``NRCatalogClassifier`` class to categorize all simulations
into categories (a)-(f) under a spin threshold of 0.001 and eccentricity threshold of 0.005.
It outputs classified lists to JSON files and prints a summary count table to stdout.
"""

import sys
import os
import json

# Ensure parent package can be found (bin/ is under nr-catalog-tools/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nrcatalogtools import NRCatalogClassifier


def main():
    print("=============================================================")
    print("Numerical Relativity Catalog Classification and Organization")
    print("=============================================================")

    # Initialize the classifier with default thresholds
    classifier = NRCatalogClassifier(spin_threshold=0.001, ecc_threshold=0.005)

    # Load the NRSur7dq4 training simulations list
    calibration_sims = classifier.load_nrsur_calibration_sims()
    print(f"Loaded {len(calibration_sims)} NRSur7dq4 training simulation IDs.")

    catalogs = ["SXS", "RIT", "MAYA"]
    category_keys = ["a", "b", "c", "d", "e", "f"]

    results = {}

    # 1. Run classifications
    for cat_name in catalogs:
        print(f"\nProcessing {cat_name} Catalog...")
        results[cat_name] = {}
        for cat_key in category_keys:
            full_category_name = classifier.CATEGORY_MAPPING[cat_key]

            # SXS gets separate identification of calibration simulations
            if cat_name == "SXS":
                all_sims = classifier.get_simulations(cat_name, cat_key)
                cal_subset = classifier.get_simulations(
                    cat_name, cat_key, only_nrsur_calibration=True
                )
                results[cat_name][full_category_name] = {
                    "total_count": len(all_sims),
                    "nrsur_calibration_count": len(cal_subset),
                    "simulations": [
                        {
                            "id": sim_id,
                            "nrsur7dq4_calibration": (sim_id in calibration_sims),
                        }
                        for sim_id in all_sims
                    ],
                }
            else:
                all_sims = classifier.get_simulations(cat_name, cat_key)
                results[cat_name][full_category_name] = {
                    "total_count": len(all_sims),
                    "simulations": all_sims,
                }

    # 2. Write JSON outputs (saved in catalog_organization/ at root)
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "catalog_organization")
    )
    os.makedirs(out_dir, exist_ok=True)

    for cat_name in catalogs:
        out_file = os.path.join(out_dir, f"{cat_name.lower()}_classification.json")
        with open(out_file, "w") as f:
            json.dump(results[cat_name], f, indent=4)
        print(f"Saved {cat_name} classification to: {out_file}")

    # 3. Print a beautiful summary table
    print("\n" + "=" * 115)
    print(
        f"{'Category Description':<35} | {'SXS (Total)':<15} | {'SXS (NRSur7dq4)':<17} | {'RIT':<15} | {'MAYA':<15}"
    )
    print("-" * 115)

    for cat_key in category_keys:
        full_name = classifier.CATEGORY_MAPPING[cat_key]
        display_name = f"({cat_key}) {full_name}"

        sxs_tot = results["SXS"][full_name]["total_count"]
        sxs_cal = results["SXS"][full_name]["nrsur_calibration_count"]
        rit_tot = results["RIT"][full_name]["total_count"]
        maya_tot = results["MAYA"][full_name]["total_count"]

        print(
            f"{display_name:<35} | {sxs_tot:<15} | {sxs_cal:<17} | {rit_tot:<15} | {maya_tot:<15}"
        )

    print("-" * 115)

    # Print overall sums
    sxs_sum = sum(
        results["SXS"][n]["total_count"] for n in classifier.CATEGORY_MAPPING.values()
    )
    sxs_cal_sum = sum(
        results["SXS"][n]["nrsur_calibration_count"]
        for n in classifier.CATEGORY_MAPPING.values()
    )
    rit_sum = sum(
        results["RIT"][n]["total_count"] for n in classifier.CATEGORY_MAPPING.values()
    )
    maya_sum = sum(
        results["MAYA"][n]["total_count"] for n in classifier.CATEGORY_MAPPING.values()
    )

    print(
        f"{'TOTALS':<35} | {sxs_sum:<15} | {sxs_cal_sum:<17} | {rit_sum:<15} | {maya_sum:<15}"
    )
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()
