# Numerical Relativity Catalog Organization

This page describes the organization and classification of the SXS, RIT, and MAYA numerical relativity catalogs into six physical categories based on spin and eccentricity thresholds, as well as the mapping of the `NRSur7dq4` calibration set.

---

## 1. Classification Results Summary

Applying a **spin component threshold of 0.001** and an **initial eccentricity threshold of 0.005** yields the following counts across the three catalogs:

| Category | Description | SXS (Total) | SXS (NRSur7dq4 Calibration) | RIT | MAYA |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **(a)** | **non-spinning eccentric** | 206 | 0 | 499 | 74 |
| **(b)** | **non-spinning non-eccentric** | 177 | 60 | 54 | 34 |
| **(c)** | **aligned-spin eccentric** | 21 | 0 | 231 | 117 |
| **(d)** | **aligned-spin non-eccentric** | 687 | 282 | 541 | 40 |
| **(e)** | **precessing-spin eccentric** | 30 | 0 | 117 | 303 |
| **(f)** | **precessing-spin non-eccentric** | 3,043 | 1,389 | 437 | 67 |
| **-** | **TOTALS** | **4,164** | **1,731** | **1,879** | **635** |

*Note: In the SXS catalog, 6 simulations raised metadata errors and were bypassed. In the NRSur7dq4 calibration subset, there are exactly 0 eccentric simulations across categories (a), (c), and (e) because the surrogate model is strictly quasicircular.*

### Key Observations:
1. **NRSur7dq4 Calibration Quasicircular Cut**: Categories **(a)**, **(c)**, and **(e)** (the eccentric categories) contain **exactly 0 simulations** in the NRSur7dq4 calibration subset. This is physically correct, as the `NRSur7dq4` surrogate is a quasicircular (non-eccentric) model calibrated only on simulations with negligible eccentricity ($e < 0.005$).
2. **Generic Spin Precession**: Precessing non-eccentric systems (category **f**) dominate the SXS catalog (3,043 total, 1,389 used for NRSur7dq4). This represents the main body of generic, fully-precessing BBH simulations.
3. **Eccentricity Abundance**: The RIT catalog has a highly significant portion of eccentric simulations (499 non-spinning, 231 aligned-spin, 117 precessing-spin), making it a valuable catalog for eccentric waveform studies.

---

## 2. Organized Files Structure

Classified simulation ID lists are exported in structured JSON format and committed directly under the root **`catalog_organization/`** directory of the repository:
- **`sxs_classification.json`**: Lists all categorized SXS simulations. Each simulation entry includes an explicit `"nrsur7dq4_calibration"` boolean flag:
  ```json
  "non-spinning non-eccentric": {
      "total_count": 177,
      "nrsur_calibration_count": 60,
      "simulations": [
          {
              "id": "SXS:BBH:0001",
              "nrsur7dq4_calibration": true
          },
          ...
      ]
  }
  ```
- **`rit_classification.json`**: Plain list of RIT simulation IDs mapped under each of the six category descriptions.
- **`maya_classification.json`**: Plain list of MAYA simulation IDs mapped under each category.

---

## 3. How to Use the JSON Files in Python

You can easily read these classification files to select sub-category sets of simulations. Below is a complete Python guide illustrating how to load the JSON outputs and query lists of simulations:

```python
import json
import os

# 1. Resolve path to catalog_organization directory under the repository root
base_dir = "/home/prayush/src/nr-catalog-tools" # replace with your repo root
org_dir = os.path.join(base_dir, "catalog_organization")

# -------------------------------------------------------------
# A. Select SXS Simulations & Filter by NRSur7dq4 Calibration
# -------------------------------------------------------------
sxs_file = os.path.join(org_dir, "sxs_classification.json")

with open(sxs_file, "r") as f:
    sxs_data = json.load(f)

# Get ALL aligned-spin non-eccentric SXS simulations (Category d)
aligned_non_ecc_sxs = [
    sim["id"] 
    for sim in sxs_data["aligned-spin non-eccentric"]["simulations"]
]
print(f"Total SXS Aligned-Spin Non-Eccentric: {len(aligned_non_ecc_sxs)}") # 687

# Get only the NRSur7dq4 Calibration subset for that category
aligned_nrsur_cal = [
    sim["id"] 
    for sim in sxs_data["aligned-spin non-eccentric"]["simulations"]
    if sim["nrsur7dq4_calibration"]
]
print(f"NRSur7dq4 Calibration subset in Category d: {len(aligned_nrsur_cal)}") # 282


# -------------------------------------------------------------
# B. Select RIT or MAYA Simulations
# -------------------------------------------------------------
rit_file = os.path.join(org_dir, "rit_classification.json")
maya_file = os.path.join(org_dir, "maya_classification.json")

with open(rit_file, "r") as f:
    rit_data = json.load(f)

with open(maya_file, "r") as f:
    maya_data = json.load(f)

# Get all precessing-spin eccentric RIT simulations (Category e)
rit_precessing_ecc = rit_data["precessing-spin eccentric"]["simulations"]
print(f"Total RIT Precessing-Spin Eccentric: {len(rit_precessing_ecc)}") # 117

# Get all non-spinning non-eccentric MAYA simulations (Category b)
maya_circular_nonspinning = maya_data["non-spinning non-eccentric"]["simulations"]
print(f"Total MAYA Non-Spinning Non-Eccentric: {len(maya_circular_nonspinning)}") # 34


# -------------------------------------------------------------
# C. Waveform Pipeline Loading Integration
# -------------------------------------------------------------
import nrcatalogtools as nrcat

rit_cat = nrcat.RITCatalog.load()

# Load the actual waveform data for the first selected eccentric simulation
if rit_precessing_ecc:
    target_sim_id = rit_precessing_ecc[0]
    print(f"\nLoading waveform for: {target_sim_id}")
    waveform = rit_cat.get(target_sim_id)
    print("Waveform loaded successfully!", type(waveform))
```

---

## 4. Reusable Class API Usage

All classification operations are powered by the `NRCatalogClassifier` class integrated into the `nrcatalogtools` library. Users can also import this class to dynamically query simulation groups under any desired spin or eccentricity thresholds:

```python
from nrcatalogtools import NRCatalogClassifier

# 1. Initialize classifier with custom or default thresholds
classifier = NRCatalogClassifier(spin_threshold=0.001, ecc_threshold=0.005)

# 2. Dynamically query simulations by category key ('a'-'f') or category name
# Example: Get all precessing-spin eccentric simulations in MAYA
maya_precessing_ecc = classifier.get_simulations('MAYA', 'e')
print(f"MAYA precessing eccentric count: {len(maya_precessing_ecc)}")

# Example: Get only NRSur7dq4 calibration simulations that are non-spinning non-eccentric in SXS
sxs_nrsur_nonspinning_circular = classifier.get_simulations(
    'SXS', 'b', only_nrsur_calibration=True
)
print(f"SXS non-spinning circular training count: {len(sxs_nrsur_nonspinning_circular)}")

# 3. Classify a single simulation directly
sxs_sim_cat = classifier.classify_simulation('SXS', 'SXS:BBH:0001')
print(f"SXS:BBH:0001 Category: {sxs_sim_cat}")
# Output: "non-spinning non-eccentric"
```
