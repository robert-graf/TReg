import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from TPTBox import Print_Logger

out = str(Path(__file__).parent.parent.parent)
sys.path.append(out)
from constants import out_userstudy, rater_key

df_all = pd.read_excel(out_userstudy / "all_angles.xlsx")
df_all["evaluator"] = df_all.apply(lambda row: rater_key[row.rater][0], axis=1)
df_all["version"] = df_all.apply(lambda row: rater_key[row.rater][1], axis=1)


angles = [
    "tibia_torsion_2D",
    "femoral_torsion_2D",
    "mLDFA",
    "MPTA",
    "HKA_2D",
    "PDFA_sagittal_2D",
    "PDFA_medial_3D",
    "PDFA_lateral_3D",
    "tibial_slope_medial",
    "tibial_slope_lateral",
]


def angular_diff_deg(a, b):
    """Compute minimal absolute difference between two angles in degrees."""
    diff = np.abs(a - b) % 180
    return np.minimum(diff, 180 - diff)


def circular_mean_deg(angles):
    """Compute mean angle in degrees."""
    angles_rad = np.deg2rad(angles)  # convert to radians
    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))
    mean_angle_rad = np.arctan2(mean_sin, mean_cos)
    mean_angle_deg = np.rad2deg(mean_angle_rad) % 180
    return mean_angle_deg


def get_groups():
    intra = df_all[df_all.evaluator == "Julius"]
    inter = df_all[df_all.version == "V1"]
    model = df_all[df_all.evaluator == "Robert_Model"]
    return {"intra": intra, "inter": inter, "model": model}


def compute_icc(df, angle, rater_col="rater", subject_col="file"):
    """
    Computes ICC(2,1) for a given angle.
    """
    df_icc = df[[subject_col, rater_col, angle]].dropna()

    icc = pg.intraclass_corr(
        data=df_icc,
        targets=subject_col,
        raters=rater_col,
        ratings=angle,
    )

    # ICC(2,1) = two-way random, absolute agreement, single measurement
    return icc[icc["Type"] == "ICC2"]["ICC"].values[0]


def get_icc_groups():
    intra = df_all[df_all.evaluator == "Julius"].copy()
    inter = df_all[df_all.version == "V1"].copy()
    model = df_all[df_all.evaluator == "Robert_Model"].copy()

    humans = df_all[df_all.evaluator != "Robert_Model"].copy()

    return {
        "intra": intra,
        "inter": inter,
        "human-model": pd.concat([humans, model]),
        "inter-model": pd.concat([inter, model]),
        "intra-model": pd.concat([intra, model]),
    }


icc_results = []

groups = get_icc_groups()

for group_name, df_g in groups.items():
    for angle in angles:
        try:
            icc_value = compute_icc(
                df=df_g,
                angle=angle,
                rater_col="rater",
                subject_col="file",
            )

            icc_results.append(
                {
                    "Group": group_name,
                    "Angle": angle,
                    "ICC_2_1": icc_value,
                }
            )
        except Exception:
            icc_results.append(
                {
                    "Group": group_name,
                    "Angle": angle,
                    "ICC_2_1": np.nan,
                }
            )

icc_df = pd.DataFrame(icc_results)
Print_Logger().on_save(out_userstudy / "04c_angles_ICC.xlsx")
icc_df.to_excel(out_userstudy / "04c_angles_ICC.xlsx", index=False)

print(icc_df)
