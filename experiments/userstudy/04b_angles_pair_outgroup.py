import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from TPTBox import Print_Logger

out = str(Path(__file__).parent.parent.parent)
sys.path.append(out)
from constants import out_userstudy, rater_key

df_all = pd.read_excel(out_userstudy / "all_angles.xlsx")
df_all["evaluator"] = df_all.apply(lambda row: rater_key[row.rater][0], axis=1)
df_all["version"] = df_all.apply(lambda row: rater_key[row.rater][1], axis=1)


angles = (
    "tibia_torsion_2D",
    "femoral_torsion_2D",
    "mLDFA",
    "MPTA",
    "HKA_2D",
    "LPFA",
    "MPFA",
    "NSA",
    "sagittal_HKA",
    "JLCA",
    "tibial_slope_medial",
    "tibial_slope_lateral",
    "patella_tilt",
    "PDFA",
    # "trochlea_depth_lateral",
    # "trochlea_depth_medial",
    # "trochlea_angle",
    # "PDFA_sagittal_2D",
    # "PDFA_medial_3D",
    # "PDFA_lateral_3D",
)


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
    mean_angle_deg = np.rad2deg(mean_angle_rad) % 360
    return mean_angle_deg


def get_groups():
    intra = df_all[df_all.evaluator == "Julius"]
    inter = df_all[df_all.version == "V1"]
    model = df_all[df_all.evaluator == "Robert_Model"]
    return {"intra": intra, "inter": inter, "model": model}


def angular_outgroup_fig():
    groups = get_groups()
    result = []

    # Define the comparisons you want
    comparisons = [
        ("intra", "inter"),
        ("intra", "model"),
        ("inter", "model"),
    ]

    for g1_name, g2_name in comparisons:
        df1, df2 = groups[g1_name], groups[g2_name]

        # Align by file and angle
        common_files = set(df1.file.unique()) & set(df2.file.unique())

        for fname in common_files:
            for a in angles:
                rows1 = df1[df1.file == fname][a].dropna().to_numpy()
                rows2 = df2[df2.file == fname][a].dropna().to_numpy()

                if len(rows1) == 0 or len(rows2) == 0:
                    continue

                # Compute all pairwise angular differences between groups
                for val1, val2 in product(rows1, rows2):
                    d = angular_diff_deg(val1, val2)
                    result.append(
                        {
                            "file": fname,
                            "angle": a,
                            "group": f"{g1_name}-vs-{g2_name}",
                            "abs_angle_diff": d,
                        }
                    )

    df_plot = pd.DataFrame(result)

    Print_Logger().on_save(out_userstudy / "04b_outgroup_angles.xlsx")
    df_plot.to_excel(out_userstudy / "04b_outgroup_angles.xlsx", index=False)

    fig = px.box(
        df_plot,
        x="angle",
        y="abs_angle_diff",
        color="group",
        points="outliers",
    )
    fig.update_layout(
        yaxis_title="Absolute Angular Difference [deg]",
        xaxis_title="Angle (cross-group)",
        boxmode="group",
        # template="simple_white",
    )
    Print_Logger().on_save(out_userstudy / "04b_outgroup_angles.svg")
    fig.write_image(out_userstudy / "04b_outgroup_angles.svg", width=1500, height=500)

    fig = px.box(df_plot, x="angle", y="abs_angle_diff", color="group", points="outliers")  # log_y=True
    fig.update_layout(
        yaxis_title="Absolute Diffrence Angle [deg].",
        xaxis_title="Angle (in-group only)",
        boxmode="group",
        # template="simple_white",
    )
    # if pairwise:
    #    fig.update_yaxes(range=[-3, 2])
    fig.write_image(out_userstudy / "04b_outgroup_angles_small.svg", width=1500, height=500)
    return df_plot


df_plot = angular_outgroup_fig()
# Compute MAE and SD
stats = df_plot.groupby(["group", "angle"])["abs_angle_diff"].agg(MAE="mean", SD="std", N="count").reset_index()

# Save table
stats["MAE±SD"] = stats.apply(lambda r: f"{r.MAE:.2f} ± {r.SD:.2f}", axis=1)

table = stats.pivot(index="angle", columns="group", values="MAE±SD")
Print_Logger().on_save(out_userstudy / "04b_outgroup_angles_stats.xlsx")
table.to_excel(out_userstudy / "04b_outgroup_angles_stats.xlsx", index=True)

print("\nMAE ± SD (deg)")
print(table.to_string())
