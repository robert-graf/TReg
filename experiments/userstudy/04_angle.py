import sys
from itertools import combinations
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


def diff_fig(pairwise=False):
    result = {}
    for name, df_task in get_groups().items():
        pairs = []
        for a in angles:
            for (fname), g in df_task.groupby(["file"]):
                v = g[a].to_numpy()
                if pairwise:
                    dists = []
                    for i, j in combinations(range(len(v)), 2):
                        d = (angular_diff_deg(v[i], v[j]),)
                        dists.append(d)

                    for d in dists:
                        pairs.append(
                            {
                                "filename": fname,
                                "angle": a,
                                "group": f"{name}-rater",
                                "abs_3d_diff": d,
                            }
                        )
                else:
                    mean_angle = circular_mean_deg(v)
                    d = np.mean([angular_diff_deg(val, mean_angle) for val in v])
                    pairs.append(
                        {
                            "filename": fname,
                            "angle": a,
                            "group": f"{name}-rater",
                            "abs_3d_diff": d,
                        }
                    )

            df_inter = pd.DataFrame(pairs)
            result[name] = df_inter

    df_plot = pd.concat(result.values(), ignore_index=True)
    s = "pairwise" if pairwise else "to-mean"
    Print_Logger().on_save(out_userstudy / f"04_ingroup_all_angles_{s}.xlsx")

    df_plot.to_excel(out_userstudy / f"04_ingroup_all_angles_{s}.xlsx", index=False)
    fig = px.box(df_plot, x="angle", y="abs_3d_diff", color="group", points="outliers")
    fig.update_layout(
        yaxis_title="Absolute Diffrence Angle",
        xaxis_title="Angle (in-group only)",
        boxmode="group",
        template="simple_white",
    )
    Print_Logger().on_save(out_userstudy / f"04_ingroup_angles_{s}.svg")
    fig.write_image(out_userstudy / f"04_ingroup_angles_{s}.svg", width=2000, height=800)


diff_fig(False)
