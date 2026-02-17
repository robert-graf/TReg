import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from TPTBox import Print_Logger

out = str(Path(__file__).parent.parent.parent)
sys.path.append(out)
from constants import extract_xyz, out_userstudy, rater_key

df = pd.read_excel(out_userstudy / "all_coordinates.xlsx")

print(rater_key)
all_rater = [
    extract_xyz(df, a + "_X", a + "_Y", a + "_Z", evaluator=evaluator, version=version) for a, (evaluator, version) in rater_key.items()
]
df_all = pd.concat(all_rater, ignore_index=True)


def get_groups():
    intra = df_all[df_all.evaluator == "Julius"].dropna(subset=["X", "Y", "Z"])
    inter = df_all[df_all.version == "V1"].dropna(subset=["X", "Y", "Z"])
    model = df_all[df_all.evaluator == "Robert_Model"].dropna(subset=["X", "Y", "Z"])
    return {"intra": intra, "inter": inter, "model": model}


def outgroup_euclid_fig():
    groups = get_groups()
    result = []

    # Define the pairs you want to compare
    comparisons = [
        ("inter", "intra"),
        ("inter", "model"),
        ("intra", "model"),
    ]

    for g1_name, g2_name in comparisons:
        df1, df2 = groups[g1_name], groups[g2_name]

        # Align by filename and POI
        common_files = set(df1.filename.unique()) & set(df2.filename.unique())
        common_pois = set(df1.POI_name.unique()) & set(df2.POI_name.unique())

        for fname, poi in product(common_files, common_pois):
            rows1 = df1[(df1.filename == fname) & (df1.POI_name == poi)]
            rows2 = df2[(df2.filename == fname) & (df2.POI_name == poi)]

            if rows1.empty or rows2.empty:
                continue

            coords1 = rows1[["X", "Y", "Z"]].values
            coords2 = rows2[["X", "Y", "Z"]].values

            # Compute all pairwise distances between groups
            for c1, c2 in product(coords1, coords2):
                d = np.linalg.norm(c1 - c2)
                result.append(
                    {
                        "filename": fname,
                        "POI_name": poi,
                        "group": f"{g1_name}-vs-{g2_name}",
                        "abs_3d_diff": d,
                    }
                )

    df_plot = pd.DataFrame(result)
    Print_Logger().on_save(out_userstudy / "03b_outgroup_distances_pair.xlsx")

    df_plot.to_excel(out_userstudy / "03b_outgroup_distances_pair.xlsx", index=False)

    fig = px.box(df_plot, x="POI_name", y="abs_3d_diff", color="group", points="outliers")
    fig.update_layout(
        yaxis_title="Absolute 3D Euclidean Deviation [mm]",
        xaxis_title="Landmark (cross-group)",
        boxmode="group",
        template="simple_white",
    )
    Print_Logger().on_save(out_userstudy / "03b_outgroup_distances_pair.svg")
    fig.write_image(out_userstudy / "03b_outgroup_distances_pair.svg", width=2000, height=800)


outgroup_euclid_fig()
