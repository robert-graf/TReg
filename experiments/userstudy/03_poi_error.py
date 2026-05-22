import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

out = str(Path(__file__).parent.parent.parent)
sys.path.append(out)
from constants import POI_MAP, euclidean_3d, extract_xyz, out_userstudy, rater_key

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
    Veerman_Model = df_all[df_all.evaluator == "Veerman_Model"].dropna(subset=["X", "Y", "Z"])
    return {"intra": intra, "inter": inter, "treg-naive": model, "treg-veerman": Veerman_Model}


# def euclid_fig(pairwise=False):
#    result = {}
#    for name, df_task in get_groups().items():
#        pairs = []
#        for (fname, poi), g in df_task.groupby(["filename", "POI_name"]):
#            if g.shape[0] != 3:
#                continue
#            coords = g[["X", "Y", "Z"]].values
#            if pairwise:
#                dists = []
#                for i, j in combinations(range(len(coords)), 2):
#                    d = euclidean_3d(coords[[i]], coords[[j]])[0]
#                    dists.append(d)
#                for d in dists:
#                    pairs.append(  # noqa: PERF401
#                        {
#                            "filename": fname,
#                            "POI_name": poi,
#                            "group": f"{name}-rater",
#                            "abs_3d_diff": d,
#                        }
#                    )
#            else:
#                subject_mean = coords.mean(axis=0, keepdims=True)  # mean across versions
#                d = np.linalg.norm(coords - subject_mean, axis=1).mean()
#                pairs.append(
#                    {
#                        "filename": fname,
#                        "POI_name": poi,
#                        "group": f"{name}-rater",
#                        "abs_3d_diff": d,
#                    }
#                )
#
#        df_inter = pd.DataFrame(pairs)
#        result[name] = df_inter
#
#    df_plot = pd.concat(result.values(), ignore_index=True)
#    s = "pairwise" if pairwise else "to-mean"
#    df_plot.to_excel(out_userstudy / f"03_ingroup_all_distances_{s}.xlsx", index=False)
#    fig = px.box(df_plot, x="POI_name", y="abs_3d_diff", color="group", points="outliers")
#    fig.update_layout(
#        yaxis_title="Absolute mean Diffrence [mm]",
#        xaxis_title="Landmark (in-group only)",
#        boxmode="group",
#        template="simple_white",
#    )
#    fig.write_image(out_userstudy / f"03_ingroup_distances_{s}.svg", width=2000, height=800)
#
#
# euclid_fig()
ticks = [
    "TGT",
    "FHC",
    "FNC",
    "FAAP",
    "FLCD",
    "FMCD",
    "FLCP",
    "FMCP",
    "FNP",
    "FADP",
    "TGPP",
    "TGCP",
    "FMCPC",
    "FLCPC",
    "TRMP",
    "TRLP",
    "TLCL",
    "TMCM",
    "TKC",
    "TLCA",
    "TLCP",
    "TMCA",
    "TMCP",
    "TTP",
    "TAAP",
    "TMIT",
    "TLIT",
    "FLM",
    "TMM",
    "TAC",
    "TADP",
    "PPP",
    "PDP",
    "PMP",
    "PLP",
    "PRPP",
    "PRDP",
    "PRHP",
]
all_results = []
for name, df_task in get_groups().items():
    df_task["rater_version"] = df_task["evaluator"].astype(str) + df_task["version"].astype(str)
    for poi_name in POI_MAP.keys():
        poi_df = df_task[df_task.POI_name == poi_name]
        if poi_df.empty:
            continue

        # Extract coordinates as (num_versions, num_files, 3)
        coords_list = []
        versions = sorted(poi_df.rater_version.unique())
        names_list = []
        files = sorted(poi_df.filename.unique())

        for v in versions:
            v_coords = []
            for f in files:
                row = poi_df[(poi_df.rater_version == v) & (poi_df.filename == f)]
                if row.empty:
                    v_coords.append([np.nan, np.nan, np.nan])
                else:
                    v_coords.append(row[["X", "Y", "Z"]].values[0])
            coords_list.append(v_coords)
            names_list.append(v)

        coords = np.array(coords_list)  # shape: (raters, subjects, 3)
        pairs = []

        for i, j in combinations(range(len(coords)), 2):
            # d = euclidean_3d(coords[[i]], coords[[j]])[0]
            euclid = np.sqrt(np.sum((coords[i] - coords[j]) ** 2, axis=-1))
            # exit()
            absolute_dif = np.sum(np.abs(coords[i] - coords[j]), axis=-1)
            for e, a, f in zip(euclid, absolute_dif, files):
                all_results.append(  # noqa: PERF401
                    {
                        "Task": name,
                        "POI": poi_name,
                        "group": f"{name}-rater",
                        "mean_euclid": e,
                        "mean_diffrence": a,
                        "Versions": versions,
                        "pair": f"{names_list[i]}-{names_list[j]}",
                        "file": str(f),
                    }
                )
        # all_results = pairs
        ## Compute per-subject mean
        # subject_mean = coords.mean(axis=0, keepdims=True)  # mean across versions
        ## distances = np.linalg.norm(coords - subject_mean, axis=2).mean(0)
        # euclid = np.sqrt(np.sum((coords - subject_mean) ** 2, axis=2)).reshape(-1)
        # absolute_dif = np.sqrt(np.sum(np.abs(coords - subject_mean), axis=2)).reshape(-1)
        # for i in range(euclid.shape[0]):
        #    all_results.append(  # noqa: PERF401
        #        {
        #            "Task": name,
        #            "POI": poi_name,
        #            # "Mean_3D_Error_mm": distances.mean(),
        #            "mean_euclid": euclid[i],
        #            "mean_diffrence": absolute_dif[i],
        #            "Versions": versions,
        #        }
        #    )

# Convert the list of dicts to a DataFrame
results_df = pd.DataFrame(all_results)
print(results_df[results_df.POI == "TGT"])
results_df.to_excel(out_userstudy / "03_ingroup_distances.xlsx", index=False)

fig = px.box(results_df, x="POI", y="mean_diffrence", color="Task", points="outliers", category_orders={"POI": ticks})
fig.update_layout(
    yaxis_title="Absolute Diffrence [mm]",
    xaxis_title="Landmark (repeatability)",
    boxmode="group",
    # template="simple_white",
)
# fig.update_yaxes(range=[-0.5, 11.5], dtick=1)
fig.write_image(out_userstudy / "03_ingroup_mean_abs_diffrence.svg", width=2000, height=500)


fig = px.box(results_df, x="POI", y="mean_euclid", color="Task", points="outliers", category_orders={"POI": ticks})
fig.update_layout(
    yaxis_title="Absolute Euclid Diffrence [mm]",
    xaxis_title="Landmark (repeatability)",
    boxmode="group",
    # template="simple_white",
)
# fig.update_yaxes(range=[-0.5, 50], dtick=5)
fig.write_image(out_userstudy / "03_ingroup_mean_mean_euclid_diffrence.svg", width=2000, height=500)

stats = results_df.groupby(["group", "POI"])["mean_diffrence"].agg(MAE="mean", SD="std", N="count").reset_index()
stats["MAE±SD"] = stats.apply(lambda r: f"{r.MAE:.2f} ± {r.SD:.2f}", axis=1)

table = stats.pivot(index="POI", columns="group", values="MAE±SD")

print("\nMAE ± SD (deg)")
print(table.to_string())


stats = results_df.groupby(["group", "POI"])["mean_euclid"].agg(MAE="mean", SD="std", N="count").reset_index()
stats["MAE±SD"] = stats.apply(lambda r: f"{r.MAE:.2f} ± {r.SD:.2f}", axis=1)

table = stats.pivot(index="POI", columns="group", values="MAE±SD")

print("\nmean_euclid ± SD (deg)")
print(table.to_string())
