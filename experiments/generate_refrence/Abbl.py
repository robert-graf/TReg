import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import ttest_rel, wilcoxon
from TPTBox import BIDS_FILE, NII, POI_Global, Print_Logger, to_nii
from TPTBox.core.vert_constants import Full_Body_Instance, Lower_Body
from TPTBox.registration import Template_Registration

out = str(Path(__file__).parent.parent.parent)
sys.path.append(out)

from constants import POI_MAP, flips_model, mapp_models_filp, out_userstudy, out_voting, path_mrk, path_train_poi, raters_all

logger = Print_Logger()

leg_ids = [
    Full_Body_Instance.femur_left,
    Full_Body_Instance.patella_left,
    Full_Body_Instance.tibia_left,
    Full_Body_Instance.fibula_left,
]
mapping_mirror = {
    Full_Body_Instance.femur_right.value: Full_Body_Instance.femur_left.value,
    Full_Body_Instance.patella_right.value: Full_Body_Instance.patella_left.value,
    Full_Body_Instance.tibia_right.value: Full_Body_Instance.tibia_left.value,
    Full_Body_Instance.fibula_right.value: Full_Body_Instance.fibula_left.value,
    Full_Body_Instance.femur_left.value: Full_Body_Instance.femur_right.value,
    Full_Body_Instance.patella_left.value: Full_Body_Instance.patella_right.value,
    Full_Body_Instance.tibia_left.value: Full_Body_Instance.tibia_right.value,
    Full_Body_Instance.fibula_left.value: Full_Body_Instance.fibula_right.value,
}


def fetch(path: Path | str, seg: bool):
    path = Path(path)
    if path.exists():
        return path
    p = str(path).split(local_folder.name)[-1][1:]
    p = Path("/run/user/1000/gvfs/smb-share:server=172.21.251.64,share=nas/datasets_processed/CT_spine/dataset-myelom", p)
    if p.exists():
        to_nii(p, seg).save(path)
    else:
        logger.on_fail(path, "does not exist")
    # assert p.exists(), p


def to_mrk(path_train_poi: Path, out_folder: Path):
    mapping1 = None
    mapping2 = None
    name__ = None

    for i in (path_train_poi).iterdir():
        if "xlsx" in i.name:
            continue
        left = "left" in i.name.lower()
        right = "right" in i.name.lower()
        assert left or right, i

        bf = BIDS_FILE(str(i).replace("subCT", "sub-CT").replace("ses2", "ses-2"), path_train_poi)
        sub = bf.get("sub")
        ses = bf.get("ses")
        sequ = bf.get("sequ")
        assert ses is not None, i
        assert sub is not None, i
        side = "left" if left else "right"
        key = f"sub-{sub}_ses-{ses}_sequ-{sequ}_side-{side}"
        out_mrk = out_folder / (key + ".mrk.json")
        if out_mrk.exists():
            continue
        poi = POI_Global.load(i)
        if mapping1 is None:
            mapping1 = poi.info["label_name"].copy()
            mapping2 = poi.info["label_group_name"].copy()
            name__ = key
        else:
            if mapping1 != poi.info["label_name"]:
                for k, v in mapping1.items():
                    if k not in poi.info["label_name"] or v != poi.info["label_name"][k]:
                        print(k, v, key, name__, k not in poi.info["label_name"], raters_all)

            assert mapping2 == poi.info["label_group_name"], (
                mapping2,
                poi.info["label_group_name"],
            )
        if poi.info.get("Side") is None:
            poi.info["Side"] = side.upper()
        out_mrk.parent.mkdir(exist_ok=True, parents=True)
        poi.sort().save_mrk(out_mrk)


def run_all(
    folder: Path,
    target: NII,
    out_folder: Path,
    lr: float = 0.001,
    max_steps: int = 1500,
    min_delta: float = 0.000001,
    pyramid_levels: int = 4,
    coarsest_level: int = 3,
    finest_level: int = 0,
    be=0.00001,
    mse=1,
    dice=0.01,
    com=0.001,
    no_inference=False,
):

    weights: dict = {
        "be": be,
        "seg": mse,
        "Dice": dice,
        "Tether": com,
    }
    out_paths = []
    for i in (folder).iterdir():
        poi = POI_Global.load(i)
        bf = BIDS_FILE(i, local_folder)
        moving_path = (
            local_folder
            / derivatives_folder
            / str(bf.get("sub"))
            / f"ses-{bf.get('ses')!s}"
            / f"sub-{bf.get('sub')!s}_ses-{bf.get('ses')!s}_sequ-{bf.get('sequ')!s}_seg-VIBESeg-11-lr_msk.nii.gz"
        )
        fetch(moving_path, True)

        # continue
        if not moving_path.exists():
            continue
        # exit()

        mirror = "LEFT" not in i.name.upper()

        out = out_folder / i.name
        out_paths.append(out)
        if out.exists():
            continue
        if no_inference:
            return None
        # assert not mirror
        moving_img = to_nii(moving_path, True)
        if mirror:
            moving_img = moving_img.map_labels(mapping_mirror)

        logger.on_debug(f"{mirror=} {out=}")
        seg = target.extract_label(leg_ids, True)
        reg = Template_Registration(
            seg,  # Target segmentation
            moving_img.extract_label(leg_ids, True),  # Starting Segmentation (not the split one)
            same_side=not mirror,
            lr=lr,
            max_steps=max_steps,
            min_delta=min_delta,
            pyramid_levels=pyramid_levels,
            coarsest_level=coarsest_level,
            finest_level=finest_level,
            # loss_terms=loss_terms,
            # poi_target_cms=None,
            # poi_cms=poi_atlas_cms,  # Can be None, than it will be computed automatically
            weights=weights,
            gpu=0,
        )
        logger.print("make atlas_reg", moving_img)
        atlas_reg = reg.transform_poi(poi)  # Transferring the atlas
        atlas_reg.info = poi.info
        atlas_reg.to_global().save_mrk(out)
    return out_paths


def robust_mean(points, k=3.5):
    """
    points: (N, 3) array-like
    returns: (3,) robust mean
    """
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N,3) points, got {points.shape}")

    # ── 1. Remove invalid points ──────────────────────────────────────────────
    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]

    if len(points) == 0:
        return np.array([np.nan, np.nan, np.nan])

    if len(points) == 1:
        return points[0]

    # ── 2. Median-based robust filtering ──────────────────────────────────────
    median = np.median(points, axis=0)

    if not np.isfinite(median).all():
        # extreme corner case
        return np.mean(points, axis=0)

    dists = np.linalg.norm(points - median, axis=1)

    med_dist = np.median(dists)
    mad = np.median(np.abs(dists - med_dist))

    # ── 3. Degenerate case ────────────────────────────────────────────────────
    if mad == 0 or not np.isfinite(mad):
        return median

    mask = dists <= k * mad
    inliers = points[mask]

    # ── 4. Fallback if filtering removed everything ───────────────────────────
    if len(inliers) == 0:
        return median

    return np.mean(inliers, axis=0)


def aggregate(out_folder: Path):
    poi_out = POI_Global(itk_coords=False)
    poi_out_mean = POI_Global(itk_coords=False)

    # collect points per landmark id
    points_by_id = defaultdict(list)
    poi_in = None
    for idx, file in enumerate(out_folder.glob("*.mrk.json"), 1):
        poi_in = POI_Global.load(file)  #
        assert poi_in.itk_coords == poi_out.itk_coords
        for k1, k2, v in poi_in.items():
            lid = k1 * 100 + k2
            poi_out[idx, lid] = v
            points_by_id[lid].append(v)

    # compute robust representative per landmark
    for lid, pts in points_by_id.items():
        pts = np.asarray(pts)
        poi_out_mean[lid // 100, lid % 100] = robust_mean(pts)
    if poi_in is None:
        return
    poi_out_mean.info = poi_in.info  # type: ignore
    (out_voting / out_folder.name).mkdir(exist_ok=True, parents=True)
    poi_out.save_mrk(out_voting / out_folder.name / "all.mrk.json", split_by_subregion=True)

    poi_out_mean.save_mrk(out_voting / out_folder.name / "mean.mrk.json")


def compute_agreement_scores(all_mrk: Path, mean_mrk: Path):
    """
    Returns per-landmark and global agreement scores.
    """
    poi_all = POI_Global.load(all_mrk)
    poi_mean = POI_Global.load(mean_mrk)

    dists_by_lid = defaultdict(list)
    # collect distances
    for e, k, v in poi_all.items():
        k1 = k // 100
        k2 = k % 100
        lid = k
        # lid = k1 * 10 + k2
        if (k1, k2) not in poi_mean:
            continue
        v_mean = poi_mean[k1, k2]
        d = np.linalg.norm(np.asarray(v) - np.asarray(v_mean))
        dists_by_lid[lid].append(d)
    # summarize per landmark
    rows = []
    for lid, dists in dists_by_lid.items():
        dists = np.asarray(dists)
        rows.append(
            {
                "lid": lid,
                "mean_dist": dists.mean(),
                "median_dist": np.median(dists),
                "p95_dist": np.percentile(dists, 95),
                "std_dist": dists.std(),
            }
        )

    df_lid = pd.DataFrame(rows)
    # global scores
    global_scores = {
        "mean_dist": df_lid["mean_dist"].mean(),
        "median_dist": df_lid["median_dist"].median(),
        "p95_dist": df_lid["p95_dist"].mean(),
        "std_dist": df_lid["std_dist"].mean(),
    }

    return df_lid, global_scores


def evaluate_all_experiments(out_voting: Path):
    rows = []
    per_lid_tables = {}

    for exp in sorted(out_voting.iterdir()):
        if "_" not in exp.name:
            continue
        logger.on_log(exp)
        mean_mrk = exp / "mean.mrk.json"
        all_mrk = exp / "all.mrk.json"
        if not mean_mrk.exists() or not all_mrk.exists():
            continue

        df_lid, scores = compute_agreement_scores(all_mrk, mean_mrk)

        scores["experiment"] = exp.name  # type: ignore
        rows.append(scores)
        per_lid_tables[exp.name] = df_lid

    df_global = pd.DataFrame(rows).set_index("experiment")
    return df_global, per_lid_tables


def compute_pvalues(per_lid_tables, baseline_key, score="mean_dist"):
    base = per_lid_tables[baseline_key].set_index("lid")[score]

    rows = []
    for key, df in per_lid_tables.items():
        if key == baseline_key:
            continue

        cur = df.set_index("lid")[score]
        common = base.index.intersection(cur.index)

        b = base.loc[common]
        c = cur.loc[common]

        # choose test
        try:
            stat, p = wilcoxon(b, c)
            test = "wilcoxon"
        except ValueError:
            stat, p = ttest_rel(b, c)
            test = "ttest"

        rows.append(
            {
                "experiment": key,
                "score": score,
                "p_value": p,
                "test": test,
            }
        )

    return pd.DataFrame(rows).set_index("experiment")


def plot_score(df, score, baseline=None):
    fig = px.bar(
        df.reset_index(),
        x="experiment",
        y=score,
        title=f"Ablation Study – {score}",
    )

    if baseline is not None:
        y0 = df.loc[baseline, score]
        fig.add_hline(
            y=y0,
            line_dash="dash",
            annotation_text="baseline",
            annotation_position="top left",
        )

    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_title="Distance [mm]",
        template="plotly_white",
    )
    fig.write_image(out_voting / f"boxplot_{score}.png")


def parse_experiment_name(name: str):
    """
    'lr-0.01_dice-0.1' → {'lr': 0.01, 'dice': 0.1}
    """
    params = {}
    for part in name.split("_"):
        if "-" not in part:
            continue
        k, v = part.split("-", 1)
        try:
            v = float(v)
            if v.is_integer():
                v = int(v)
        except ValueError:
            pass
        params[k] = v
    return params


def diff_label(exp_params, base_params):
    """
    Returns compact label showing only changed parameters.
    """
    diffs = []
    for k, v in exp_params.items():
        if k not in base_params:
            continue
        if base_params[k] != v:
            diffs.append(f"{k}={v}")

    if not diffs:
        return "baseline"

    return ",".join(diffs)


def file_creation_time(path: Path) -> float:
    """
    Returns creation time in seconds since epoch.
    On Linux, this is metadata change time (still monotonic enough for deltas).
    """
    return os.path.getctime(path)


def compute_run_times(paths: list[Path], drop_n=3):
    """
    paths: list of output files returned by run_all (in execution order)
    drop_n: number of largest outliers to remove
    """
    times = np.array([file_creation_time(p) for p in paths])

    # duration per run = difference between consecutive creations
    durations = np.diff(times)

    if len(durations) <= drop_n:
        raise ValueError("Not enough runs to drop outliers", len(durations))

    # drop largest outliers
    durations_sorted = np.sort(durations)
    trimmed = durations_sorted[:-drop_n]

    return {
        "time mean [s]": trimmed.mean(),
        "time median[s]": np.median(trimmed),
        "time std [s]": trimmed.std(),
        "all_durations": durations,
        "used_durations": trimmed,
    }


def change_to_label(change: dict):
    if not change:
        return "baseline"
    return ",".join(f"{k}={v}" for k, v in change.items())


if __name__ == "__main__":
    derivatives_folder = "derivatives-VIBESeg-12-"
    local_folder = Path("/media/data/robert/dataset-myelom/dataset-myelom")
    if not local_folder.exists():
        local_folder = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-myelom")

    assert local_folder.exists(), local_folder
    out_mrk = path_train_poi.parent / "mrk"
    to_mrk(path_train_poi, out_mrk)

    target = (
        local_folder
        / derivatives_folder
        / "CTFU04045"
        / "ses-20220303"
        / "sub-CTFU04045_ses-20220303_sequ-204_seg-VIBESeg-11-lr_msk.nii.gz"
    )
    fetch(target, True)
    default_dict = {
        "lr": 0.001,
        "max_steps": 1500,
        "min_delta": 0.000001,
        "pyramid_levels": 4,
        "coarsest_level": 3,
        "finest_level": 0,
        "be": 0.00001,
        "mse": 1,
        "dice": 0.01,
        "com": 0.001,
    }
    changes = [
        {},
        ###########
        {"lr": 0.1},
        {"lr": 0.01},
        # {"lr": 0.001},
        {"lr": 0.0001},
        {"lr": 0.00001},
        ###########
        {"min_delta": 0.01},
        {"min_delta": 0.001},
        {"min_delta": 0.0001},
        # {"min_delta": 0.000001},
        {"min_delta": 0.0000001},
        {"min_delta": 0.00000001},
        {"min_delta": 0.000000001},
        ###########
        # {"pyramid_levels": 4, "coarsest_level": 3},
        {"pyramid_levels": 5, "coarsest_level": 4},
        {"pyramid_levels": 3, "coarsest_level": 2},
        ##########
        {"be": 0.0},
        {"be": 0.001},
        # {"be": 0.00001},
        {"be": 0.0000001},
        {"be": 0.000000001},
        {"be": 0.1},
        ##########
        # {"mse": 1},
        {"mse": 0},
        {"mse": 10},
        {"mse": 0.1},
        #########
        {"dice": 0.1},
        {"dice": 0},
        {"dice": 0.001},
        #########
        {"com": 0},
        {"com": 0.1},
        {"com": 0.0000000001},
        #########
        {"dice": 0, "com": 0},
    ]
    time = {}
    for c in changes:
        di = default_dict.copy()
        for k, v in c.items():
            di[k] = v
        key = "_".join(f"{k}-{v}" for k, v in di.items())
        target = to_nii(target, True)
        out_folder = path_train_poi.parent / f"treg_{key}"
        out_folder.mkdir(exist_ok=True)

        out_file = run_all(out_mrk, target, out_folder, **di, no_inference=True)
        if out_file is None:
            continue
        stats = compute_run_times(out_file, drop_n=3)
        # print(str(c))
        # print(f"Average runtime (trimmed): {stats['mean']:.2f} s")
        # print(f"Median runtime: {stats['median']:.2f} s")
        # print(f"Std runtime: {stats['std']:.2f} s")
        del stats["all_durations"]
        del stats["used_durations"]
        time["treg_" + key] = stats
        aggregate(out_folder)
        # break

    df_global, per_lid = evaluate_all_experiments(out_voting)
    # TODO replace the name with only what changed in the dict. Remove _ and -
    baseline = df_global.index[0]  # or explicit name

    df_p = compute_pvalues(per_lid, baseline, score="mean_dist")
    df_final = df_global.join(df_p[["p_value"]])

    ###
    rename_map = {}
    df_index = list(df_final.index)

    for i, change in enumerate(changes):
        di = default_dict.copy()
        for k, v in change.items():
            di[k] = v
        key = "_".join(f"{k}-{v}" for k, v in di.items())

        old_name = "treg_" + key
        # assert old_name in df_index, (old_name, df_index)
        new_name = change_to_label(change)
        rename_map[old_name] = new_name
    rename_map["treg"] = "baseline-"
    rename_map["treg10"] = "min_delta=0.000001"
    rename_map["treg100"] = "min_delta=0.0000001"
    rename_map["treg1000"] = "min_delta=0.00000001"
    df_time = pd.DataFrame.from_dict(time, orient="index")
    print(df_time)
    # Now join with df_final safely
    df_final = df_final.join(df_time, how="left")
    # rename_map["baseline"] = "treg"
    # rename_map["baseline"] = "treg"
    df_final = df_final.rename(index=rename_map)
    baseline = rename_map.get(baseline, baseline)
    ###

    print(df_final.round(5).sort_values("mean_dist"))
    df_final.to_excel(out_voting / "summery.xlsx")
    for score in ["mean_dist", "median_dist", "p95_dist"]:
        plot_score(df_final.sort_values("mean_dist"), score, baseline)
