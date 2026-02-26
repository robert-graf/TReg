import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
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
        # if u in flips_model:
        #    poi.map_labels_(mapp_models_filp)
        # if "Robert_Model" not in u:
        if mapping1 is None:
            mapping1 = poi.info["label_name"].copy()
            mapping2 = poi.info["label_group_name"].copy()
            name__ = key
        else:
            if mapping1 != poi.info["label_name"]:
                for k, v in mapping1.items():
                    if k not in poi.info["label_name"] or v != poi.info["label_name"][k]:
                        print(
                            k,
                            v,
                            key,
                            name__,
                            k not in poi.info["label_name"],
                            raters_all,
                        )

            assert mapping2 == poi.info["label_group_name"], (
                mapping2,
                poi.info["label_group_name"],
            )
        # else:
        #    mapping = Lower_Body.get_mapping()
        #    assert mapping2 is not None
        #    label_map_full = {}
        #    for k, v in mapping1.items():
        #        # print(k, v)
        #        # print(type(k))
        #        a, b = mapping[v]
        #        c, d = str(k).replace("(", "").replace(")", "").split(",")
        #        label_map_full[a.value, b.value] = int(c), int(d)
        #    poi.map_labels_(label_map_full)
        #    poi.info["label_name"] = mapping1
        #    poi.info["label_group_name"] = mapping2
        # all_data[u][key] = {"file": i, "poi": poi}
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
):

    weights: dict = {
        "be": be,
        "seg": mse,
        "Dice": dice,
        "Tether": com,
    }
    print(f"{min_delta=}")
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
        if out.exists():
            continue
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
        print(mirror)


def robust_mean(points, k=3.5):
    """
    points: (N, 3) array
    returns: (3,) robust mean
    """
    points = np.asarray(points)

    if len(points) == 1:
        return points[0]

    median = np.median(points, axis=0)
    dists = np.linalg.norm(points - median, axis=1)

    mad = np.median(np.abs(dists - np.median(dists)))
    if mad == 0:
        return median

    mask = dists < k * mad
    return points[mask].mean(axis=0)


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
            lid = k1 * 10 + k2
            poi_out[idx, lid] = v
            points_by_id[lid].append(v)

    # compute robust representative per landmark
    for lid, pts in points_by_id.items():
        pts = np.asarray(pts)
        poi_out_mean[lid // 10, lid % 10] = robust_mean(pts)
    poi_out_mean.info = poi_in.info  # type: ignore
    (out_voting / out_folder.name).mkdir(exist_ok=True, parents=True)
    poi_out.save_mrk(out_voting / out_folder.name / "all.mrk.json", split_by_subregion=True)

    poi_out_mean.save_mrk(out_voting / out_folder.name / "mean.mrk.json")


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
        {"min_delta": 0.000001},
        # {"min_delta": 0.001},
        {"min_delta": 0.000000001},
        ###########
        # {"pyramid_levels": 4, "coarsest_level": 3},
        {"pyramid_levels": 5, "coarsest_level": 4},
        {"pyramid_levels": 3, "coarsest_level": 2},
        ##########
        {"be": 0.0},
        {"be": 0.001},
        {"be": 0.00001},
        {"be": 0.0000001},
        {"be": 0.000000001},
        {"be": 0.1},
        ##########
        {"mse": 1},
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
    for c in changes:
        di = default_dict.copy()
        for k, v in c.items():
            di[k] = v
        key = "_".join(f"{k}-{v}" for k, v in di.items())
        target = to_nii(target, True)
        out_folder = path_train_poi.parent / f"treg_{key}"
        out_folder.mkdir(exist_ok=True)
        run_all(out_mrk, target, out_folder, **di)
        aggregate(out_folder)
