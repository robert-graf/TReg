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

from constants import out_voting, path_train_poi, raters_all
from treg.angle import compute_angles

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
leg_ids_other = [
    Full_Body_Instance.femur_right,
    Full_Body_Instance.patella_right,
    Full_Body_Instance.tibia_right,
    Full_Body_Instance.fibula_right,
]


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
    key = "_".join(f"{k}-{v}" for k, v in default_dict.items())
    target = to_nii(target, True)
    out_folder = path_train_poi.parent / f"treg_{key}"
    right = target.extract_label(leg_ids_other, True)
    target = target.extract_label(leg_ids, True)
    # poi_out = out_voting / out_folder.name / "all.mrk.json"
    poi_out_mean = out_voting / out_folder.name / "mean.mrk.json"

    crop = target.compute_crop(0, 100)
    target = target.apply_crop(crop)
    poi = POI_Global.load(poi_out_mean)
    target.save("data/sub-atlas_seg-TotalVibe-12_msk.nii.gz")
    right.apply_crop(right.compute_crop(0, 100)).save("data/sub-right_seg-TotalVibe-12_msk.nii.gz")

    poi.save("data/sub-atlas_seg-poi_poi.json")
    poi.save_mrk("data/sub-atlas_seg-poi_poi.mrk.json")
    compute_angles(poi, "data/lines.mrk.json")
