from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from TPTBox import NII, Image_Reference, POI_Global, Print_Logger, to_nii
from TPTBox.core.vert_constants import Full_Body_Instance
from TPTBox.registration import Template_Registration

from treg.angle import compute_angles
from treg.veerman_rules_based import run_single_case

logger = Print_Logger()

# IDs from the segmentation model, for you own segmentation update it with the label ID you want to use.
leg_ids = [
    Full_Body_Instance.femur_left,  # <- you can use integer for the labels here.
    Full_Body_Instance.patella_left,
    Full_Body_Instance.tibia_left,
    Full_Body_Instance.fibula_left,
]
# Mirrors the ids for left and right legs. Update for you own. If not needed, make an empty dict.
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
weights_default = {"be": 0.00001, "seg": 1, "Dice": [0.01, 0.01, 0.01, 0.1], "Tether": 0.001}


def resolve_device(ddevice: Literal["cpu", "cuda", "mps"], gpu: int = 0) -> torch.device:
    if ddevice == "cuda":
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu)
            return torch.device(f"cuda:{gpu}")
        else:
            print("⚠️ CUDA requested but not available → falling back to CPU")

    if ddevice == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        else:
            print("⚠️ MPS requested but not available → falling back to CPU")

    return torch.device("cpu")


def bin_mask(path: Path, override=False):
    logger.on_log("load mask")
    bin_msk = to_nii(path, True)
    if len(bin_msk.shape) == 4:
        bin_msk = bin_msk.set_array(bin_msk.get_array().sum(-1)).clamp(0, 1)
        if override:
            bin_msk.set_dtype_("smallest_uint").save(path)
    return bin_msk


def run_all(
    files: dict,
    sides: list,
    # Atlas
    atlas_seg_file: str | Path = "data/leg/sub-atlas_seg-VIBESeg-12_msk.nii.gz",  # default is a left leg
    atlas_file: str | Path = "data/leg/sub-atlas_seg-poi_poi.json",
    atlas_seg_subdivided_file: str | Path | None = "data/leg/sub-atlas_seg-subregion_msk.nii.gz",
    # Parameters
    lr: float = 0.001,
    max_steps: int = 1500,
    min_delta: float = 0.000001,
    pyramid_levels: int = 4,
    coarsest_level: int = 3,
    finest_level: int = 0,
    weights: dict | None = None,
    ids=leg_ids,  # TODO rename
    mapping_mirror=mapping_mirror,
    ddevice: Literal["cpu", "cuda", "mps"] = "cuda",
    gpu=0,
):
    if weights is None:
        weights = weights_default
    binary_msk = bin_mask(files["bin_msk"]) if "bin_msk" in files else None

    for side in sides:
        seg = files["seg"]
        logger.prefix = "TREG"
        logger.on_log(seg.name, f"{side=}")
        reg = treg_one_leg(
            target=seg,
            target_out_poi=files[side]["target_out_poi"],
            target_out_subdivided=files[side]["target_out_subdivided"],
            atlas_seg_file=atlas_seg_file,
            atlas_poi_file=atlas_file,
            atlas_seg_subdivided_file=atlas_seg_subdivided_file,
            mirror=files[side]["mirror"],
            lr=lr,
            max_steps=max_steps,
            min_delta=min_delta,
            pyramid_levels=pyramid_levels,
            coarsest_level=coarsest_level,
            finest_level=finest_level,
            weights=weights,
            binary_msk=binary_msk,
            ids=ids,  # TODO rename
            mapping_mirror=mapping_mirror,
            ddevice=ddevice,
            gpu=gpu,
        )
        logger.prefix = "POI"
        logger.on_log("save poi excel")
        compute_angles_excel(
            files[side]["target_out_poi"],
            files[side]["target_out_angle"],
        )
        logger.prefix = "veerman"
        logger.on_log("save veerman csv")
        run_single_case(
            nii=files[side]["target_out_subdivided"],
            stl_folder=Path(files[side]["target_out_subdivided"]).parent / "stl",
            side="L" if side.lower() == "left" else "R",
            output_csv=str(files[side]["target_out_angle2"]).split(".")[0] + ".csv",
        )
        return reg


def treg_one_leg(
    # new target
    target: Image_Reference,
    target_out_poi: str | Path,
    target_out_subdivided: str | Path,
    # Atlas
    atlas_seg_file: Image_Reference,
    atlas_poi_file: Image_Reference,
    atlas_seg_subdivided_file: Image_Reference | None,
    # parameter
    lr,
    max_steps,
    min_delta,
    pyramid_levels,
    coarsest_level,
    finest_level,
    weights=weights_default,
    mirror=False,
    binary_msk: Image_Reference | None = None,
    ids=leg_ids,  # TODO rename
    mapping_mirror=mapping_mirror,
    ddevice: Literal["cpu", "cuda", "mps"] = "cuda",
    gpu=0,
):

    if binary_msk is not None:
        mask_nii = to_nii(binary_msk, True)
        if len(mask_nii.shape) == 4:
            mask_nii.set_array_(mask_nii.get_array().sum(-1))
    else:
        mask_nii = None

    # load
    moving_img = to_nii(atlas_seg_file, True)
    target_nii = to_nii(target, True)
    # change label for mirroring
    if mirror:
        target_nii = target_nii.map_labels(mapping_mirror)
    # Limit to only used labels
    print(target_nii.unique(), moving_img.unique())
    seg = target_nii.extract_label(ids, True)
    print("unique", seg.unique(), moving_img.unique())

    # Run Template_Registration
    reg = Template_Registration(
        seg,  # Target segmentation
        moving_img.extract_label(ids, True),  # Starting Atlas Segmentation (not the split one)
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
        gpu=gpu,
        ddevice=ddevice,
        fixed_mask=mask_nii,
    )
    if atlas_poi_file is not None:
        # Transfer atlas to target
        logger.print("Transfer atlas to target")
        poi_in = POI_Global.load(atlas_poi_file)
        atlas_reg = reg.transform_poi(poi_in)  # Transferring the atlas points
        atlas_reg.info = poi_in.info
        target_out_poi = Path(target_out_poi)
        target_out_poi.parent.mkdir(exist_ok=True)
        atlas_reg.to_global().save_mrk(target_out_poi)
        atlas_reg.save(target_out_poi)

    if atlas_seg_subdivided_file is not None:
        n = reg.transform_nii(
            to_nii(atlas_seg_subdivided_file, True), allow_only_same_grid_as_moving=False
        )  # Transferring the atlas subdivisions
        s = seg.extract_label([13, 15, 113, 115], keep_label=True) % 100
        s.map_labels_({13: 7, 15: 8})
        a = (1 - mask_nii).dilate_msk(2) if mask_nii is not None else 0
        n = n.infect_(s) * (seg + a).clamp(0, 1)
        s = s.erode_msk_euclid(3)
        n[s != 0] = s[s != 0]
        n.save(target_out_subdivided)
    return reg


def compute_angles_excel(target_out_poi, target_out_angle):
    atlas_reg = POI_Global.load(target_out_poi)
    angle_dict, _, _ = compute_angles(atlas_reg.to_global(), target_out_angle)
    df = pd.DataFrame([angle_dict])
    df.to_excel(str(target_out_angle).split(".")[0] + ".xlsx")
