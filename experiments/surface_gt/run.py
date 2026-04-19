import sys
from pathlib import Path

from TPTBox import BIDS_FILE, NII, POI_Global, Print_Logger, to_nii
from TPTBox.registration import Template_Registration

out = str(Path(__file__).parent.parent.parent)
sys.path.append(out)
from experiments.generate_refrence.reg import derivatives_folder, fetch, leg_ids, local_folder, mapping_mirror

logger = Print_Logger()


def run_all(
    file: Path,
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
    mirror=False,
):

    weights: dict = {"be": be, "seg": mse, "Dice": dice, "Tether": com}

    # poi = POI_Global.load(file)
    tamplate = to_nii(file, True)
    bf = BIDS_FILE(file, local_folder)
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
        print("not exits")
        return

    out = out_folder / file.name
    if out.exists():
        print("already exits", out)
        return
    if no_inference:
        print("no_inference")
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
    atlas_reg = reg.transform_nii(tamplate, allow_only_same_grid_as_moving=False)  # Transferring the atlas
    # atlas_reg.info = poi.info
    atlas_reg.save(out)


if __name__ == "__main__":
    target = (
        local_folder
        / derivatives_folder
        / "CTFU04045"
        / "ses-20220303"
        / "sub-CTFU04045_ses-20220303_sequ-204_seg-VIBESeg-11-lr_msk.nii.gz"
    )
    fetch(target, True)
    target = to_nii(target, True)
    files = [
        "/media/data/robert/code/TReg/experiments/surface_gt/sub-CTFU03127_ses-20171206_sequ-202_seg-fov2-reg-julius_V2_msk.nii.gz.seg.nrrd",
        "/media/data/robert/code/TReg/experiments/surface_gt/sub-CTFU05100_ses-20210421_sequ-203_seg-fov2-reg-julius_V2_msk.nii.gz.seg.nrrd",
        "/media/data/robert/code/TReg/experiments/surface_gt/sub-MM00019_ses-20160420_sequ-3_seg-fov2-reg-julius_V2_msk.nii.gz.seg.nrrd",
    ]
    out_folder = Path(__file__).parent / "out"
    for f in files:
        run_all(Path(f), target, out_folder)
