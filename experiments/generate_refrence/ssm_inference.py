import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial import cKDTree
from skimage import measure
from sklearn.decomposition import PCA
from TPTBox import BIDS_FILE, NII, POI_Global, to_nii

out = str(Path(__file__).parent.parent.parent)
sys.path.append(out)

from constants import (
    POI_MAP,
    flips_model,
    mapp_models_filp,
    out_atlas,
    out_userstudy,
    out_voting,
    path_annotator_poi,
    path_mrk,
    path_train_poi,
    raters_all,
)
from experiments.generate_refrence.reg import derivatives_folder, fetch, local_folder, run
from experiments.generate_refrence.ssm_train import SurfaceSSM, infer_pois


def _get_all(folder: Path = Path(path_annotator_poi / "Robert_Model_1")):

    for i in (folder).iterdir():
        # poi = POI_Global.load(i)
        bf = BIDS_FILE(i, local_folder)
        moving_path = (
            local_folder
            / derivatives_folder
            / str(bf.get("sub"))
            / f"ses-{bf.get('ses')!s}"
            / f"sub-{bf.get('sub')!s}_ses-{bf.get('ses')!s}_sequ-{bf.get('sequ')!s}_seg-VIBESeg-11-lr_msk.nii.gz"
        )
        fetch(moving_path, True)
        yield moving_path, i


if __name__ == "__main__":
    template_poi = POI_Global.load(out_voting / "treg" / "ssm_mean.mrk.json")
    ssm = SurfaceSSM.load(out_atlas / "ssm.pkl")

    for i in range(1, 4):
        for infrence_target, poi_path in _get_all():
            out_path = out_atlas.parent / f"input/userstudy/pois/SSM_Model_{i}/{poi_path.name.split('.')[0]}.mrk.json"
            out_path.parent.mkdir(exist_ok=True, parents=True)
            if out_path.exists():
                continue
            mirror = "LEFT" not in poi_path.name.upper()
            right_leg = False
            atlas = (
                local_folder
                / derivatives_folder
                / "CTFU04045"
                / "ses-20220303"
                / "sub-CTFU04045_ses-20220303_sequ-204_seg-VIBESeg-11-lr_msk.nii.gz"
            )
            inf_nii = to_nii(infrence_target, True)
            reg = run(inf_nii, to_nii(atlas, True), mirror=right_leg)  # for ssm atlas is target and inference is moving
            moved = reg.transform_nii(inf_nii)
            poi = infer_pois(ssm, moved, template_poi)

            poi_out = reg.transform_poi_inverse(poi)
            poi_out.to_global().save_mrk(out_path)
            exit()
