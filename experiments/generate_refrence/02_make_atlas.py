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

from constants import atlas_path, default_dict, out_voting, path_train_poi, raters_all, target_path
from experiments.generate_refrence.reg import leg_ids, leg_ids_other, mapping_mirror, to_mrk
from treg.angle import compute_angles

logger = Print_Logger()


if __name__ == "__main__":
    out_mrk = path_train_poi.parent / "mrk"
    to_mrk(path_train_poi, out_mrk)

    target = target_path

    target = to_nii(target, True)
    out_folder = path_train_poi.parent / "treg_"
    right = target.extract_label(leg_ids_other, True)
    target = target.extract_label(leg_ids, True)
    # poi_out = out_voting / out_folder.name / "all.mrk.json"
    poi_out_mean = out_voting / out_folder.name / "mean.mrk.json"

    crop = target.compute_crop(0, 100)
    target = target.apply_crop(crop)
    poi = POI_Global.load(poi_out_mean)
    target.save("data/sub-atlas_seg-VIBESeg-12_msk.nii.gz")
    right.apply_crop(right.compute_crop(0, 100)).save("data/sub-right_seg-VIBESeg-12_msk.nii.gz")

    poi.save("data/sub-atlas_seg-poi_poi.json")
    poi.save_mrk("data/sub-atlas_seg-poi_poi.mrk.json")
    compute_angles(poi, "data/lines.mrk.json")

    to_nii(atlas_path, True).save("data/sub-atlas_seg-subregion_msk.nii.gz")
