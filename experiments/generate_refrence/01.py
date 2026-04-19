import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from TPTBox import POI_Global, to_nii

out = str(Path(__file__).parent.parent.parent)
sys.path.append(out)

from constants import out_voting, path_train_poi, target_path
from experiments.generate_refrence.reg import aggregate, run_all_to_atlas, to_mrk

if __name__ == "__main__":
    out_mrk = path_train_poi.parent / "mrk"
    to_mrk(path_train_poi, out_mrk)
    target = target_path
    time = {}
    key = "_".join(f"{k}-{v}" for k, v in {}.items() if v != "SVFFD")
    target = to_nii(target, True)
    out_folder = path_train_poi.parent / f"treg_{key}"
    out_folder.mkdir(exist_ok=True)
    run_all_to_atlas(out_mrk, target, out_folder)
    aggregate(out_folder)
