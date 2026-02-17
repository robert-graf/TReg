import sys
from pathlib import Path

import numpy as np
import pandas as pd
from TPTBox import BIDS_FILE, POI_Global
from TPTBox.core.vert_constants import Lower_Body

out = str(Path(__file__).parent.parent.parent)
sys.path.append(out)

from constants import POI_MAP, flips_model, mapp_models_filp, out_userstudy, path_annotator_poi, path_mrk, path_train_poi, raters_all

target = "sub-CTFU04045_ses-20220303_sequ-204_mod-ct_seg-fov6_msk.nii.gz"

path_train_poi
