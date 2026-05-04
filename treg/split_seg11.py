import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from TPTBox import BIDS_FILE, NII, BIDS_Global_info, Print_Logger, to_nii
from TPTBox.core.vert_constants import Full_Body_Instance
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
ignore_label = 99


def get_nii_lr(status):
    if "nii_lr" in status:
        return status["nii_lr"]
    a: BIDS_FILE = status["out_file_lr"]
    if a.exists():
        return to_nii(a, True)
    return get_nii(status)


def get_nii(status) -> NII:
    if isinstance(status["nii"], BIDS_FILE):
        print("load", status["nii"])
    nii = to_nii(status["nii"], True)
    nii.max()
    status["nii"] = nii
    return nii


def split_step_one(seg10: NII, roi_lr: NII):
    split = {}
    status = {"unique": seg10.unique()}
    ret = True
    for x in Full_Body_Instance:
        if "right" not in x.name and x.value not in [Full_Body_Instance.spleen.value, Full_Body_Instance.liver.value]:
            continue
        if x.value not in status["unique"] and x.value not in [61, 60]:
            continue
        if str(x.value) in split:
            continue
        ret = False
    if ret:
        return

    nii = seg10.copy()
    lr = roi_lr
    left = lr.extract_label(2)
    right = lr.extract_label(1)
    for x in tqdm(
        [
            Full_Body_Instance.femur_right,
            Full_Body_Instance.femur_left,
            Full_Body_Instance.patella_right,
            Full_Body_Instance.patella_left,
            Full_Body_Instance.tibia_right,
            Full_Body_Instance.tibia_left,
            Full_Body_Instance.fibula_right,
            Full_Body_Instance.fibula_left,
        ],
        desc="split",
    ):
        if "right" not in x.name and x.value not in [Full_Body_Instance.spleen.value, Full_Body_Instance.liver.value]:
            continue
        if x.value not in status["unique"] and x.value not in [61, 60]:
            continue
        if str(x.value) in split:
            continue
        # if x.value == 4:
        #    break
        struct = nii.extract_label(x.value).get_connected_components(1)
        trigger_step_two = False
        has_left = False
        has_right = False
        out = []
        for i in struct.unique():
            cc = struct.extract_label(i)

            l = cc * left
            l_sum = l.sum()
            r = cc * right
            r_sum = r.sum()
            s = r_sum + l_sum
            out.append((int(r_sum), int(l_sum)))
            if s <= 20:
                nii[cc != 0] = ignore_label
                continue
            th = 0.8 if x.value not in [Full_Body_Instance.spleen.value, Full_Body_Instance.liver.value] else 0
            if max(l_sum, r_sum) / s < th:
                trigger_step_two = True
                print("TRIGGER - second step")
            if l_sum > r_sum:
                has_left = True
                if x.value == Full_Body_Instance.liver.value:
                    nii[cc != 0] = Full_Body_Instance.spleen.value
                elif x.value == Full_Body_Instance.spleen.value:
                    pass
                else:
                    nii[cc != 0] = 100 + x.value
            else:
                has_right = True
                # x.value not in [Full_Body_Instance.spleen.value, Full_Body_Instance.liver.value]
                if x.value == Full_Body_Instance.spleen.value:
                    nii[cc != 0] = Full_Body_Instance.liver.value
                elif x.value == Full_Body_Instance.liver.value:
                    pass
                else:
                    nii[cc != 0] = x.value

        if not (has_right and has_left):  # and x.value not in [Full_Body_Instance.spleen.value, Full_Body_Instance.liver.value]:
            trigger_step_two = True
        if trigger_step_two:
            nii = split_step_two(status, nii, x, out)
            print("TRIGGER One sided - second step")
        split[x.value] = out
    # status["split"] = split
    return nii


def split_step_two(status, nii: NII, idx: Full_Body_Instance, out: list):
    i = [idx.value, idx.value + 100]
    left_idx = idx.value + 100
    if idx.value in [Full_Body_Instance.liver.value]:
        return nii
    if idx.value in [Full_Body_Instance.spleen.value]:
        i = [Full_Body_Instance.spleen.value, Full_Body_Instance.liver.value]
        left_idx = Full_Body_Instance.spleen.value
    print("split_step_two", idx)
    # get reference for infect
    target = nii.extract_label(i)

    if idx in [Full_Body_Instance.pelvis_left, Full_Body_Instance.pelvis_right]:
        ct = status["ct_file"]
        f = ct.get_changed_path("nii.gz", "msk", parent="derivatives-total-old", info={"seg": "totalSeg"})
        ref = (
            to_nii(f, True)
            .resample_from_to_(nii)
            .map_labels(
                {
                    77: Full_Body_Instance.pelvis_left.value,
                    78: Full_Body_Instance.pelvis_right.value,
                }
            )
        ) * target
    else:
        raise NotImplementedError("100")
        # ref = get_100(status, nii) * target
    u = ref.unique()
    print(u, left_idx)
    if left_idx not in u:
        out.append("split_step_two does not work for this label")
        print("split_step_two does not work for this label")
        return nii
    out.append("split_step_two")
    print("split_step_two", i)
    # infect in s than in all directions
    ref = ref.extract_label(i, keep_label=True)
    ref = ref.infect(target, axis="S", verbose=False).infect(target, verbose=False)
    print(f"{ref.unique()=}")
    nii[target != 0] = ignore_label
    nii[ref != 0] = ref[ref != 0]
    return nii
