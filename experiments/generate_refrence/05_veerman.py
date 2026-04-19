import sys
from pathlib import Path

import pandas as pd
from TPTBox import POI, POI_Global, to_nii

in_folder = str(Path(__file__).parent.parent.parent)
sys.path.append(in_folder)

from constants import atlas_path, out_userstudy, path_annotator_poi, target_path
from experiments.generate_refrence.reg import (
    fetch,
)
from treg.veerman_rules_based import run_single_case


def change_to_label(change: dict):
    if not change:
        return "baseline"
    return ",".join(f"{k}={v}" for k, v in change.items())


if __name__ == "__main__":
    target = target_path
    fetch(target, True)
    target = to_nii(target, True)
    atlas = to_nii(atlas_path, True)
    atlas_poi = POI_Global.load("/media/data/robert/code/TReg/results/voting/treg_/mean.mrk.json")

    rows = []  # <-- collect all results here

    for model_idx in range(1, 2):
        key = str(model_idx)
        in_folder = path_annotator_poi / f"Robert_Model_{model_idx}"
        out_folder = path_annotator_poi / f"Veerman_Model_{model_idx}"
        out_folder.mkdir(exist_ok=True)

        for file in in_folder.iterdir():
            if "_desc-target" in file.name:
                continue
            if ".mrk.json" in file.name:
                continue

            left = "LEFT" not in file.name.upper()

            out_stl = out_folder / "stl" / file.name.split(".")[0]
            out = out_folder / (file.name.split(".")[0] + ".json")

            if out.exists():
                poi = POI.load(out)
                info = poi.info
                angles = info.get("angles", {})
                row = {
                    "rater": f"Model_{model_idx}",
                    "file": file.name,
                    **angles,  # unpack all computed angles into columns
                }

                rows.append(row)
                continue

            out_stl.mkdir(exist_ok=True, parents=True)

            subreg_path = str(file).split(".")[0].replace("_poi", "_msk") + ".nii.gz"
            subreg_nii = to_nii(subreg_path, True)

            poi = run_single_case(subreg_nii, out_stl, "L" if left else "R", verbose=False)
            poi.to_local(subreg_nii).save(out)

            info = poi.info
            angles = info.get("angles", {})

            # ---- Build one row (adapt keys to your structure) ----
            row = {
                "rater": f"Model_{model_idx}",
                "file": file.name,
                **angles,  # unpack all computed angles into columns
            }

            rows.append(row)

    # ---- Create table ----
    df = pd.DataFrame(rows)

    # Optional: sort columns nicely
    cols = ["rater", "file", *sorted([c for c in df.columns if c not in ["rater", "file"]])]

    # df = df[cols]

    # ---- Save ----
    df.round(3).to_excel(out_userstudy / "all_angles_veerman.xlsx", index=False)
    print(len(df))
    print(df.head())
