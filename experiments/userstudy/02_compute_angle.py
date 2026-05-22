import sys
from pathlib import Path

import numpy as np
import pandas as pd
from TPTBox import POI_Global

out = str(Path(__file__).parent.parent.parent)
sys.path.append(out)


from constants import POI_MAP, out_userstudy, raters_all
from treg import angle
from treg.angle import compute_angles

# raters_all = [
#    "Julius/Julius_V1",
#    "Julius/Julius_V2",
#    "Julius/Julius_V3",
#    "Leon",
#    "Philipp",
#    # "Robert_Model",
#    "Robert_Model_1",
#    "Robert_Model_2",
#    "Robert_Model_3",
#    "SSM_Model_1",
# ]

copy_from_rules_based = {
    "Robert_Model_1": "Julius/Julius_V1",
    "Robert_Model_2": "Julius/Julius_V2",
    "Robert_Model_3": "Julius/Julius_V3",
}


def export_angles_to_excel(raters, base_dir="pois_mrk", out_xlsx=out_userstudy / "angles_all_raters.xlsx"):

    ref = {}
    rows = []

    for rater in raters:
        rater_dir = Path(base_dir) / rater
        ref[rater] = {}
        for f in sorted(rater_dir.glob("*.mrk.json")):
            print(f)
            if "Veerman_Model" in rater:
                s = str(f.absolute()).replace("mrk.json", "json")
                poi_original = POI_Global.load(s)
                assert "angles" in poi_original.info, (s, poi_original.info)
            else:
                poi_original = POI_Global.load(f)

            ref[rater][f.name] = poi_original
            # if rater in copy_from_rules_based:
            #    rater_other = copy_from_rules_based[rater]
            #    poi_other = ref[rater_other][f.name]
            #    for s in {
            #        "FHC",
            #        "FNC",
            #        "FADP",
            #        "FAAP",
            #    }:  # "FADP", "FAAP"
            #        a, b = POI_MAP[s]
            #        poi_original[a, b] = poi_other[a, b]
            if "angles" in poi_original.info:
                angles = poi_original.info["angles"]
            else:
                assert "Veerman_Model" not in rater, (rater, poi_original.info)
                print(rater)
                angles, _, _ = compute_angles(poi_original, lagacy=True)

            row = {"rater": rater, "file": f.name}

            # add all angles as columns
            row.update(angles)
            rows.append(row)
    # Build dataframe
    df = pd.DataFrame(rows)

    # Sort by file, then rater (stable & readable)
    df = df.sort_values(by=["file", "rater"]).reset_index(drop=True)

    # Save Excel
    df.to_excel(out_xlsx, index=False)

    return df


if __name__ == "__main__":
    df = export_angles_to_excel(raters=raters_all, base_dir="pois_mrk", out_xlsx=out_userstudy / "all_angles.xlsx")

    print(df.head())
    poi = POI_Global.load(
        Path(
            "pois_mrk",
            "Julius/Julius_V1",
            "sub-CTFU00159_ses-20140801_side-left.mrk.json",
        )
    )
    angles, _, _ = compute_angles(
        poi,
        "sub-CTFU00159_ses-20140801_side-left_desc-annotated.mrk.json",
        "sub-CTFU00159_ses-20140801_side-left_desc-annotated_frame-orto.mrk.json",
    )
