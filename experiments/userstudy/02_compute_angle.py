import sys
from pathlib import Path

import numpy as np
import pandas as pd
from TPTBox import POI_Global

out = str(Path(__file__).parent.parent.parent)
sys.path.append(out)


from constants import out_userstudy, raters_all
from treg import angle
from treg.angle import compute_angles


def export_angles_to_excel(raters, base_dir="pois_mrk", out_xlsx=out_userstudy / "angles_all_raters.xlsx"):
    rows = []

    for rater in raters:
        rater_dir = Path(base_dir) / rater
        for f in sorted(rater_dir.glob("*.mrk.json")):
            poi_original = POI_Global.load(f)
            angles, _, _ = compute_angles(poi_original)

            row = {
                "rater": rater,
                "file": f.name,
            }

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
