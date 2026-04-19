import sys
from pathlib import Path

import numpy as np
import pandas as pd
from TPTBox import BIDS_FILE, POI_Global
from TPTBox.core.vert_constants import Lower_Body

out = str(Path(__file__).parent.parent.parent)
sys.path.append(out)

from constants import POI_MAP, flips_model, mapp_models_filp, out_userstudy, path_annotator_poi, path_mrk, raters_all


############################################################################
### Reading ###
############################################################################
def main():
    all_data = {u: {} for u in raters_all}
    mapping2 = {
        1: "Femur proximal",
        2: "Femur distal",
        3: "Tibia proximal",
        4: "Tibia distal",
        5: "Patella",
    }  # poi.info["label_name"].copy()
    mapping1 = {
        "(1, 1)": "TGT",
        "(1, 2)": "FHC",
        "(1, 3)": "FNC",
        "(1, 4)": "FAAP",
        "(2, 1)": "FLCD",
        "(2, 2)": "FMCD",
        "(2, 3)": "FLCP",
        "(2, 4)": "FMCP",
        "(2, 5)": "FNP",
        "(2, 6)": "FADP",
        "(2, 7)": "TGPP",
        "(2, 8)": "TGCP",
        "(2, 9)": "FMCPC",
        "(2, 10)": "FLCPC",
        "(2, 11)": "TRMP",
        "(2, 12)": "TRLP",
        "(3, 1)": "TLCL",
        "(3, 2)": "TMCM",
        "(3, 3)": "TKC",
        "(3, 4)": "TLCA",
        "(3, 5)": "TLCP",
        "(3, 6)": "TMCA",
        "(3, 7)": "TMCP",
        "(3, 8)": "TTP",
        "(3, 9)": "TAAP",
        "(3, 10)": "TMIT",
        "(3, 11)": "TLIT",
        "(4, 1)": "FLM",
        "(4, 2)": "TMM",
        "(4, 3)": "TAC",
        "(4, 4)": "TADP",
        "(5, 1)": "PPP",
        "(5, 2)": "PDP",
        "(5, 3)": "PMP",
        "(5, 4)": "PLP",
        "(5, 5)": "PRPP",
        "(5, 6)": "PRDP",
        "(5, 7)": "PRHP",
    }.copy()
    name__ = None

    ## Normalize data so i have not to deal with diffrences in path names.
    for u in raters_all:
        assert (path_annotator_poi / u).exists(), path_annotator_poi / u
        for i in (path_annotator_poi / u).iterdir():
            if "xlsx" in i.name:
                continue
            left = "left" in i.name.lower()
            right = "right" in i.name.lower()
            assert left or right, i

            bf = BIDS_FILE(str(i).replace("subCT", "sub-CT").replace("ses2", "ses-2"), u)
            sub = bf.get("sub")
            ses = bf.get("ses")
            assert ses is not None, i
            assert sub is not None, i
            side = "left" if left else "right"
            key = f"sub-{sub}_ses-{ses}_side-{side}"
            if sub == "CTFU04731":
                continue
            poi = POI_Global.load(i)

            if u in flips_model:
                poi.map_labels_(mapp_models_filp)
            if "Robert_Model" not in u:
                if "SSM" in u.upper():
                    poi.info["label_name"] = mapping1.copy()
                    poi.info["label_group_name"] = mapping2.copy()

                if mapping1 != poi.info["label_name"]:
                    for k, v in mapping1.items():
                        if k not in poi.info["label_name"] or v != poi.info["label_name"][k]:
                            print(
                                k,
                                v,
                                key,
                                name__,
                                k not in poi.info["label_name"],
                                raters_all,
                            )

                    # if "SSM" not in u.upper():

                    assert mapping2 == poi.info["label_group_name"], (u, mapping2, poi.info["label_group_name"])

            else:
                mapping = Lower_Body.get_mapping()
                assert mapping2 is not None
                label_map_full = {}
                for k, v in mapping1.items():
                    # print(k, v)
                    # print(type(k))
                    a, b = mapping[v]
                    c, d = str(k).replace("(", "").replace(")", "").split(",")
                    label_map_full[a.value, b.value] = int(c), int(d)
                poi.map_labels_(label_map_full)
                poi.info["label_name"] = mapping1
                poi.info["label_group_name"] = mapping2
            all_data[u][key] = {"file": i, "poi": poi}
            if poi.info.get("Side") is None:
                poi.info["Side"] = side.upper()
            (path_mrk / u).mkdir(exist_ok=True, parents=True)
            poi.sort().save_mrk(path_mrk / u / (key + ".mrk.json"))


def export_master_poi_table(raters, base_dir="pois_mrk", out_xlsx="master_poi_coordinates.xlsx"):
    rows = []

    base_dir = Path(base_dir)

    # Alle Dateinamen sammeln (rater-unabhängig)
    all_files = set()
    for rater in raters:
        for f in (base_dir / rater).glob("*.mrk.json"):
            all_files.add(f.name)
            print(len(all_files))

    all_files = sorted(all_files)

    for fname in all_files:
        for poi_name, key in POI_MAP.items():
            row = {
                "filename": fname,
                "POI_name": poi_name,
                "id": POI_MAP[poi_name],
            }

            for rater in raters:
                fpath = base_dir / rater / fname

                if not fpath.exists():
                    row[f"{rater}_X"] = np.nan
                    row[f"{rater}_Y"] = np.nan
                    row[f"{rater}_Z"] = np.nan
                    continue

                poi = POI_Global.load(fpath)

                if key in poi:
                    x, y, z = poi[key]
                else:
                    x = y = z = np.nan

                row[f"{rater}_X"] = x
                row[f"{rater}_Y"] = y
                row[f"{rater}_Z"] = z

            rows.append(row)

    df = pd.DataFrame(rows)

    # hübsche Sortierung
    df = df.sort_values(["filename", "id"]).reset_index(drop=True)
    print(out_xlsx)
    df.to_excel(out_xlsx, index=False)

    return df


if __name__ == "__main__":
    main()
    df = export_master_poi_table(
        raters=raters_all,
        base_dir="pois_mrk",
        out_xlsx=out_userstudy / "all_coordinates.xlsx",
    )

    print(df.head())
