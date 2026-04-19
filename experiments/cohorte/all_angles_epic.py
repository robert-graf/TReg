import sys
from pathlib import Path

import pandas as pd
from TPTBox import BIDS_Global_info, POI_Global, Print_Logger
from TPTBox.core.vert_constants import Abstract_lvl, Full_Body_Instance

out = str(Path(__file__).parent.parent.parent)
sys.path.append(out)
from treg.angle import POI_MAP, compute_angles


class Lower_Body(Abstract_lvl):
    # Patella
    PATELLA_PROXIMAL_POLE = 1
    PATELLA_DISTAL_POLE = 2
    PATELLA_MEDIAL_POLE = 3
    PATELLA_LATERAL_POLE = 4
    PATELLA_RIDGE_PROXIMAL_POLE = 5
    PATELLA_RIDGE_DISTAL_POLE = 6
    PATELLA_RIDGE_HIGH_POINT = 7

    # Trochlea ossis femoris
    TROCHLEAR_RIDGE_MEDIAL_POINT = 8
    TROCHLEAR_RIDGE_LATERAL_POINT = 9
    TROCHLEA_GROOVE_CENTRAL_POINT = 10

    # Femur
    PELVIS_CENTER = 11
    NECK_CENTER = 12
    TIP_OF_GREATER_TROCHANTER = 13
    LATERAL_CONDYLE_POSTERIOR = 14
    LATERAL_CONDYLE_POSTERIOR_CRANIAL = 15
    LATERAL_CONDYLE_DISTAL = 16  # 16 flcd
    MEDIAL_CONDYLE_DISTAL = 17  # 17 fmcd
    NOTCH_POINT = 18
    # Femur, Tibia
    ANATOMICAL_AXIS_PROXIMAL = 19
    ANATOMICAL_AXIS_DISTAL = 20
    MEDIAL_CONDYLE_POSTERIOR = 21  # flc
    MEDIAL_CONDYLE_POSTERIOR_CRANIAL = 22

    # Tibia
    KNEE_CENTER = 23
    MEDIAL_INTERCONDYLAR_TUBERCLE = 24
    LATERAL_INTERCONDYLAR_TUBERCLE = 25
    MEDIAL_CONDYLE_ANTERIOR = 26
    LATERAL_CONDYLE_ANTERIOR = 27
    MEDIAL_CONDYLE_MEDIAL = 28
    LATERAL_CONDYLE_LATERAL = 29
    ANKLE_CENTER = 30
    MEDIAL_MALLEOLUS = 31
    TGPP = 99
    TTP = 98
    # Fibula
    LATERAL_MALLEOLUS = 32

    @classmethod
    def get_mapping(cls):
        return _ABBREVIATION_TO_ENUM


_ABBREVIATION_TO_ENUM = {
    # Patella
    "PPP": (Full_Body_Instance.patella_right, Lower_Body.PATELLA_PROXIMAL_POLE),
    "PDP": (Full_Body_Instance.patella_right, Lower_Body.PATELLA_DISTAL_POLE),
    "PMP": (Full_Body_Instance.patella_right, Lower_Body.PATELLA_MEDIAL_POLE),
    "PLP": (Full_Body_Instance.patella_right, Lower_Body.PATELLA_LATERAL_POLE),
    "PRPP": (Full_Body_Instance.patella_right, Lower_Body.PATELLA_RIDGE_PROXIMAL_POLE),
    "PRDP": (Full_Body_Instance.patella_right, Lower_Body.PATELLA_RIDGE_DISTAL_POLE),
    "PRHP": (Full_Body_Instance.patella_right, Lower_Body.PATELLA_RIDGE_HIGH_POINT),
    # Femur
    "TRMP": (Full_Body_Instance.femur_right, Lower_Body.TROCHLEAR_RIDGE_MEDIAL_POINT),
    "TRLP": (Full_Body_Instance.femur_right, Lower_Body.TROCHLEAR_RIDGE_LATERAL_POINT),
    "TGCP": (Full_Body_Instance.femur_right, Lower_Body.TROCHLEA_GROOVE_CENTRAL_POINT),
    "FHC": (Full_Body_Instance.femur_right, Lower_Body.PELVIS_CENTER),
    "FNC": (Full_Body_Instance.femur_right, Lower_Body.NECK_CENTER),
    "TGT": (Full_Body_Instance.femur_right, Lower_Body.TIP_OF_GREATER_TROCHANTER),
    "FLCP": (Full_Body_Instance.femur_right, Lower_Body.LATERAL_CONDYLE_POSTERIOR),
    "FLCPC": (Full_Body_Instance.femur_right, Lower_Body.LATERAL_CONDYLE_POSTERIOR_CRANIAL),
    "FMCP": (Full_Body_Instance.femur_right, Lower_Body.MEDIAL_CONDYLE_POSTERIOR),
    "FMCPC": (Full_Body_Instance.femur_right, Lower_Body.MEDIAL_CONDYLE_POSTERIOR_CRANIAL),
    "FLCD": (Full_Body_Instance.femur_right, Lower_Body.LATERAL_CONDYLE_DISTAL),
    "FMCD": (Full_Body_Instance.femur_right, Lower_Body.MEDIAL_CONDYLE_DISTAL),
    "FNP": (Full_Body_Instance.femur_right, Lower_Body.NOTCH_POINT),
    "FAAP": (Full_Body_Instance.femur_right, Lower_Body.ANATOMICAL_AXIS_PROXIMAL),
    "FADP": (Full_Body_Instance.femur_right, Lower_Body.ANATOMICAL_AXIS_DISTAL),
    # Tibia
    "TKC": (Full_Body_Instance.tibia_right, Lower_Body.KNEE_CENTER),
    "TMIT": (Full_Body_Instance.tibia_right, Lower_Body.MEDIAL_INTERCONDYLAR_TUBERCLE),
    "TLIT": (Full_Body_Instance.tibia_right, Lower_Body.LATERAL_INTERCONDYLAR_TUBERCLE),
    "TMCP": (Full_Body_Instance.tibia_right, Lower_Body.MEDIAL_CONDYLE_POSTERIOR),
    "TLCP": (Full_Body_Instance.tibia_right, Lower_Body.LATERAL_CONDYLE_POSTERIOR),
    "TMCA": (Full_Body_Instance.tibia_right, Lower_Body.MEDIAL_CONDYLE_ANTERIOR),
    "TLCA": (Full_Body_Instance.tibia_right, Lower_Body.LATERAL_CONDYLE_ANTERIOR),
    "TMCM": (Full_Body_Instance.tibia_right, Lower_Body.MEDIAL_CONDYLE_MEDIAL),
    "TLCL": (Full_Body_Instance.tibia_right, Lower_Body.LATERAL_CONDYLE_LATERAL),
    "TAC": (Full_Body_Instance.tibia_right, Lower_Body.ANKLE_CENTER),
    "TMM": (Full_Body_Instance.tibia_right, Lower_Body.MEDIAL_MALLEOLUS),
    "TAAP": (Full_Body_Instance.tibia_right, Lower_Body.ANATOMICAL_AXIS_PROXIMAL),
    "TADP": (Full_Body_Instance.tibia_right, Lower_Body.ANATOMICAL_AXIS_DISTAL),
    "TGPP": (Full_Body_Instance.tibia_right, Lower_Body.TGPP),
    "TTP": (Full_Body_Instance.tibia_right, Lower_Body.TTP),
    # Fibula
    "FLM": (Full_Body_Instance.fibula_right, Lower_Body.LATERAL_MALLEOLUS),
}


logger = Print_Logger()
bgi = BIDS_Global_info(
    "/DATA/NAS/datasets_processed/CT_spine/dataset-myelom/",
    ["derivatives-reg-post-v2", "derivatives-VIBESeg-12-"],
)


def export_angles_to_excel(out_xlsx="angles_dataset.xlsx", all_img=False):
    rows = []

    for subj, sub in bgi.enumerate_subjects():
        logger.on_log(subj)
        q = sub.new_query()
        q.filter("seg", "VIBESeg-11")
        for fam in q.loop_dict():
            try:
                u = fam["msk_seg-VIBESeg-11"][0].open_nii().unique()
                if Full_Body_Instance.pelvis_right.value not in u:
                    # print("skip 1")
                    continue
                if Full_Body_Instance.metatarsals_right.value not in u:
                    # print("skip 2")
                    continue
                if Full_Body_Instance.phalanges_right.value not in u:
                    # print("skip 3")
                    continue
                try:
                    for key in ["poi_seg-fov2-reg", "poi_seg-fov3-reg"]:
                        file = fam[key][0]
                        print(file)

                        # check if compleat (verse + Pelvise)
                        poi_original = POI_Global.load(file)
                        mapping = Lower_Body.get_mapping()
                        label_map_full = {}
                        for k, v in POI_MAP.items():
                            # print(k, v)
                            # print(type(k))
                            a, b = mapping[k]
                            c, d = v
                            label_map_full[a.value, b.value] = int(c), int(d)
                            label_map_full[a.value + 100, b.value] = int(c), int(d)
                        poi_original.map_labels_(label_map_full)
                        poi_original.info["label_name"] = POI_MAP.copy()
                        # print(poi_original.keys())
                        # poi.info["label_group_name"] = mapping2
                        angles, _, _ = compute_angles(poi_original)

                        row = {"rater": "full dataset", "file": str(file), "leg": "left" if key == "poi_seg-fov3-reg" else "right"}

                        # add all angles as columns
                        row.update(angles)
                        rows.append(row)
                except Exception as e:
                    logger.on_fail(e)
                if not all_img:
                    # only one per sample
                    break
            except Exception as e:
                logger.on_fail(e)

    print(rows)
    # Build dataframe
    df = pd.DataFrame(rows)

    # Sort by file, then rater (stable & readable)
    df = df.reset_index(drop=True)

    # Save Excel
    df.to_excel(out_xlsx, index=False)

    return df


if __name__ == "__main__":
    export_angles_to_excel()
    export_angles_to_excel(out_xlsx="angles_dataset_with_duplicate.xlsx", all_img=True)
