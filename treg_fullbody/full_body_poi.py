import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from TPTBox import BIDS_FILE, NII, POI, BIDS_Global_info, Image_Reference, No_Logger, POI_Global, calc_centroids, to_nii
from TPTBox.core.bids_files import Buffered_BIDS_Global_info
from TPTBox.core.vert_constants import Full_Body_Instance, Vertebra_Instance

sys.path.append(str(Path(__file__).parent.parent))
from TPTBox.registration._deformable.multilabel_segmentation import Template_Registration

from treg.mesh_analysis import AnalysisError
from treg_fullbody.fix_rib_poi import fix_rib, is_rib_fixed
from treg_fullbody.poi_dict import pois_full as poi_naming_schema
from treg_fullbody.subreg_2_poi import get_centroid, get_sphere_center

logger = No_Logger(prefix="T-REG")

ds = "/DATA/NAS/datasets_processed/CT_spine/dataset-myelom/"
root_atlas = Path("data/full_body")
atlas_poi_folder = root_atlas / "pois"
atlas_templates_folder = root_atlas / "templates"
buffer_folder = root_atlas / "buffer_folder"
buffer_folder.mkdir(exist_ok=True)


@dataclass
class SubregPOI:
    idx: int = 0
    idx_subreg: int = 0
    poi_idx: tuple[int, int] = (0, 0)
    algorithm: Literal["cms", "sphere"] = "cms"


@dataclass
class Task:
    task_id: int | str
    label: list[Full_Body_Instance]
    input_pois: list[Path]
    label_target: list | None = None
    ribs: bool = False
    ribs_exclusive: bool = True
    _ids_subreg: dict[int, list[int]] | None = None
    _ids_subreg_2: list[int] | None = None
    subreg_POIs: list[SubregPOI] | None = None  # noqa: N815
    # vert_thorax: bool = False
    # crop: int = 150
    mirror: bool = False  # Warning: only works with labels that have acually mirrors, because we set the labels to +100 blindly!
    lr: float = 0.001
    max_steps: int = 1500
    min_delta: float = 0.000001
    pyramid_levels: int = 4
    coarsest_level: int = 3
    finest_level: int = 0
    others: dict[str, Path] = field(default_factory=dict)
    mapping_target: dict | None = None
    gt_rib: Path = atlas_templates_folder / "sub-CTFU04045_ses-02480_sequ-204_mod-ct_seg-vert_msk.nii.gz"
    gt_12: Path = atlas_templates_folder / "sub-CTFU04045_ses-02480_sequ-204_mod-ct_seg-VIBESeg-12_msk.nii.gz"
    weights: dict = field(default_factory=lambda: {"be": 0.00001, "seg": 1, "Dice": [0.01, 0.01, 0.01, 0.1], "Tether": [1, 0.1, 0.001, 0]})


def get_tasks_ct():
    FB = Full_Body_Instance
    return [
        Task(
            "shoulder-left",
            [FB.costal_cartilage, FB.clavicula_left, FB.scapula_left, FB.sternum],
            [
                atlas_poi_folder / "clavicula_l.mrk.json",
                atlas_poi_folder / "costal_cartilage_new.mrk.json",
                atlas_poi_folder / "sternum_new.mrk.json",
                atlas_poi_folder / "scapula_l.mrk.json",
            ],
            others={"subreg": atlas_templates_folder / "shoulder.nii.gz"},
        ),
        Task(
            "shoulder-right",
            [FB.clavicula_right, FB.scapula_right, FB.sternum],
            [atlas_poi_folder / "clavicula_l.mrk.json", atlas_poi_folder / "scapula_l.mrk.json"],
            others={"subreg": atlas_templates_folder / "shoulder.nii.gz"},
            mirror=True,
            mapping_target={x.value: FB[x.name.replace("_right", "_left")].value for x in [FB.clavicula_right, FB.scapula_right]},
        ),
        # Task(
        #    "leg-left",
        #    [FB.femur_left, FB.patella_left, FB.tibia_left, FB.fibula_left],
        #    [root_atlas.parent / "leg" / "sub-atlas_seg-poi_poi.json"],
        #    others={
        #        "veerman": root_atlas.parent / "leg" / "sub-atlas_seg-subregion_msk.nii.gz",
        #        #"veerman-raw": root_atlas.parent / "leg" / "sub-atlas_seg-subregion_msk.nii.gz",
        #    },
        # ),
        # Task(
        #    "leg-right",
        #    [FB.femur_right, FB.patella_right, FB.tibia_right, FB.fibula_right],
        #    [root_atlas.parent / "leg" / "sub-atlas_seg-poi_poi.json"],
        #    mirror=True,
        #    others={"veerman": root_atlas.parent / "leg" / "sub-atlas_seg-subregion_msk.nii.gz"},
        #    mapping_target={
        #        Full_Body_Instance.femur_right.value: Full_Body_Instance.femur_left.value,
        #        Full_Body_Instance.patella_right.value: Full_Body_Instance.patella_left.value,
        #        Full_Body_Instance.tibia_right.value: Full_Body_Instance.tibia_left.value,
        #        Full_Body_Instance.fibula_right.value: Full_Body_Instance.fibula_left.value,
        #    },
        # ),
        Task(
            "leg-left-2",
            [FB.femur_left, FB.patella_left, FB.tibia_left, FB.fibula_left],
            [root_atlas.parent / "leg2" / "sub-atlas_seg-poi_poi.json"],
            others={
                "veerman": root_atlas.parent / "leg2" / "sub-atlas_seg-subregion_msk.nii.gz",
                # "veerman-raw": root_atlas.parent / "leg2" / "sub-atlas_seg-subregion_msk.nii.gz",
            },
            gt_12=root_atlas.parent / "leg2" / "sub-atlas_seg-VIBESeg-12_msk.nii.gz",
            # weights={"be": 0.00001, "seg": 1, "Dice": [0.01, 0.01, 0.01, 0.1], "Tether": [0.01, 0.01, 0.001, 0]},
        ),
        Task(
            "leg-right-2",
            [FB.femur_right, FB.patella_right, FB.tibia_right, FB.fibula_right],
            [root_atlas.parent / "leg2" / "sub-atlas_seg-poi_poi.json"],
            mirror=True,
            others={"veerman": root_atlas.parent / "leg2" / "sub-atlas_seg-subregion_msk.nii.gz"},
            mapping_target={
                Full_Body_Instance.femur_right.value: Full_Body_Instance.femur_left.value,
                Full_Body_Instance.patella_right.value: Full_Body_Instance.patella_left.value,
                Full_Body_Instance.tibia_right.value: Full_Body_Instance.tibia_left.value,
                Full_Body_Instance.fibula_right.value: Full_Body_Instance.fibula_left.value,
            },
            gt_12=root_atlas.parent / "leg2" / "sub-atlas_seg-VIBESeg-12_msk.nii.gz",
        ),
        Task(
            "hip",
            [FB.sacrum, FB.pelvis_left, FB.pelvis_right],
            [atlas_poi_folder / "pelvis_l_new.mrk.json", atlas_poi_folder / "pelvis_r_new.mrk.json", atlas_poi_folder / "sacrum.mrk.json"],
            others={
                "sacrum-s5": atlas_templates_folder / "sacrum.nii.gz",
                "subreg": atlas_templates_folder / "pelvis.nii.gz",
            },
            # TODO add 6 glider sacrum.
            _ids_subreg_2=[FB.pelvis_left.value, FB.pelvis_right.value],
        ),
        Task(
            "arm-left",
            [FB.hand_left, FB.radius_left, FB.ulna_left, FB.humerus_left],
            [atlas_poi_folder / "UpperArm_left.json", atlas_poi_folder / "Forearm_left.json"],
            others={"subreg": atlas_templates_folder / "hand.nii.gz"},
            # _ids_subreg={105: [5, 20, 22, 24, 26, 27]},
            # _ids_subreg_2=[105, 104],
            subreg_POIs=[SubregPOI(idx=104, idx_subreg=1, poi_idx=(104, 1), algorithm="sphere")],
        ),
        Task(
            "arm-right",
            [FB.hand_right, FB.radius_right, FB.ulna_right, FB.humerus_right],
            [atlas_poi_folder / "UpperArm_left.json", atlas_poi_folder / "Forearm_left.json"],
            others={"subreg": atlas_templates_folder / "hand.nii.gz"},
            # _ids_subreg={105: [5, 19, 21, 23, 25, 28]},
            # _ids_subreg_2=[105, 104],
            mirror=True,
            mapping_target={
                x.value: FB[x.name.replace("_right", "_left")].value
                for x in [FB.hand_right, FB.radius_right, FB.ulna_right, FB.humerus_right]
            },
            subreg_POIs=[SubregPOI(idx=104, idx_subreg=1, poi_idx=(4, 1), algorithm="sphere")],
        ),
        Task(
            "ribs-left",
            [FB.rib_left],
            [atlas_poi_folder / "ribcage_l.mrk.json"],
            ribs=True,  # TODO add logic for RIBs
            weights={"be": 0.0001, "seg": 1, "Dice": [0.01, 0.1, 0.1, 0.1], "Tether": [1, 0.1, 0.001, 0]},
            others={"rib": atlas_templates_folder / "rib_left.nii.gz"},
        ),
        Task(
            "ribs-right",
            [FB.rib_right],
            [atlas_poi_folder / "ribcage_r.mrk.json"],
            ribs=True,  # TODO add logic for RIBs
            weights={"be": 0.0001, "seg": 1, "Dice": [0.01, 0.1, 0.1, 0.1], "Tether": [1, 0.1, 0.001, 0]},
            others={"rib": atlas_templates_folder / "rib_right.nii.gz"},
        ),
        Task(
            "feet-left",
            [x for x in FB.feet() if "left" in x.name],
            [],
            others={"subreg": atlas_templates_folder / "foot.nii.gz"},
        ),
        Task(
            "feet-right",
            [x for x in FB.feet() if "left" not in x.name],
            [],
            others={"subreg": atlas_templates_folder / "foot.nii.gz"},
            mirror=True,
            mapping_target={x.value: FB[x.name.replace("_right", "_left")].value for x in FB.feet() if "left" not in x.name},
        ),
    ]


tasks = get_tasks_ct()


def change_rib_reference(
    task: Task,
    rib_pois: POI_Global,
    poi_atlas: POI,
    atlas_fov: NII,
    others: dict[str, NII],
    T13_offset=30,
    rib_length=None,
    ribs_shorten=None,
):
    key_others = "rib"
    assert key_others in others, others
    if ribs_shorten is None:
        ribs_shorten = [Vertebra_Instance.T11, Vertebra_Instance.T12, Vertebra_Instance.L1, Vertebra_Instance.T13]
    if rib_length is None:
        rib_lengths = [38, 30, 20]  # [38, 24, 10]
    logger.print("change_rib_reference")
    assert atlas_fov.orientation == ("P", "I", "R"), atlas_fov.orientation
    offset = 0 if "left" not in str(task.task_id) else 100
    # Remove T12
    if (Vertebra_Instance.T12.RIB + offset, 0) not in rib_pois:
        logger.print("Remove T12 from Atlas")
        rib = atlas_fov.extract_label(Vertebra_Instance.T12.RIB)
        atlas_fov[rib != 0] = 0
        for k1, k2 in poi_atlas.extract_region(Vertebra_Instance.T12.RIB + offset):
            poi_atlas.remove_((k1, k2))
    # Add T13 (label is for L1 instead of T13)
    if (Vertebra_Instance.L1.RIB + offset, 0) in rib_pois:
        logger.print("Add T13 to Atlas")
        rib = atlas_fov.extract_label(Vertebra_Instance.T12.RIB)
        atlas_fov.resample_from_to_(rib)
        rib *= others[key_others]
        rib2 = rib * 0
        rib2[:, T13_offset:, :] = rib[:, :-T13_offset, :]

        others[key_others][rib2 != 0] = rib2[rib2 != 0]
        atlas_fov[rib2 != 0] = Vertebra_Instance.L1.RIB
        poi_atlas.assert_affine(rib2)
        for _, k2, cord in poi_atlas.extract_region(Vertebra_Instance.T12.RIB + offset).items():
            cord = list(cord)
            cord[1] += T13_offset
            poi_atlas[(Vertebra_Instance.L1.RIB + offset, k2)] = cord
    for r in ribs_shorten:
        rib_length = rib_pois.info["ribs"]["left" if "left" in str(task.task_id) else "right"]
        v = str(r.value)
        if v not in rib_length:
            continue
        rib_length = rib_length[v]["rib_length"]
        if rib_length > rib_lengths[0]:
            continue
        rib = atlas_fov.extract_label(r.RIB)
        subs = others[key_others]
        ids = [i for i, l in enumerate(rib_lengths, 1) if rib_length <= l]
        print(r, "Shorten", r, "by", ids)
        # change reference for short ribs
        rm = subs.extract_label(ids) * rib
        others[key_others][rm != 0] = 0
        atlas_fov[rm != 0] = 0
        # remove points from short ribs
        if 1 in ids:
            poi_atlas.remove_((r.RIB + offset, 4))
        if 2 in ids:
            poi_atlas.remove_((r.RIB + offset, 3))
        if 3 in ids:
            poi_atlas.remove_((r.RIB + offset, 2))

    return poi_atlas, atlas_fov, others


def extract_label_for_task(
    task: Task,
    VIBESeg_12: Image_Reference,
    ribs_instance: Image_Reference,
    mirror=False,
    is_target=False,
    crop=20,
) -> NII:
    arr = to_nii(VIBESeg_12, True)
    label = task.label
    if is_target and task.label_target is not None:
        # VIBESeg 100 and VIBESeg 12 have diffrent label ids. This is a manuam matching
        label = task.label_target
    if mirror:
        # VIBESeg 12 mirroring
        label = [a.value + 100 if a.value < 100 else a.value - 100 for a in label]
    selected = arr.extract_label(label, True)
    # Optinal remapping
    if task.mapping_target is not None and is_target:
        selected.map_labels_(task.mapping_target)
    if task.ribs:
        rib = to_nii(ribs_instance, True)
        selected.resample_from_to_(rib)
        # Pull Labels from ribs
        if task.ribs_exclusive:
            selected[np.logical_and(selected != 0, rib != 0)] = rib[np.logical_and(selected != 0, rib != 0)]
        else:
            rib = rib.extract_label(list(range(38, 54)), keep_label=True)
            selected[rib != 0] = rib[rib != 0]

    selected = selected.reorient()
    selected.set_dtype_("smallest_uint")
    selected.apply_crop_(selected.compute_crop(0, crop, raise_error=False))
    return selected


atlas_poi_buffer = {}


def get_poi(path):
    poi = POI_Global.load(path)
    label_name = {}
    mapping = {}
    for k1, k2 in poi.keys():
        key = poi.info["label_name"][str((k1, k2))]
        key_new = poi_naming_schema.get(key.replace("sacrum_1", "sacrum"), {"name": key, "value": (k1, k2)})
        label_name[str(key_new["value"])] = key_new["name"]
        mapping[(k1, k2)] = key_new["value"]
    poi.map_labels_(label_map_full=mapping)
    poi.info["label_name"] = label_name
    poi.level_one_info = Full_Body_Instance
    return poi


def get_atlas_poi(task: Task, rib_pois: POI_Global | None = None, save_debug=False) -> tuple[POI, NII, POI, dict[str, NII]]:
    if task.task_id in atlas_poi_buffer:
        return atlas_poi_buffer[task.task_id]
    logger.on_text("input_pois", task.input_pois)
    # Load and merge POI files
    poi_atlas = POI_Global({}, itk_coords=True)
    poi_atlas.info["label_name"] = {}
    for e, p in enumerate(task.input_pois, start=1):
        poi_curr = get_poi(p).to_cord_system(poi_atlas.itk_coords, True)
        logger.on_text(p.name, poi_curr.keys())
        for k, k2, cord in poi_curr.items():
            label = poi_curr.info["label_name"][str((k, k2))]
            if task.mirror:
                if k > 100:
                    k -= 100  # noqa: PLW2901
                else:
                    k += 100  # noqa: PLW2901
            assert (k, k2) not in poi_atlas, (k, k2, poi_atlas.keys(), e)
            poi_atlas[k, k2] = cord
            poi_atlas.info["label_name"][str((k, k2))] = label

    atlas_fov: NII = extract_label_for_task(task, task.gt_12, task.gt_rib, task.mirror)

    def x(v):
        nii = to_nii(v, True)
        ret = nii.resample_from_to(atlas_fov) * atlas_fov.clamp(0, 1)
        if task.mirror:
            ret.set_array_(ret.get_array()[:, :, ::-1])
        return ret

    others = {k: x(v) for k, v in task.others.items()}

    # MIRROR
    poi_atlas = poi_atlas.to_local(atlas_fov)
    if task.mirror:
        atlas_fov.set_array_(atlas_fov.get_array()[:, :, ::-1])
        for k1, k2, (x, y, z) in poi_atlas.items(sort=True):  # type: ignore
            assert poi_atlas.get_axis("R") == 2
            poi_atlas[k1, k2] = (x, y, poi_atlas.shape[2] - 1 - z)  # type: ignore
    # Center of Mass
    cms_path = buffer_folder / f"cms-{task.task_id}.json" if buffer_folder is not None else None
    if cms_path is not None and cms_path.exists():
        poi_cms_atlas = POI.load(cms_path)
    else:
        poi_cms_atlas = calc_centroids(atlas_fov, second_stage=40, bar=True)
        if cms_path is not None:
            poi_cms_atlas.to_global().save_mrk(cms_path)
            poi_cms_atlas.save(cms_path)

    if rib_pois is not None and task.ribs:
        poi_atlas, atlas_fov, others = change_rib_reference(task, rib_pois, poi_atlas, atlas_fov, others)
    # save_debug
    if save_debug and buffer_folder is not None:
        task_path = buffer_folder / f"task-{task.task_id}.json"
        task_nii_ref = buffer_folder / f"task-{task.task_id}-VIBESeg-12.nii.gz"

        if not task_path.exists() or not task_nii_ref.exists():
            poi_atlas.save(task_path)
            poi_atlas.to_global(itk_coords=True).save_mrk(task_path)
            atlas_fov.save(task_nii_ref)
            for k, v in others.items():
                v.save(buffer_folder / f"task-{task.task_id}-{k}.nii.gz")

    # return
    out = (poi_atlas, atlas_fov, poi_cms_atlas, others)
    atlas_poi_buffer[task.task_id] = out
    return out


def _path(img: BIDS_FILE, parent, task: Task):
    out_poi_final = img.get_changed_path("json", "poi", parent=parent, info={"seg": f"treg-{task.task_id}"})
    out_atlas_final = img.get_changed_path("nii.gz", "msk", parent=parent, info={"seg": f"treg-{task.task_id}"})
    return out_poi_final, out_atlas_final


def post_reg(
    reg: Template_Registration,
    task: Task,
    img: BIDS_FILE,
    poi_atlas: POI,
    nii_atlas_fov: NII,
    data_target: NII,
    others: dict[str, NII],
    parent: str,
    mask_nii: NII | None = None,
):
    out_poi_final, out_atlas_final = _path(img, parent, task)
    # logger.print("make atlas_reg", nii_atlas_fov)
    # atlas_reg = reg.transform_nii(nii_atlas_fov).set_dtype("smallest_uint")  # Transferring the atlas
    # if data_target.clamp(0, 1) * atlas_reg.sum() == 0:
    #    logger.on_fail("output empty. Saved nothing")
    #    return
    # atlas_reg.save(out_atlas_final)
    atlas_reg2 = None
    for k, v in others.items():
        out = img.get_changed_path(bids_format="msk", parent=parent, info={"seg": f"fov-{task.task_id}-{k}"})
        atlas_reg2 = reg.transform_nii(to_nii(v, True).resample_from_to(nii_atlas_fov)).set_dtype("smallest_uint")
        # print(k, 1, atlas_reg2.unique())
        if k == "veerman":
            s = data_target.extract_label([13, 15, 113, 115], keep_label=True) % 100
            s.map_labels_({13: 7, 15: 8})
            a = (1 - mask_nii).dilate_msk(2) if mask_nii is not None else 0
            atlas_reg2 = atlas_reg2.infect_(s) * (data_target + a).clamp(0, 1)
            s = s.erode_msk_euclid(3)
            atlas_reg2[s != 0] = s[s != 0]
            from treg.veerman_rules_based import run_single_case

            l = "R" if "right" in out.name else "L"
            try:
                run_single_case(atlas_reg2, out.parent / f"stl-{task.task_id}_{l}", l)
            except AnalysisError:
                logger.print_error()
        elif "sacrum" in k:
            atlas_reg2 = atlas_reg2.infect_(data_target.extract_label(Full_Body_Instance.sacrum), verbose=False)
        else:
            u = data_target.unique()
            data_target2 = data_target.extract_label(task._ids_subreg_2, True) if task._ids_subreg_2 else data_target.copy()
            atlas_reg2[data_target2 == 0] = 0
            # print(k, 2, u)
            for i in u:
                mask = data_target2.extract_label(i)
                c = atlas_reg2 * mask
                if task._ids_subreg:
                    if i in task._ids_subreg:
                        c = c.extract_label(task._ids_subreg[i], True)
                        # print(k, 4, i)
                    else:
                        atlas_reg2[c != 0] = 0
                        # print(k, 3, i)
                        continue

                # print(k, 4.5, c.unique())
                c = c.filter_connected_components_(keep_label=True, min_volume=400, connectivity=1)
                c = c.infect_(mask, verbose=False)
                # print(k, 5, c.unique())
                atlas_reg2[c != 0] = c[c != 0]
        # print(k, 6, c.unique())
        atlas_reg2.save(out)

    logger.print("make atlas_reg poi")
    if len(poi_atlas) != 0:
        poi_reg = reg.transform_poi(poi_atlas)
        poi_reg.info = poi_atlas.info
        if task.subreg_POIs is not None:
            assert atlas_reg2 is not None
            a = poi_reg.to_global()
            for t in task.subreg_POIs:
                try:
                    print(t)
                    print(atlas_reg2.unique())  # [1, 108]
                    print(data_target.unique())  # [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 108]
                    print((atlas_reg2 * data_target.extract_label(t.idx)).unique(), t.idx)

                    if t.algorithm == "sphere":
                        # try:
                        a = get_sphere_center(
                            atlas_reg2 * data_target.extract_label(t.idx),
                            t.idx_subreg,
                            t.poi_idx,
                            a,
                        )
                        # except Exception as e:
                        #    print(e)

                        assert a is not None
                    elif t.algorithm == "cms":
                        try:
                            a = get_centroid(
                                atlas_reg2 * data_target.extract_label(t.idx),
                                t.idx_subreg,
                                t.poi_idx,
                                a,
                            )
                        except Exception as e:
                            print(e)
                        assert a is not None
                    else:
                        raise NotImplementedError(t.algorithm)
                except Exception as e:
                    logger.on_fail(e)
            poi_reg = a.resample_from_to(poi_reg)
        poi_reg.save(out_poi_final, make_parents=True)
        poi_reg.to_global().save_mrk(out_poi_final, main_key=f"fov-{task.task_id}")


def reg(task: Task, img: BIDS_FILE, seg_vibe12: Image_Reference, ribs: Image_Reference, parent: str, rib_pois: POI_Global | None = None):
    out_poi_final, out_final = _path(img, parent, task)
    if len(task.input_pois) == 0 and len(task.others) != 0:
        out = img.get_changed_path(
            bids_format="msk",
            parent=parent,
            info={"seg": f"fov-{task.task_id}-{next(task.others.keys().__iter__())}"},
        )
        if out.exists():
            return _path(img, parent, task)
    if out_poi_final.exists() or out_final.exists():
        logger.on_save(f"{out_poi_final.name=} exists; skip!")
        return _path(img, parent, task)

    ### Segs ###
    data_target = extract_label_for_task(task, seg_vibe12, ribs, is_target=True, crop=10)
    if data_target.max() == 0:
        return _path(img, parent, task)
    poi_atlas, nii_atlas_fov, poi_atlas_cms, others = get_atlas_poi(task, rib_pois=rib_pois)

    u = data_target.unique()
    if len(u) <= 1:
        logger.on_warning("No Segmentation in target")
        return _path(img, parent, task)
    nii_atlas_fov = nii_atlas_fov.extract_label(u, True)
    ##################
    logger.on_log(f"Running task: {task!s};\n{data_target.shape=}; {nii_atlas_fov.shape=}")
    # logger.on_debug(f"{u}; {nii_atlas_fov.unique()=}")
    # logger.on_debug(f"{data_target.seg=}; {nii_atlas_fov.seg=}")
    # nii_atlas_fov.save("/DATA/NAS/ongoing_projects/robert/code/TReg/data/leg2/test/nii_atlas_fov.nii.gz")
    # data_target.save("/DATA/NAS/ongoing_projects/robert/code/TReg/data/leg2/test/data_target.nii.gz")
    # for k, v in others.items():
    #    v.save(f"/DATA/NAS/ongoing_projects/robert/code/TReg/data/leg2/test/{k}.nii.gz")
    # if (
    #    (img.get("sub") == "CTFU01051" and img.get("ses") == "02340")
    #    or (img.get("sub") == "MM00052" and img.get("ses") == "00380")
    #    or (img.get("sub") == "MM00161" and img.get("ses") == "00000")
    # ):  # TODO Remove
    #    from copy import deepcopy
    #
    #    task = deepcopy(task)
    #    task.finest_level += 1
    #    # sub-MM00052_ses-00380_sequ-203_mod-ct_seg-VIBESeg-12_msk.['nii.gz'] data/full_body/pois/ribcage_l.mrk.json
    reg = Template_Registration(
        data_target,  # [::2, ::2, ::2],  # Target segmentation
        nii_atlas_fov,  # Starting Segmentation (not the split one)
        same_side=True,
        lr=task.lr,
        max_steps=task.max_steps,
        min_delta=task.min_delta,
        pyramid_levels=task.pyramid_levels,
        coarsest_level=task.coarsest_level,
        finest_level=task.finest_level,
        weights=task.weights,
        poi_cms=poi_atlas_cms,  # Can be None, than it will be computed automatically
        gpu=gpu,
    )
    post_reg(reg, task, img, poi_atlas, nii_atlas_fov, data_target, others, parent)
    return _path(img, parent, task)


def get_rib_info(img_file: BIDS_FILE, rib_instance: Path, parent, compute_rib_special_cases=True):
    ds_rib = BIDS_FILE(rib_instance, img_file.dataset)
    rib_stats = ds_rib.get_changed_path("json", "poi", parent=parent, info={"seg": "rib-lengths"})
    rib_stats2 = Path(str(rib_stats).replace(".json", ".mrk.json"))
    if compute_rib_special_cases and (not rib_stats.exists() or not rib_stats2.exists()):
        spine_seg = rib_instance.parent / (rib_instance.name.replace("vert", "spine"))
        assert spine_seg.exists() and "spine" in spine_seg.name, spine_seg
        poi = POI_Global(itk_coords=False)
        poi.info["label_name"] = {}
        from treg_fullbody.rib_length_measurement_algorithm import measure_ribs_length_subject, result_to_dict

        result = measure_ribs_length_subject(to_nii(rib_instance, True), to_nii(spine_seg, True), vert_ids=None)
        for r in result:
            if r.vertebra.RIB == Vertebra_Instance.T13:
                r.vertebra = Vertebra_Instance.L1

            x = r.vertebra.RIB + (100 if r.leftside else 0)
            poi[x, 0] = r.start_point
            poi.info["label_name"][f"({x}, {0})"] = "rib_start_point"
            if r.end_point is not None:
                poi.info["label_name"][f"({x}, {7})"] = "rib_end_point"
                poi[x, 7] = r.end_point
        poi.info["ribs"] = result_to_dict(result)
        poi.to_cord_system(itk_coords=True).save(rib_stats, make_parents=True)
        poi.to_cord_system(itk_coords=True).save_mrk(rib_stats2)
        return poi
    else:
        if rib_stats.exists():
            return POI_Global.load(rib_stats)
        else:
            return None


skip_feet = True
parent_final = "derivatives-final-points"


def run_all(
    img_file: BIDS_FILE,
    VIBESeg_12: Path,
    rib_instance: Path,
    parent="derivatives-treg",
    compute_rib_special_cases=True,
    override=False,
    make_bone=True,
):
    os.nice(15)
    if isinstance(rib_instance, BIDS_FILE):
        rib_instance = rib_instance.get_nii_file()  # type: ignore
    if not VIBESeg_12.exists():
        logger.on_fail(VIBESeg_12, "missing; Skip!")
        return
    out_poi_final = img_file.get_changed_path("json", "poi", parent=parent_final, info={"seg": "torso"})
    out_poi_final_leg = img_file.get_changed_path("json", "poi", parent=parent_final, info={"seg": "leg"})
    out_atlas_final = img_file.get_changed_path("nii.gz", "msk", parent=parent_final, info={"seg": "treg"})

    bone = img_file.get_changed_path("nii.gz", "msk", parent=parent, info={"seg": "bone"})
    if not bone.exists() and make_bone:
        to_nii(VIBESeg_12, True).extract_label(Full_Body_Instance.bone(), True).save(bone)
    # if not is_rib_fixed(img_file, parent):
    #    override = True
    # if not img_file.get_changed_path("nii.gz", "msk", parent=parent, info={"seg": "fov-leg-left-2-veerman"}).exists():
    #    override = True
    # override = True  # TODO REMOVE
    # if out_atlas_final.exists() and to_nii(out_atlas_final, True).sum() == 0:
    #    print("unlink defective atlas")
    #    out_atlas_final.unlink(True)
    if out_poi_final.exists() and out_atlas_final.exists() and not override:  # out_poi_final_leg.exists()
        logger.on_ok(out_atlas_final.name, "exist; Skip!")
        return
    logger.on_log(VIBESeg_12)
    poi_final = POI_Global(itk_coords=True)
    poi_final_leg = POI_Global(itk_coords=True)
    poi_final_leg.info["label_name"] = {}
    rib_pois = None
    try:
        rib_pois = get_rib_info(img_file, rib_instance, parent="derivatives-treg", compute_rib_special_cases=compute_rib_special_cases)
    except AssertionError as e:
        logger.on_fail(e)
    # return
    if rib_pois is not None:
        poi_final.join_left_(rib_pois.to_cord_system(poi_final.itk_coords))
    fail = False
    for task in tasks:
        leg_keys = ["leg-left", "leg-right", "leg-left-2", "leg-right-2"]
        try:
            if skip_feet and "feet" in str(task.task_id):
                continue
            poi_file, _ = reg(task, img_file, VIBESeg_12, rib_instance, parent=parent, rib_pois=rib_pois)
            if not poi_file.exists():
                continue
            poi_file = fix_rib(task, rib_instance, img_file, poi_file, parent)

            poi = POI_Global.load(poi_file, itk_coords=True)
            if task.task_id in leg_keys:
                from TPTBox.core.vert_constants import _ABBREVIATION_TO_ENUM

                def mk_tuple(v):
                    v = str(v).replace("(", "").replace(")", "").replace(" ", "").split(",")
                    return int(v[0]), int(v[1])

                m = {
                    mk_tuple(v): (
                        _ABBREVIATION_TO_ENUM[k][0].value + (0 if task.task_id in ["leg-right", "leg-right-2"] else 100),
                        _ABBREVIATION_TO_ENUM[k][1].value,
                    )
                    for v, k in poi.info["label_name"].items()
                }
                label_name = {
                    f"({_ABBREVIATION_TO_ENUM[k][0].value + (0 if task.task_id in ['leg-right', 'leg-right-2'] else 100)}, {_ABBREVIATION_TO_ENUM[k][1].value})": (
                        k
                    )
                    for v, k in poi.info["label_name"].items()
                }
                poi.map_labels_(label_map_full=m)
                poi.info["label_name"] = label_name

            assert len([a for a in poi.keys() if a in poi_final]) == 0, [a for a in poi.keys() if a in poi_final]
            poi_final_leg.join_left_(poi) if task.task_id in leg_keys else poi_final.join_left_(poi)

        except Exception:
            logger.print_error()
            fail = True
        #     continue
        # for i in
    if fail:
        return
    poi_veerman_left = out_poi_final.parent / "stl_L" / "poi.json"
    poi_veerman_right = out_poi_final.parent / "stl_R" / "poi.json"

    # return
    from TPTBox.core.vert_constants import _ABBREVIATION_TO_ENUM

    old = img_file.get_changed_path("json", "poi", parent=parent, info={"seg": "torso"})
    poi_veerman_left = old.parent / "stl-leg-left-2_L" / "poi.json"
    poi_veerman_right = old.parent / "stl-leg-right-2_R" / "poi.json"

    from treg.veerman_rules_based import run_single_case

    for poi, l, offset in [(poi_veerman_left, "L", 100), (poi_veerman_right, "R", 0)]:
        verman_seg = img_file.get_changed_path(
            "nii.gz",
            "msk",
            parent=parent,
            info={"seg": f"fov-leg-{'right-2' if l == 'R' else 'left-2'}-veerman"},
        )
        if not poi.exists() and verman_seg.exists():
            print("run_single_case")
            run_single_case(verman_seg, poi.parent, l, allow_partial=True)
        if not poi.exists():
            continue
        poi = POI_Global.load(poi, itk_coords=True)
        for k1, k2, coord in poi.items():
            name = poi.info["label_name"][f"({k1}, {k2})"]
            of = 0 if name in ["FNC", "FHC"] else 100
            if name == "FNC":
                k1 = 13
                k2 = 12
            if name == "FHC":
                k1 = 13
                k2 = 11
            poi_final_leg[k1 + offset, k2 + of] = coord
            poi_final_leg.info["label_name"][f"({k1 + offset}, {k2 + of})"] = name + "-veerman"
    out_seg = to_nii(VIBESeg_12, True) * 0
    poi_final = poi_final.to_local(out_seg).filter_points_inside_shape(inplace=True).to_global(itk_coords=True)
    poi_final_leg = poi_final_leg.to_local(out_seg).filter_points_inside_shape(inplace=True).to_global(itk_coords=True)

    poi_final.save(out_poi_final, make_parents=True)
    poi_final.save_mrk(out_poi_final, split_by_region=True)
    poi_final_leg.save(out_poi_final_leg, make_parents=True)
    poi_final_leg.save_mrk(out_poi_final_leg, split_by_region=True)

    for f in poi_file.parent.glob(f"*sequ-{img_file.get('sequ')}*seg-fov*"):
        nii = to_nii(f, True).resample_from_to(out_seg, mode="constant")
        out_seg[nii != 0] = nii[nii != 0]
    out_seg.set_dtype("smallest_uint").save(out_atlas_final)


gpu = 4
if __name__ == "__main__":
    # ds = Path("/media/data/robert/dataset-myelom/dataset-myelom/")
    # ds = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-myelom")
    #
    # x = ds / "derivatives-final/sub-CTFU00066/ses-02970"
    #
    # img = BIDS_FILE(x / "sub-CTFU00066_ses-02970_sequ-3_ct.nii.gz", ds)
    #
    # run_all(
    #    img,
    #    x / "sub-CTFU00066_ses-02970_sequ-3_mod-ct_seg-VIBESeg-12_msk.nii.gz",
    #    x / "sub-CTFU00066_ses-02970_sequ-3_mod-ct_seg-vert_msk.nii.gz",
    #    # override=True,
    # )
    p = Path("/DATA/NAS/datasets_processed/CT_spine/dataset-myelom/")
    if p.exists():
        bgi = Buffered_BIDS_Global_info(p, ["derivatives-final"])
    else:
        # Load private NAS path from local config file
        config_path = Path(__file__).parent.parent / ".config.json"

        with open(config_path) as f:
            config = json.load(f)
        nas_path = config["dataset_myelom_path"]
        bgi = BIDS_Global_info(nas_path, ["derivatives-final"])
        gpu = 0

    ## Create job list
    all_files = []
    for sub, subj in bgi.enumerate_subjects(shuffle=True, sort=False):
        q = subj.new_query()
        # q.flatten()
        q.filter_filetype("nii.gz")
        q.filter_format("ct")
        q.filter("seg", "spine")
        q.filter("seg", "vert")
        q.filter("seg", "VIBESeg-12")
        # q.filter("sub", ["CTFU00354"])  #
        # q.filter("ses", ["03470"])  #
        for fam in q.loop_dict():
            img_file = fam["ct"][0]
            vert: BIDS_FILE = fam["msk_seg-vert"][0]
            vibe_seg = fam["msk_seg-VIBESeg-12"][0]
            if not vert.exists():
                continue
            if not vibe_seg.exists():
                continue
            all_files.append([img_file, vibe_seg, vert])

    import random

    #
    # random.seed(42)
    # random.shuffle(all_files)
    print()
    print(len(all_files))
    # all_files = all_files[:10]
    for e, f in enumerate(all_files, 1):
        print(f"{e:3}/{len(all_files):3}                   ")
        try:
            run_all(*f)
        except Exception as e:
            print("FAIL", e)
            raise
