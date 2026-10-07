from TPTBox import BIDS_Global_info, to_nii
from TPTBox.segmentation import run_spineps, run_vibeseg
from TPTBox.segmentation.rib import add_ribs_to_vert_spine

from treg_fullbody.full_body_poi import run_all

root = "/DATA/NAS/datasets_processed/CT_fullbody/dataset-bonescreen-test2"

gpu = 0
bgi = BIDS_Global_info(root)

for _, sub in bgi.enumerate_subjects(sort=True):
    q = sub.new_query()
    q.filter_format("ct")
    for fam in q.loop_dict():
        assert len(fam.get("ct", [])) == 1, fam.get("ct")  # type: ignore
        ct = fam["ct"][0]
        out_seg = ct.get_changed_path("nii.gz", "msk", info={"seg": "VIBESeg-12", "res": "iso"})
        out_vert = ct.get_changed_path("nii.gz", "msk", info={"seg": "vert-rib"})
        out_spine = ct.get_changed_path("nii.gz", "msk", info={"seg": "spine-rib"})

        run_vibeseg(ct, out_seg, gpu=gpu, keep_size=True, dataset_id=12)

        out = run_spineps(
            ct, model_instance="ct_instance", model_semantic="ct", model_labeling="ct_labeling", ignore_compatibility_issues=True
        )
        print(out["out_vert"])
        if not (out_vert.exists() and out_spine.exists()):
            add_ribs_to_vert_spine(
                out["out_vert"], out["out_spine"], out_seg, ct, root, spine_path_out=out_spine, vert_path_out=out_vert, save=True
            )
        run_all(ct, out_seg, out_vert, out_spine)
        exit()
