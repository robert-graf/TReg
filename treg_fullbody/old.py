def get_tasks_vibe():
    return [
        Task(
            101,
            [
                Full_Body_Instance.costal_cartilage,
                Full_Body_Instance.clavicula_right,
                Full_Body_Instance.scapula_right,
                Full_Body_Instance.clavicula_left,
                Full_Body_Instance.scapula_left,
                Full_Body_Instance.sternum,
                Full_Body_Instance.humerus_left,
                Full_Body_Instance.humerus_right,
                # Do not exist in Version 80 to 100
                # Full_Body_Instance.hand_left,
                # Full_Body_Instance.radius_left,
                # Full_Body_Instance.ulna_left,
                # Full_Body_Instance.hand_right,
                # Full_Body_Instance.radius_right,
                # Full_Body_Instance.ulna_right,
            ],
            [
                atlas_poi_folder / "clavicula_l.mrk.json",
                atlas_poi_folder / "clavicula_r.mrk.json",
                atlas_poi_folder / "costal_cartilage_new.mrk.json",
                atlas_poi_folder / "sternum_new.mrk.json",
                atlas_poi_folder / "scapula_l.mrk.json",
                atlas_poi_folder / "scapula_r.mrk.json",
                atlas_poi_folder / "UpperArm_right_vibe.json",
                atlas_poi_folder / "UpperArm_left_vibe.json",
                atlas_poi_folder / "ribcage_l_new.mrk.json",  # TODO in 40 only?
                atlas_poi_folder / "ribcage_r_new.mrk.json",  # TODO in 40 only?
                # Do not exist in 80-100
                # fov.parent / "Forearm_left.json",
                # fov.parent / "Forearm_right.json",
            ],
            label_target=[
                Full_Body_Instance_Vibe.costal_cartilages,
                Full_Body_Instance_Vibe.clavicula_right,
                Full_Body_Instance_Vibe.scapula_right,
                Full_Body_Instance_Vibe.clavicula_left,
                Full_Body_Instance_Vibe.scapula_left,
                Full_Body_Instance_Vibe.sternum,
                Full_Body_Instance_Vibe.humerus_left,
                Full_Body_Instance_Vibe.humerus_right,
                # Full_Body_Instance_Vibe.rib,
            ],
            ribs=True,
            ribs_exclusive=False,
            others={
                "split-seg": atlas_templates_folder / "all.nii.gz",
            },
            mapping_target=Full_Body_Instance.get_VIBESeg_mapping(),
        ),
        Task(
            102,
            [
                Full_Body_Instance.femur_left,
                Full_Body_Instance.femur_right,
                Full_Body_Instance.sacrum,
                Full_Body_Instance.pelvis_left,
                Full_Body_Instance.pelvis_right,
            ],
            [
                atlas_poi_folder / "poi_type_avg_mrk_r_vibe.mrk.json",
                atlas_poi_folder / "poi_type_avg_mrk_vibe.mrk.json",
                atlas_poi_folder / "pelvis_l_new.mrk.json",
                atlas_poi_folder / "pelvis_r_new.mrk.json",
                atlas_poi_folder / "sacrum.mrk.json",
            ],
            label_target=[
                Full_Body_Instance_Vibe.femur_left,
                Full_Body_Instance_Vibe.femur_right,
                Full_Body_Instance_Vibe.sacrum,
                Full_Body_Instance_Vibe.pelvis_left,
                Full_Body_Instance_Vibe.pelvis_right,
            ],
            others={
                "split-seg": atlas_templates_folder / "all.nii.gz",
                "split-seg-leg": root_atlas / "leg" / "sub-atlas_seg-subregion_msk.nii.gz",  # TODO Mirror?
            },
            mapping_target=Full_Body_Instance.get_VIBESeg_mapping(),
        ),
    ]
