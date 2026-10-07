"""Full-body POI inference over a BIDS dataset.

For every CT in the dataset this script:
  1) runs VIBESeg-12 (12-label body segmentation) via :func:`run_vibeseg`,
  2) runs SPINEPS (vertebra + spine segmentation) and merges in rib labels via
     :func:`add_ribs_to_vert_spine`,
  3) runs the TReg full-body POI pipeline (:func:`run_all`), which registers
     per-region atlases to the subject and writes landmark POI files.

Outputs land next to the CT in BIDS-style derivatives subfolders. Re-running
is cheap: each stage skips its work if the output file already exists.
"""

import argparse
from pathlib import Path

from TPTBox import BIDS_Global_info
from TPTBox.segmentation import run_spineps, run_vibeseg
from TPTBox.segmentation.rib import add_ribs_to_vert_spine

from treg_fullbody.full_body_poi import run_all


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run VIBESeg-12 + SPINEPS + TReg full-body POI on every CT in a BIDS dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "root",
        nargs="?",
        default="/DATA/NAS/datasets_processed/CT_fullbody/dataset-bonescreen-test2",
        help="Path to the BIDS dataset root (contains rawdata/, derivatives/, ...).",
    )
    p.add_argument("--gpu", type=int, default=0, help="GPU index to use.")
    p.add_argument(
        "--sort",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Iterate subjects in sorted order (deterministic).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N CT scans across the dataset (None = all).",
    )
    p.add_argument(
        "--skip-spineps",
        action="store_true",
        help="Skip the SPINEPS vertebra/spine stage (expects spine + vert outputs to already exist).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    bgi = BIDS_Global_info(str(root))
    processed = 0
    for _, sub in bgi.enumerate_subjects(sort=args.sort):
        q = sub.new_query()
        q.filter_format("ct")
        for fam in q.loop_dict():
            assert len(fam.get("ct", [])) == 1, fam.get("ct")  # type: ignore
            ct = fam["ct"][0]
            out_seg = ct.get_changed_path("nii.gz", "msk", info={"seg": "VIBESeg-12", "res": "iso"})
            out_vert = ct.get_changed_path("nii.gz", "msk", info={"seg": "vert-rib"})
            out_spine = ct.get_changed_path("nii.gz", "msk", info={"seg": "spine-rib"})

            run_vibeseg(ct, out_seg, gpu=args.gpu, keep_size=True, dataset_id=12)

            if not args.skip_spineps:
                out = run_spineps(
                    ct,
                    model_instance="ct_instance",
                    model_semantic="ct",
                    model_labeling="ct_labeling",
                    ignore_compatibility_issues=True,
                )
                print(out["out_vert"])
                if not (out_vert.exists() and out_spine.exists()):
                    add_ribs_to_vert_spine(
                        out["out_vert"],
                        out["out_spine"],
                        out_seg,
                        ct,
                        str(root),
                        spine_path_out=out_spine,
                        vert_path_out=out_vert,
                        save=True,
                    )

            run_all(ct, out_seg, out_vert, out_spine)

            processed += 1
            if args.limit is not None and processed >= args.limit:
                return


if __name__ == "__main__":
    main()
