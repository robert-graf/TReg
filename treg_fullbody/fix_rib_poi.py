import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import numpy as np
from TPTBox import BIDS_FILE, NII, BIDS_Global_info, POI_Global, Print_Logger, to_nii
from TPTBox.core.bids_files import Buffered_BIDS_Global_info
from TPTBox.core.vert_constants import Full_Body_Instance, Location
from TPTBox.mesh3D.snapshot3D import make_snapshot3D_parallel
from TPTBox.segmentation import run_vibeseg
from tqdm import tqdm


class RegionDef(NamedTuple):
    key: str
    label: str
    idx: list[int]
    poi_source: str  # bids family key for the poi file
    bone_key: str  # bids family key for background bone seg
    seg_key: str  # bids family key for the highlighted seg
    crop_margin: int  # voxel padding around crop
    parent: str  # output derivatives folder
    category: str  # GUI filter category
    sub_idx: list[int] = None  # type: ignore
    poi_idx: None | list[tuple[int, int]] = None


REGIONS: list[RegionDef] = [
    *[
        RegionDef(
            f"rib-right-{n}",
            f"Rib R {n}",
            list(range(40, 53)),
            "poi_seg-torso",
            "msk_seg-vert",
            "msk_seg-vert",
            30,
            "derivatives-VIBESeg-12-points-snp",
            "Rib",
            sub_idx=[n],
        )
        for n in range(1, 7)
    ],
    *[
        RegionDef(
            f"rib-left-{n}",
            f"Rib L {n}",
            list(range(140, 153)),
            "poi_seg-torso",
            "msk_seg-vert",
            "msk_seg-vert",
            30,
            "derivatives-VIBESeg-12-points-snp",
            "Rib",
            sub_idx=[n],
        )
        for n in range(1, 7)
    ],
]
logger = Print_Logger()
# Pois 40 - 52 / 140 - 152; fixable ribs [1-6]
REGIONS_: list[RegionDef] = REGIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fit_curve(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(coords)
    t = np.arange(n, dtype=float)
    coeffs = np.polyfit(t, coords, deg=3)  # degree 3 for better extrapolation
    return coeffs, t


def _predict(coeffs: np.ndarray, t: float) -> np.ndarray:
    return np.array([np.polyval(coeffs[:, ax], t) for ax in range(3)])


# ---------------------------------------------------------------------------
# Centerline-based rib voxel placement
# ---------------------------------------------------------------------------


def _rib_centerline_point(
    ribs_nii: "NII", rib_label: int, seed_world: np.ndarray, neighborhood_mm: float = 15.0, itk_coords: bool = False
) -> np.ndarray | None:
    """
    Return the center-of-mass of the rib voxels within `neighborhood_mm` of
    `seed_world`.  Doubles the radius once if the neighbourhood is empty.
    Returns None only if the rib label is absent entirely.
    """
    arr = ribs_nii.get_array()
    vox_coords = np.argwhere(arr == rib_label)
    if len(vox_coords) == 0:
        return None

    affine = ribs_nii.affine
    ones = np.ones((len(vox_coords), 1))
    world_coords = (affine @ np.hstack([vox_coords, ones]).T).T[:, :3]
    if itk_coords:
        world_coords[:, 0] *= -1
        world_coords[:, 1] *= -1

    dists = np.linalg.norm(world_coords - seed_world, axis=1)

    for radius in (neighborhood_mm, neighborhood_mm * 2):
        mask = dists <= radius
        if mask.any():
            return world_coords[mask].mean(axis=0)

    # Last resort for very short ribs
    return world_coords[int(np.argmin(dists))]


# ---------------------------------------------------------------------------
# Validity checks for a newly proposed point
# ---------------------------------------------------------------------------


def _is_clustered(new_coord: np.ndarray, placed_coords: list[np.ndarray], cluster_threshold_mm: float) -> bool:
    """
    Return True if `new_coord` is within `cluster_threshold_mm` of any
    coord in `placed_coords`.  Used to detect when two different columns
    land on the same short rib at nearly the same location.
    """
    return any(np.linalg.norm(new_coord - p) < cluster_threshold_mm for p in placed_coords)


# ---------------------------------------------------------------------------
# Main fix function
# ---------------------------------------------------------------------------
def is_rib_fixed(img_file: BIDS_FILE, parent):
    for i in ["ribs-left", "ribs-right"]:
        out_rib = img_file.get_changed_path("json", "poi", parent=parent, info={"seg": i + "-post"})
        if not out_rib.exists():
            return False
    return True


def fix_rib(task, rib_instance, img_file, poi_file, parent):
    if task.task_id in ["ribs-left", "ribs-right"]:
        spine_seg = rib_instance.parent / (rib_instance.name.replace("vert", "spine"))
        out_rib = img_file.get_changed_path("json", "poi", parent=parent, info={"seg": task.task_id + "-post"})
        _fix_rib(BIDS_FILE(poi_file, img_file.dataset), rib_instance, spine_seg, out_rib)
        return out_rib
    return poi_file


def _fix_rib(poi_bids: BIDS_FILE, vert, spine, out_path):
    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    subj = str(poi_bids.get("sub")) + "-" + str(poi_bids.get("ses"))
    vert = to_nii(vert, True)
    spine = to_nii(spine, True)
    ribs_left: NII = vert * spine.extract_label(Location.Rib_Left.value)
    ribs_right: NII = vert * spine.extract_label(Location.Rib_Right.value)
    poi = POI_Global.load(poi_bids, True)
    # Clustering is detected across columns for the same (rib_index, side):
    # if two columns place rib 11 at nearly the same spot, the second is rejected.
    # Key: (rib_index, left_str) → list of world coords already placed this run.
    placed_per_rib: dict[tuple[int, str], list[np.ndarray]] = {}
    for region in REGIONS_:
        column_idx = region.sub_idx[0]
        idxs = region.idx  # [40-52] or [140-152]
        left = "left" if idxs[0] > 100 else "right"

        # Collect existing POI world coords for all ribs on this side
        existing_coords: dict[int, np.ndarray] = {}
        for rib_num, poi_idx in zip(range(1, 14), region.idx):
            try:
                coord = poi[poi_idx, column_idx]
                if coord is not None:
                    existing_coords[rib_num] = np.asarray(coord, dtype=float)
            except (KeyError, TypeError):
                pass

        rib = ribs_left if left == "left" else ribs_right
        u = rib.unique()

        # Median inter-rib spacing from good ribs (1–10) for neighborhood / cluster threshold
        ref_ribs_good = sorted(k for k in existing_coords if k <= 10)
        if len(ref_ribs_good) >= 2:
            ref_coords_good = np.stack([existing_coords[k] for k in ref_ribs_good])
            spacings = [np.linalg.norm(ref_coords_good[i + 1] - ref_coords_good[i]) for i in range(len(ref_coords_good) - 1)]
            median_spacing = float(np.median(spacings))
        else:
            median_spacing = 20.0

        neighborhood_mm = max(10.0, median_spacing * 0.5)
        cluster_threshold_mm = median_spacing * 0.3

        # stop_placing is per column_idx: once this column can no longer produce
        # valid points (rib absent in segmentation, not enough ref ribs, …)
        # we clean up any remaining higher rib_index POIs and move on.
        stop_placing = False

        for rib_index in (11, 12, 13):
            target_label = idxs[0] + (rib_index - 1)
            target_poi_idx = target_label

            # Always remove existing POI — re-added below only if valid.
            if had_existing := (target_poi_idx, column_idx) in poi:
                poi.remove_((target_poi_idx, column_idx))
            if column_idx != 1 and (target_poi_idx, column_idx - 1) not in poi:
                continue

            if stop_placing:
                if had_existing:
                    logger.on_debug(f"[{subj}] rib {rib_index}-{column_idx}-{left}: column stopped — removed existing POI")
                continue

            if target_label % 100 not in u:
                stop_placing = True
                logger.on_debug("No rib", target_label)
                continue

            # Curve from all ribs placed so far for this column (1–10 + any
            # accepted lower ribs).  Missing rib_index entries mean that
            # rib was rejected — curve naturally skips them.
            ref_ribs = sorted(k for k in existing_coords if k <= rib_index - 1)
            if len(ref_ribs) < 4:
                # logger.on_debug(f"[{subj}] rib {rib_index}-{column_idx}-{left}: not enough ref ribs")
                stop_placing = True
                continue

            ref_coords = np.stack([existing_coords[k] for k in ref_ribs])
            coeffs, t_ref = _fit_curve(ref_coords)
            rib_min, rib_max = ref_ribs[0], ref_ribs[-1]
            t_for_target = t_ref[-1] * (rib_index - rib_min) / (rib_max - rib_min)
            seed_world = _predict(coeffs, t_for_target)

            new_coord = _rib_centerline_point(
                rib, target_label % 100, seed_world, neighborhood_mm=neighborhood_mm, itk_coords=poi.itk_coords
            )

            if new_coord is None:
                stop_placing = True
                continue

            # --- Clustering check: same rib_index, different column_idx ---
            # If another column already placed a point for this rib at nearly
            # the same location, the short rib is not contributing new info.
            key = (rib_index, left)
            same_rib_placed = placed_per_rib.get(key, [])
            if _is_clustered(new_coord, same_rib_placed, cluster_threshold_mm):
                logger.on_debug(f"[{subj}] rib {rib_index}-{column_idx}-{left}: clusters with same rib_index in another column — skipping")
                # Don't stop the column — other rib_indices may still be fine.
                # Don't add to existing_coords so next rib_index curve skips it.
                stop_placing = True
                continue

            # Point is valid — write it and register for future cluster checks
            poi[target_poi_idx, column_idx] = tuple(new_coord)
            # poi_new[target_poi_idx, column_idx] = tuple(new_coord)
            existing_coords[rib_index] = new_coord
            placed_per_rib.setdefault(key, []).append(new_coord)

            logger.on_bold(
                f"[{subj}] rib {rib_index}-{column_idx}-{left}: "
                f"placed POI at {np.round(new_coord, 1)} "
                f"(seed {np.round(seed_world, 1)}, r={neighborhood_mm:.1f} mm)"
            )
    # exit()
    poi.save(out_path)
    poi.save_mrk(out_path)
    return out_path


def _worker(args):
    raise NotImplementedError()
    subj, fam = args
    return _fix_rib(subj, fam)


def post_process(
    dataset_path: str | Path,
    cpus: int = 8,
    region_keys: list[str] | None = None,
    progress_callback=None,
) -> list[Path]:
    dataset = Path(dataset_path)
    bgi = BIDS_Global_info(dataset, ["derivatives-final-points", "derivatives-final", "derivatives-treg"])

    all_subjects = list(bgi.iter_subjects(sort=True))
    tasks = []

    for subj, sub in all_subjects:
        q = sub.new_query()
        q.filter_format("msk")
        q.filter("seg", "torso")
        q.flatten()
        q.filter_self(lambda x: x.get("seg") != "leg" or x.parent == "derivatives-final-points")
        q.unflatten()
        q.filter("sub", "CTFU00065")
        q.filter("ses", "00000")
        for fam in q.loop_dict():
            tasks.append((subj, fam))  # noqa: PERF401

    for task in tasks:
        _worker(task)
    exit()

    created = []
    with ProcessPoolExecutor(max_workers=cpus) as ex:
        for a in tqdm(ex.map(_worker, tasks), total=len(tasks)):
            if a is not None and a[0] is not None:
                created.append(a)
    return created


if __name__ == "__main__":
    dataset = "/DATA/NAS/datasets_processed/CT_spine/dataset-myelom"
    created = post_process(dataset, progress_callback=None)
