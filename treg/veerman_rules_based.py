"""
Veerman 3D Leg Alignment Analysis — Standalone (No Blender)
===========================================================

Pure-Python reimplementation of the Veerman alignment pipeline from
veerman_alignment_analysis.py, with all Blender (bpy) dependencies
replaced by trimesh + numpy + scipy.

Reference: Veerman et al. (2025) — KSSTA 2025;33:2276-2292.

Dependencies: trimesh>=4.0, numpy, scipy
Usage: Import functions and call step-by-step, or use batch_runner.py.
"""

import csv
import logging
import math
import os
import sys
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import stl
import trimesh
from numpy.linalg import eigh, lstsq, norm, svd
from TPTBox import NII, POI, POI_Global, Print_Logger, to_nii
from TPTBox.core.poi_fun.save_mkr import MKR_Lines

out = str(Path(__file__).parent.parent)
sys.path.append(out)
from treg.mesh_analysis import AnalysisError, MeshWrapper, fit_cylinder

logger = Print_Logger(prefix="veerman")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

POI_MAP = {
    "TGT": (1, 1),
    "FHC": (1, 2),
    "FNC": (1, 3),
    "FAAP": (1, 4),
    "FLCD": (2, 1),
    "FMCD": (2, 2),
    "FLCP": (2, 3),
    "FMCP": (2, 4),
    "FNP": (2, 5),
    "FADP": (2, 6),
    "TGPP": (2, 7),
    "TGCP": (2, 8),
    "FMCPC": (2, 9),
    "FLCPC": (2, 10),
    "TRMP": (2, 11),
    "TRLP": (2, 12),
    "TLCL": (3, 1),
    "TMCM": (3, 2),
    "TKC": (3, 3),
    "TLCA": (3, 4),
    "TLCP": (3, 5),
    "TMCA": (3, 6),
    "TMCP": (3, 7),
    "TTP": (3, 8),
    "TAAP": (3, 9),
    "TMIT": (3, 10),
    "TLIT": (3, 11),
    "FLM": (4, 1),
    "TMM": (4, 2),
    "TAC": (4, 3),
    "TADP": (4, 4),
    "ankle_center": (4, 5),
    "PPP": (5, 1),
    "PDP": (5, 2),
    "PMP": (5, 3),
    "PLP": (5, 4),
    "PRPP": (5, 5),
    "PRDP": (5, 6),
    "PRHP": (5, 7),
    "cylinder-center": (6, 1),
    "cylinder-axis-point-1": (6, 2),
    "cylinder-axis-point-2": (6, 3),
    "cylinder-radius-point-1": (6, 4),
    "cylinder-radius-point-2": (6, 5),
    "cylinder-radius-point-3": (6, 6),
    "cylinder-radius-point-4": (6, 7),
    "cylinder-radius-point-5": (6, 8),
    "cylinder-radius-point-6": (6, 9),
    "cylinder-radius-point-7": (6, 10),
    "cylinder-radius-point-8": (6, 11),
    "dist_fem_center": (7, 1),
    "dist_fem_center-ray1": (7, 2),
    "dist_fem_center-ray2": (7, 3),
}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ID_TO_MESH_NAME = {
    10: "fem_head",
    4: "fem_neck",
    15: "fem_condyle_medial",
    14: "fem_condyle_lateral",
    12: "fem_trochlea_medial",
    13: "fem_trochlea_lateral",
    11: "tibia_plateau_medial",
    5: "tibia_plateau_lateral",
    18: "ankle_malleolus_medial",
    17: "ankle_malleolus_lateral",
    9: "ankle_malleolus_mid",
}

MESH_NAME_TO_ID = {v: k for k, v in ID_TO_MESH_NAME.items()}
EXPECTED_MESHES = list(ID_TO_MESH_NAME.values())


# =============================================================================
# HELPERS
# =============================================================================


def _check_required_keys(results, required, step_name):
    missing = [k for k in required if k not in results]
    if missing:
        raise AnalysisError(f"Missing required keys for {step_name}: {missing}.")


# =============================================================================
# STEP 0 — SYNTHETIC VALIDATION
# =============================================================================


def stl_to_trimesh(cube: stl.mesh.Mesh) -> trimesh.Trimesh:
    # cube.vectors shape: (n_faces, 3, 3)
    vertices = cube.vectors.reshape(-1, 3)

    # Each triangle gets 3 consecutive vertices
    faces = np.arange(len(vertices)).reshape(-1, 3)

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


_buffer = {}


def get_stl(nii: NII, stem="fem_head", stl_folder: Path | None = None, side: Literal["R", "L"] = "L"):
    out = Path(stl_folder) / f"{stem}_{side}.stl" if stl_folder else None
    if out is not None and out in _buffer:
        return _buffer[out]
    if out is not None and out.exists():
        wrapper = MeshWrapper.load(out, name=stem)
    else:
        # nii.save(Path(stl_folder, "all.nii.gz"))
        stl = nii.to_stl(MESH_NAME_TO_ID[stem], out_path=out)
        wrapper = MeshWrapper(stem, stl_to_trimesh(stl))
    if out is not None:
        _buffer[out] = wrapper
    return wrapper


def load_save_stls(nii: "str | Path | NII", results: dict, stl_folder: Path, side: Literal["R", "L"], verbose=True):
    """Generate STL files from NII segmentation."""
    logger.info("=" * 60, verbose=verbose)
    logger.info("Load STLs", verbose=verbose)
    logger.info("=" * 60, verbose=verbose)

    results["side"] = side
    results["stl_folder"] = stl_folder
    results["meshes"] = {}
    results["warnings"] = []

    nii = to_nii(nii, True)
    loaded = []
    for stem in ID_TO_MESH_NAME.values():
        results["meshes"][stem] = get_stl(nii, stem, stl_folder, side)
        loaded.append(stem)
    if len(loaded) != len(EXPECTED_MESHES):
        missing = set(EXPECTED_MESHES) - set(loaded)
        raise AnalysisError(f"Missing meshes: {missing}")
    logger.info(f"\n{'Mesh':<30} {'Vertices':>10} {'Faces':>10} {'BBox X':>10} {'BBox Y':>10} {'BBox Z':>10}", verbose=verbose)
    logger.info("-" * 82, verbose=verbose)
    for stem in EXPECTED_MESHES:
        wrapper = results["meshes"][stem]
        coords = wrapper.vertices
        n_verts = len(coords)
        n_faces = len(wrapper.faces)
        bb_size = coords.max(axis=0) - coords.min(axis=0)
        logger.info(f"{stem:<30} {n_verts:>10} {n_faces:>10} {bb_size[0]:>10.1f} {bb_size[1]:>10.1f} {bb_size[2]:>10.1f}", verbose=verbose)

        if n_verts < 20:
            raise AnalysisError(f"{stem} has only {n_verts} vertices (< 20)")
        for dim, sz in zip(["X", "Y", "Z"], bb_size):
            if sz < 5:
                w = f"WARN: {stem} bounding box {dim} = {sz:.1f} mm (< 5 mm)"
                logger.warning(w)
                results["warnings"].append(w)
            if sz > 500:
                w = f"WARN: {stem} bounding box {dim} = {sz:.1f} mm (> 500 mm)"
                logger.warning(w)
                results["warnings"].append(w)

        faces = wrapper.faces
        v0 = coords[faces[:, 0]]
        v1 = coords[faces[:, 1]]
        v2 = coords[faces[:, 2]]
        areas = 0.5 * norm(np.cross(v1 - v0, v2 - v0), axis=1)
        n_degen = int((areas < 1e-10).sum())
        n_tris = len(faces)
        if n_tris > 0 and n_degen / n_tris > 0.01:
            w = f"WARN: {stem} has {n_degen}/{n_tris} degenerate faces ({100 * n_degen / n_tris:.1f}%)"
            logger.warning(w)
            results["warnings"].append(w)

    logger.info(f"\nSTEP 1a COMPLETE: {len(loaded)} meshes loaded, side={side}", verbose=verbose)


# =============================================================================
# STEP 2a — COMPUTE SIMPLE JOINT CENTERS
# =============================================================================


def get_FHC(
    nii: NII,
    out_poi: POI_Global | None = None,
    stl_folder: Path | None = None,
    side: Literal["R", "L"] = "L",
    warnings: list | None = None,
    verbose=True,
):
    if warnings is None:
        warnings = []
    mesh = get_stl(nii, stem="fem_head", stl_folder=stl_folder, side=side)
    sph = mesh.fit_sphere()
    if out_poi is not None:
        out_poi[POI_MAP["FHC"]] = sph["center"]
        out_poi.info["fem_head_radius"] = sph["radius"]
        out_poi.info["fem_head_rmse"] = sph["rmse"]
    if verbose:
        logger.info(f"  Center: ({sph['center'][0]:.2f}, {sph['center'][1]:.2f}, {sph['center'][2]:.2f})")
        logger.info(f"  Radius: {sph['radius']:.2f} mm")
        logger.info(f"  RMSE: {sph['rmse']:.4f} mm")
        logger.info(f"  Max residual: {sph['max_residual']:.4f} mm")
        logger.info(f"  Mean signed residual: {sph['mean_signed_residual']:.4f} mm")
        logger.info(f"  Residual std: {sph['residual_std']:.4f} mm")

    if sph["rmse"] > 5.0:
        raise AnalysisError(f"Sphere RMSE {sph['rmse']:.2f} > 5 mm")
    if not (20 <= sph["radius"] <= 28):
        w = f"WARN: Sphere radius {sph['radius']:.2f} mm outside 20-28 mm range"
        logger.warning(w)
        warnings.append(w)
    if sph["rmse"] > 2.0:
        w = f"WARN: Sphere RMSE {sph['rmse']:.2f} > 2 mm"
        logger.warning(w)
        warnings.append(w)
    if abs(sph["mean_signed_residual"]) > 0.5:
        w = f"WARN: Mean signed residual {sph['mean_signed_residual']:.4f} > 0.5 mm (bias)"
        logger.warning(w)
        warnings.append(w)
    return sph


def _get_centroid(
    nii: NII,
    out_poi: POI_Global | None = None,
    stl_folder: Path | None = None,
    side: Literal["R", "L"] = "L",
    warnings: list | None = None,
    verbose=True,
    stem: str | list[str] = None,  # noqa: RUF013 # type: ignore
    name: str | None = None,
    key: str = None,  # noqa: RUF013 # type: ignore
):
    if name is None:
        name = str(stem)
    if warnings is None:
        warnings = []

    if out_poi is not None and POI_MAP[key] in out_poi:
        neck_centroid = out_poi[POI_MAP[key]]
    else:
        if isinstance(stem, str):
            mesh = get_stl(nii, stem=stem, stl_folder=stl_folder, side=side)
        else:
            mesh = MeshWrapper.concatenate([get_stl(nii, stem=s, stl_folder=stl_folder, side=side) for s in stem])
        neck_centroid = mesh.area_weighted_centroid()
        if out_poi is not None:
            out_poi[POI_MAP[key]] = neck_centroid

    logger.info(f"{name} centroid: ({neck_centroid[0]:.2f}, {neck_centroid[1]:.2f}, {neck_centroid[2]:.2f})", verbose=verbose)
    return np.array(neck_centroid)


get_FNC = partial(_get_centroid, stem="fem_neck", name="Femoral neck", key="FNC")
get_TMCM = partial(_get_centroid, stem="tibia_plateau_medial", name="Medial plateau", key="TMCM")
get_TLCL = partial(_get_centroid, stem="tibia_plateau_lateral", name="Lateral plateau centroid", key="TLCL")
get_TMM = partial(_get_centroid, stem="ankle_malleolus_medial", name="Ankle medial malleolus", key="TMM")
get_FLM = partial(_get_centroid, stem="ankle_malleolus_lateral", name="Ankle lateral malleolus", key="FLM")
get_TAC = partial(_get_centroid, stem="ankle_malleolus_mid", name="Ankle plafond", key="TAC")
get_ankle_center = partial(
    _get_centroid,
    stem=["ankle_malleolus_medial", "ankle_malleolus_lateral", "ankle_malleolus_mid"],
    name="ankle center",
    key="ankle_center",
)


def get_TKC(
    nii: NII,
    out_poi: POI_Global | None = None,
    stl_folder: Path | None = None,
    side: Literal["R", "L"] = "L",
    warnings: list | None = None,
    verbose=True,
    _name="Proximal tibial",
    _p1=get_TMCM,
    _p2=get_TLCL,
    _key="TKC",
):
    logger.info() if verbose else None
    med_plat_centroid = _p1(nii, out_poi, stl_folder, side, warnings, verbose)
    lat_plat_centroid = _p2(nii, out_poi, stl_folder, side, warnings, verbose)
    prox_tib_center = (med_plat_centroid + lat_plat_centroid) / 2.0
    if out_poi is not None:
        out_poi[POI_MAP[_key]] = prox_tib_center

    logger.info(f"{_name} center: ({prox_tib_center[0]:.2f}, {prox_tib_center[1]:.2f}, {prox_tib_center[2]:.2f})", verbose=verbose)


get_TRMP = partial(_get_centroid, stem="fem_trochlea_medial", name=None, key="TRMP")
get_TRLP = partial(_get_centroid, stem="fem_trochlea_lateral", name=None, key="TRLP")
get_TGCP = partial(get_TKC, _name="trochlea_center", _p1=get_TRMP, _p2=get_TRLP, _key="TGCP")

get_FLCD = partial(_get_centroid, stem="fem_condyle_lateral", name="Ankle plafond", key="FLCD")
get_FMCD = partial(_get_centroid, stem="fem_condyle_medial", name="Ankle plafond", key="FMCD")
get_FNP = partial(get_TKC, _name="trochlea_center", _p1=get_FLCD, _p2=get_FMCD, _key="FNP")


def compute_points(
    nii: NII,
    poi: POI_Global,
    stl_folder: Path | None = None,
    side: Literal["R", "L"] = "L",
    warnings: list | None = None,
    verbose=True,
):
    """Compute femoral head center (sphere fit), neck center, proximal tibial
    center, and ankle center using area-weighted centroids.
    All coordinates stored in local (voxel) space; converted to global for export.
    """
    if warnings is None:
        warnings = []
    logger.info("=" * 60)
    logger.info("STEP 2a: Compute Simple Joint Centers")
    logger.info("=" * 60)

    get_FHC(nii, poi, stl_folder, side, warnings, verbose)
    # --- Femoral neck center ---
    get_FNC(nii, poi, stl_folder, side, warnings, verbose)
    get_TKC(nii, poi, stl_folder, side, warnings, verbose)
    print()
    get_TMM(nii, poi, stl_folder, side, warnings, verbose)
    get_FLM(nii, poi, stl_folder, side, warnings, verbose)
    get_TAC(nii, poi, stl_folder, side, warnings, verbose)
    get_ankle_center(nii, poi, stl_folder, side, warnings, verbose)
    print()
    get_TGCP(nii, poi, stl_folder, side, warnings, verbose)
    get_FNP(nii, poi, stl_folder, side, warnings, verbose)
    poi.info["cranial_dir"] = _comp_dir(poi, "FHC", "ankle_center", name="Anatomical cranial")
    ## Cylinder
    posterior_dir = _comp_dir(poi, "FNP", "TGCP", name="Posterior direction (trochlea->condyle)")
    posterior_pts_all = []
    for condyle_name in ["fem_condyle_medial", "fem_condyle_lateral"]:
        wrapper = get_stl(nii, stem=condyle_name, stl_folder=stl_folder, side=side)
        pts = wrapper.vertices
        centroid = wrapper.area_weighted_centroid()
        offsets = pts - centroid
        posterior_proj = np.dot(offsets, posterior_dir)
        posterior_pts = pts[posterior_proj > 0]
        logger.info(f"{condyle_name}: {len(pts)} total vertices, {len(posterior_pts)} posterior")
        posterior_pts_all.append(posterior_pts)
    all_posterior = np.vstack(posterior_pts_all)
    logger.info(f"Total posterior vertices for cylinder fit: {len(all_posterior)}")
    if len(all_posterior) < 100:
        raise AnalysisError(f"Only {len(all_posterior)} posterior vertices (< 100)")
    cyl = fit_cylinder(all_posterior)
    poi.info["cylinder_axis"] = cyl["axis"]
    poi.info["cylinder_point"] = cyl["point"]
    poi.info["cylinder_radius"] = cyl["radius"]
    poi.info["cylinder_rmse"] = cyl["rmse"]

    poi[POI_MAP["cylinder-center"]] = cyl["point"]
    axis_length = norm(np.array(poi[POI_MAP["FLCD"]]) - np.array(poi[POI_MAP["FMCD"]]))
    axis = cyl["axis"]
    center = cyl["point"]
    radius = cyl["radius"]

    tmp = np.array([1.0, 0.0, 0.0])
    tmp2 = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(tmp, axis)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
        tmp2 = np.array([0.0, 0.0, 1.0])
    ortho = np.cross(axis, tmp)
    ortho = ortho / np.linalg.norm(ortho)
    ortho2 = np.cross(axis, tmp2)
    ortho2 = ortho2 / np.linalg.norm(ortho2)
    n_diag = np.linalg.norm(ortho + ortho2)

    poi[POI_MAP["cylinder-axis-point-1"]] = center + axis_length * axis
    poi[POI_MAP["cylinder-axis-point-2"]] = center - axis_length * axis
    poi[POI_MAP["cylinder-radius-point-1"]] = center + radius * ortho
    poi[POI_MAP["cylinder-radius-point-2"]] = center - radius * ortho
    poi[POI_MAP["cylinder-radius-point-3"]] = center + radius * ortho2
    poi[POI_MAP["cylinder-radius-point-4"]] = center - radius * ortho2
    poi[POI_MAP["cylinder-radius-point-5"]] = center + radius * (ortho + ortho2) / n_diag
    poi[POI_MAP["cylinder-radius-point-6"]] = center - radius * (ortho + ortho2) / n_diag
    poi[POI_MAP["cylinder-radius-point-7"]] = center + radius * (ortho - ortho2) / n_diag
    poi[POI_MAP["cylinder-radius-point-8"]] = center - radius * (ortho - ortho2) / n_diag

    logger.info("\nCylinder fit results:")
    logger.info(f"  Axis: ({cyl['axis'][0]:.4f}, {cyl['axis'][1]:.4f}, {cyl['axis'][2]:.4f})")
    logger.info(f"  Point (local): ({cyl['point'][0]:.2f}, {cyl['point'][1]:.2f}, {cyl['point'][2]:.2f})")
    logger.info(f"  Radius: {cyl['radius']:.2f} mm")
    logger.info(f"  RMSE: {cyl['rmse']:.4f} mm")
    logger.info(f"  Max residual: {cyl['max_residual']:.4f} mm")
    logger.info(f"  Mean signed residual: {cyl['mean_signed_residual']:.4f} mm")

    if cyl["rmse"] > 8.0:
        raise AnalysisError(f"Cylinder RMSE {cyl['rmse']:.2f} > 8 mm")
    if not (18 <= cyl["radius"] <= 30):
        w = f"WARN: Cylinder radius {cyl['radius']:.2f} mm outside 18-30 mm range"
        logger.warning(w)
        warnings.append(w)
    if cyl["rmse"] > 3.0:
        w = f"WARN: Cylinder RMSE {cyl['rmse']:.2f} > 3 mm"
        logger.warning(w)
        warnings.append(w)
    if abs(cyl["mean_signed_residual"]) > 1.0:
        w = f"WARN: Cylinder mean signed residual {cyl['mean_signed_residual']:.4f} > 1 mm"
        logger.warning(w)
        warnings.append(w)
    condyle_line_norm = _comp_dir(poi, "FMCD", "FLCD", name="condyle_line_norm", verbose=False)
    axis_angle_to_condyle = math.degrees(math.acos(min(1.0, abs(np.dot(cyl["axis"], condyle_line_norm)))))
    logger.info(f"  Axis angle to condyle line: {axis_angle_to_condyle:.1f} deg")
    if axis_angle_to_condyle > 15:
        w = f"WARN: Cylinder axis-to-condyle-line angle {axis_angle_to_condyle:.1f} > 15 deg"
        logger.warning(w)
        warnings.append(w)

    # --- Distal femoral center via ray-cast along cylinder axis ---
    axis_dir = cyl["axis"]
    axis_pt = cyl["point"]

    all_cond_pts = np.vstack(
        [get_stl(nii, "fem_condyle_medial", stl_folder, side).vertices, get_stl(nii, "fem_condyle_lateral", stl_folder, side).vertices]
    )
    max_extent = np.ptp(all_cond_pts, axis=0).max()
    ray_distance = max_extent * 2.0

    perp1 = np.cross(axis_dir, np.array([0.0, 0.0, 1.0]))
    if norm(perp1) < 1e-6:
        perp1 = np.cross(axis_dir, np.array([0.0, 1.0, 0.0]))
    perp1 = perp1 / norm(perp1)
    perp2 = np.cross(axis_dir, perp1)
    perp2 = perp2 / norm(perp2)

    GRID_HALF = 15.0
    GRID_N = 7
    grid_offsets = np.linspace(-GRID_HALF, GRID_HALF, GRID_N)
    ray_origins_list = []
    ray_dirs_list = []
    for o1 in grid_offsets:
        for o2 in grid_offsets:
            base = axis_pt + o1 * perp1 + o2 * perp2
            ray_origins_list.append(base - ray_distance * axis_dir)
            ray_dirs_list.append(axis_dir.copy())
            ray_origins_list.append(base + ray_distance * axis_dir)
            ray_dirs_list.append(-axis_dir.copy())
    grid_origins = np.array(ray_origins_list)
    grid_dirs = np.array(ray_dirs_list)

    all_hit_projs = []
    for condyle_name in ["fem_condyle_medial", "fem_condyle_lateral"]:
        wrapper = get_stl(nii, condyle_name, stl_folder, side)
        hits_before = len(all_hit_projs)
        locs, idx_ray, idx_tri = wrapper._tm.ray.intersects_location(
            ray_origins=grid_origins,
            ray_directions=grid_dirs,
            multiple_hits=True,
        )
        if len(locs) > 0:
            projs = np.dot(locs - axis_pt, axis_dir)
            all_hit_projs.extend(projs.tolist())
        condyle_hits = len(all_hit_projs) - hits_before
        n_rays_hit = len(np.unique(idx_ray)) if len(locs) > 0 else 0
        logger.info(f"  {condyle_name}: {condyle_hits} ray-cast hits ({n_rays_hit}/{len(grid_origins)} rays)")

    if len(all_hit_projs) < 2:
        logger.warning("Grid ray-cast found < 2 hits — falling back to vertex projection")
        for condyle_name in ["fem_condyle_medial", "fem_condyle_lateral"]:
            pts = get_stl(nii, condyle_name, stl_folder, side).vertices
            projections = np.dot(pts - axis_pt, axis_dir)
            all_hit_projs.extend([projections.min(), projections.max()])

    all_min_proj = min(all_hit_projs)
    all_max_proj = max(all_hit_projs)
    mid_proj = (all_min_proj + all_max_proj) / 2.0
    dist_fem_center = axis_pt + mid_proj * axis_dir
    p1 = axis_pt + all_min_proj * axis_dir
    p2 = axis_pt + all_max_proj * axis_dir

    poi.info["dist_fem_center"] = dist_fem_center
    poi[POI_MAP["dist_fem_center"]] = dist_fem_center
    poi[POI_MAP["dist_fem_center-ray1"]] = p1
    poi[POI_MAP["dist_fem_center-ray2"]] = p2

    logger.info(f"\nDistal femoral center : ({dist_fem_center[0]:.2f}, {dist_fem_center[1]:.2f}, {dist_fem_center[2]:.2f})")
    logger.info(f"  Ray-cast intersection 1 : ({p1[0]:.2f}, {p1[1]:.2f}, {p1[2]:.2f})")
    logger.info(f"  Ray-cast intersection 2: ({p2[0]:.2f}, {p2[1]:.2f}, {p2[2]:.2f})")
    logger.info(f"  Condylar width: {norm(p1 - p2):.2f} mm")

    logger.info("\ncompute points COMPLETE: Distal femoral center computed.")


def _comp_dir(poi, name1, name2, name="Anatomical cranial", verbose=True):
    cranial_vec = np.array(poi[POI_MAP[name1]]) - np.array(poi[POI_MAP[name2]])
    cranial_dir = cranial_vec / norm(cranial_vec)
    logger.info(f"{name} direction: ({cranial_dir[0]:.4f}, {cranial_dir[1]:.4f}, {cranial_dir[2]:.4f})", verbose=verbose)
    return cranial_dir


# =============================================================================
# STEP 3 — BUILD COORDINATE SYSTEMS
# =============================================================================


def step_3(poi: POI_Global):
    """Construct femoral and tibial coordinate systems and mechanical axes."""
    logger.info("=" * 60)
    logger.info("STEP 3: Build Coordinate Systems")
    logger.info("=" * 60)

    required = [
        "fem_head_center",
        "dist_fem_center",
        "prox_tib_center",
        "ankle_center",
        "cylinder_axis",
        "med_plateau_centroid",
        "lat_plateau_centroid",
        "cond_med_centroid",
        "cond_lat_centroid",
        "side",
        "cranial_dir",
    ]
    _check_required_keys(poi.info, required, "step_3")
    results = poi.info
    cranial_dir = results["cranial_dir"]
    fem_head = results["fem_head_center"]
    dist_fem = results["dist_fem_center"]
    prox_tib = results["prox_tib_center"]
    ankle = results["ankle_center"]

    mech_fem_axis = fem_head - dist_fem
    mech_fem_length = norm(mech_fem_axis)
    mech_fem_axis_unit = mech_fem_axis / mech_fem_length

    mech_tib_axis = prox_tib - ankle
    mech_tib_length = norm(mech_tib_axis)
    mech_tib_axis_unit = mech_tib_axis / mech_tib_length

    logger.info(f"Mechanical femoral axis length: {mech_fem_length:.1f} mm")
    logger.info(f"Mechanical tibial axis length: {mech_tib_length:.1f} mm")

    if not (350 <= mech_fem_length <= 500):
        w = f"WARN: Femoral axis length {mech_fem_length:.1f} mm outside 350-500 mm"
        logger.warning(w)
        results["warnings"].append(w)
    if not (300 <= mech_tib_length <= 450):
        w = f"WARN: Tibial axis length {mech_tib_length:.1f} mm outside 300-450 mm"
        logger.warning(w)
        results["warnings"].append(w)

    # --- Femoral CS ---
    fem_Z = mech_fem_axis_unit.copy()
    if np.dot(fem_Z, cranial_dir) < 0:
        fem_Z = -fem_Z

    cyl_axis = results["cylinder_axis"]
    fem_X = cyl_axis - np.dot(cyl_axis, fem_Z) * fem_Z
    fem_X = fem_X / norm(fem_X)

    # Use pre-computed condyle centroids (no recomputation)
    cond_med_centroid = results["cond_med_centroid"]
    cond_lat_centroid = results["cond_lat_centroid"]
    med_to_lat = cond_lat_centroid - cond_med_centroid
    med_to_lat_projected = med_to_lat - np.dot(med_to_lat, fem_Z) * fem_Z
    if np.dot(fem_X, med_to_lat_projected) < 0:
        fem_X = -fem_X

    # Y = anatomic anterior, derived from trochlea (anterior) → condyle (posterior).
    # This avoids the chirality of cross(Z, X), which would make Y posterior on
    # one side and anterior on the other.
    med_cond_centroid = np.array(poi[POI_MAP["FMCD"]])  # "fem_condyle_medial"
    lat_cond_centroid = np.array(poi[POI_MAP["FLCD"]])  # "fem_condyle_lateral"
    med_troch_centroid = np.array(poi[POI_MAP["TRMP"]])  # fem_trochlea_medial
    lat_troch_centroid = np.array(poi[POI_MAP["TRLP"]])  # fem_trochlea_lateral
    trochlea_center = (med_troch_centroid + lat_troch_centroid) / 2.0
    condyle_center = (med_cond_centroid + lat_cond_centroid) / 2.0
    anterior_anat = trochlea_center - condyle_center
    anterior_anat = anterior_anat / norm(anterior_anat)
    results["anterior_anat"] = anterior_anat

    # Gram-Schmidt: project anterior_anat onto plane orthogonal to BOTH
    # fem_Z and fem_X. Without the second projection, fem_Y inherits any
    # lateral component of anterior_anat (relevant in patients with
    # lateralized trochlea, e.g. dysplasia), making the (X, Y, Z) basis
    # non-orthogonal and breaking sign rules that rely on fem_Y direction.
    fem_Y = (anterior_anat
             - np.dot(anterior_anat, fem_Z) * fem_Z
             - np.dot(anterior_anat, fem_X) * fem_X)
    fem_Y_norm = norm(fem_Y)
    if fem_Y_norm < 1e-10:
        raise AnalysisError(
            "anterior_anat aligned with fem_Z or fem_X — cannot define orthogonal fem_Y"
        )
    fem_Y = fem_Y / fem_Y_norm
    assert abs(np.dot(fem_X, fem_Y)) < 1e-6, "fem_X . fem_Y must be ~0 after Gram-Schmidt"

    results["fem_cs"] = {"origin": dist_fem, "X": fem_X, "Y": fem_Y, "Z": fem_Z}

    logger.info("\nFemoral CS (origin = dist_fem_center):")
    logger.info(f"  X (lateral):  ({fem_X[0]:.4f}, {fem_X[1]:.4f}, {fem_X[2]:.4f})")
    logger.info(f"  Y (anterior): ({fem_Y[0]:.4f}, {fem_Y[1]:.4f}, {fem_Y[2]:.4f})")
    logger.info(f"  Z (cranial):  ({fem_Z[0]:.4f}, {fem_Z[1]:.4f}, {fem_Z[2]:.4f})")

    # --- Tibial CS ---
    tib_Z = mech_tib_axis_unit.copy()
    if np.dot(tib_Z, cranial_dir) < 0:
        tib_Z = -tib_Z

    med_centroid = results["med_plateau_centroid"]
    lat_centroid = results["lat_plateau_centroid"]
    plateau_line = lat_centroid - med_centroid
    tib_X = plateau_line - np.dot(plateau_line, tib_Z) * tib_Z
    tib_X = tib_X / norm(tib_X)
    if np.dot(tib_X, plateau_line) < 0:
        tib_X = -tib_X

    # Y = anatomic anterior, derived from the same anterior_anat reference
    # as fem_Y. Avoids the chirality of cross(tib_Z, tib_X), which produces
    # opposite anatomical directions for left vs right legs and causes
    # TTA, posterior_slope_*, mMPPTA, mLPPTA, PTJ_AP* to flip sign by side.
    # Gram-Schmidt against tib_X for orthonormality of the (X, Y, Z) basis.
    tib_Y = (anterior_anat
             - np.dot(anterior_anat, tib_Z) * tib_Z
             - np.dot(anterior_anat, tib_X) * tib_X)
    tib_Y_norm = norm(tib_Y)
    if tib_Y_norm < 1e-10:
        raise AnalysisError(
            "anterior_anat aligned with tib_Z or tib_X — cannot define orthogonal tib_Y"
        )
    tib_Y = tib_Y / tib_Y_norm
    assert abs(np.dot(tib_X, tib_Y)) < 1e-6, "tib_X . tib_Y must be ~0 after Gram-Schmidt"
    results["tib_cs"] = {"origin": prox_tib, "X": tib_X, "Y": tib_Y, "Z": tib_Z}

    logger.info("\nTibial CS (origin = prox_tib_center):")
    logger.info(f"  X (lateral):  ({tib_X[0]:.4f}, {tib_X[1]:.4f}, {tib_X[2]:.4f})")
    logger.info(f"  Y (anterior): ({tib_Y[0]:.4f}, {tib_Y[1]:.4f}, {tib_Y[2]:.4f})")
    logger.info(f"  Z (cranial):  ({tib_Z[0]:.4f}, {tib_Z[1]:.4f}, {tib_Z[2]:.4f})")

    # --- Validation ---
    fem_rh = np.dot(np.cross(fem_Z, fem_X), fem_Y)
    tib_rh = np.dot(np.cross(tib_Z, tib_X), tib_Y)
    fem_hand = "right-handed" if fem_rh > 0 else "left-handed"
    tib_hand = "right-handed" if tib_rh > 0 else "left-handed"
    # With anatomic Y derived from anterior_anat (a patient-fixed reference
    # that does not change sign with leg side), the basis chirality differs
    # between left and right legs by construction. This is expected and
    # required for measurements like TTA and posterior slope to be
    # side-invariant. Logged for transparency, not raised as an error.
    logger.info(
        f"Handedness: fem={fem_hand} ({fem_rh:+.4f}), "
        f"tib={tib_hand} ({tib_rh:+.4f})"
    )

    fem_Z_align = np.dot(fem_Z, cranial_dir)
    tib_Z_align = np.dot(tib_Z, cranial_dir)
    logger.info(f"Z cranial check: fem_Z·cranial={fem_Z_align:.4f}, tib_Z·cranial={tib_Z_align:.4f} (should be > 0.9)")
    if fem_Z_align < 0.7:
        raise AnalysisError(f"Femoral Z·cranial={fem_Z_align:.4f} < 0.7")
    elif fem_Z_align < 0.9:
        w = f"WARN: Femoral Z·cranial={fem_Z_align:.4f} < 0.9 (axis strongly tilted)"
        logger.warning(w)
        results["warnings"].append(w)
    if tib_Z_align < 0.7:
        raise AnalysisError(f"Tibial Z·cranial={tib_Z_align:.4f} < 0.7")
    elif tib_Z_align < 0.9:
        w = f"WARN: Tibial Z·cranial={tib_Z_align:.4f} < 0.9 (axis strongly tilted)"
        logger.warning(w)
        results["warnings"].append(w)

    # --- Knee extension check ---
    mfa_2d = np.array([np.dot(mech_fem_axis_unit, fem_Y), np.dot(mech_fem_axis_unit, fem_Z)])
    mta_2d = np.array([np.dot(mech_tib_axis_unit, fem_Y), np.dot(mech_tib_axis_unit, fem_Z)])
    mfa_2d_len = norm(mfa_2d)
    mta_2d_len = norm(mta_2d)
    if mfa_2d_len > 1e-10 and mta_2d_len > 1e-10:
        cos_flex = np.clip(np.dot(mfa_2d / mfa_2d_len, mta_2d / mta_2d_len), -1.0, 1.0)
        flexion_angle = math.degrees(math.acos(cos_flex))
        logger.info(f"\nKnee flexion (sagittal): {flexion_angle:.1f} deg")
        if flexion_angle > 10.0:
            w = f"WARN: Knee flexion {flexion_angle:.1f} deg > 10 deg — angles may be unreliable."
            logger.warning(w)
            results["warnings"].append(w)
        elif flexion_angle > 5.0:
            w = f"WARN: Knee flexion {flexion_angle:.1f} deg > 5 deg — mild flexion detected"
            logger.warning(w)
            results["warnings"].append(w)
        else:
            logger.info(f"  PASS: Knee near full extension ({flexion_angle:.1f} deg)")
        results["knee_flexion_angle"] = flexion_angle

    logger.info("\nSTEP 3 COMPLETE: Coordinate systems built.")


# =============================================================================
# STEP 4 — COMPUTE JOINT ORIENTATIONS & ANGLES
# =============================================================================


def fit_plane(points, orient_toward=None):
    """Fit plane to point cloud using SVD."""
    pts = np.asarray(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    U, S, Vt = svd(centered, full_matrices=False)
    normal = Vt[2]
    if orient_toward is not None:
        if np.dot(normal, orient_toward) < 0:
            normal = -normal
    else:
        if normal[2] < 0:
            normal = -normal
    dists = np.dot(centered, normal)
    rmse = math.sqrt(np.mean(dists**2))
    return {"normal": normal, "centroid": centroid, "rmse": rmse}


def step_4(poi: POI_Global):
    """Compute all Veerman Figure 7 alignment angles."""
    logger.info("=" * 60)
    logger.info("STEP 4: Compute Joint Orientations & Angles (Veerman Fig. 7)")
    logger.info("=" * 60)

    required = [
        "fem_cs",
        "tib_cs",
        "fem_head_center",
        "dist_fem_center",
        "fem_neck_center",
        "prox_tib_center",
        "ankle_center",
        "cylinder_axis",
        "meshes",
        "med_plateau_centroid",
        "lat_plateau_centroid",
        "ankle_med_centroid",
        "ankle_lat_centroid",
        "cond_med_centroid",
        "cond_lat_centroid",
        "side",
        "cranial_dir",
    ]
    results = poi.info
    _check_required_keys(results, required, "step_4")

    fem_cs = results["fem_cs"]
    tib_cs = results["tib_cs"]
    side = results["side"]
    cranial_dir = results["cranial_dir"]
    meshes = results["meshes"]
    angles = {}

    fem_Z = fem_cs["Z"]
    fem_X = fem_cs["X"]
    fem_Y = fem_cs["Y"]
    tib_Z = tib_cs["Z"]
    tib_Y = tib_cs["Y"]

    dist_fem = results["dist_fem_center"]
    fem_head = results["fem_head_center"]
    prox_tib = results["prox_tib_center"]
    ankle = results["ankle_center"]
    cond_med_centroid = results["cond_med_centroid"]
    cond_lat_centroid = results["cond_lat_centroid"]

    mech_fem_unit = (fem_head - dist_fem) / norm(fem_head - dist_fem)
    mech_tib_unit = (prox_tib - ankle) / norm(prox_tib - ankle)

    # =====================================================================
    # A) DISTAL FEMORAL JOINT ORIENTATION (DFJ)
    # =====================================================================
    logger.info("\n--- Distal femoral condylar orientation (DFJ) ---")
    for condyle_name, label in [("fem_condyle_medial", "med"), ("fem_condyle_lateral", "lat")]:
        pts = meshes[condyle_name].vertices
        projections = np.dot(pts - dist_fem, fem_Z)
        n_distal = max(1, int(math.ceil(len(pts) * 0.02)))
        distal_indices = np.argpartition(projections, n_distal)[:n_distal]
        distal_pt = pts[distal_indices].mean(axis=0)
        results[f"{label}_condyle_distal_pt"] = distal_pt
        logger.info(
            f"  Distal centroid ({condyle_name}, n={n_distal}): "
            f"({distal_pt[0]:.2f}, {distal_pt[1]:.2f}, {distal_pt[2]:.2f}), "
            f"Z-proj range: [{projections[distal_indices].min():.2f}, "
            f"{projections[distal_indices].max():.2f}] mm"
        )

    dfj_vec = results["lat_condyle_distal_pt"] - results["med_condyle_distal_pt"]
    dfj_unit = dfj_vec / norm(dfj_vec)
    results["dfj_line"] = dfj_unit

    # =====================================================================
    # B) PROXIMAL TIBIAL JOINT ORIENTATION (PTJ)
    # =====================================================================
    logger.info("\n--- Proximal tibial joint orientation (PTJ) ---")
    med_plat = results["med_plateau_centroid"]
    lat_plat = results["lat_plateau_centroid"]
    ptj_unit = (lat_plat - med_plat) / norm(lat_plat - med_plat)
    results["ptj_line"] = ptj_unit
    logger.info(f"  PTJ direction: ({ptj_unit[0]:.4f}, {ptj_unit[1]:.4f}, {ptj_unit[2]:.4f})")

    # =====================================================================
    # C) SUPRACONDYLAR FEMORAL JOINT ORIENTATION (SFJ)
    # =====================================================================
    logger.info("\n--- Supracondylar femoral joint orientation (SFJ) ---")
    proximal_border_pts = []
    for pts_array, name in [
        (meshes["fem_trochlea_medial"].vertices, "trochlea_med"),
        (meshes["fem_trochlea_lateral"].vertices, "trochlea_lat"),
        (meshes["fem_condyle_medial"].vertices, "condyle_med"),
        (meshes["fem_condyle_lateral"].vertices, "condyle_lat"),
    ]:
        z_proj = np.dot(pts_array - dist_fem, fem_Z)
        proximal_mask = z_proj >= (z_proj.max() - 5.0)
        proximal_border_pts.append(pts_array[proximal_mask])
        logger.info(f"  {name}: {proximal_mask.sum()} proximal border vertices")

    all_proximal = np.vstack(proximal_border_pts)
    supra_plane = fit_plane(all_proximal, orient_toward=cranial_dir)
    sfj_normal = supra_plane["normal"]
    results["sfj_normal"] = sfj_normal
    results["sfj_centroid"] = supra_plane["centroid"]
    logger.info(f"  SFJ plane normal: ({sfj_normal[0]:.4f}, {sfj_normal[1]:.4f}, {sfj_normal[2]:.4f}), RMSE={supra_plane['rmse']:.2f} mm")

    # =====================================================================
    # D) TIBIAL PLATEAU PLANES
    # =====================================================================
    logger.info("\n--- Tibial plateau orientations ---")
    plateau_planes = {}
    for plat_name, label in [("tibia_plateau_medial", "medial"), ("tibia_plateau_lateral", "lateral")]:
        pts = meshes[plat_name].vertices
        plane = fit_plane(pts, orient_toward=tib_Z)
        plateau_planes[label] = {"normal": plane["normal"], "centroid": plane["centroid"], "rmse": plane["rmse"]}
        pn = plane["normal"]
        logger.info(f"  {label} plateau: normal=({pn[0]:.4f}, {pn[1]:.4f}, {pn[2]:.4f}), RMSE={plane['rmse']:.4f} mm")

    combined_plane = fit_plane(
        np.vstack([meshes["tibia_plateau_medial"].vertices, meshes["tibia_plateau_lateral"].vertices]),
        orient_toward=tib_Z,
    )
    plateau_planes["combined"] = {
        "normal": combined_plane["normal"],
        "centroid": combined_plane["centroid"],
        "rmse": combined_plane["rmse"],
    }
    results["plateau_planes"] = plateau_planes

    # =====================================================================
    # E) CORONAL PLANE ANGLES (mLDFA, mMPTA, HKAA)
    # =====================================================================
    logger.info("\n--- Coronal plane angles ---")
    mikulicz_unit = (fem_head - ankle) / norm(fem_head - ankle)
    leg_Z = mikulicz_unit if np.dot(mikulicz_unit, cranial_dir) > 0 else -mikulicz_unit

    dfj_perp = dfj_unit - np.dot(dfj_unit, leg_Z) * leg_Z
    if norm(dfj_perp) < 1e-10:
        raise AnalysisError("DFJ aligned with leg Z — cannot define coronal plane")
    leg_X = dfj_perp / norm(dfj_perp)
    if np.dot(leg_X, cond_lat_centroid - cond_med_centroid) < 0:
        leg_X = -leg_X
    leg_Y = np.cross(leg_Z, leg_X)
    leg_Y = leg_Y / norm(leg_Y)
    results["leg_cs"] = {"X": leg_X, "Y": leg_Y, "Z": leg_Z}

    def _project_2d(vec_3d, axis_h, axis_v):
        vec_2d = np.array([np.dot(vec_3d, axis_h), np.dot(vec_3d, axis_v)])
        l = norm(vec_2d)
        return vec_2d / l if l > 1e-10 else vec_2d

    mfa_cor = _project_2d(mech_fem_unit, leg_X, leg_Z)
    dfj_cor = _project_2d(dfj_unit, leg_X, leg_Z)
    mldfa = math.degrees(math.atan2(abs(dfj_cor[0] * mfa_cor[1] - dfj_cor[1] * mfa_cor[0]), np.dot(dfj_cor, mfa_cor)))
    angles["mLDFA"] = mldfa
    logger.info(f"  mLDFA: {mldfa:.1f} deg")

    mta_cor = _project_2d(mech_tib_unit, leg_X, leg_Z)
    ptj_cor = _project_2d(ptj_unit, leg_X, leg_Z)
    mmpta = math.degrees(math.atan2(abs(ptj_cor[0] * mta_cor[1] - ptj_cor[1] * mta_cor[0]), np.dot(ptj_cor, mta_cor)))
    angles["mMPTA"] = mmpta
    logger.info(f"  mMPTA: {mmpta:.1f} deg")

    deviation = math.degrees(math.acos(np.clip(np.dot(mfa_cor, mta_cor), -1.0, 1.0)))
    cross_2d = mfa_cor[0] * mta_cor[1] - mfa_cor[1] * mta_cor[0]  # * (-1.0 if side == "R" else 1.0)
    if cross_2d > 0:
        deviation = -deviation
    angles["HKAA"] = 180.0 - deviation
    angles["mHKA"] = deviation
    logger.info(f"  HKAA: {180.0 - deviation:.1f} deg | mHKA: {deviation:.1f} deg")

    # =====================================================================
    # F) SAGITTAL PLANE ANGLES
    # =====================================================================
    logger.info("\n--- Sagittal plane angles ---")

    def _sagittal_posterior_angle(mech_axis_unit, joint_normal, cs_Y, cs_Z):
        ma_sag = np.array([np.dot(mech_axis_unit, cs_Y), np.dot(mech_axis_unit, cs_Z)])
        ma_sag = ma_sag / norm(ma_sag)
        jn_y = np.dot(joint_normal, cs_Y)
        jn_z = np.dot(joint_normal, cs_Z)
        jl_sag = np.array([-jn_z, jn_y])
        jl_len = norm(jl_sag)
        if jl_len < 1e-10:
            return float("nan")
        jl_sag = jl_sag / jl_len
        return math.degrees(math.acos(np.clip(abs(np.dot(ma_sag, jl_sag)), 0.0, 1.0)))

    med_normal = plateau_planes["medial"]["normal"]
    lat_normal = plateau_planes["lateral"]["normal"]

    angles["mMPPTA"] = _sagittal_posterior_angle(mech_tib_unit, med_normal, tib_Y, tib_Z)
    angles["mLPPTA"] = _sagittal_posterior_angle(mech_tib_unit, lat_normal, tib_Y, tib_Z)
    angles["mPDFA"] = _sagittal_posterior_angle(mech_fem_unit, sfj_normal, fem_Y, fem_Z)
    logger.info(f"  mMPPTA: {angles['mMPPTA']:.1f} deg | mLPPTA: {angles['mLPPTA']:.1f} deg | mPDFA: {angles['mPDFA']:.1f} deg")

    for label in ["medial", "lateral", "combined"]:
        pn = plateau_planes[label]["normal"]
        slope = math.degrees(math.atan2(-np.dot(pn, tib_Y), np.dot(pn, tib_Z)))
        angles[f"posterior_slope_{label}"] = slope
        logger.info(f"  Posterior slope ({label}): {slope:.1f} deg (positive = posterior tilt)")

    # =====================================================================
    # G) PTJ AP ORIENTATION
    # =====================================================================
    logger.info("\n--- PTJ AP orientation ---")

    angles["PTJ_APM"] = math.degrees(math.atan2(-np.dot(med_normal, tib_Y), np.dot(med_normal, tib_Z)))
    angles["PTJ_APL"] = math.degrees(math.atan2(-np.dot(lat_normal, tib_Y), np.dot(lat_normal, tib_Z)))
    logger.info(f"  PTJ APM: {angles['PTJ_APM']:.1f} deg | PTJ APL: {angles['PTJ_APL']:.1f} deg")

    # =====================================================================
    # H) FEMORAL VERSION (FVA) & TIBIAL TORSION (TTA)
    # =====================================================================
    logger.info("\n--- Femoral version (FVA) ---")
    fem_Z_axis = fem_cs["Z"]
    neck_center = results["fem_neck_center"]

    neck_long_axis = fem_head - neck_center
    neck_long_axis = neck_long_axis / norm(neck_long_axis)

    prox_projected = neck_long_axis - np.dot(neck_long_axis, fem_Z_axis) * fem_Z_axis
    if norm(prox_projected) < 1e-10:
        raise AnalysisError("Neck long axis aligned with femoral Z")
    prox_projected = prox_projected / norm(prox_projected)

    med_cond_centroid = np.array(poi[POI_MAP["FMCD"]])  # "fem_condyle_medial"
    lat_cond_centroid = np.array(poi[POI_MAP["FLCD"]])  # "fem_condyle_lateral"

    medial_dir_fva = med_cond_centroid - lat_cond_centroid
    medial_dir_fva = medial_dir_fva - np.dot(medial_dir_fva, fem_Z_axis) * fem_Z_axis
    medial_dir_fva = medial_dir_fva / norm(medial_dir_fva)

    cyl_axis = results["cylinder_axis"]
    if np.dot(cyl_axis, medial_dir_fva) < 0:
        cyl_axis = -cyl_axis
    dist_projected = cyl_axis - np.dot(cyl_axis, fem_Z_axis) * fem_Z_axis
    if norm(dist_projected) < 1e-10:
        raise AnalysisError("Cylinder axis aligned with femoral Z")
    dist_projected = dist_projected / norm(dist_projected)

    version_angle = math.degrees(math.acos(np.clip(np.dot(prox_projected, dist_projected), -1.0, 1.0)))
    delta_anterior_fva = np.dot(prox_projected - dist_projected, fem_Y)
    if delta_anterior_fva < 0:
        version_angle = -version_angle
    if version_angle < -90:
        version_angle = version_angle + 180
    if version_angle > 90:
        version_angle = 180 - version_angle
    angles["FVA"] = version_angle
    angles["femoral_version"] = version_angle
    logger.info(f"  FVA: {version_angle:.1f} deg (positive = anteversion)")

    logger.info("\n--- Tibial torsion (TTA) ---")
    tib_Z_axis = tib_cs["Z"]
    prox_tib_vec = results["lat_plateau_centroid"] - results["med_plateau_centroid"]
    prox_tib_projected = prox_tib_vec - np.dot(prox_tib_vec, tib_Z_axis) * tib_Z_axis
    if norm(prox_tib_projected) < 1e-10:
        raise AnalysisError("Plateau line aligned with tibial Z")
    prox_tib_projected = prox_tib_projected / norm(prox_tib_projected)

    dist_tib_vec = results["ankle_lat_centroid"] - results["ankle_med_centroid"]
    dist_tib_projected = dist_tib_vec - np.dot(dist_tib_vec, tib_Z_axis) * tib_Z_axis
    if norm(dist_tib_projected) < 1e-10:
        raise AnalysisError("Intermalleolar axis aligned with tibial Z")
    dist_tib_projected = dist_tib_projected / norm(dist_tib_projected)

    torsion_angle = math.degrees(math.acos(np.clip(np.dot(prox_tib_projected, dist_tib_projected), -1.0, 1.0)))
    delta_anterior_tta = np.dot(prox_tib_projected - dist_tib_projected, tib_Y)
    if delta_anterior_tta < 0:
        torsion_angle = -torsion_angle
    angles["TTA"] = torsion_angle
    angles["tibial_torsion"] = torsion_angle
    logger.info(f"  TTA: {torsion_angle:.1f} deg (positive = external)")

    # =====================================================================
    # STORE & VALIDATE
    # =====================================================================
    results["angles"] = angles

    logger.info("\n--- Angle range checks ---")
    angle_checks = {
        "mLDFA": {"pass": (85, 90), "warn": (80, 95), "fail": (70, 100)},
        "mMPTA": {"pass": (85, 90), "warn": (80, 95), "fail": (70, 100)},
        "HKAA": {"pass": (175, 185), "warn": (170, 190), "fail": (160, 200)},
        "mPDFA": {"pass": (80, 90), "warn": (75, 95), "fail": (65, 100)},
        "mMPPTA": {"pass": (80, 90), "warn": (75, 95), "fail": (65, 100)},
        "mLPPTA": {"pass": (80, 90), "warn": (75, 95), "fail": (65, 100)},
        "posterior_slope_medial": {"pass": (5, 15), "warn": (2, 20), "fail": (0, 25)},
        "posterior_slope_lateral": {"pass": (5, 15), "warn": (2, 20), "fail": (0, 25)},
        "FVA": {"pass": (10, 20), "warn": (5, 30), "fail": (0, 40)},
        "TTA": {"pass": (15, 30), "warn": (10, 40), "fail": (0, 50)},
        "mHKA": {"pass": (-5, 5), "warn": (-10, 10), "fail": (-15, 15)},
    }
    for angle_name, ranges in angle_checks.items():
        if angle_name not in angles:
            continue
        val = angles[angle_name]
        lo_p, hi_p = ranges["pass"]
        lo_w, hi_w = ranges["warn"]
        lo_f, hi_f = ranges["fail"]
        if lo_p <= val <= hi_p:
            logger.info(f"  {angle_name}: {val:.1f} deg — PASS")
        elif lo_w <= val <= hi_w:
            w = f"WARN: {angle_name}: {val:.1f} deg outside PASS range ({lo_p}-{hi_p})"
            logger.warning(w)
            results["warnings"].append(w)
        elif lo_f <= val <= hi_f:
            w = f"WARN: {angle_name}: {val:.1f} deg outside WARN range ({lo_w}-{hi_w})"
            logger.warning(w)
            results["warnings"].append(w)
        else:
            logger.error(f"FAIL: {angle_name}: {val:.1f} deg outside physiological range ({lo_f}-{hi_f})")
            results["warnings"].append(f"FAIL: {angle_name}: {val:.1f} deg outside physiological range")

    logger.info("\n--- All angles summary ---")
    for name, val in sorted(angles.items()):
        logger.info(f"  {name:30s}: {val:7.1f} deg")

    logger.info("\nSTEP 4 COMPLETE: All Veerman Fig. 7 angles computed.")


# =============================================================================
# STEP 5 — EXPORT RESULTS (all coordinates in global/world mm)
# =============================================================================


def step_5(poi: POI_Global, results, output_path=None):
    """Export all results to per-case CSV (long format).

    All joint-center coordinates are converted to global (ITK/world) mm space
    via poi.local_to_global() before writing.
    """
    logger.info("=" * 60)
    logger.info("STEP 5: Export Results (global coordinates)")
    logger.info("=" * 60)

    required = ["stl_folder", "angles", "fem_head_center", "dist_fem_center", "prox_tib_center", "ankle_center", "side"]
    _check_required_keys(results, required, "step_5")

    if output_path is None:
        output_path = os.path.join(os.path.dirname(results["stl_folder"]), "veerman_analysis_results.csv")

    rows = []
    warnings_dict = {}
    for w in results.get("warnings", []):
        for key in ["fem_head", "cylinder", "ankle", "slope", "version", "torsion", "mHKA"]:
            if key.lower() in w.lower():
                warnings_dict.setdefault(key, []).append(w)

    def _add_center_row(name, cord, rmse=None, radius=None, desc=""):
        warn_str = "; ".join(warnings_dict.get(name.split("_")[0], []))
        g = cord
        logger.info(f"  {name} global: ({g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f}) mm")
        rows.append(
            {
                "parameter": name,
                "type": "center",
                "x_mm": f"{g[0]:.4f}",
                "y_mm": f"{g[1]:.4f}",
                "z_mm": f"{g[2]:.4f}",
                "angle_deg": "",
                "rmse_mm": f"{rmse:.4f}" if rmse is not None else "",
                "radius_mm": f"{radius:.4f}" if radius is not None else "",
                "description": desc,
                "warnings": warn_str,
            }
        )

    def _add_angle_row(name, angle, desc=""):
        warn_str = "; ".join(warnings_dict.get(name, []))
        rows.append(
            {
                "parameter": name,
                "type": "angle",
                "x_mm": "",
                "y_mm": "",
                "z_mm": "",
                "angle_deg": f"{angle:.2f}",
                "rmse_mm": "",
                "radius_mm": "",
                "description": desc,
                "warnings": warn_str,
            }
        )

    def _add_axis_row(name, vec, desc=""):
        # Axes are unit vectors (dimensionless) — no coordinate conversion needed
        rows.append(
            {
                "parameter": name,
                "type": "axis",
                "x_mm": f"{vec[0]:.6f}",
                "y_mm": f"{vec[1]:.6f}",
                "z_mm": f"{vec[2]:.6f}",
                "angle_deg": "",
                "rmse_mm": "",
                "radius_mm": "",
                "description": desc + " (dimensionless unit vector)",
                "warnings": "",
            }
        )

    # --- Centers ---
    _add_center_row(
        "fem_head_center",
        np.array(poi.info.get("fem_head_center", results["fem_head_center"])),
        rmse=poi.info.get("fem_head_rmse"),
        radius=poi.info.get("fem_head_radius"),
        desc="Sphere fit to fem_head",
    )
    _add_center_row("fem_neck_center", results["fem_neck_center"], desc="Area-weighted centroid of fem_neck")
    _add_center_row(
        "dist_fem_center",
        np.array(poi.info.get("dist_fem_center", results["dist_fem_center"])),
        rmse=poi.info.get("cylinder_rmse"),
        radius=poi.info.get("cylinder_radius"),
        desc="Cylinder axis ray-cast midpoint",
    )
    _add_center_row("prox_tib_center", results["prox_tib_center"], desc="Midpoint of plateau centroids")
    _add_center_row("ankle_center", results["ankle_center"], desc="Area-weighted centroid of all 3 ankle surfaces combined")
    _add_center_row("med_plateau_centroid", results["med_plateau_centroid"], desc="Area-weighted centroid of medial plateau")
    _add_center_row("lat_plateau_centroid", results["lat_plateau_centroid"], desc="Area-weighted centroid of lateral plateau")
    _add_center_row("ankle_med_centroid", results["ankle_med_centroid"], desc="Medial malleolus centroid")
    _add_center_row("ankle_lat_centroid", results["ankle_lat_centroid"], desc="Lateral malleolus centroid")
    _add_center_row("ankle_mid_centroid", results["ankle_mid_centroid"], desc="Plafond centroid")

    # --- Axes (dimensionless) ---
    if "cylinder_axis" in results:
        _add_axis_row("cylinder_axis", results["cylinder_axis"], desc="PCA-derived cylinder axis")
    if "fem_cs" in results:
        cs = results["fem_cs"]
        _add_axis_row("fem_cs_X", cs["X"], desc="Femoral CS X-axis (lateral)")
        _add_axis_row("fem_cs_Y", cs["Y"], desc="Femoral CS Y-axis (anterior)")
        _add_axis_row("fem_cs_Z", cs["Z"], desc="Femoral CS Z-axis (cranial)")
    if "tib_cs" in results:
        cs = results["tib_cs"]
        _add_axis_row("tib_cs_X", cs["X"], desc="Tibial CS X-axis (lateral)")
        _add_axis_row("tib_cs_Y", cs["Y"], desc="Tibial CS Y-axis (anterior)")
        _add_axis_row("tib_cs_Z", cs["Z"], desc="Tibial CS Z-axis (cranial)")

    # --- Angles ---
    angle_descriptions = {
        "mLDFA": "Mech. lateral distal femoral angle (Fig.7b2)",
        "mMPTA": "Mech. medial proximal tibial angle (Fig.7b3)",
        "HKAA": "Hip-knee-ankle angle (Fig.7b4, 180=neutral)",
        "mHKA": "Mechanical HKA deviation (positive = varus)",
        "mPDFA": "Mech. posterior distal femoral angle (Fig.7c3)",
        "mMPPTA": "Mech. medial posterior proximal tibial angle (Fig.7c1)",
        "mLPPTA": "Mech. lateral posterior proximal tibial angle (Fig.7c2)",
        "PTJ_APM": "Proximal tibial joint AP orientation medial",
        "PTJ_APL": "Proximal tibial joint AP orientation lateral",
        "FVA": "Femoral version angle (Fig.7d1, positive = anteversion)",
        "TTA": "Tibial torsion angle (Fig.7d2, positive = external)",
        "femoral_version": "Femoral version (legacy alias for FVA)",
        "tibial_torsion": "Tibial torsion (legacy alias for TTA)",
        "posterior_slope_medial": "Medial posterior tibial slope (normal-based)",
        "posterior_slope_lateral": "Lateral posterior tibial slope (normal-based)",
        "posterior_slope_combined": "Combined posterior tibial slope (normal-based)",
    }
    for name, val in results.get("angles", {}).items():
        _add_angle_row(name, val, desc=angle_descriptions.get(name, ""))

    # --- Write CSV ---
    fieldnames = ["parameter", "type", "x_mm", "y_mm", "z_mm", "angle_deg", "rmse_mm", "radius_mm", "description", "warnings"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"\nCSV written: {output_path} ({len(rows)} rows, global coordinates)")

    # Summary printout
    a = results.get("angles", {})
    side = results["side"]
    case_name = os.path.basename(os.path.dirname(results["stl_folder"]))
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Case: {case_name} | Side: {side}")
    logger.info(f"  HKAA: {a.get('HKAA', float('nan')):.1f} deg | mHKA: {a.get('mHKA', float('nan')):.1f} deg")
    logger.info(f"  mLDFA: {a.get('mLDFA', float('nan')):.1f} deg | mMPTA: {a.get('mMPTA', float('nan')):.1f} deg")
    logger.info(
        f"  mPDFA: {a.get('mPDFA', float('nan')):.1f} deg | mMPPTA: {a.get('mMPPTA', float('nan')):.1f} deg | mLPPTA: {a.get('mLPPTA', float('nan')):.1f} deg"
    )
    logger.info(f"  FVA: {a.get('FVA', float('nan')):.1f} deg | TTA: {a.get('TTA', float('nan')):.1f} deg")
    logger.info(
        f"  Med slope: {a.get('posterior_slope_medial', float('nan')):.1f} deg | Lat slope: {a.get('posterior_slope_lateral', float('nan')):.1f} deg"
    )
    logger.info(f"{'=' * 60}")

    if results.get("warnings"):
        logger.info(f"\nAccumulated warnings ({len(results['warnings'])}):")
        for w in results["warnings"]:
            logger.info(f"  - {w}")

    logger.info("\nSTEP 5 COMPLETE: Results exported in global coordinates.")


def step_5_flat(results, case_id):
    """Return a wide-format dict (one row per case) for master CSV.
    Joint-center coordinates are in global (ITK/world) mm space.
    """
    a = results.get("angles", {})
    poi: POI_Global = results["poi"]

    def _safe_global(key, idx):
        v = results.get(key)
        if v is None:
            return None
        g = np.array(v, dtype=np.float64)
        return float(g[idx])

    def _z(x):
        return x if x is not None else None

    row = {
        "case_id": case_id,
        "side": results.get("side", ""),
        "stl_folder": results.get("stl_folder", ""),
        # Fit quality (mm)
        "fem_head_radius_mm": _z(poi.info.get("fem_head_radius")),
        "fem_head_rmse_mm": _z(poi.info.get("fem_head_rmse")),
        "cylinder_radius_mm": _z(poi.info.get("cylinder_radius")),
        "cylinder_rmse_mm": _z(poi.info.get("cylinder_rmse")),
        # CS validation
        "knee_flexion_deg": results.get("knee_flexion_angle"),
        # Coronal angles
        "mLDFA_deg": a.get("mLDFA"),
        "mMPTA_deg": a.get("mMPTA"),
        "HKAA_deg": a.get("HKAA"),
        "mHKA_deg": a.get("mHKA"),
        # Sagittal angles
        "mPDFA_deg": a.get("mPDFA"),
        "mMPPTA_deg": a.get("mMPPTA"),
        "mLPPTA_deg": a.get("mLPPTA"),
        "PTJ_APM_deg": a.get("PTJ_APM"),
        "PTJ_APL_deg": a.get("PTJ_APL"),
        "posterior_slope_medial_deg": a.get("posterior_slope_medial"),
        "posterior_slope_lateral_deg": a.get("posterior_slope_lateral"),
        "posterior_slope_combined_deg": a.get("posterior_slope_combined"),
        # Torsion
        "FVA_deg": a.get("FVA"),
        "TTA_deg": a.get("TTA"),
        # Joint centers in global mm
        "fem_head_center_x_mm": _safe_global("fem_head_center", 0),
        "fem_head_center_y_mm": _safe_global("fem_head_center", 1),
        "fem_head_center_z_mm": _safe_global("fem_head_center", 2),
        "dist_fem_center_x_mm": _safe_global("dist_fem_center", 0),
        "dist_fem_center_y_mm": _safe_global("dist_fem_center", 1),
        "dist_fem_center_z_mm": _safe_global("dist_fem_center", 2),
        "prox_tib_center_x_mm": _safe_global("prox_tib_center", 0),
        "prox_tib_center_y_mm": _safe_global("prox_tib_center", 1),
        "prox_tib_center_z_mm": _safe_global("prox_tib_center", 2),
        "ankle_center_x_mm": _safe_global("ankle_center", 0),
        "ankle_center_y_mm": _safe_global("ankle_center", 1),
        "ankle_center_z_mm": _safe_global("ankle_center", 2),
        # QC
        "n_warnings": len(results.get("warnings", [])),
        "warnings": "; ".join(results.get("warnings", [])),
    }
    return row


# =============================================================================
# CONVENIENCE: run full pipeline on a single case
# =============================================================================


def color_from_idx(i, n=12):
    phi = i / n
    return [
        0.5 + 0.5 * np.cos(2 * np.pi * phi),
        0.5 + 0.5 * np.cos(2 * np.pi * (phi + 1 / 3)),
        0.5 + 0.5 * np.cos(2 * np.pi * (phi + 2 / 3)),
    ]


def run_single_case(nii, stl_folder: "Path | str", side: Literal["R", "L"], output_csv=None, verbose=True):
    """Run the full pipeline for one case and return results dict."""
    stl_folder = Path(stl_folder)

    angle_lines: list[MKR_Lines] = [
        {"key_points": [POI_MAP["FHC"], (4, 5)], "color": color_from_idx(1), "name": "Mikulicz line [FHC-ankle]"},
        {
            "key_points": [POI_MAP["dist_fem_center"], POI_MAP["dist_fem_center-ray1"]],
            "color": color_from_idx(2),
            "name": "dist_fem_center ray-1",
        },
        {
            "key_points": [POI_MAP["dist_fem_center"], POI_MAP["dist_fem_center-ray2"]],
            "color": color_from_idx(2),
            "name": "dist_fem_center ray-2",
        },
        {"key_points": [POI_MAP["FHC"], POI_MAP["dist_fem_center"]], "color": color_from_idx(3), "name": "Mechanical femoral axis"},
        {"key_points": [POI_MAP["TKC"], POI_MAP["ankle_center"]], "color": color_from_idx(3), "name": "Mechanical tibial axis"},
    ]

    nii = to_nii(nii, True)

    poi = nii.make_empty_POI().to_global(itk_coords=False)
    poi.info["label_name"] = {f"({k1}, {k2})": v for v, (k1, k2) in POI_MAP.items()}
    poi.info["label_group_name"] = {"1": "Femur proximal", "2": "Femur distal", "3": "Tibia proximal", "4": "Tibia distal", "5": "Patella"}
    poi.info["warnings"] = []
    load_save_stls(nii, poi.info, stl_folder, side, verbose=verbose)
    compute_points(nii, poi, stl_folder, side, poi.info["warnings"], verbose=verbose)

    poi.info["fem_head_center"] = np.array(poi[POI_MAP["FHC"]])
    poi.info["fem_neck_center"] = np.array(poi[POI_MAP["FNC"]])
    poi.info["med_plateau_centroid"] = np.array(poi[POI_MAP["TMCM"]])
    poi.info["lat_plateau_centroid"] = np.array(poi[POI_MAP["TLCL"]])
    poi.info["prox_tib_center"] = np.array(poi[POI_MAP["TKC"]])
    poi.info["ankle_med_centroid"] = np.array(poi[POI_MAP["TMM"]])
    poi.info["ankle_lat_centroid"] = np.array(poi[POI_MAP["FLM"]])
    poi.info["ankle_mid_centroid"] = np.array(poi[POI_MAP["TAC"]])
    poi.info["ankle_center"] = np.array(poi[POI_MAP["ankle_center"]])
    poi.info["cond_med_centroid"] = np.array(poi[POI_MAP["FMCD"]])
    poi.info["cond_lat_centroid"] = np.array(poi[POI_MAP["FLCD"]])

    step_3(poi)
    step_4(poi)

    if output_csv:
        step_5(poi, poi.info, output_path=output_csv)

    # Convert POI to global before saving
    global_poi = poi
    print(poi.info.keys())
    del poi.info["meshes"]
    global_poi.save_mrk(stl_folder / "poi.mrk.json", split_by_region=True, add_lines=angle_lines)
    poi.to_local(nii).save(stl_folder / "poi.json")

    return global_poi


if __name__ == "__main__":
    run_single_case(
        nii="/media/data/robert/code/TReg/pois_mrk/sub-CTFU03127_ses-20171206_sequ-202_seg-fov2-reg-julius_V2_msk.nii.gz.seg.nrrd",
        stl_folder="/media/data/robert/code/TReg/results/veerman/sub-CTFU03127_ses-20171206_sequ-202_seg-fov2",
        side="L",
        output_csv="/media/data/robert/code/TReg/results/veerman/sub-CTFU03127_ses-20171206_sequ-202_seg-fov2/angle_L.csv",
    )
