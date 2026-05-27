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
from numpy.linalg import norm, svd
from TPTBox import NII, POI, POI_Global, Print_Logger, to_nii
from TPTBox.core.poi_fun.save_mkr import MKR_Lines

out = str(Path(__file__).parent.parent)
sys.path.append(out)
from treg.mesh_analysis import AnalysisError, MeshWrapper, fit_cylinder
from treg.veerman_rules_based import stl_to_trimesh

logger = Print_Logger(prefix="subreg_2_poi")


def get_stl(nii: NII, idx):
    stl = nii.to_stl(idx)
    wrapper = MeshWrapper(str(idx), stl_to_trimesh(stl))
    return wrapper


def get_sphere_center(nii: NII, idx, idx_poi: tuple[int, int] | None = None, out_poi: POI_Global | None = None, verbose=False):
    if idx_poi is None:
        idx_poi = (0, idx)
    mesh = get_stl(nii, idx)
    sph = mesh.fit_sphere()
    if out_poi is not None:
        out_poi[idx_poi] = sph["center"]
        # out_poi.info["fem_head_radius"] = sph["radius"]
        # out_poi.info["fem_head_rmse"] = sph["rmse"]
    if verbose:
        logger.info(f"  Center: ({sph['center'][0]:.2f}, {sph['center'][1]:.2f}, {sph['center'][2]:.2f})")
        logger.info(f"  Radius: {sph['radius']:.2f} mm")
        logger.info(f"  RMSE: {sph['rmse']:.4f} mm")
        logger.info(f"  Max residual: {sph['max_residual']:.4f} mm")
        logger.info(f"  Mean signed residual: {sph['mean_signed_residual']:.4f} mm")
        logger.info(f"  Residual std: {sph['residual_std']:.4f} mm")
    return out_poi


def get_centroid(
    nii: NII,
    idx,
    idx_poi: tuple[int, int] | None = None,
    out_poi: POI_Global | None = None,
    verbose=True,
):
    if idx_poi is None:
        idx_poi = (0, idx)
    if out_poi is not None and idx_poi in out_poi:
        neck_centroid = out_poi[idx_poi]
    else:
        mesh = get_stl(nii, idx)
        neck_centroid = mesh.area_weighted_centroid()
        if out_poi is not None:
            out_poi[idx_poi] = neck_centroid

    logger.info(f"{idx_poi} centroid: ({neck_centroid[0]:.2f}, {neck_centroid[1]:.2f}, {neck_centroid[2]:.2f})", verbose=verbose)
    return out_poi
