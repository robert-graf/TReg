import csv
import logging
import math
import os
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
from numpy.linalg import eigh, lstsq, norm, svd
from TPTBox import NII, POI, Print_Logger, to_nii
from TPTBox.core.poi_fun.save_mkr import MKR_Lines

logger = Print_Logger(prefix="veerman")


class AnalysisError(Exception):
    """Raised when a pipeline step encounters a FAIL condition."""


class MeshWrapper:
    """Thin adapter: trimesh.Trimesh -> pipeline-compatible mesh interface."""

    def __init__(self, name: str, tm: trimesh.Trimesh):
        self.name = name
        self._tm = tm

    @property
    def vertices(self) -> np.ndarray:
        """(N, 3) float64 — replaces get_vertices_np(obj)."""
        return np.asarray(self._tm.vertices, dtype=np.float64)

    @property
    def faces(self) -> np.ndarray:
        """(M, 3) int32 — replaces calc_loop_triangles + foreach_get."""
        return np.asarray(self._tm.faces, dtype=np.int32)

    def area_weighted_centroid(self) -> np.ndarray:
        """Vectorized area-weighted centroid."""
        verts, faces = self.vertices, self.faces
        if len(faces) == 0:
            logger.on_fail(f"Mesh '{self.name}' has no faces — cannot compute centroid")
            return np.zeros(3)
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        areas = 0.5 * norm(np.cross(v1 - v0, v2 - v0), axis=1)
        centroids = (v0 + v1 + v2) / 3.0
        total_area = areas.sum()
        if total_area < 1e-12:
            logger.on_fail(f"Mesh '{self.name}' has zero total area")
            return np.zeros(3)
        return (centroids * areas[:, np.newaxis]).sum(axis=0) / total_area

    @staticmethod
    def concatenate(wrappers: list, name: str = "combined"):
        """Combine multiple meshes."""
        combined = trimesh.util.concatenate([w._tm for w in wrappers])
        return MeshWrapper(name, combined)

    @staticmethod
    def load(filepath: str | Path, name: str | None = None):
        """Load STL file and return MeshWrapper."""
        if name is None:
            name = os.path.splitext(os.path.basename(filepath))[0]
        m = trimesh.load_mesh(filepath, process=False)
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(list(m.geometry.values()))
        return MeshWrapper(name, m)

    def fit_sphere(self):
        """Algebraic least-squares sphere fit."""
        vertices = self.vertices
        if len(vertices) < 50:
            raise AnalysisError(f"{self.name} has {len(vertices)} vertices (< 50)")

        pts = np.asarray(vertices, dtype=np.float64)
        n = pts.shape[0]
        A = np.column_stack([2 * pts, np.ones(n)])
        b = np.sum(pts**2, axis=1)
        x, _, _, _ = lstsq(A, b, rcond=None)
        center = x[:3]
        radius = math.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3])
        dists = norm(pts - center, axis=1)
        residuals = dists - radius
        rmse = math.sqrt(np.mean(residuals**2))
        return {
            "center": center,
            "radius": radius,
            "rmse": rmse,
            "max_residual": float(np.max(np.abs(residuals))),
            "mean_signed_residual": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
            "residuals": residuals,
        }


# =============================================================================
# PURE FITTING FUNCTIONS
# =============================================================================


def _refine_axis(pts, centroid, centered, initial_axis, n_steps=3, n_candidates=36, cranial_dir=None):
    best = _circle_fit_along_axis(pts, centroid, centered, initial_axis, cranial_dir=cranial_dir)
    if best is None:
        return None
    current_axis = best["axis"].copy()
    half_angle = math.radians(30)
    for _ in range(n_steps):
        ref = np.array([1.0, 0.0, 0.0]) if abs(current_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        b1 = np.cross(current_axis, ref)
        b1 = b1 / norm(b1)
        b2 = np.cross(current_axis, b1)
        for i in range(n_candidates):
            phi = 2 * math.pi * i / n_candidates
            tilt = half_angle * (0.5 + 0.5 * (i % 3) / 2)
            candidate = math.cos(tilt) * current_axis + math.sin(tilt) * (math.cos(phi) * b1 + math.sin(phi) * b2)
            candidate = candidate / norm(candidate)
            result = _circle_fit_along_axis(pts, centroid, centered, candidate, cranial_dir=cranial_dir)
            if result is not None and result["rmse"] < best["rmse"]:
                best = result
                current_axis = best["axis"].copy()
        half_angle *= 0.4
    return best


def _circle_fit_along_axis(pts, centroid, centered, candidate_axis, cranial_dir=None):
    axis = candidate_axis.copy()
    if cranial_dir is None:
        cranial_dir = np.array([0.0, 0.0, 1.0])
    if np.dot(axis, cranial_dir) < 0:
        axis = -axis
    axis = axis / norm(axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u_basis = np.cross(axis, ref)
    u_basis = u_basis / norm(u_basis)
    v_basis = np.cross(axis, u_basis)
    v_basis = v_basis / norm(v_basis)
    u_coords = np.dot(centered, u_basis)
    v_coords = np.dot(centered, v_basis)
    n = len(pts)
    A_2d = np.column_stack([2 * u_coords, 2 * v_coords, np.ones(n)])
    b_2d = u_coords**2 + v_coords**2
    x_2d, _, _, _ = lstsq(A_2d, b_2d, rcond=None)
    cu, cv = x_2d[0], x_2d[1]
    disc = cu**2 + cv**2 + x_2d[2]
    if disc < 0:
        return None
    radius = math.sqrt(disc)
    point_3d = centroid + cu * u_basis + cv * v_basis
    vecs = pts - point_3d
    along_axis = np.dot(vecs, axis)[:, np.newaxis] * axis
    perp_vecs = vecs - along_axis
    perp_dists = norm(perp_vecs, axis=1)
    residuals = perp_dists - radius
    rmse = math.sqrt(np.mean(residuals**2))
    return {
        "axis": axis,
        "point": point_3d,
        "radius": radius,
        "rmse": rmse,
        "max_residual": float(np.max(np.abs(residuals))),
        "mean_signed_residual": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residuals": residuals,
    }


def fit_cylinder(points, cranial_dir=None):
    """PCA + 2D circle fit for cylinder estimation."""
    pts = np.asarray(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = np.dot(centered.T, centered) / len(pts)
    eigenvalues, eigenvectors = eigh(cov)
    best_ratio_idx = 0
    best_ratio = float("inf")
    for i in range(3):
        others = [eigenvalues[j] for j in range(3) if j != i]
        if others[0] < 1e-12 or others[1] < 1e-12:
            continue
        ratio = max(others) / min(others)
        if ratio < best_ratio:
            best_ratio = ratio
            best_ratio_idx = i
    results_list = []
    for i in range(3):
        candidate = eigenvectors[:, i]
        result = _circle_fit_along_axis(pts, centroid, centered, candidate, cranial_dir=cranial_dir)
        if result is not None:
            results_list.append((i, result))
    if not results_list:
        raise ValueError("Cylinder fit failed: no valid circle fit for any axis candidate")
    ratio_result = None
    best_rmse_result = None
    for i, result in results_list:
        if i == best_ratio_idx:
            ratio_result = result
        if best_rmse_result is None or result["rmse"] < best_rmse_result["rmse"]:
            best_rmse_result = result
    if ratio_result is not None:
        if best_rmse_result is not None and ratio_result["rmse"] > 2.0 * best_rmse_result["rmse"]:
            initial = best_rmse_result
        else:
            initial = ratio_result
    else:
        initial = best_rmse_result
    refined = _refine_axis(pts, centroid, centered, initial["axis"], cranial_dir=cranial_dir)
    ret = refined if refined is not None else initial
    assert ret is not None
    return ret


def area_weighted_centroid_from_points(vertices, faces):
    verts = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if len(faces) == 0:
        return np.zeros(3)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    areas = 0.5 * norm(np.cross(v1 - v0, v2 - v0), axis=1)
    face_centroids = (v0 + v1 + v2) / 3.0
    total_area = areas.sum()
    if total_area < 1e-12:
        return np.zeros(3)
    return (areas[:, np.newaxis] * face_centroids).sum(axis=0) / total_area


def fit_plane(points, orient_toward=None):
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
