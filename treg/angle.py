import sys
from pathlib import Path

import numpy as np
from TPTBox import POI_Global
from TPTBox.core.poi_fun.save_mkr import MKR_Lines

out = str(Path(__file__).parent.parent)
sys.path.append(out)

from constants import POI_MAP

rad2deg = 180.0 / np.pi


def calculate_angle_ad(v1, v2, eps=1e-8):
    """
    Calculate angle between two vectors in radians.

    Parameters
    ----------
    v1, v2 : array-like, shape (3,)
        Input vectors
    eps : float
        Numerical stability threshold

    Returns
    -------
    angle : float
        Angle in radians, in range [0, pi]
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < eps or n2 < eps:
        return np.nan

    cosang = np.dot(v1, v2) / (n1 * n2)

    # Clamp for numerical safety
    cosang = np.clip(cosang, -1.0, 1.0)

    return np.arccos(cosang)


def angle_deg(v1, v2):
    return calculate_angle_ad(v1, v2) * rad2deg


def color_from_idx(i, n=12):
    phi = i / n
    return [
        0.5 + 0.5 * np.cos(2 * np.pi * (phi)),
        0.5 + 0.5 * np.cos(2 * np.pi * (phi + 1 / 3)),
        0.5 + 0.5 * np.cos(2 * np.pi * (phi + 2 / 3)),
    ]


def lotfus_punkt(FHC, P1, P2):
    # Richtungsvektor der Geraden a
    a_vec = P2 - P1
    # Projektion von FHC auf die Gerade
    t = np.dot(FHC - P1, a_vec) / np.dot(a_vec, a_vec)
    Orth_FHC = P1 + t * a_vec
    return Orth_FHC


def ortho(FHC, Orth_FHC, P1, P2, poi_original, k=100):
    X_axis = P2 - P1
    Z_axis = FHC - Orth_FHC
    Y_axis = np.cross(Z_axis, X_axis)
    Y_axis /= np.linalg.norm(Y_axis)
    TMCA = np.array(poi_original[POI_MAP["TMCA"]])
    TMCP = np.array(poi_original[POI_MAP["TMCP"]])
    if np.dot(TMCA - TMCP, Y_axis) < 0:
        Y_axis = -Y_axis
    Y_point = Orth_FHC + Y_axis * k
    return Y_point, Y_axis


def build_frame(P1, P2, Orth_FHC, Y_point, FHC):
    # Raw axes
    X = P2 - P1
    Y = Y_point - Orth_FHC
    Z = FHC - Orth_FHC

    # Normalize Z first (most stable anatomical reference)
    Z = Z / np.linalg.norm(Z)

    # Make X orthogonal to Z
    X = X - np.dot(X, Z) * Z
    X = X / np.linalg.norm(X)

    # Enforce orthogonality
    Y = np.cross(Z, X)
    Y = Y / np.linalg.norm(Y)

    # Rotation matrix (columns = axes)
    R = np.column_stack((X, Y, Z))

    return R, Orth_FHC


def compute_refrence_system(poi):
    FMCP = np.array(poi[POI_MAP["FMCP"]])
    FMCD = np.array(poi[POI_MAP["FMCD"]])
    FLCP = np.array(poi[POI_MAP["FLCP"]])
    FLCD = np.array(poi[POI_MAP["FLCD"]])
    FHC = np.array(poi[POI_MAP["FHC"]])
    P1 = np.array((FMCP[0], FMCD[1], FMCP[2]))
    P2 = np.array((FLCP[0], FLCD[1], FLCP[2]))
    # Vector a =  P1(X-FMCP, Y-FMCD, Z-FMCP) nach P2 (X-FLCP, Y-FLCD, Z-FLCP)
    # Z-Achse == Orthogonalel zu å durch FHC
    Orth_FHC = lotfus_punkt(FHC, P1, P2)
    Y_point, _ = ortho(FHC, Orth_FHC, P1, P2, poi)
    poi[99, 1] = P1
    poi[99, 2] = P2
    poi[99, 3] = Y_point
    poi[99, 4] = Orth_FHC

    poi.info["label_name"]["(99,1)"] = "P1"
    poi.info["label_name"]["(99,2)"] = "P2"
    poi.info["label_name"]["(99,3)"] = "Y_point"
    poi.info["label_name"]["(99,4)"] = "Orth_FHC"
    return P1, P2, Orth_FHC, Y_point, FHC


def compute_angles(poi_original: POI_Global, out_poi=None, out_poi_orthosys=None):
    P1, P2, Orth_FHC, Y_point, FHC = compute_refrence_system(poi_original)
    R, origin = build_frame(P1, P2, Orth_FHC, Y_point, FHC)

    def world_to_local(x, y, z):
        p = np.asarray((x, y, z))
        return (p - origin) @ R

    def local_to_world(x, y, z):
        p_local = np.asarray((x, y, z))
        return p_local @ R.T + origin

    poi = poi_original.apply_all(world_to_local)

    FHC = np.array(poi[POI_MAP["FHC"]])
    FNC = np.array(poi[POI_MAP["FNC"]])
    FNP = np.array(poi[POI_MAP["FNP"]])

    FMCP = np.array(poi[POI_MAP["FMCP"]])
    FLCP = np.array(poi[POI_MAP["FLCP"]])

    FMCD = np.array(poi[POI_MAP["FMCD"]])
    FLCD = np.array(poi[POI_MAP["FLCD"]])

    FMCPC = np.array(poi[POI_MAP["FMCPC"]])
    FLCPC = np.array(poi[POI_MAP["FLCPC"]])

    TAC = np.array(poi[POI_MAP["TAC"]])
    TKC = np.array(poi[POI_MAP["TKC"]])

    TMCP = np.array(poi[POI_MAP["TMCP"]])
    TLCP = np.array(poi[POI_MAP["TLCP"]])

    TMCA = np.array(poi[POI_MAP["TMCA"]])
    TLCA = np.array(poi[POI_MAP["TLCA"]])

    TMCM = np.array(poi[POI_MAP["TMCM"]])
    TLCL = np.array(poi[POI_MAP["TLCL"]])

    TADP = np.array(poi[POI_MAP["TADP"]])
    TAAP = np.array(poi[POI_MAP["TAAP"]])

    FLM = np.array(poi[POI_MAP["FLM"]])
    TMM = np.array(poi[POI_MAP["TMM"]])
    TGCP = np.array(poi[POI_MAP["TGCP"]])
    TGPP = np.array(poi[POI_MAP["TGPP"]])

    angles = {}
    angle_lines: list[MKR_Lines] = [
        {"key_points": [(99, 1), (99, 2)], "color": [1, 0.0, 0.0]},
        {"key_points": [POI_MAP["FHC"], (99, 4)], "color": [1, 0, 0.0]},
        {"key_points": [(99, 3), (99, 4)], "color": [1, 0, 0]},
    ]

    idx = 0
    v1 = FLM - TMM
    v2 = TLCP - TMCP
    v1[2] = 0
    v2[2] = 0

    angles["tibia_torsion_2D"] = angle_deg(v1, v2)

    angle_lines.append({"key_points": [POI_MAP["TMCP"], POI_MAP["TLCP"]], "color": color_from_idx(idx)})
    angle_lines.append({"key_points": [POI_MAP["FLM"], POI_MAP["TMM"]], "color": color_from_idx(idx)})
    idx += 1
    v1 = FNC - FHC
    v2 = FMCP - FLCP
    v1[2] = 0
    v2[2] = 0

    angles["femoral_torsion_2D"] = 180 - angle_deg(v1, v2)

    angle_lines.append({"key_points": [POI_MAP["FMCP"], POI_MAP["FLCP"]], "color": color_from_idx(idx)})
    angle_lines.append({"key_points": [POI_MAP["FNC"], POI_MAP["FHC"]], "color": color_from_idx(idx)})
    idx += 1
    v1 = FHC - FNP
    v2 = FLCD - FMCD
    v1[1] = 0
    v2[1] = 0

    angles["mLDFA"] = angle_deg(v1, v2)

    angle_lines.append({"key_points": [POI_MAP["FMCD"], POI_MAP["FLCD"]], "color": color_from_idx(idx)})
    angle_lines.append({"key_points": [POI_MAP["FHC"], POI_MAP["FNP"]], "color": color_from_idx(idx)})
    idx += 1

    v1 = TAC - TKC
    v2 = TMCM - TLCL
    v1[1] = 0
    v2[1] = 0

    angles["MPTA"] = angle_deg(v1, v2)

    angle_lines.append({"key_points": [POI_MAP["TMCM"], POI_MAP["TLCL"]], "color": color_from_idx(idx)})
    angle_lines.append({"key_points": [POI_MAP["TAC"], POI_MAP["TKC"]], "color": color_from_idx(idx)})
    idx += 1
    v1 = FHC - FNP
    v2 = TAC - TKC
    v3 = TMCM - TLCL
    v1[1] = v2[1] = v3[1] = 0

    phi1 = angle_deg(v3, v1)
    phi2 = angle_deg(v3, v2)

    angles["HKA_2D"] = phi1 + phi2

    angle_lines.append({"key_points": [POI_MAP["TAC"], POI_MAP["TKC"]], "color": color_from_idx(idx)})
    angle_lines.append({"key_points": [POI_MAP["TMCM"], POI_MAP["TLCL"]], "color": color_from_idx(idx)})
    angle_lines.append({"key_points": [POI_MAP["FHC"], POI_MAP["FNP"]], "color": color_from_idx(idx)})
    idx += 1
    v1 = FHC - FNP
    v2 = FMCPC - FLCPC

    x = (TGCP[0] - FMCPC[0]) / v2[0]
    FMLCP = FMCPC + x * v2
    v3 = FMLCP - TGPP
    poi[99, 99] = v3
    angles["PDFA_sagittal_2D"] = angle_deg(v1, v3)

    angle_lines.append({"key_points": [POI_MAP["TGPP"], (99, 99)], "color": color_from_idx(idx)})
    angle_lines.append({"key_points": [POI_MAP["FHC"], POI_MAP["FNP"]], "color": color_from_idx(idx)})
    idx += 1
    angles["PDFA_medial_3D"] = angle_deg(FMCP - TGCP, FHC - FNP)

    angle_lines.append({"key_points": [POI_MAP["FMCP"], POI_MAP["TGCP"]], "color": color_from_idx(idx)})
    angle_lines.append({"key_points": [POI_MAP["FHC"], POI_MAP["FNP"]], "color": color_from_idx(idx)})
    idx += 1
    angles["PDFA_lateral_3D"] = angle_deg(FLCP - TGCP, FHC - FNP)

    angle_lines.append({"key_points": [POI_MAP["FLCP"], POI_MAP["TGCP"]], "color": color_from_idx(idx)})
    angle_lines.append({"key_points": [POI_MAP["FHC"], POI_MAP["FNP"]], "color": color_from_idx(idx)})
    idx += 1
    angles["tibial_slope_medial"] = 90 - angle_deg(TADP - TAAP, TMCP - TMCA)
    angles["tibial_slope_lateral"] = 90 - angle_deg(TADP - TAAP, TLCP - TLCA)

    angle_lines.append({"key_points": [POI_MAP["TMCP"], POI_MAP["TMCA"]], "color": color_from_idx(idx)})
    angle_lines.append({"key_points": [POI_MAP["TADP"], POI_MAP["TAAP"]], "color": color_from_idx(idx)})
    idx += 1

    angle_lines.append({"key_points": [POI_MAP["TADP"], POI_MAP["TAAP"]], "color": color_from_idx(idx)})
    angle_lines.append({"key_points": [POI_MAP["TLCP"], POI_MAP["TLCA"]], "color": color_from_idx(idx)})
    idx += 1
    # print(angle_lines)
    if out_poi_orthosys is not None:
        poi.save_mrk(out_poi_orthosys, add_lines=angle_lines)
    angles_out = {k: round(v, 2) for k, v in angles.items()}
    poi_back = poi.apply_all(local_to_world)
    if out_poi is not None:
        poi_back.save_mrk(out_poi, add_lines=angle_lines)
    return (angles_out, poi_back, poi)
