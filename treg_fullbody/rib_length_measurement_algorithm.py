# source https://github.com/Hendrik-code/rib-segmentation/blob/main/rib_length_measurement/rib_length_measurement_algorithm.py
# Apache-2.0 license
# Hendrik Möller
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.linalg import norm
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist
from TPTBox import NII, POI, Location, Log_Type, No_Logger, Vertebra_Instance, calc_poi_labeled_buffered, np_utils, to_nii
from TPTBox.core.vert_constants import COORDINATE, vert_subreg_labels

logger = No_Logger(prefix="RibLengthMeasurementAlgorithm")

vidx2name = Vertebra_Instance.idx2name()
vname2idx = Vertebra_Instance.name2idx()

from TPTBox.core.poi_fun.ray_casting import max_distance_ray_cast_convex, unit_vector


def get_raycasted_point(rib_nii: NII, point_arr: np.ndarray, cur_point_idx: int, direction_vector: COORDINATE) -> int:
    end_point_coords = max_distance_ray_cast_convex(rib_nii, point_arr[cur_point_idx], direction_vector)
    end_point_coords = np.asarray([round(i) for i in end_point_coords])
    end_point_idx = get_idx_point_closest_to_point(point_arr, end_point_coords)
    return end_point_idx


def get_point_arr(arr):
    X, Y, Z = np.where(arr)
    point_arr = np.asarray([[X[i], Y[i], Z[i]] for i in range(len(X))])
    return point_arr


def get_point_arr_mm(arr, resolution):
    point_arr = get_point_arr(arr)
    zms_stacked = np.vstack([resolution] * point_arr.shape[0])
    point_arr_mm = np.multiply(point_arr, zms_stacked)
    return point_arr_mm


def cdist_to_point(point, a):
    return cdist([point], a)[0]


def np_index(arr: np.ndarray, entry) -> np.ndarray:
    bool_arr = entry == arr
    idxs = np.flatnonzero((bool_arr).all(1))
    return idxs


def get_idx_point_closest_to_point(point_arr: np.ndarray, point: COORDINATE | np.ndarray) -> int:
    point = tuple(round(i) for i in point)
    idxs = np_index(point_arr, point)
    if len(idxs) == 0:
        distance_vectors = cdist(point_arr, np.asarray([point]))
        idxs = [np.argmin(distance_vectors)]
    return idxs[0]  # type: ignore


def array_slice(a, axis, start, end, step=1):
    return a[(slice(None),) * (axis % a.ndim) + (slice(start, end, step),)]


def change_one_slice(s: slice, change: tuple[int, int], shape: int) -> slice:
    return slice(max(s.start + change[0], 0), min(s.stop + change[1], shape))


def change_slice_tuple(
    slices: tuple[slice, slice, slice], change: int | tuple[tuple[int, int], tuple[int, int], tuple[int, int]], shape: tuple[int, int, int]
) -> tuple[slice, slice, slice]:
    if isinstance(change, int):
        change = ((-change, change), (-change, change), (-change, change))
    return tuple(change_one_slice(slices[d], change[d], shape[d]) for d in range(3))  # type: ignore


def slices_border_shape(slices: tuple[slice, slice, slice], shp: tuple[int, int, int], voxel_tolerance: int = 2):
    seg_at_border = False
    for d in range(3):
        if slices[d].start <= voxel_tolerance or slices[d].stop - 1 >= shp[d] - voxel_tolerance:
            seg_at_border = True
            break
    return seg_at_border


def refine_start_points(rib_cropped, start_point_idx, point_arr, logger):  # noqa: ARG001
    """Refines start point by slicing at its location and using the center point of that slices segmentation"""
    start_point_coord = point_arr[start_point_idx]
    # refine start point coord by using center of mass of this L/R slice
    start_slice = array_slice(rib_cropped.get_seg_array(), 2, start_point_coord[2], start_point_coord[2] + 1)
    # dim 0
    start_slice[0 : max(start_point_coord[0] - 20, 0)] = 0
    start_slice[min(start_point_coord[0] + 21, rib_cropped.shape[0] - 1) : rib_cropped.shape[0] - 1] = 0
    # dim 1
    start_slice[:, 0 : max(start_point_coord[1] - 20, 0)] = 0
    start_slice[:, min(start_point_coord[1] + 21, rib_cropped.shape[1] - 1) : rib_cropped.shape[1] - 1] = 0
    start_com = [round(i) for i in np_utils.np_center_of_mass(start_slice)[1]]
    start_coord = np.asarray([start_com[0], start_com[1], start_point_coord[2]])
    try:
        start_point_idx = get_idx_point_closest_to_point(point_arr, start_coord)
    except Exception:
        return None
    # refinement done
    return start_point_idx


def find_all_candidate_points(
    point_arr, cur_point_idx, prior_point_idx, prior_prior_point_idx, interpolation_distance_mm, interpolation_distance_mm_tol, distance_row
):
    # Get all candidate points
    point_candidate_idxs = np.where(
        (interpolation_distance_mm - interpolation_distance_mm_tol <= distance_row)
        & (distance_row <= interpolation_distance_mm + interpolation_distance_mm_tol)
    )[0]

    # if prior points exist, remove candidates closer to prior point
    # remove all candidates that are closer to the prior point than the current
    point_candidate_idxs, removed_idxs = remove_candidates_idxs_closer_to_prior_points(
        prior_point_idx,
        point_candidate_idxs,
        interpolation_distance_mm,
        interpolation_distance_mm_tol,
        point_arr,
        cur_point_idx,
    )
    point_candidate_idxs, removed_idxs = remove_candidates_idxs_closer_to_prior_points(
        prior_prior_point_idx,
        point_candidate_idxs,
        interpolation_distance_mm,
        interpolation_distance_mm_tol,
        point_arr,
        cur_point_idx,
    )
    point_candidates_coords = [point_arr[idx] for idx in point_candidate_idxs]
    return point_candidate_idxs, point_candidates_coords


def remove_candidates_idxs_closer_to_prior_points(
    prior_point_idx, point_candidates_idxs, interpolation_distance_mm, interpolation_distance_mm_tol, point_arr, cur_point_idx
):
    before_remove = point_candidates_idxs.copy()
    if prior_point_idx is not None:
        point_distance_tolerance_circle = (interpolation_distance_mm / 2) - (2 * interpolation_distance_mm_tol)
        point_candidates_idxs = [
            p
            for p in point_candidates_idxs
            if np.linalg.norm(point_arr[p] - point_arr[cur_point_idx])
            < np.linalg.norm(point_arr[p] - point_arr[prior_point_idx]) - point_distance_tolerance_circle
        ]
    removed = [p for p in before_remove if p not in point_candidates_idxs]
    return point_candidates_idxs, removed


def find_end_point(
    point_arr,
    rib_seg_cropped,
    cur_point_idx,
    prior_point_idx,
    precision_resolution,
    interpolation_distance_mm,
    interpolation_distance_mm_tol,
    distance_row,
):
    # handle end of path
    # if prior point exists, just raycast it
    if prior_point_idx is not None:
        end_point_idxs = get_possible_end_points(rib_seg_cropped, point_arr, cur_point_idx, prior_point_idx, precision_resolution)
        end_point_idxs = list(set(end_point_idxs))
        prior_distances = [distance_row[i] for i in end_point_idxs]
        end_point_idx = end_point_idxs[np.argmax(prior_distances)]
        prior_distance = distance_row[end_point_idx]
        #
        end_point_coords = point_arr[end_point_idx]
    else:
        # find farthest point that is also not closer to previous point
        distance_row = np.asarray([i if i < interpolation_distance_mm + interpolation_distance_mm_tol else 0 for i in distance_row])
        end_point_idx = np.argmax(distance_row)
        end_point_coords = point_arr[end_point_idx]
        prior_distance = distance_row[end_point_idx]
    return end_point_idx, end_point_coords, prior_distance


def get_possible_end_points(
    rib_nii: NII, point_arr: np.ndarray, cur_point_idx: int, prior_point_idx: int, resolution_precision: float
) -> list[int]:
    initial_direction_vector = point_arr[cur_point_idx] - point_arr[prior_point_idx]
    end_points = [get_raycasted_point(rib_nii, point_arr, cur_point_idx, initial_direction_vector)]

    for d in range(3):
        for change in range(-8, 9, 2):
            d_vector = initial_direction_vector.copy()
            d_vector[d] += change * resolution_precision
            end_points.append(get_raycasted_point(rib_nii, point_arr, cur_point_idx, d_vector))
    return end_points


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def calc_orientation_from_poi(poi: POI, region: int):
    poi_v: POI = poi.extract_vert(region)

    point_keys = [
        Location.Vertebra_Corpus,
        Location.Vertebra_Direction_Posterior,
        Location.Vertebra_Direction_Inferior,
        Location.Vertebra_Direction_Right,
    ]
    for p in point_keys:
        assert p in poi.keys_region(), f"POI {p} not found, got {poi.keys_region()}"

    point_keys = [i.value for i in point_keys]
    points = {s: np.asarray(v) for r, s, v in poi_v.items() if s in point_keys}
    # calc corpus - three other to get directional vectors (and normalize)
    rel_to_corpus = {
        s: unit_vector(v - points[Location.Vertebra_Corpus.value]) for s, v in points.items() if s != Location.Vertebra_Corpus.value
    }
    pir_global_vectors = {
        Location.Vertebra_Direction_Posterior.value: np.array([1, 0, 0]),
        Location.Vertebra_Direction_Inferior.value: np.array([0, 1, 0]),
        Location.Vertebra_Direction_Right.value: np.array([0, 0, 1]),
    }
    PIR_angles = [angle_between(v, pir_global_vectors[s]) for s, v in rel_to_corpus.items()]
    PIR_angle_degrees = [math.degrees(i) for i in PIR_angles]

    # R = [x_x, y_x, z_x; x_y, y_y, y_z; z_x, z_y, z_z]
    R = np.asarray([[v[idx] for v in rel_to_corpus.values()] for idx in range(3)])
    corpus_com = points[Location.Vertebra_Corpus.value]

    return R, corpus_com, rel_to_corpus, PIR_angle_degrees


@dataclass
class RibResult:
    vertebra: Vertebra_Instance
    leftside: bool
    last_v: bool | None

    rib_length: int
    stump_rib: bool | None
    rib_volume: float
    seg_at_border: bool

    start_point: tuple
    end_point: tuple | None

    fixed_points_along_path: dict = field(repr=False)
    orig_zoom: list[float] | None = field(repr=False, default=None)
    vert_ori_rel_to_corpus: dict | None = field(repr=False, default=None)
    PIR_angle_degrees: list[float] | None = field(repr=False, default=None)  # noqa: N815


def _rib_length_algorithm(
    sem_vr: NII,
    vert_id,
    leftside,
    stump_rib_threshold_in_mm: float = 38.0,
    interpolation_distance_mm: float = 15.0,
    max_iterations: int = 150,
    do_dilate_erode: bool = True,
    verbose: bool | int = 0,
) -> RibResult:
    """The Rib Length Measurement Algorithm, calculating the length of the provided segmentation

    Args:
        sem_vr (NII): Semantic Mask
        stump_rib_threshold_in_mm (int, optional): Threshold for stump rib length. Defaults to 38.
        interpolation_distance_mm (int, optional): The circular distance for each iteration. Defaults to 15.
        max_iterations (int, optional): Maximum number of iteration until it should crash. Defaults to 150.
        round_digits (int, optional): _description_. Defaults to 5.
        return_debug_data (bool, optional): If true, will return debug data. Defaults to False.
        verbose (bool | int, optional): Verbosity level for the algorithm. Defaults to 0.

    Returns:
        dict: A data dictionary
    """
    verbose = int(verbose)
    precision_resolution = sem_vr.zoom[0]
    assert 0 < precision_resolution <= 1.0, f"precision_resolution must be in (0, 1.0], got {precision_resolution}"
    #
    debug_data: dict = {}
    with logger:
        sem_vr.reorient_()
        sem_vr_labels = sem_vr.unique()
        assert Location.Vertebra_Corpus_border.value in sem_vr_labels, f"no corpus ({Location.Vertebra_Corpus_border.value}) in input"
        non_vert_subreg_labels = [i for i in sem_vr_labels if not (40 < i < 51)]

        assert len(non_vert_subreg_labels) == 1, f"Not exactly one non-subregion label present as rib label, got {non_vert_subreg_labels}"
        rib_label = non_vert_subreg_labels[0]
        expected_rib_labels = [Location.Rib_Left.value, Location.Rib_Right.value]
        if rib_label not in expected_rib_labels:
            logger.print(f"Unusual rib label {rib_label}, expected {expected_rib_labels}", Log_Type.STRANGE)

        # Extract label
        init_shp = sem_vr.shape
        rib_seg = sem_vr.extract_label(rib_label)
        rib_crop = rib_seg.compute_crop(dist=0)
        seg_at_border = slices_border_shape(rib_crop, init_shp, voxel_tolerance=2)
        # Crop down
        init_crop = sem_vr.compute_crop(dist=0)
        init_crop = change_slice_tuple(init_crop, change=6, shape=init_shp)  # TODO change to < 6 changes rib length
        sem_vr.apply_crop_(init_crop)
        rib_seg_cropped = rib_seg.apply_crop_(init_crop)
        logger.print(f"Cropped from {init_shp} to {sem_vr.shape}", verbose=verbose > 1)
        sem_vert = sem_vr.extract_label(vert_subreg_labels(True), keep_label=True)
        sem_corpus = sem_vert.extract_label(Location.Vertebra_Corpus_border.value)
        zooms = rib_seg_cropped.zoom

        if do_dilate_erode:
            rib_seg_cropped = (
                rib_seg_cropped.dilate_msk_(n_pixel=1, connectivity=1, labels=1, verbose=verbose > 0)
                .erode_msk_(n_pixel=1, connectivity=1, labels=1, verbose=verbose > 0)
                .dilate_msk_(n_pixel=1, connectivity=1, labels=1, verbose=verbose > 0)
                .erode_msk_(n_pixel=1, connectivity=1, labels=1, verbose=verbose > 0)
            )

        # calculate center of vertebra in question
        vert_com = sem_corpus.center_of_masses()[1]
        debug_data["sem_corpus_crop"] = sem_corpus
        #####################f
        # initialize path
        fixed_points_along_path: dict = {}
        rib_volume = rib_seg_cropped.volumes(in_mm3=True)[1]
        logger.print(f"rib_volume = {rib_volume}", verbose=verbose > 0)

        # get all points on segmentation mask
        point_arr = get_point_arr(rib_seg_cropped)
        # stack resolution
        try:
            zms_stacked = np.vstack([zooms] * point_arr.shape[0])
        except Exception:
            logger.print("could not compute zms_stacked, arr must be empty", Log_Type.FAIL)
            raise  # {"debug": debug_data}
        # get points in mm
        point_arr_mm = np.multiply(point_arr, zms_stacked)
        # get center of vertebra in mm
        vert_com_mm = np.multiply(vert_com, zooms)

        distance_to_vertebra = cdist_to_point(vert_com_mm, point_arr_mm)
        # minimum distance to vertebra corpus becomes start point of path
        start_point_idx = np.argmin(distance_to_vertebra)
        refined_start_idx = refine_start_points(rib_seg_cropped, start_point_idx, point_arr, logger)
        if refined_start_idx is None:
            logger.print("Start point refinement failed", Log_Type.FAIL)
        else:
            start_point_idx = refined_start_idx
        start_point_coord = point_arr[start_point_idx]
        # add startpoint to path
        fixed_points_along_path[start_point_idx] = {"coord": start_point_coord, "prior": None, "prior_distance": None, "endpoint": True}
        ################# START ALGORITHM #################
        # initialize algorithm values
        start_interpolation_distance_mm = interpolation_distance_mm
        interpolation_distance_mm = start_interpolation_distance_mm
        interpolation_distance_mm_tol = precision_resolution
        min_interpolation_distance_mm = (interpolation_distance_mm / 2) - (2 * interpolation_distance_mm_tol)
        # Moving variables
        prior_prior_point_idx = None
        prior_point_idx = None
        prior_distance = None
        cur_point_idx = start_point_idx
        #
        end_point = None

        # Get array
        darr = rib_seg_cropped.get_seg_array()
        darr[start_point_coord[0], start_point_coord[1], start_point_coord[2]] = 100
        debug_arr = darr.copy()

        # Loop counts
        loop_count = 0
        iter_count = 1

        ################# ITERATION LOOP #################
        while True:
            loop_count += 1
            #
            logger.print(f"ITERATION {iter_count}", verbose=verbose > 1)

            # Get distances to current point
            distance_row = cdist_to_point(point_arr_mm[cur_point_idx], point_arr_mm)
            # Get all candidate points
            point_candidate_idxs, point_candidates_coords = find_all_candidate_points(
                point_arr,
                cur_point_idx,
                prior_point_idx,
                prior_prior_point_idx,
                interpolation_distance_mm,
                interpolation_distance_mm_tol,
                distance_row,
            )
            n_candidates = len(point_candidate_idxs)
            logger.print("Possible_candidates", n_candidates, verbose=verbose > 1)
            #

            if len(point_candidates_coords) == 0:
                logger.print("No point candidates, move to end sequence", verbose=verbose > 1)

                end_point_idx, end_point_coords, prior_distance = find_end_point(
                    point_arr,
                    rib_seg_cropped,
                    cur_point_idx,
                    prior_point_idx,
                    precision_resolution,
                    interpolation_distance_mm,
                    interpolation_distance_mm_tol,
                    distance_row,
                )
                if end_point_idx not in fixed_points_along_path:
                    # delete prior point if too close to end point
                    if prior_point_idx is not None and prior_distance < min_interpolation_distance_mm:
                        logger.print("Deleted point prior to end because of proximity", verbose=verbose > 1)
                        prior_distance = np.linalg.norm(point_arr_mm[prior_point_idx] - point_arr_mm[end_point_idx])
                        fixed_points_along_path.pop(prior_point_idx, None)
                    #
                    fixed_points_along_path[end_point_idx] = {
                        "coord": end_point_coords,
                        "prior": cur_point_idx,
                        "prior_distance": prior_distance,
                        "endpoint": True,
                    }
                else:
                    fixed_points_along_path[end_point_idx]["endpoint"] = True
                #
                end_point = end_point_coords
                logger.print(f"Found end point at {end_point_coords}", verbose=verbose > 0)
                darr[end_point[0], end_point[1], end_point[2]] = 200
                debug_data[f"iterations_{iter_count}"] = rib_seg_cropped.set_array(darr.copy())
                #
                # Update moving values
                prior_prior_point_idx = prior_point_idx
                prior_point_idx = cur_point_idx
                cur_point_idx = end_point_coords
                ################# End loop because we found end #################
                break
                #################
            else:
                # there are multiple candidates
                # take average as new point
                avg_candidate_coord = [round(i) for i in np.sum(point_candidates_coords, axis=0) / n_candidates]
                # move half-distance in that direction
                cur_coord = point_arr[cur_point_idx]
                avg_candidate_coord = tuple(np.add(cur_coord, (np.subtract(avg_candidate_coord, cur_coord) * 0.5)))
                #
                new_point_idx = get_idx_point_closest_to_point(point_arr, avg_candidate_coord)
                new_point_coord = point_arr[new_point_idx]

                prior_distance = distance_row[new_point_idx]
                # if it didn't move far enough, then the circle is not wide enough
                if prior_distance < min_interpolation_distance_mm or len(point_candidates_coords) < 3:
                    interpolation_distance_mm += 2 * precision_resolution
                    logger.print(
                        f"Increased interpolation distance, distance={prior_distance} and threshold={min_interpolation_distance_mm}",
                        verbose=verbose > 1,
                    )
                    continue
                else:
                    interpolation_distance_mm = start_interpolation_distance_mm

                # else we found a good new point
                for p in point_candidates_coords:
                    darr[p[0], p[1], p[2]] = iter_count + 1
                darr[new_point_coord[0], new_point_coord[1], new_point_coord[2]] = iter_count + 100
                debug_arr[new_point_coord[0], new_point_coord[1], new_point_coord[2]] = iter_count + 100
                if new_point_idx in fixed_points_along_path:
                    logger.print("New point already in the path, something went wrong", Log_Type.FAIL, verbose=True)
                    break
                fixed_points_along_path[new_point_idx] = {
                    "coord": new_point_coord,
                    "prior": cur_point_idx,
                    "prior_distance": prior_distance,
                    "endpoint": False,
                }
                ###########
                # Update moving values
                prior_prior_point_idx = prior_point_idx
                prior_point_idx = cur_point_idx
                cur_point_idx = new_point_idx

            debug_data[f"iterations_{iter_count}"] = rib_seg_cropped.set_array(darr.copy())
            iter_count += 1
            loop_count = 0
            if iter_count >= max_iterations:
                logger.print(f"Did not converge after max_iterations={max_iterations}", Log_Type.FAIL)
                end_point = start_point_coord
                break
        ################# End Algorithm #################
        if verbose > 2:
            for idx, (i, g) in enumerate(fixed_points_along_path.items()):
                logger.print(idx, i, g)

        # Calculating rib length with piece-wise linear interpolation
        rib_length = sum([g["prior_distance"] for i, g in fixed_points_along_path.items() if g["prior"] is not None])

        # stump rib detection
        # stump_rib if not segmentation at border, else None (not determinable)
        is_stump_rib = False if rib_length > stump_rib_threshold_in_mm else True if not seg_at_border else None

        start_point = start_point_coord
        # debug_data["final_algo"] = rib_seg_cropped.set_array(darr.copy())
        # if not return_debug_data:
        #    debug_data = None
        return RibResult(
            vertebra=vert_id,
            leftside=leftside,
            last_v=None,
            rib_length=round(rib_length, 3),
            stump_rib=is_stump_rib,
            start_point=rib_seg_cropped.local_to_global(start_point),
            end_point=rib_seg_cropped.local_to_global(end_point) if end_point is not None else None,
            fixed_points_along_path=fixed_points_along_path,
            seg_at_border=seg_at_border,
            rib_volume=rib_volume,
        )


def measure_ribs_length_subject(
    inst_seg: NII,
    sem_seg: NII,
    poi: POI | None = None,
    calc_orientation: bool = False,
    vert_ids: Sequence[Vertebra_Instance] | None = (
        Vertebra_Instance.T11,
        Vertebra_Instance.T12,
        Vertebra_Instance.T13,
        Vertebra_Instance.L1,
    ),
) -> list[RibResult]:
    """Calculates the length of the lowest two ribs in a subject

    Args:
        sem_seg (NII): Semantic Mask
        inst_seg (NII): Instance Mask
        poi (POI | None, optional): Center of Corpus and Direction Points if available. Defaults to None.
        calc_orientation (bool, optional): If true, will compute the orientation of the vertebra and return that. Defaults to False.
        vert_ids: Vert IDs that should be computed if None all will be computed
    Returns:
        list[dict]: A list of datapoints, where each datapoint is a dictionary mapping keys to values.
    """
    vert_ids_int = [a.value for a in vert_ids] if vert_ids is not None else [i for i in range(6, 30) if (i > 7 and i < 21) or i == 28]
    results = []
    # Orientation consistent
    sem_seg.reorient_().map_labels_({50: 49}, verbose=False)
    inst_seg.reorient_()
    try:
        sem_seg.assert_affine(other=inst_seg)
    except AssertionError:
        return results
    # get last 2 vertebra
    u = inst_seg.unique()
    rib_labels = [v for v in u if v in Vertebra_Instance.rib_label()]
    last_vertebrae = [i for i in u if i in vert_ids_int]
    last_vertebrae.sort()
    selected_vertebrae: list[Vertebra_Instance] = [Vertebra_Instance(v) for v in last_vertebrae if Vertebra_Instance(v).RIB in rib_labels]

    if len(selected_vertebrae) == 0:
        logger.print("No selected rib visible")
        return results
    last_v = selected_vertebrae[-1]
    seg_u = sem_seg.unique()
    # Loop over vertebra
    for vert in selected_vertebrae:
        is_last_v = vert == last_v
        rib_label = vert.RIB
        if rib_label == Vertebra_Instance.T13.RIB and rib_label not in rib_labels:
            rib_label = Vertebra_Instance.L1.RIB
        if rib_label not in rib_labels:
            logger.print(f"has no rib at vertebra {vert.name}")
            continue
        # vertebra level
        logger.on_log(f"Process vertebra {vert}")
        vert_vr = inst_seg.extract_label([vert.value, rib_label], keep_label=True)

        # Orientation of vertebra in image space
        rel_to_corpus, PIR_angle_degrees = None, None
        if calc_orientation:
            if poi is None:
                poi = calc_poi_labeled_buffered(
                    inst_seg,
                    sem_seg,
                    subreg_id=[Location.Vertebra_Corpus, Location.Vertebra_Direction_Posterior],
                    out_path=__file__.join("poi.json"),
                )
            poi_vr = poi.extract_vert(vert.value)
            _, _, rel_to_corpus, PIR_angle_degrees = calc_orientation_from_poi(poi_vr, vert.value)
            rel_to_corpus = {k: list(v) for k, v in rel_to_corpus.items()}
        # For both left and right side rib

        for leftside in [False, True]:  #
            leftside_str = "left" if leftside else "right"
            sem_rib_label = Location.Rib_Left.value if leftside else Location.Rib_Right.value

            if sem_rib_label not in seg_u:
                logger.print(f"has no rib at vertebra {vert.name} on {leftside_str} side; {seg_u=}", Log_Type.STRANGE)
                continue

            # extract correct label
            sem_vr = sem_seg.extract_label([sem_rib_label, 41, 42, 43, 44, 45, 46, 47, 48, 49], keep_label=True)
            sem_vr[vert_vr == 0] = 0
            # hand it over to one rib function handle
            try:
                data_dict = _measure_one_rib_length(sem_vr, leftside=leftside, vert_id=vert)
            except Exception as e:
                logger.on_fail(e)
                continue
            data_dict.last_v = is_last_v
            data_dict.vert_ori_rel_to_corpus = rel_to_corpus
            data_dict.PIR_angle_degrees = PIR_angle_degrees
            results.append(data_dict)
    return results


def _measure_one_rib_length(sem_vr: NII, vert_id: Vertebra_Instance, leftside: bool, resolution: float = 0.5):
    left_side_str = "left" if leftside else "right"
    init_shp = sem_vr.shape
    # crop
    sem_vr_crop = sem_vr.compute_crop(dist=8)
    sem_vr_cropped = sem_vr.apply_crop(sem_vr_crop)
    logger.print(f"Cropped down from {init_shp}, to {sem_vr_cropped.shape}")
    # then rescale
    orig_zoom = sem_vr.zoom
    sem_vr2 = sem_vr_cropped.rescale((resolution, resolution, resolution), verbose=False, mode="nearest")
    #
    logger.print(f"Calc rib stats, vertebra {vert_id}, {left_side_str} side, resolution={resolution}", verbose=True)
    #########################
    # Call to Rib length measurement algorithm
    data_dict = _rib_length_algorithm(sem_vr2, vert_id, leftside)

    #########################
    data_dict.orig_zoom = list(orig_zoom)

    logger.print(
        f"is stump rib = {data_dict.stump_rib}; length = {data_dict.rib_length:.3f}",
    )

    return data_dict


def sanitize_json_key(k):
    if isinstance(k, np.integer):
        return int(k)

    if isinstance(k, Path):
        return str(k)

    return k


def sanitize_json(obj):
    if isinstance(obj, dict):
        return {sanitize_json_key(k): sanitize_json(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, Path):
        return str(obj.absolute())

    return obj


def result_to_dict(res: list[RibResult]) -> dict[Literal["left", "right"], dict[int, dict]]:
    out: dict[Literal["left", "right"], dict[int, dict]] = {"left": {}, "right": {}}
    for i in res:
        out["left" if i.leftside else "right"][i.vertebra.value] = asdict(i)
        out["left" if i.leftside else "right"][i.vertebra.value]["vertebra"] = out["left" if i.leftside else "right"][i.vertebra.value][
            "vertebra"
        ].name
    return sanitize_json(out)  # type: ignore


if __name__ == "__main__":
    sem_vr = to_nii(
        "/media/data/robert/code/TReg/data/full_body/templates/sub-CTFU04045_ses-02480_sequ-204_mod-ct_seg-vert_msk.nii.gz", True
    )
    sem_vr2 = to_nii(
        "/media/data/robert/code/TReg/data/full_body/templates/sub-CTFU04045_ses-02480_sequ-204_mod-ct_seg-spine_msk.nii.gz", True
    )
    out = measure_ribs_length_subject(sem_vr, sem_vr2)
    print(result_to_dict(out))
    for i in out:
        print(i)
