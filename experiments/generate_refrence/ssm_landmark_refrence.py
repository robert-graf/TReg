import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from TPTBox import POI_Global

out = str(Path(__file__).parent.parent.parent)
sys.path.append(out)


def load_shapes(out_folder: Path):
    """
    Load POI files and convert them into landmark matrices
    """
    shapes = {}
    landmark_ids = None
    poi_template = None

    for file in sorted(out_folder.glob("*.mrk.json")):
        poi = POI_Global.load(file)

        if poi_template is None:
            poi_template = poi

        pts = []
        ids = []

        for k1, k2, v in poi.items():
            lid = k1 * 100 + k2
            ids.append(lid)
            pts.append(v)

        pts = np.array(pts)

        if landmark_ids is None:
            landmark_ids = ids
        else:
            assert landmark_ids == ids, "Landmark ordering mismatch"

        shapes[file.name] = pts

    return shapes, landmark_ids, poi_template


def procrustes_align(shapes):
    """
    Align shapes via generalized Procrustes analysis
    """
    keys = list(shapes.keys())

    X = np.stack([shapes[k] for k in keys])

    mean_shape = np.mean(X, axis=0)

    for _ in range(10):
        aligned = []

        for s in X:
            s_center = s - s.mean(axis=0)
            m_center = mean_shape - mean_shape.mean(axis=0)

            U, _, Vt = np.linalg.svd(s_center.T @ m_center)

            R = U @ Vt

            s_aligned = s_center @ R

            aligned.append(s_aligned)

        X = np.stack(aligned)

        mean_shape = np.mean(X, axis=0)

    aligned_shapes = {k: X[i] for i, k in enumerate(keys)}

    return aligned_shapes, mean_shape


def build_ssm(shapes):
    """
    Train PCA Statistical Shape Model
    """
    keys = list(shapes.keys())

    X = []

    for k in keys:
        X.append(shapes[k].reshape(-1))

    X = np.stack(X)

    pca = PCA()
    pca.fit(X)

    return pca


def reconstruct_consensus(pca, n_landmarks):
    """
    Reconstruct mean shape from PCA
    """
    mean_shape = pca.mean_.reshape(n_landmarks, 3)
    return mean_shape


def project_shape_to_ssm(pca, shape, clamp=3.0):
    """
    Project shape into valid SSM space
    """
    x = shape.reshape(1, -1)

    b = pca.transform(x)

    std = np.sqrt(pca.explained_variance_)

    b = np.clip(b, -clamp * std, clamp * std)

    x_rec = pca.inverse_transform(b)

    return x_rec.reshape(-1, 3)


def save_poi(mean_shape, landmark_ids, poi_template, out_file):
    """
    Convert numpy landmarks back to POI format
    """

    poi_out = POI_Global(itk_coords=False)

    for lid, pt in zip(landmark_ids, mean_shape):
        k1 = lid // 100
        k2 = lid % 100

        poi_out[k1, k2] = pt

    poi_out.info = poi_template.info

    poi_out.save_mrk(out_file)


def aggregate_ssm(out_folder: Path, out_voting: Path):
    """
    Main SSM aggregation pipeline
    """

    print("Loading shapes")
    out_voting / out_folder.name / "ssm_mean.mrk.json"
    shapes, landmark_ids, poi_template = load_shapes(out_folder)
    out_dir = out_voting / out_folder.name
    out_dir.mkdir(exist_ok=True, parents=True)

    # save_poi(
    #    shapes[next(shapes.keys().__iter__())],
    #    landmark_ids,
    #    poi_template,
    #    out_dir / "test.mrk.json",
    # )
    print(f"Loaded {len(shapes)} shapes")

    print("Running Procrustes alignment")

    aligned_shapes, mean_shape = procrustes_align(shapes)

    print("Training PCA Statistical Shape Model")

    pca = build_ssm(aligned_shapes)

    print("Reconstructing consensus shape")

    mean_shape = reconstruct_consensus(pca, len(landmark_ids))

    # restore global position
    translations = [s.mean(axis=0) for s in shapes.values()]
    global_shift = np.mean(translations, axis=0)

    mean_shape += global_shift

    save_poi(
        mean_shape,
        landmark_ids,
        poi_template,
        out_dir / "ssm_mean.mrk.json",
    )

    print("SSM consensus saved")

    return pca


def reconstruct_all_cases(out_folder: Path, pca, out_voting: Path):
    """
    Optional: project all cases into valid SSM space
    """

    shapes, landmark_ids, poi_template = load_shapes(out_folder)

    out_dir = out_voting / out_folder.name / "ssm_projected"
    out_dir.mkdir(exist_ok=True, parents=True)

    for name, shape in shapes.items():
        shape_rec = project_shape_to_ssm(pca, shape)

        save_poi(shape_rec, landmark_ids, poi_template, out_dir / name)


if __name__ == "__main__":
    from constants import out_voting, path_train_poi

    out_folder = path_train_poi.parent / "treg"

    print("Running Statistical Shape Model aggregation")

    pca = aggregate_ssm(out_folder, out_voting)
    # exit()
    reconstruct_all_cases(out_folder, pca, out_voting)

    print("Done.")
