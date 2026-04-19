import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial import cKDTree
from skimage import measure
from sklearn.decomposition import PCA
from TPTBox import BIDS_FILE, NII, POI_Global, to_nii

out = str(Path(__file__).parent.parent.parent)
sys.path.append(out)

from constants import POI_MAP, flips_model, mapp_models_filp, out_atlas, out_userstudy, out_voting, path_mrk, path_train_poi, raters_all
from experiments.generate_refrence.reg import derivatives_folder, fetch, local_folder

# ---------------------------------------------------------
# Surface extraction
# ---------------------------------------------------------


def sample_mesh_points(verts, faces, n_points=5000):
    """
    Uniformly sample points on mesh faces
    """
    tri_vertices = verts[faces]

    areas = (
        np.linalg.norm(
            np.cross(
                tri_vertices[:, 1] - tri_vertices[:, 0],
                tri_vertices[:, 2] - tri_vertices[:, 0],
            ),
            axis=1,
        )
        / 2
    )

    probs = areas / areas.sum()
    face_ids = np.random.choice(len(faces), n_points, p=probs)

    r1 = np.sqrt(np.random.rand(n_points))
    r2 = np.random.rand(n_points)

    a = tri_vertices[face_ids, 0]
    b = tri_vertices[face_ids, 1]
    c = tri_vertices[face_ids, 2]

    samples = (1 - r1)[:, None] * a + (r1 * (1 - r2))[:, None] * b + (r1 * r2)[:, None] * c

    return samples


from skimage import measure


def segmentation_to_mesh(seg: NII):

    arr = seg.get_array()
    affine = seg.affine

    verts, faces, _, _ = measure.marching_cubes(arr, level=0)

    verts_world = voxel_to_world(verts, affine)

    return verts_world, faces


def voxel_to_world(voxels, affine):
    """
    Convert voxel coordinates (N,3) to world coordinates using affine
    """
    voxels = np.asarray(voxels)

    ones = np.ones((voxels.shape[0], 1))
    vox_h = np.concatenate([voxels, ones], axis=1)

    world = vox_h @ affine.T

    return world[:, :3]


def segmentation_to_points(nii: NII, n_points=5000):

    verts, faces = segmentation_to_mesh(nii)
    pts = sample_mesh_points(verts, faces, n_points)

    return pts


# ---------------------------------------------------------
# Procrustes alignment
# ---------------------------------------------------------


def rigid_align(A, B):
    """
    Rigid alignment of A to B
    """
    Ac = A - A.mean(0)
    Bc = B - B.mean(0)

    U, _, Vt = np.linalg.svd(Ac.T @ Bc)

    R = U @ Vt
    t = B.mean(0) - A.mean(0) @ R

    return R, t


def apply_rigid(X, R, t):
    return X @ R + t


# ---------------------------------------------------------
# Correspondence building
# ---------------------------------------------------------


def compute_correspondence(reference, points):

    tree = cKDTree(points)
    d, idx = tree.query(reference)

    return points[idx]


# ---------------------------------------------------------
# Build Surface SSM
# ---------------------------------------------------------


class SurfaceSSM:
    def __init__(self, n_components=20):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.mean = None
        self.reference = None

    def fit(self, point_sets):

        ref = point_sets[0]
        aligned = []

        for pts in point_sets:
            R, t = rigid_align(pts, ref)
            pts = apply_rigid(pts, R, t)

            corr = compute_correspondence(ref, pts)

            aligned.append(corr)

        X = np.stack([p.reshape(-1) for p in aligned])

        self.pca.fit(X)

        self.mean = self.pca.mean_.reshape(ref.shape)
        self.reference = ref

    def reconstruct(self, b):

        x = self.pca.mean_ + self.pca.components_.T @ b
        return x.reshape(self.mean.shape)

    def save(self, path):
        data = {
            "n_components": self.n_components,
            "pca": self.pca,
            "mean": self.mean,
            "reference": self.reference,
        }
        joblib.dump(data, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)

        obj = cls(data["n_components"])
        obj.pca = data["pca"]
        obj.mean = data["mean"]
        obj.reference = data["reference"]

        return obj


# ---------------------------------------------------------
# Fit SSM to new segmentation
# ---------------------------------------------------------


def fit_ssm_to_points(ssm, seg_points, n_iter=25):

    shape = ssm.mean.copy()

    for _ in range(n_iter):
        tree = cKDTree(seg_points)

        d, idx = tree.query(shape)

        closest = seg_points[idx]

        R, t = rigid_align(shape, closest)

        shape = apply_rigid(shape, R, t)

        x = shape.reshape(1, -1)

        b = ssm.pca.transform(x)

        std = np.sqrt(ssm.pca.explained_variance_)
        b = np.clip(b, -3 * std, 3 * std)

        shape = ssm.pca.inverse_transform(b).reshape(shape.shape)

    return shape


# ---------------------------------------------------------
# Landmark extraction from surface
# ---------------------------------------------------------


def project_landmarks_to_surface(surface_pts, landmark_pts):

    tree = cKDTree(surface_pts)

    d, idx = tree.query(landmark_pts)

    return surface_pts[idx]


# ---------------------------------------------------------
# Train SSM
# ---------------------------------------------------------

from tqdm import tqdm


def train_surface_ssm(seg_paths):

    point_sets = []

    for p in tqdm(seg_paths, desc="extract_mesh"):
        nii = to_nii(p, True)[::10, ::10, ::10]

        pts = segmentation_to_points(nii)

        point_sets.append(pts)
    print("SurfaceSSM")
    ssm = SurfaceSSM(n_components=min(20, len(seg_paths)))
    print("fit SurfaceSSM")
    ssm.fit(point_sets)

    return ssm


# ---------------------------------------------------------
# Infer POIs
# ---------------------------------------------------------


def infer_pois(ssm, new_seg_path, landmark_template):

    nii = to_nii(new_seg_path, True)

    seg_pts = segmentation_to_points(nii)

    fitted_surface = fit_ssm_to_points(ssm, seg_pts)

    template_landmarks = []

    ids = []

    for k1, k2, v in landmark_template.items():
        ids.append((k1, k2))
        template_landmarks.append(v)

    template_landmarks = np.array(template_landmarks)

    predicted = project_landmarks_to_surface(
        fitted_surface,
        template_landmarks,
    )

    poi_out = POI_Global(itk_coords=False)

    for (k1, k2), pt in zip(ids, predicted):
        poi_out[k1, k2] = pt

    poi_out.info = landmark_template.info
    return poi_out
    # poi_out.save_mrk(out_file)


# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------

if __name__ == "__main__":
    # Train
    out_mrk = path_train_poi.parent / "treg"

    # Folder of all registered segmentations
    train_segmentations = sorted(out_mrk.glob("*.nii.gz"))
    assert len(train_segmentations) != 0

    template_poi = POI_Global.load(out_voting / "treg" / "ssm_mean.mrk.json")
    # SSM
    ssm = train_surface_ssm(train_segmentations)
    ssm.save(out_atlas / "ssm.pkl")
    # infer_pois(
    #    ssm,
    #    train_segmentations[-1],
    #    template_poi,
    #    out_atlas.parent / f"{train_segmentations[-1].name.split('.')[0]}_predicted_landmarks.mrk.json",
    # )
