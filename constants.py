from pathlib import Path

import numpy as np

basepath = str(Path(__file__).parent)
path_annotator_poi = Path(basepath, "input", "userstudy", "pois")
path_train_poi = Path(basepath, "input", "train", "pois")
path_mrk = Path(basepath, "pois_mrk")
out_userstudy = Path(basepath, "results", "userstudy")

out_userstudy.mkdir(exist_ok=True, parents=True)

raters_all = [
    "Julius/Julius_V1",
    "Julius/Julius_V2",
    "Julius/Julius_V3",
    "Leon",
    "Philipp",
    # "Robert_Model",
    "Robert_Model_1",
    "Robert_Model_2",
    "Robert_Model_3",
]

flips_model = [
    "Robert_Model_1",
    "Robert_Model_2",
    "Robert_Model_3",
]
mapp_models_filp = {
    (13, 16): (13, 17),
    (13, 17): (13, 16),
    (13, 21): (13, 14),
    (13, 14): (13, 21),
    (13, 15): (13, 22),
    (13, 22): (13, 15),
}

rater_key: dict[str, tuple[str, str]] = {}
for rater in raters_all:
    version = rater.split("_")[-1]
    if "_" not in rater:
        version = "V1"
    evaluator = rater.split("/")[0].rsplit("_", maxsplit=1)[0]
    rater_key[rater] = (evaluator, version)

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
    "PPP": (5, 1),
    "PDP": (5, 2),
    "PMP": (5, 3),
    "PLP": (5, 4),
    "PRPP": (5, 5),
    "PRDP": (5, 6),
    "PRHP": (5, 7),
}


def extract_xyz(df, x_col, y_col, z_col, evaluator, version):
    out = df[["filename", "POI_name", x_col, y_col, z_col]].copy()
    out.columns = ["filename", "POI_name", "X", "Y", "Z"]
    out["evaluator"] = evaluator
    out["version"] = version
    return out


def euclidean_3d(a, b):
    return np.linalg.norm(a - b, axis=1)


def icc_2_1(data):
    n, k = data.shape
    mean_subject = np.mean(data, axis=1)
    mean_rater = np.mean(data, axis=0)
    grand_mean = np.mean(data)

    ss_subject = k * np.sum((mean_subject - grand_mean) ** 2)
    ss_rater = n * np.sum((mean_rater - grand_mean) ** 2)
    ss_error = np.sum((data - mean_subject[:, None] - mean_rater + grand_mean) ** 2)

    ms_subject = ss_subject / (n - 1)
    ms_rater = ss_rater / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    # https://de.wikipedia.org/wiki/Intraklassen-Korrelation
    icc = (ms_subject - ms_error) / (ms_subject + (k - 1) * ms_error + k * (ms_rater - ms_error) / n)

    return icc, ms_error


def icc(Y, icc_type="ICC(2,1)"):
    """Calculate intraclass correlation coefficient

    ICC Formulas are based on:
    Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
    assessing rater reliability. Psychological bulletin, 86(2), 420.
    icc1:  x_ij = mu + beta_j + w_ij
    icc2/3:  x_ij = mu + alpha_i + beta_j + (ab)_ij + epsilon_ij
    Code modifed from nipype algorithms.icc
    https://github.com/nipy/nipype/blob/master/nipype/algorithms/icc.py

    Args:
        Y: The data Y are entered as a 'table' ie. subjects are in rows and repeated
            measures in columns
        icc_type: type of ICC to calculate. (ICC(2,1), ICC(2,k), ICC(3,1), ICC(3,k))
    Returns:
        ICC: (np.array) intraclass correlation coefficient
    """

    [n, k] = Y.shape

    # Degrees of Freedom
    dfc = k - 1
    dfe = (n - 1) * (k - 1)
    dfr = n - 1

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
    x0 = np.tile(np.eye(n), (k, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T), Y.flatten("F"))
    residuals = Y.flatten("F") - predicted_Y
    SSE = (residuals**2).sum()

    MSE = SSE / dfe

    # Sum square column effect - between colums
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
    MSC = SSC / dfc  # / n (without n in SPSS results)

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    if icc_type == "icc1":
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        # ICC = (MSR - MSRW) / (MSR + (k-1) * MSRW)
        NotImplementedError("This method isn't implemented yet.")

    elif icc_type == "ICC(2,1)" or icc_type == "ICC(2,k)":
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        if icc_type == "ICC(2,k)":
            k = 1
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)

    elif icc_type == "ICC(3,1)" or icc_type == "ICC(3,k)":
        # ICC(3,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error)
        if icc_type == "ICC(3,k)":
            k = 1
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE)

    return ICC, SSE
