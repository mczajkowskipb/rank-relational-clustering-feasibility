# step5_dist_matrices.py
# Full pairwise distance matrices (Footrule + Kendall) on rank-encoded data
# Minimal refactor for multiple datasets
#
# Outputs (per dataset):
#   results/cache_<ds>_footrule.npz  (contains D_footrule)
#   results/cache_<ds>_kendall.npz   (contains D_kendall)
# and summary:
#   results/step5_dist.txt

import os
import time
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# CONFIG (common section)
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SEED = 123

DATASETS = ["golub", "colon", "ALL", "DLBCL", "vantVeer"]

os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_FILE = os.path.join(RESULTS_DIR, "step5_dist.txt")


# ------------------------------------------------------------
# helpers (common)
# ------------------------------------------------------------
def x_path(ds: str) -> str:
    return os.path.join(DATA_DIR, ds, "X.csv")


def y_path(ds: str) -> str:
    return os.path.join(DATA_DIR, ds, "y.csv")


def read_X(ds: str) -> pd.DataFrame:
    if ds == "golub":
        return pd.read_csv(x_path(ds), index_col=0)
    return pd.read_csv(x_path(ds))


def read_y(ds: str) -> pd.Series:
    y_df = pd.read_csv(y_path(ds))
    return y_df.iloc[:, 0]


def r_top500_path(ds: str) -> str:
    return os.path.join(RESULTS_DIR, f"{ds}_R_top500.csv")


def read_R(ds: str) -> np.ndarray:
    return pd.read_csv(r_top500_path(ds)).values


def cache_footrule_path(ds: str) -> str:
    return os.path.join(RESULTS_DIR, f"cache_{ds}_footrule.npz")


def cache_kendall_path(ds: str) -> str:
    return os.path.join(RESULTS_DIR, f"cache_{ds}_kendall.npz")


def footrule_matrix(R: np.ndarray) -> np.ndarray:
    n, p = R.shape
    D = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            s = int(np.abs(R[i] - R[j]).sum())
            D[i, j] = s
            D[j, i] = s
    return D


def kendall_matrix(R: np.ndarray) -> np.ndarray:
    n, p = R.shape
    D = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        Ri = R[i]
        for j in range(i + 1, n):
            Rj = R[j]
            cnt = 0
            for a in range(p):
                for b in range(a + 1, p):
                    if (Ri[a] - Ri[b]) * (Rj[a] - Rj[b]) < 0:
                        cnt += 1
            D[i, j] = cnt
            D[j, i] = cnt
    return D


def stats_min_median_max(D: np.ndarray):
    return int(D.min()), float(np.median(D)), int(D.max())


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
all_lines = []

for ds in DATASETS:
    r_path = r_top500_path(ds)
    if not os.path.exists(r_path):
        raise FileNotFoundError(f"Missing input: {r_path} (run step3 first)")

    print(f"\n=== {ds} ===")
    R = read_R(ds)
    n, p = R.shape

    # Footrule
    t0 = time.perf_counter()
    Df = footrule_matrix(R)
    t1 = time.perf_counter()
    tf = t1 - t0

    np.savez_compressed(cache_footrule_path(ds), D_footrule=Df)

    # Kendall
    t0 = time.perf_counter()
    Dk = kendall_matrix(R)
    t1 = time.perf_counter()
    tk = t1 - t0

    np.savez_compressed(cache_kendall_path(ds), D_kendall=Dk)

    fmin, fmed, fmax = stats_min_median_max(Df)
    kmin, kmed, kmax = stats_min_median_max(Dk)

    lines = []
    lines.append(f"Dataset: {ds}")
    lines.append(f"D_footrule shape: {Df.shape}")
    lines.append(f"D_kendall shape: {Dk.shape}")
    lines.append(f"Footrule time [s]: {tf:.3f}")
    lines.append(f"Kendall time [s]: {tk:.3f}")
    lines.append(f"Footrule stats (min/median/max): {fmin} / {fmed:.3f} / {fmax}")
    lines.append(f"Kendall stats (min/median/max): {kmin} / {kmed:.3f} / {kmax}")
    lines.append(f"Saved: {cache_footrule_path(ds)}")
    lines.append(f"Saved: {cache_kendall_path(ds)}")
    lines.append("-" * 40)

    for line in lines:
        print(line)
    all_lines.extend(lines)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for line in all_lines:
        f.write(line + "\n")
