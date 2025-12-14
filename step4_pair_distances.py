# step4_pair_distances.py
# Pairwise distance sanity test (Footrule + Kendall) on rank-encoded data
# Minimal refactor for multiple datasets

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
OUT_FILE = os.path.join(RESULTS_DIR, "step4_pairdist.txt")


# ------------------------------------------------------------
# helpers (common)
# ------------------------------------------------------------
def x_path(ds: str) -> str:
    return os.path.join(DATA_DIR, ds, "X.csv")


def y_path(ds: str) -> str:
    return os.path.join(DATA_DIR, ds, "y.csv")


def read_X(ds: str) -> pd.DataFrame:
    # exception: golub has an extra index column in data/<ds>/X.csv
    if ds == "golub":
        return pd.read_csv(x_path(ds), index_col=0)
    return pd.read_csv(x_path(ds))


def read_y(ds: str) -> pd.Series:
    y_df = pd.read_csv(y_path(ds))
    return y_df.iloc[:, 0]


def r_top500_path(ds: str) -> str:
    return os.path.join(RESULTS_DIR, f"{ds}_R_top500.csv")


def read_R(ds: str) -> np.ndarray:
    # step3 writes without index for all datasets
    return pd.read_csv(r_top500_path(ds)).values


def footrule(R0: np.ndarray, R1: np.ndarray) -> int:
    return int(np.abs(R0 - R1).sum())


def kendall_discordant_pairs(R0: np.ndarray, R1: np.ndarray) -> int:
    p = R0.shape[0]
    cnt = 0
    for i in range(p):
        for j in range(i + 1, p):
            if (R0[i] - R0[j]) * (R1[i] - R1[j]) < 0:
                cnt += 1
    return cnt


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
all_lines = []

for ds in DATASETS:
    r_path = r_top500_path(ds)
    if not os.path.exists(r_path):
        raise FileNotFoundError(f"Missing input: {r_path} (run step3 first)")

    R = read_R(ds)
    if R.shape[0] < 2:
        raise ValueError(f"{ds}: need at least 2 samples, got {R.shape[0]}")

    R0, R1 = R[0], R[1]

    lines = []
    lines.append(f"Dataset: {ds}")
    lines.append(f"R shape: {R.shape}")

    # Footrule
    t0 = time.perf_counter()
    fr = footrule(R0, R1)
    t1 = time.perf_counter()
    fr_ms = (t1 - t0) * 1000.0

    # Kendall (naive O(p^2))
    t0 = time.perf_counter()
    kd = kendall_discordant_pairs(R0, R1)
    t1 = time.perf_counter()
    kd_ms = (t1 - t0) * 1000.0

    lines.append(f"Footrule value: {fr}")
    lines.append(f"Footrule time [ms]: {fr_ms:.3f}")
    lines.append(f"Kendall value (discordant pairs): {kd}")
    lines.append(f"Kendall time [ms]: {kd_ms:.3f}")
    lines.append("-" * 40)

    for line in lines:
        print(line)
    all_lines.extend(lines)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for line in all_lines:
        f.write(line + "\n")
