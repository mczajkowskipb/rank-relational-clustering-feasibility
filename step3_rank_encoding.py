# step3_rank_encoding.py
# Deterministic per-sample rank encoding (0..p-1), minimal refactor for multiple datasets

import os
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
OUT_FILE = os.path.join(RESULTS_DIR, "step3_ranks.txt")


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


def x_top500_path(ds: str) -> str:
    return os.path.join(RESULTS_DIR, f"{ds}_X_top500.csv")


def r_top500_path(ds: str) -> str:
    return os.path.join(RESULTS_DIR, f"{ds}_R_top500.csv")


def rank_encode_row(values: np.ndarray) -> np.ndarray:
    """
    Deterministic rank encoding for one sample.
    Ranks: 0 .. p-1
    Tie-break: stable sort + original column index.
    """
    p = values.shape[0]
    idx = np.arange(p)
    order = np.lexsort((idx, values))  # primary: values, secondary: column index
    ranks = np.empty(p, dtype=int)
    ranks[order] = np.arange(p)
    return ranks


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
all_lines = []

for ds in DATASETS:
    in_path = x_top500_path(ds)
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Missing input: {in_path} (run step2 first)")

    X = pd.read_csv(in_path)  # step2 now writes without index for all datasets
    X_values = X.values
    n, p = X_values.shape

    R = np.zeros((n, p), dtype=int)
    for i in range(n):
        R[i, :] = rank_encode_row(X_values[i, :])

    R_df = pd.DataFrame(R, columns=X.columns)
    out_path = r_top500_path(ds)
    R_df.to_csv(out_path, index=False)

    # sanity checks
    lines = []
    lines.append(f"Dataset: {ds}")
    lines.append(f"R shape: {R_df.shape}")

    for i in range(min(3, n)):
        ok_perm = np.array_equal(np.sort(R[i, :]), np.arange(p))
        has_dup = (len(np.unique(R[i, :])) != p)
        lines.append(f"Sample {i}: permutation_ok={ok_perm}, duplicates={has_dup}")

    lines.append(f"Rank min: {R.min()}, Rank max: {R.max()}")
    lines.append(f"Saved: {out_path}")
    lines.append("-" * 40)

    for line in lines:
        print(line)
    all_lines.extend(lines)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for line in all_lines:
        f.write(line + "\n")
