# step2_mad_top500.py
# MAD-based top-500 feature filtering (minimal refactor, same logic)

import os
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
OUT_FILE = os.path.join(RESULTS_DIR, "step2_mad.txt")


# ------------------------------------------------------------
# helpers (common)
# ------------------------------------------------------------
def x_path(ds: str) -> str:
    return os.path.join(DATA_DIR, ds, "X.csv")


def y_path(ds: str) -> str:
    return os.path.join(DATA_DIR, ds, "y.csv")


def read_X(ds: str) -> pd.DataFrame:
    # exception: golub has an extra index column
    if ds == "golub":
        return pd.read_csv(x_path(ds), index_col=0)
    return pd.read_csv(x_path(ds))


def read_y(ds: str) -> pd.Series:
    y_df = pd.read_csv(y_path(ds))
    return y_df.iloc[:, 0]


def compute_mad(X: pd.DataFrame) -> pd.Series:
    """
    MAD(j) = median_i |x_ij - median_i(x_ij)|
    computed column-wise (same definition as before)
    """
    med = X.median(axis=0)
    mad = (X.sub(med, axis=1)).abs().median(axis=0)
    return mad


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
all_lines = []

for ds in DATASETS:
    print(f"\n=== {ds} ===")
    X = read_X(ds)
    orig_shape = X.shape

    mad = compute_mad(X).sort_values(ascending=False)

    top = mad.iloc[:500]
    top_cols = top.index.tolist()

    X_top = X.loc[:, top_cols]

    # outputs
    out_x = os.path.join(RESULTS_DIR, f"{ds}_X_top500.csv")
    out_cols = os.path.join(RESULTS_DIR, f"{ds}_top500_cols.txt")

    X_top.to_csv(out_x, index=False)  # keep no index to stay consistent with exports
    with open(out_cols, "w", encoding="utf-8") as f:
        for c in top_cols:
            f.write(str(c) + "\n")

    # sanity
    has_dups = len(top_cols) != len(set(top_cols))

    lines = []
    lines.append(f"Dataset: {ds}")
    lines.append(f"X original shape: {orig_shape}")
    lines.append(f"X top500 shape: {X_top.shape}")
    lines.append(
        f"MAD(top500) stats: min={top.min():.6g}, median={top.median():.6g}, max={top.max():.6g}"
    )
    lines.append(f"Duplicates in top500: {has_dups}")
    lines.append(f"Saved: {out_x}")
    lines.append(f"Saved: {out_cols}")
    lines.append("-" * 40)

    for line in lines:
        print(line)
    all_lines.extend(lines)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for line in all_lines:
        f.write(line + "\n")
