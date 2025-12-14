# step1_io_sanity.py
# I/O sanity check for multiple datasets (minimal refactor, same logic)

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
OUT_FILE = os.path.join(RESULTS_DIR, "step1_io.txt")


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
    # y is a single-column CSV (header may be 'y' or something else)
    y_df = pd.read_csv(y_path(ds))
    return y_df.iloc[:, 0]


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
all_lines = []

for ds in DATASETS:
    X = read_X(ds)
    y = read_y(ds)

    vc = y.value_counts(dropna=False)

    lines = []
    lines.append(f"Dataset: {ds}")
    lines.append(f"X.shape: {X.shape}")
    lines.append(f"y.shape: {y.shape}")
    lines.append("y value counts:")
    lines.append(str(vc))
    lines.append(f"X contains NaN: {X.isna().any().any()}")
    lines.append("-" * 40)

    for line in lines:
        print(line)
    all_lines.extend(lines)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for line in all_lines:
        f.write(line + "\n")
