# step6_baseline_clustering.py
# Baseline clustering for multiple datasets (load cached distances from step5)
#
# Outputs:
#   results/step6_baseline.csv

import os
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# ------------------------------------------------------------
# CONFIG (common section)
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SEED = 123

DATASETS = ["golub", "colon", "ALL", "DLBCL", "vantVeer"]

os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_FILE = os.path.join(RESULTS_DIR, "step6_baseline.csv")


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


def read_X_top500(ds: str) -> np.ndarray:
    return pd.read_csv(x_top500_path(ds)).values


def cache_path(ds: str, metric: str) -> str:
    # metric in {"footrule","kendall"}
    return os.path.join(RESULTS_DIR, f"cache_{ds}_{metric}.npz")


def load_D(ds: str, metric: str) -> np.ndarray:
    path = cache_path(ds, metric)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing cache for {ds}/{metric}: {path}. Run step5_dist_matrices.py first."
        )
    z = np.load(path)
    key = f"D_{metric}"  # step5 stores D_footrule / D_kendall
    if key not in z:
        raise KeyError(f"Cache {path} does not contain key '{key}'. Found: {list(z.keys())}")
    return z[key]


# ------------------------------------------------------------
# Simple PAM (k-medoids) on precomputed distances
# ------------------------------------------------------------
def pam(D: np.ndarray, k: int, n_init: int = 1, random_state: int = 123, max_iter: int = 20) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n = D.shape[0]

    best_labels = None
    best_cost = np.inf

    for _ in range(n_init):
        medoids = rng.choice(n, size=k, replace=False)

        for _ in range(max_iter):
            labels = np.argmin(D[:, medoids], axis=1)

            new_medoids = medoids.copy()
            for i in range(k):
                cluster_idx = np.where(labels == i)[0]
                if len(cluster_idx) == 0:
                    continue
                costs = D[np.ix_(cluster_idx, cluster_idx)].sum(axis=1)
                new_medoids[i] = cluster_idx[np.argmin(costs)]

            if np.array_equal(new_medoids, medoids):
                break
            medoids = new_medoids

        total_cost = np.sum(D[np.arange(n), medoids[labels]])
        if total_cost < best_cost:
            best_cost = total_cost
            best_labels = labels.copy()

    return best_labels


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
rows = []

for ds in DATASETS:
    print(f"\n=== Dataset: {ds} ===")

    # Inputs
    if not os.path.exists(x_top500_path(ds)):
        raise FileNotFoundError(f"Missing {x_top500_path(ds)} (run step2 first)")
    X = read_X_top500(ds)

    if not os.path.exists(cache_path(ds, "footrule")):
        raise FileNotFoundError(f"Missing {cache_path(ds, 'footrule')} (run step5 first)")
    if not os.path.exists(cache_path(ds, "kendall")):
        raise FileNotFoundError(f"Missing {cache_path(ds, 'kendall')} (run step5 first)")

    Df = load_D(ds, "footrule")
    Dk = load_D(ds, "kendall")

    y = read_y(ds).values
    if pd.isna(y).any():
        raise ValueError(f"{ds}: y contains NaN (fix export).")

    # Shape consistency checks
    n = len(y)
    if X.shape[0] != n:
        raise ValueError(f"{ds}: X rows {X.shape[0]} != len(y) {n}")
    if Df.shape != (n, n) or Dk.shape != (n, n):
        raise ValueError(f"{ds}: D shape mismatch with y (Df {Df.shape}, Dk {Dk.shape}, n {n})")

    k = len(np.unique(y))
    print(f"k = {k}")

    # 1) Value baseline: k-means
    t0 = time.perf_counter()
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED)
    labels_km = km.fit_predict(Xs)
    t1 = time.perf_counter()

    ari_km = adjusted_rand_score(y, labels_km)
    dt_km = t1 - t0
    print(f"kmeans_euclid: ARI={ari_km:.4f}, time={dt_km:.2f}s")
    rows.append(dict(dataset=ds, method="kmeans_euclid", k=k, ARI=ari_km, runtime_seconds=dt_km))

    # 2) Rank-space: Footrule + PAM (from cache)
    t0 = time.perf_counter()
    labels_f = pam(Df, k, n_init=1, random_state=SEED, max_iter=20)
    t1 = time.perf_counter()

    ari_f = adjusted_rand_score(y, labels_f)
    dt_f = t1 - t0
    print(f"kmedoids_footrule: ARI={ari_f:.4f}, time={dt_f:.2f}s")
    rows.append(dict(dataset=ds, method="kmedoids_footrule", k=k, ARI=ari_f, runtime_seconds=dt_f))

    # 3) Rank-space: Kendall + PAM (from cache)
    t0 = time.perf_counter()
    labels_k = pam(Dk, k, n_init=1, random_state=SEED, max_iter=20)
    t1 = time.perf_counter()

    ari_k = adjusted_rand_score(y, labels_k)
    dt_k = t1 - t0
    print(f"kmedoids_kendall: ARI={ari_k:.4f}, time={dt_k:.2f}s")
    rows.append(dict(dataset=ds, method="kmedoids_kendall", k=k, ARI=ari_k, runtime_seconds=dt_k))

df = pd.DataFrame(rows)
df.to_csv(OUT_FILE, index=False)

print("\nSaved:", OUT_FILE)
print(df)
