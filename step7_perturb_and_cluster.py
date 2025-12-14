# step7_perturb_and_cluster.py
# Robustness under perturbations: baseline / dropout / noise / scaling
# Datasets: golub, colon, DLBCL
#
# Inputs:
#   results/<ds>_X_top500.csv
#   data/<ds>/y.csv
#
# Output (append after each run):
#   results/step7_perturb_runs.csv
#   results/step7_log.txt

import os
import time
import hashlib
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
SEED = 123  # kept for consistency
BASE_SEED = 123

DATASETS = ["golub", "colon", "DLBCL"]
SCENARIOS = ["baseline", "dropout", "noise", "scaling"]
N_REPS = 10

os.makedirs(RESULTS_DIR, exist_ok=True)

RUNS_CSV = os.path.join(RESULTS_DIR, "step7_perturb_runs.csv")
LOG_TXT = os.path.join(RESULTS_DIR, "step7_log.txt")


# ------------------------------------------------------------
# Optional numba
# ------------------------------------------------------------
USE_NUMBA = False
NUMBA_THREADS = None
try:
    import numba
    from numba import njit, prange

    USE_NUMBA = True
    try:
        NUMBA_THREADS = numba.get_num_threads()
    except Exception:
        NUMBA_THREADS = None
except Exception:
    USE_NUMBA = False


# ------------------------------------------------------------
# helpers (common)
# ------------------------------------------------------------
def x_top500_path(ds: str) -> str:
    return os.path.join(RESULTS_DIR, f"{ds}_X_top500.csv")


def y_path(ds: str) -> str:
    return os.path.join(DATA_DIR, ds, "y.csv")


def read_X_top500(ds: str) -> pd.DataFrame:
    pth = x_top500_path(ds)
    if not os.path.exists(pth):
        raise FileNotFoundError(f"Missing input {pth} (run step2 first)")
    X = pd.read_csv(pth)
    return X


def read_y(ds: str) -> np.ndarray:
    pth = y_path(ds)
    if not os.path.exists(pth):
        raise FileNotFoundError(f"Missing input {pth}")
    y_df = pd.read_csv(pth)
    y = y_df.iloc[:, 0].values
    return y


def make_seed(ds: str, scenario: str, rep: int) -> int:
    s = f"{BASE_SEED}|{ds}|{scenario}|{rep}"
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32-bit-ish deterministic seed


def log(msg: str) -> None:
    print(msg)
    with open(LOG_TXT, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def ensure_finite(X: np.ndarray, context: str) -> None:
    if np.isnan(X).any():
        raise ValueError(f"{context}: X contains NaN")
    if np.isinf(X).any():
        raise ValueError(f"{context}: X contains inf")


def rank_encode_matrix(X: np.ndarray) -> np.ndarray:
    """
    Deterministic ranks per sample:
      - ranks 0..p-1
      - tie-break via feature index (lexsort)
    """
    n, p = X.shape
    idx = np.arange(p)
    R = np.empty((n, p), dtype=np.int32)
    for i in range(n):
        order = np.lexsort((idx, X[i, :]))  # primary: value, secondary: feature index
        r = np.empty(p, dtype=np.int32)
        r[order] = np.arange(p, dtype=np.int32)
        R[i, :] = r
    return R


# ------------------------------------------------------------
# Distance matrices (numba if available; fallback otherwise)
# ------------------------------------------------------------
if USE_NUMBA:

    @njit(parallel=True, fastmath=False)
    def footrule_matrix_numba(R):
        n, p = R.shape
        D = np.zeros((n, n), dtype=np.int64)
        for i in prange(n):
            for j in range(i + 1, n):
                s = 0
                for t in range(p):
                    a = R[i, t] - R[j, t]
                    if a < 0:
                        a = -a
                    s += a
                D[i, j] = s
                D[j, i] = s
        return D

    @njit(parallel=True, fastmath=False)
    def kendall_matrix_numba(R):
        n, p = R.shape
        D = np.zeros((n, n), dtype=np.int64)
        for i in prange(n):
            for j in range(i + 1, n):
                cnt = 0
                for a in range(p):
                    ra = R[i, a]
                    rj_a = R[j, a]
                    for b in range(a + 1, p):
                        if (ra - R[i, b]) * (rj_a - R[j, b]) < 0:
                            cnt += 1
                D[i, j] = cnt
                D[j, i] = cnt
        return D

else:

    def footrule_matrix_numba(R):
        n, p = R.shape
        D = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = int(np.abs(R[i] - R[j]).sum())
                D[j, i] = D[i, j]
        return D

    def kendall_matrix_numba(R):
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


# ------------------------------------------------------------
# PAM (k-medoids) on precomputed distances (same as step6)
# ------------------------------------------------------------
def pam(D: np.ndarray, k: int, n_init: int = 1, random_state: int = 123, max_iter: int = 30) -> np.ndarray:
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
# Perturbations
# ------------------------------------------------------------
def apply_scenario(X: np.ndarray, scenario: str, rng: np.random.Generator):
    """
    Returns: Xp (n x p'), p', metadata dict
    """
    n, p = X.shape
    if scenario == "baseline":
        return X.copy(), p, {}

    if scenario == "dropout":
        keep = rng.choice(p, size=int(round(0.70 * p)), replace=False)  # keep 70%
        keep.sort()
        Xp = X[:, keep].copy()
        return Xp, Xp.shape[1], {"dropped_features": p - Xp.shape[1]}

    if scenario == "noise":
        Xp = X.copy()
        std = Xp.std(axis=0, ddof=0)
        scale = 0.20 * std
        # if std == 0 => no noise in that feature
        eps = rng.normal(loc=0.0, scale=1.0, size=Xp.shape) * scale
        Xp = Xp + eps
        return Xp, p, {}

    if scenario == "scaling":
        Xp = X.copy()
        idx = rng.choice(n, size=int(round(0.50 * n)), replace=False)
        Xp[idx, :] *= 10.0
        return Xp, p, {"scaled_samples": len(idx)}

    raise ValueError(f"Unknown scenario: {scenario}")


# ------------------------------------------------------------
# Resume / append logic
# ------------------------------------------------------------
def load_completed_set():
    if not os.path.exists(RUNS_CSV):
        return set()
    df = pd.read_csv(RUNS_CSV)
    # unique key per (dataset, scenario, rep, method)
    keys = set(
        zip(df["dataset"].astype(str), df["scenario"].astype(str), df["rep"].astype(int), df["method"].astype(str))
    )
    return keys


def append_rows(rows):
    df = pd.DataFrame(rows)
    header = not os.path.exists(RUNS_CSV)
    df.to_csv(RUNS_CSV, mode="a", header=header, index=False)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
log(f"STEP7 start | USE_NUMBA={USE_NUMBA} | NUMBA_THREADS={NUMBA_THREADS}")

completed = load_completed_set()
log(f"Loaded completed runs: {len(completed)} rows")

for ds in DATASETS:
    Xdf = read_X_top500(ds)
    y = read_y(ds)

    if pd.isna(y).any():
        raise ValueError(f"{ds}: y contains NaN (fix export).")

    X = Xdf.values.astype(np.float64, copy=False)
    ensure_finite(X, f"{ds}:input")

    n = len(y)
    if X.shape[0] != n:
        raise ValueError(f"{ds}: X rows {X.shape[0]} != len(y) {n}")

    k = len(np.unique(y))

    for scenario in SCENARIOS:
        for rep in range(N_REPS):
            seed = make_seed(ds, scenario, rep)
            rng = np.random.default_rng(seed)

            # Prepare X'
            Xp, p_eff, meta = apply_scenario(X, scenario, rng)
            ensure_finite(Xp, f"{ds}:{scenario}:{rep}:post_perturb")

            # Rank encode (for rank-based methods)
            # Note: kmeans uses Xp directly.
            # We measure times per method below.

            # --- kmeans_euclid ---
            method = "kmeans_euclid"
            key = (ds, scenario, rep, method)
            if key not in completed:
                t_total0 = time.perf_counter()
                t0 = time.perf_counter()
                Xs = StandardScaler().fit_transform(Xp)
                km = KMeans(n_clusters=k, n_init=20, random_state=SEED)
                labels = km.fit_predict(Xs)
                t1 = time.perf_counter()

                ari = adjusted_rand_score(y, labels)
                t_total1 = time.perf_counter()

                row = dict(
                    dataset=ds, scenario=scenario, rep=rep, n=int(n), p=int(p_eff),
                    method=method, k=int(k), ARI=float(ari),
                    time_total_s=float(t_total1 - t_total0),
                    time_dist_s=0.0,
                    time_cluster_s=float(t1 - t0),
                )
                append_rows([row])
                completed.add(key)
                log(f"{ds} | {scenario} | rep={rep} | {method} | ARI={ari:.4f} | total={row['time_total_s']:.2f}s")

            # For rank methods compute ranks once per (ds,scenario,rep)
            # (still counted inside time_dist_s per method; but reused to avoid recompute)
            R = None
            R_ready = False
            if scenario in ("baseline", "dropout", "noise", "scaling"):
                R = rank_encode_matrix(Xp)
                # sanity for first row only (fast)
                if R.shape[0] > 0:
                    pchk = R.shape[1]
                    if not np.array_equal(np.sort(R[0]), np.arange(pchk)):
                        raise ValueError(f"{ds}:{scenario}:{rep}: rank encoding sanity failed for sample 0")

            # --- kmedoids_footrule ---
            method = "kmedoids_footrule"
            key = (ds, scenario, rep, method)
            if key not in completed:
                t_total0 = time.perf_counter()
                t_dist0 = time.perf_counter()
                Df = footrule_matrix_numba(R)
                t_dist1 = time.perf_counter()

                t_cl0 = time.perf_counter()
                labels = pam(Df, k, n_init=1, random_state=SEED, max_iter=30)
                t_cl1 = time.perf_counter()

                ari = adjusted_rand_score(y, labels)
                t_total1 = time.perf_counter()

                row = dict(
                    dataset=ds, scenario=scenario, rep=rep, n=int(n), p=int(p_eff),
                    method=method, k=int(k), ARI=float(ari),
                    time_total_s=float(t_total1 - t_total0),
                    time_dist_s=float(t_dist1 - t_dist0),
                    time_cluster_s=float(t_cl1 - t_cl0),
                )
                append_rows([row])
                completed.add(key)
                log(f"{ds} | {scenario} | rep={rep} | {method} | ARI={ari:.4f} | dist={row['time_dist_s']:.2f}s | cl={row['time_cluster_s']:.2f}s")

            # --- kmedoids_kendall ---
            method = "kmedoids_kendall"
            key = (ds, scenario, rep, method)
            if key not in completed:
                t_total0 = time.perf_counter()
                t_dist0 = time.perf_counter()
                Dk = kendall_matrix_numba(R)
                t_dist1 = time.perf_counter()

                t_cl0 = time.perf_counter()
                labels = pam(Dk, k, n_init=1, random_state=SEED, max_iter=30)
                t_cl1 = time.perf_counter()

                ari = adjusted_rand_score(y, labels)
                t_total1 = time.perf_counter()

                row = dict(
                    dataset=ds, scenario=scenario, rep=rep, n=int(n), p=int(p_eff),
                    method=method, k=int(k), ARI=float(ari),
                    time_total_s=float(t_total1 - t_total0),
                    time_dist_s=float(t_dist1 - t_dist0),
                    time_cluster_s=float(t_cl1 - t_cl0),
                )
                append_rows([row])
                completed.add(key)
                log(f"{ds} | {scenario} | rep={rep} | {method} | ARI={ari:.4f} | dist={row['time_dist_s']:.2f}s | cl={row['time_cluster_s']:.2f}s")

log("STEP7 done. Output: " + RUNS_CSV)
