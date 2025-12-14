# Rank-Relational Clustering – Preliminary Feasibility Study

This repository contains a **reference (non-optimized) implementation** and **preliminary experiments** supporting a feasibility study of **rank-based relational clustering** included in an NCN OPUS grant proposal.

The goal of this repository is **not** to provide a production-ready framework, but to demonstrate:
- feasibility of relational clustering based on within-sample orderings,
- robustness to common perturbations in omics data,
- realistic computational costs using a simple baseline implementation.

---

## Scope of the repository

The experiments compare:
- **Value-based clustering**: k-means with Euclidean distance,
- **Relational clustering**: k-medoids (PAM) operating in rank space using:
  - Footrule distance,
  - Kendall (discordant pairs) distance.

The focus is on **robustness**, not absolute clustering accuracy.

---

## Datasets

The repository includes **three standard, publicly available gene expression benchmarks** obtained from Bioconductor:

| Dataset | Samples (N) | Features (P) | Task |
|-------|------------|--------------|------|
| Golub ALL/AML | 72 | ~7,000 | leukemia subtype |
| Colon cancer | 62 | ~2,000 | tumor vs normal |
| DLBCL | 194 | ~3,500 | lymphoma subtype |

These datasets are **widely used in methodological studies** of high-dimensional clustering and were selected to avoid dataset-specific tuning or cherry-picking.

> **Note:** Original datasets are included for transparency and reproducibility. They originate from Bioconductor packages and are redistributed here strictly for research and academic use.

---

## Preprocessing

To limit dimensionality while remaining fully unsupervised:
- the **top 500 genes** were selected using **median absolute deviation (MAD)**,
- no class labels were used during feature selection.

Each sample was then **rank-encoded**, producing within-sample permutations used for relational distances.

---

## Perturbation scenarios

Robustness was evaluated under the following controlled perturbations:

- **Baseline**  
  Original data after MAD-based feature selection (top 500 genes).

- **Feature dropout**  
  Random removal of **20% of features** per run.

- **Additive noise**  
  Additive Gaussian noise with variance equal to **20% of the empirical variance** of each feature.

- **Monotonic scaling**  
  Sample-wise multiplicative scaling applied to **50% of samples**, with scaling factors drawn from `[0.5, 2.0]`, simulating batch effects while preserving within-sample orderings.

Each scenario was repeated multiple times; clustering quality was assessed using **Adjusted Rand Index (ARI)**.

---

## Results

Final robustness results are provided as:
- **tables** (`results/step8_table.tex`),
- **figures** (`figures/fig_step8_*.png`).

The figures show that:
- value-based k-means is highly sensitive to perturbations,
- rank-based relational clustering maintains substantially higher stability,
- monotonic scaling (batch effects) is particularly well handled by rank-based methods.

---

## Computational performance

All experiments were executed using a **straightforward, non-optimized Python implementation** on a **standard consumer-grade laptop** (no GPU, no parallelization).

Observed runtimes:
- Footrule distance matrices: **seconds**,
- Kendall distance matrices: **minutes**,  
  corresponding to **roughly an order-of-magnitude slowdown compared to Footrule**,
- k-medoids clustering on cached distances: **well under one second**.

These results demonstrate that rank-based relational clustering is **already computationally feasible at this scale**, motivating further optimization and scalability work.

---

## Repository structure
├── data/ # Original datasets (Bioconductor)
│ ├── golub/
│ ├── colon/
│ └── DLBCL/
├── results/ # Intermediate results and cached distances
├── figures/ # Final robustness plots
├── step1_io_sanity.py
├── step2_mad_top500.py
├── step3_rank_encoding.py
├── step4_pair_distances.py
├── step5_dist_matrices.py
├── step6_baseline_clustering.py
├── step7_perturbations.py
├── step8_aggregation_and_plots.py
├── requirements.txt
└── README.md


Scripts are designed to be run **sequentially** (`step1` → `step8`).

---

## Reproducibility

To reproduce the experiments:

```bash
pip install -r requirements.txt
python step1_io_sanity.py
python step2_mad_top500.py
python step3_rank_encoding.py
python step4_pair_distances.py
python step5_dist_matrices.py
python step6_baseline_clustering.py
python step7_perturbations.py
python step8_aggregation_and_plots.py


Disclaimer

This repository provides a reference implementation used for a preliminary feasibility study.

The code is intentionally simple and unoptimized.

No claims are made regarding optimal performance or scalability.

Results are intended to motivate further methodological and computational development.

License

This project is released under the MIT License.

Citation

If you reference this repository in an academic context, please cite it as:

Czajkowski, M. (2025). Rank-relational clustering: preliminary feasibility study. GitHub repository.
