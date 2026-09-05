# MLEM + FONF — Fractional-Order Multi-Notch Spectral Regularization for CT

Official code for:

> A. P. Singh, M. Khurana, and S. Tiwari, "MLEM Reconstruction with
> Fractional-Order Multi-Notch Spectral Regularization," *IEEE Signal
> Processing Letters*, 2026.

MLEM+FONF embeds a fractional-order multi-notch spectral filter inside the
MLEM iteration as a training-free, geometry-agnostic prior: each iteration
applies the MLEM multiplicative update, shapes the iterate in the frequency
domain, relaxes toward the filtered image (0 < λΔt < 1), and projects to
non-negativity. A data-driven rule (τ = μ_R + 3σ_R on the
background-normalized spectrum) activates K ≥ 0 Gaussian notches only when
narrowband artifacts are detected, so one framework covers broadband Poisson
noise (K = 0) and periodic stripe/ring artifacts (K > 0). No training data
are used anywhere.

## Two implementations

| Folder | What it is | Why it exists |
|---|---|---|
| **`matlab/`** | The pipeline **as run for the paper** (MATLAB R2025b). The shipped `results_stats.csv` matches Table II digit for digit; sanitized comments, computational content verified unchanged at token level and bit-for-bit under Octave (`matlab/EQUIVALENCE_REPORT.md`). | Exact reproducibility of the published numbers. |
| **`python/`** | An independent, self-contained implementation of the closed-form transfer function of Eq. (4) and Algorithm 1 (NumPy/SciPy), with runnable experiment scripts. | Literal Eq. (4) reference; easy to run anywhere; reproduces the paper's qualitative results (rankings, monotone α-ablation, rise-and-stabilize convergence, notch gain +8.2 dB vs +7.7 dB in the paper). |

Implementation note: the as-run base low-pass in `matlab/` uses a rational
fractional-order form (1/(1 + λρ^α) with a spectral mixing factor); the
closed-form Gaussian base printed as Eq. (4) is provided in
`matlab/fonf_eq4.m` (cross-validated against `python/` to machine precision)
and is used end-to-end in `python/`. The notch construction and the
Sec. II-C selection rule are identical to the paper in both.

## Quick start

MATLAB (Table II): open `matlab/`, run `run_all_cases_stats_parallel.m`.

Python:
```bash
cd python && pip install -r requirements.txt && cd experiments
python run_comparison.py     # Table-II-style benchmark
python run_convergence.py    # Fig.-5-style curves
python run_alpha_sweep.py    # Fig.-4a-style ablation
python run_stripe_demo.py    # Fig.-6-style proof of mechanism
```

The real thoracic CT slice is not redistributable; place your own
`thoraxphantom.tif` on the MATLAB path (or drop a 256×256 slice into the
Python `PHANTOMS` registry) to reproduce that column.

## Citation and license

See `CITATION.cff`. Authors' code is MIT-licensed (`LICENSE`); third-party
components retain their own notices (`matlab/THIRD_PARTY.md`).
