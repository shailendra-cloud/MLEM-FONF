# MATLAB code (as run for the paper)

This folder contains the exact pipeline that produced the results in the
paper, with comments rewritten for publication and computational content
verified unchanged (see `EQUIVALENCE_REPORT.md`).

## Reproduce Table II
Run `run_all_cases_stats_parallel.m` (requires the Parallel Computing
Toolbox; replace `parfor` with `for` otherwise). It evaluates all six
methods over 10 Poisson-noise seeds x 1000 iterations on the three
synthetic phantoms, plus the real thorax slice if `thoraxphantom.tif` is on
the path, and writes `results_stats.csv` (the shipped copy already matches
the paper's Table II).

## Other entry points
- `main_single_case.m` — interactive single-run comparison with figures
  (reconstructions, convergence curves, line profile with zoom inset).
- `ablation_alpha.m`, `alpha_sensitivity.m` — fractional-order ablation
  (Fig. 4a; shipped CSVs included).
- `fonf_stripe_demo.m` — multi-notch K>0 proof of mechanism (Fig. 6):
  the tau = mu_R + 3*sigma_R rule of Sec. II-C with Eq. (4) notches.
- `fonf_eq4.m` — the closed form of Eq. (4) (Gaussian base + notches),
  cross-validated against the Python implementation to machine precision.

## Note on the base spectral term
The as-run base low-pass in `fonf_filter2D_improved.m` is a rational
fractional-order form, 1/(1 + lambda*rho^alpha), with a spectral mixing
factor; the closed-form Gaussian expression printed as Eq. (4) is provided
in `fonf_eq4.m` and is implemented end-to-end in `python/`, which
reproduces the paper's qualitative results. Both are fractional-order
low-pass filters; the notch construction and the Sec. II-C selection rule
are identical to the paper in `fonf_stripe_demo.m`.

Data note: the real thoracic slice is not redistributable; supply your own
`thoraxphantom.tif` to reproduce that column.
