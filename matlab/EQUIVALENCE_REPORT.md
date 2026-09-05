# Verification: sanitized code == as-run code == paper numbers

The MATLAB sources here are the code that generated every number in the
paper. Comments and internal notes were rewritten for publication; the
computational content was verified unchanged at three levels.

**1. Provenance.** `results_stats.csv` in this folder was written by
`run_all_cases_stats_parallel.m` (10 seeds x 1000 iterations x 4 datasets)
and matches Table II of the paper digit for digit (e.g., thorax MLEM+FONF
PSNR 31.0752 -> 31.08, SNR 18.4575 -> 18.46, MSSIM 0.8358 -> 0.836).
`alpha_sensitivity.csv` matches Fig. 4a.

**2. Mechanical equivalence.** A MATLAB-aware comment stripper reduced each
original and sanitized file to its executable token stream; 18/19 files are
token-identical. Documented exceptions (neither affects any computed value):
`alpha_sensitivity.m` line 1 had a stray token `ai ` that made the shipped
file unrunnable and was removed; one console banner string in
`main_single_case.m` was shortened (`fprintf` text only).

**3. Execution equivalence.** Under GNU Octave 8.4, the original and
sanitized versions of all six reconstruction algorithms (MLEM, MLEM+TV,
MLEM+AD, MLEM+FuzzyAD, MLEM+FONF, PnP-ADMM(TV)), the FONF filter, and all
eight metrics were run on identical seeded inputs: all 16 outputs were
bit-for-bit identical.

**Cross-language check.** `fonf_eq4.m` (the closed form of Eq. (4)) agrees
with the Python implementation `python/src/mlem_fonf/fonf.py` to machine
precision (max abs difference 1.1e-16 on a 64x64 grid with one notch pair).
