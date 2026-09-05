"""Sparse parallel-beam projector (system matrix A).

Geometry follows the paper: 64 detector bins, 90 angles over 360 degrees,
256 x 256 image grid, uniform initialization.  Rays are line integrals
sampled at unit steps with bilinear interpolation, assembled once into a
scipy CSR matrix so that MLEM can use exact matched forward/backprojection
pairs (A, A^T).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def build_system_matrix(n: int = 256, n_bins: int = 64, n_angles: int = 90,
                        span_deg: float = 360.0) -> sp.csr_matrix:
    """Assemble the sparse system matrix A of shape (n_bins*n_angles, n*n).

    Each row integrates the image along one parallel ray using bilinear
    weights at unit-length sample points.
    """
    angles = np.deg2rad(np.arange(n_angles) * (span_deg / n_angles))
    # detector bin centres across the field of view
    s = np.linspace(-(n - 1) / 2.0, (n - 1) / 2.0, n_bins)
    t = np.arange(-n, n + 1, 1.0)                      # samples along the ray
    c = (n - 1) / 2.0

    rows, cols, vals = [], [], []
    for j, th in enumerate(angles):
        d = np.array([np.cos(th), np.sin(th)])          # ray direction
        nv = np.array([-np.sin(th), np.cos(th)])        # detector normal
        # sample points for all bins at once: (n_bins, n_t, 2)
        px = s[:, None] * nv[0] + t[None, :] * d[0] + c
        py = s[:, None] * nv[1] + t[None, :] * d[1] + c
        x0 = np.floor(px).astype(np.int64)
        y0 = np.floor(py).astype(np.int64)
        fx, fy = px - x0, py - y0
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                          (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xx, yy = x0 + dx, y0 + dy
            ok = (xx >= 0) & (xx < n) & (yy >= 0) & (yy < n) & (w > 1e-12)
            bin_idx = np.broadcast_to(np.arange(len(s))[:, None], xx.shape)[ok]
            rows.append(j * n_bins + bin_idx)
            cols.append((yy[ok] * n + xx[ok]))
            vals.append(w[ok])
    A = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_bins * n_angles, n * n),
    ).tocsr()
    A.sum_duplicates()
    return A


def poisson_sinogram(A: sp.csr_matrix, image: np.ndarray, incident: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Photon-limited measurements: g_i ~ Poisson((A f)_i) scaled by `incident`.

    `incident` sets the mean count level (lower = noisier), mirroring the
    ill-posed low-dose regime of the paper.
    """
    proj = A @ image.ravel()
    lam = proj / max(proj.max(), 1e-12) * incident
    g = rng.poisson(lam).astype(np.float64)
    return g * (proj.max() / incident)                 # back to line-integral scale
