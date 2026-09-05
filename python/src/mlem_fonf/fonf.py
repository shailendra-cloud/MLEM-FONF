"""Fractional-order multi-notch filter (FONF).

Implements the transfer function of Eq. (4),

    H_FONF(w) = G_alpha(w) * prod_m [1 - exp(-||w - w_m||^2 / (2 sigma_m^2))]^(alpha/2),
    G_alpha(w) = exp(-alpha ||w||^2 / (4 w_N^2)),

with w_N the Nyquist radius of the discrete spectrum (fixed, no free
parameter), and the data-driven notch selection rule of Sec. II-C.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter


def _freq_grid(n: int):
    f = np.fft.fftfreq(n)                       # cycles / pixel in [-0.5, 0.5)
    wx, wy = np.meshgrid(2 * np.pi * f, 2 * np.pi * f, indexing="xy")
    return wx, wy, np.sqrt(wx ** 2 + wy ** 2)


def h_fonf(n: int, alpha: float = 0.5, notches=()) -> np.ndarray:
    """Return H_FONF on the (unshifted) FFT grid.

    `notches` is a sequence of (wx_m, wy_m, sigma_m) in radians/pixel; the
    conjugate of each notch is applied automatically so the filter stays real.
    """
    wx, wy, wr = _freq_grid(n)
    w_nyq = np.pi                                # Nyquist radius (fixed)
    H = np.exp(-alpha * wr ** 2 / (4.0 * w_nyq ** 2))
    for (cx, cy, sg) in notches:
        for sx, sy in ((cx, cy), (-cx, -cy)):    # conjugate pair
            d2 = (wx - sx) ** 2 + (wy - sy) ** 2
            H *= (1.0 - np.exp(-d2 / (2.0 * sg ** 2))) ** (alpha / 2.0)
    return H


def select_notches(image: np.ndarray, sigma_cap_frac: float = 0.05,
                   k_max: int = 8, dc_guard: int = 3):
    """Sec. II-C rule: peaks of R = |F(f)| / S_bar(||w||) above tau = mu_R + 3 sigma_R.

    Returns a list of (wx_m, wy_m, sigma_m); empty when no structured peak is
    detected (K = 0, the broadband regime).
    """
    n = image.shape[0]
    F = np.fft.fftshift(np.abs(np.fft.fft2(image)))
    cy = cx = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - cy, xx - cx)
    rbin = r.astype(int)
    # radially averaged magnitude spectrum S_bar(rho)
    s_sum = np.bincount(rbin.ravel(), F.ravel())
    s_cnt = np.bincount(rbin.ravel())
    s_bar = s_sum / np.maximum(s_cnt, 1)
    R = F / np.maximum(s_bar[rbin], 1e-12)
    R[r <= dc_guard] = 0.0                       # exclude the DC neighbourhood
    tau = R[r > dc_guard].mean() + 3.0 * R[r > dc_guard].std()
    peaks = (R == maximum_filter(R, size=7)) & (R > tau)
    pys, pxs = np.nonzero(peaks)
    if len(pys) == 0:
        return []
    order = np.argsort(-R[pys, pxs])
    notches, used = [], np.zeros(n * n, bool)
    for idx in order:
        py, px = pys[idx], pxs[idx]
        if px < cx or (px == cx and py < cy):    # keep one of each conjugate pair
            continue
        # half-width at half-maximum along x, capped
        half = R[py, px] / 2.0
        w = 1
        while px + w < n and px - w >= 0 and min(R[py, px + w], R[py, px - w]) > half:
            w += 1
        sigma = min(w * 2 * np.pi / n, sigma_cap_frac * np.pi)
        wxm = (px - cx) * 2 * np.pi / n
        wym = (py - cy) * 2 * np.pi / n
        notches.append((wxm, wym, sigma))
        if len(notches) >= k_max:
            break
    return notches


def apply_filter(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    """f_hat = F^{-1}( H_FONF . F(f) ), real part."""
    return np.real(np.fft.ifft2(H * np.fft.fft2(image)))
