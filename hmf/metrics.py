from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _gaussian_kernel(win_size: int = 11, sigma: float = 1.5) -> np.ndarray:
    coords = np.arange(win_size, dtype=np.float64) - win_size // 2
    g = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


def ssim(cover: np.ndarray, stego: np.ndarray, *, win_size: int = 11, sigma: float = 1.5) -> float:
    k1, k2, l_val = 0.01, 0.03, 255.0
    c1 = (k1 * l_val) ** 2
    c2 = (k2 * l_val) ** 2
    kernel = _gaussian_kernel(win_size, sigma)

    def plane(c_plane: np.ndarray, s_plane: np.ndarray) -> float:
        c = c_plane.astype(np.float64)
        s = s_plane.astype(np.float64)
        wv_c = sliding_window_view(c, (win_size, win_size))
        wv_s = sliding_window_view(s, (win_size, win_size))
        wv_c2 = sliding_window_view(c * c, (win_size, win_size))
        wv_s2 = sliding_window_view(s * s, (win_size, win_size))
        wv_cs = sliding_window_view(c * s, (win_size, win_size))

        mu_c = (wv_c * kernel).sum(axis=(-2, -1))
        mu_s = (wv_s * kernel).sum(axis=(-2, -1))
        mu_c2, mu_s2 = mu_c ** 2, mu_s ** 2
        mu_cs = mu_c * mu_s

        sig_c2 = (wv_c2 * kernel).sum(axis=(-2, -1)) - mu_c2
        sig_s2 = (wv_s2 * kernel).sum(axis=(-2, -1)) - mu_s2
        sig_cs = (wv_cs * kernel).sum(axis=(-2, -1)) - mu_cs

        num = (2.0 * mu_cs + c1) * (2.0 * sig_cs + c2)
        den = (mu_c2 + mu_s2 + c1) * (sig_c2 + sig_s2 + c2)
        return float(np.mean(num / den))

    cover_u8 = np.asarray(cover, dtype=np.uint8)
    stego_u8 = np.asarray(stego, dtype=np.uint8)
    if cover_u8.ndim == 2:
        return plane(cover_u8, stego_u8)
    return float(np.mean([plane(cover_u8[:, :, ch], stego_u8[:, :, ch]) for ch in range(cover_u8.shape[2])]))


def psnr(cover: np.ndarray, stego: np.ndarray) -> Tuple[float, float]:
    diff = cover.astype(np.int16) - stego.astype(np.int16)
    mse = float(np.mean(diff * diff))
    if mse == 0:
        return mse, float("inf")
    return mse, float(10.0 * np.log10((255.0 * 255.0) / mse))


def psnr_from_flips(n_values: int, n_flips: int) -> Tuple[float, float]:
    mse = float(n_flips) / max(int(n_values), 1)
    if mse == 0:
        return mse, float("inf")
    return mse, float(10.0 * np.log10((255.0 * 255.0) / mse))
