"""Utilitas mask virtual embedding (dipakai bersama antar plugin)."""

from __future__ import annotations

import numpy as np

from hmf.hamming import carrier_vector


def rgb_relation_mask_block(
    image: np.ndarray,
    idx_block: np.ndarray,
    channel_idx: int,
    *,
    margin: int = 0,
) -> np.ndarray:
    img = np.asarray(image, dtype=np.uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        return np.zeros(len(idx_block), dtype=np.uint8)

    flat_rgb = (img.reshape(-1, 3) & np.uint8(0xFE)).astype(np.int16)
    pixels = flat_rgb[idx_block]
    ch = int(channel_idx)
    m = int(margin)

    if ch == 0:
        return ((pixels[:, 0] - pixels[:, 1]) > m).astype(np.uint8)
    if ch == 1:
        return ((pixels[:, 1] - pixels[:, 2]) > m).astype(np.uint8)
    if ch == 2:
        return ((pixels[:, 2] - pixels[:, 0]) > m).astype(np.uint8)
    raise ValueError("channel_idx RGB harus 0, 1, atau 2")


def local_variance_mask(
    image: np.ndarray,
    *,
    threshold: float = 256.0,
    window_radius: int = 1,
) -> np.ndarray:
    img = np.asarray(image, dtype=np.uint8) & np.uint8(0xFE)
    planes = img[:, :, None] if img.ndim == 2 else img
    h, w, n_channels = planes.shape

    rows = np.repeat(np.arange(h), w)
    cols = np.tile(np.arange(w), h)
    r0 = np.maximum(0, rows - window_radius)
    r1 = np.minimum(h, rows + window_radius + 1)
    c0 = np.maximum(0, cols - window_radius)
    c1 = np.minimum(w, cols + window_radius + 1)
    area = (r1 - r0) * (c1 - c0)

    out = np.zeros((h * w, n_channels), dtype=np.uint8)
    for ch in range(n_channels):
        plane = planes[:, :, ch].astype(np.float64)
        integral = np.pad(plane, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        integral_sq = np.pad(plane * plane, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        sums = integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
        sums_sq = integral_sq[r1, c1] - integral_sq[r0, c1] - integral_sq[r1, c0] + integral_sq[r0, c0]
        mean = sums / area
        var = (sums_sq / area) - (mean * mean)
        out[:, ch] = (var > threshold).astype(np.uint8)
    return out


def rgb_mask_matrix(image: np.ndarray, *, margin: int = 0) -> np.ndarray:
    idx_all = np.arange(carrier_vector(image).shape[0], dtype=np.int64)
    cols = [
        rgb_relation_mask_block(image, idx_all, 0, margin=margin),
        rgb_relation_mask_block(image, idx_all, 1, margin=margin),
        rgb_relation_mask_block(image, idx_all, 2, margin=margin),
    ]
    return np.stack(cols, axis=1).astype(np.uint8)
