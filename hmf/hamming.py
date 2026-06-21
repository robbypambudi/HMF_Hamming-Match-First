from __future__ import annotations

from typing import Sequence

import numpy as np


def matrix_block_n(matrix_r: int) -> int:
    r = int(matrix_r)
    if not 2 <= r <= 7:
        raise ValueError("matrix_r harus di rentang 2..7")
    return (1 << r) - 1


def matrix_capacity(n_pixels: int, matrix_r: int, n_channels: int = 1) -> int:
    n = matrix_block_n(matrix_r)
    return (int(n_pixels) // n) * int(matrix_r) * int(n_channels)


def group_bits(bits: Sequence[int], k: int) -> np.ndarray:
    arr = np.array([int(b) & 1 for b in bits], dtype=np.uint8)
    pad = (-arr.size) % k
    if pad:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])
    return arr.reshape(-1, k)


def syndrome(lsb_block: np.ndarray, matrix_r: int) -> np.ndarray:
    out = np.zeros(matrix_r, dtype=np.uint8)
    for i, bit in enumerate(lsb_block, start=1):
        if int(bit) & 1:
            for b in range(matrix_r):
                out[b] ^= (i >> b) & 1
    return out


def syndrome_to_pos(delta: np.ndarray) -> int:
    pos = 0
    for b, bit in enumerate(delta):
        pos |= (int(bit) & 1) << b
    return pos


def perm_indices(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(n).astype(np.int64)


def carrier_vector(arr: np.ndarray, embed_scope: str = "ALL_CHANNELS") -> np.ndarray:
    arr = np.asarray(arr, dtype=np.uint8)
    scope = str(embed_scope).upper()

    if arr.ndim == 2:
        return arr.reshape(-1, 1)

    if arr.ndim == 3 and arr.shape[2] == 3:
        if scope != "ALL_CHANNELS":
            raise ValueError("untuk RGB, embed_scope wajib ALL_CHANNELS")
        return arr.reshape(-1, 3)

    raise ValueError("gambar harus 2D grayscale atau 3D RGB")
