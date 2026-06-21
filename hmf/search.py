from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Sequence, Tuple

import numpy as np

from hmf.embed import count_required_flips
from hmf.hamming import carrier_vector, matrix_capacity
from hmf.metrics import psnr_from_flips


def _running_in_subprocess() -> bool:
    return mp.current_process().name != "MainProcess"


@dataclass(frozen=True)
class _WorkerState:
    cover: np.ndarray
    payload_bits: Tuple[int, ...]
    embed_scope: str
    virtual_mode: str
    n_pixels: int
    n_channels: int
    payload_len: int


def _eval_combo(state: _WorkerState, args: Tuple[int, int]) -> Optional[Tuple[int, int, int, int, float]]:
    seed_value, matrix_r = args
    try:
        cap = matrix_capacity(state.n_pixels, matrix_r, state.n_channels)
        if state.payload_len > cap:
            return None

        flips = count_required_flips(
            state.cover,
            state.payload_bits,
            seed=int(seed_value),
            matrix_r=int(matrix_r),
            embed_scope=state.embed_scope,
            virtual_mode=state.virtual_mode,
        )
        return (
            int(seed_value),
            int(matrix_r),
            int(cap),
            int(flips),
            flips / max(state.payload_len, 1),
        )
    except Exception:
        return None


def _serial_search(
    cover: np.ndarray,
    payload_bits: Sequence[int],
    combos: List[Tuple[int, int]],
    *,
    embed_scope: str,
    virtual_mode: str,
    n_pixels: int,
    n_channels: int,
    payload_len: int,
) -> List[Tuple[int, int, int, int, float]]:
    rows: List[Tuple[int, int, int, int, float]] = []
    for seed_value, matrix_r in combos:
        try:
            cap = matrix_capacity(n_pixels, matrix_r, n_channels)
            flips = count_required_flips(
                cover,
                payload_bits,
                seed=seed_value,
                matrix_r=matrix_r,
                embed_scope=embed_scope,
                virtual_mode=virtual_mode,
            )
            rows.append((seed_value, matrix_r, cap, flips, flips / max(payload_len, 1)))
        except Exception:
            continue
    return rows


def _default_eval_row(
    n_pixels: int,
    n_channels: int,
    payload_len: int,
    r_candidates: Sequence[int],
) -> Tuple[int, int, List[Tuple[int, int, int, int, float]]]:
    r = min(int(x) for x in r_candidates)
    cap = matrix_capacity(n_pixels, r, n_channels)
    row = (0, r, cap, 0, 0.0)
    return 0, r, [row]


def choose_best_matrix_r(
    cover: np.ndarray,
    payload_bits: Sequence[int],
    *,
    seed: int,
    embed_scope: str = "ALL_CHANNELS",
    virtual_mode: str = "RGB_RELATION",
    r_candidates: Sequence[int] = (3, 4, 5, 6, 7),
) -> Tuple[int, List[Tuple[int, int, int, float]]]:
    carrier = carrier_vector(cover, embed_scope=embed_scope)
    n_pixels, n_channels = carrier.shape
    payload_len = len(payload_bits)
    eval_rows: List[Tuple[int, int, int, float]] = []

    for r in sorted(set(int(x) for x in r_candidates)):
        try:
            cap = matrix_capacity(n_pixels, r, n_channels)
            if payload_len > cap:
                continue
            flips = count_required_flips(
                cover,
                payload_bits,
                seed=seed,
                matrix_r=r,
                embed_scope=embed_scope,
                virtual_mode=virtual_mode,
            )
            eval_rows.append((r, cap, flips, flips / max(payload_len, 1)))
        except ValueError:
            continue

    if not eval_rows:
        raise ValueError("tidak ada nilai r yang muat payload")

    best_r, _, _, _ = min(eval_rows, key=lambda x: (x[2], -x[0]))
    return best_r, eval_rows


def choose_best_seed_and_matrix_r(
    cover: np.ndarray,
    payload_bits: Sequence[int],
    *,
    seed_candidates: Sequence[int],
    embed_scope: str = "ALL_CHANNELS",
    virtual_mode: str = "RGB_RELATION",
    r_candidates: Sequence[int] = (3, 4, 5, 6, 7),
    max_workers: Optional[int] = None,
    parallel: bool = True,
    allow_nested_parallel: bool = False,
) -> Tuple[int, int, List[Tuple[int, int, int, int, float]]]:
    carrier = carrier_vector(cover, embed_scope=embed_scope)
    n_pixels, n_channels = carrier.shape
    payload_len = len(payload_bits)

    seeds = sorted(set(int(x) for x in seed_candidates))
    rs = sorted(set(int(x) for x in r_candidates))
    combos = [
        (s, r)
        for s in seeds
        for r in rs
        if payload_len <= matrix_capacity(n_pixels, r, n_channels)
    ]
    if not combos:
        return _default_eval_row(n_pixels, n_channels, payload_len, rs)

    use_parallel = parallel and len(combos) >= 4
    if use_parallel and _running_in_subprocess() and not allow_nested_parallel:
        use_parallel = False

    if not use_parallel:
        eval_rows = _serial_search(
            cover, payload_bits, combos,
            embed_scope=embed_scope, virtual_mode=virtual_mode,
            n_pixels=n_pixels, n_channels=n_channels, payload_len=payload_len,
        )
    else:
        eval_rows = _parallel_search(
            cover, payload_bits, combos,
            embed_scope=embed_scope, virtual_mode=virtual_mode,
            n_pixels=n_pixels, n_channels=n_channels, payload_len=payload_len,
            max_workers=max_workers,
        )

    if not eval_rows:
        return _default_eval_row(n_pixels, n_channels, payload_len, rs)

    best_seed, best_r, _, _, _ = min(eval_rows, key=lambda x: (x[3], -x[1], x[0]))
    return best_seed, best_r, eval_rows


def _parallel_search(
    cover: np.ndarray,
    payload_bits: Sequence[int],
    combos: List[Tuple[int, int]],
    *,
    embed_scope: str,
    virtual_mode: str,
    n_pixels: int,
    n_channels: int,
    payload_len: int,
    max_workers: Optional[int],
) -> List[Tuple[int, int, int, int, float]]:
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = max(1, min(int(max_workers), len(combos), 4))

    state = _WorkerState(
        cover=np.asarray(cover, dtype=np.uint8),
        payload_bits=tuple(int(b) & 1 for b in payload_bits),
        embed_scope=embed_scope,
        virtual_mode=virtual_mode,
        n_pixels=n_pixels,
        n_channels=n_channels,
        payload_len=payload_len,
    )

    eval_combo = partial(_eval_combo, state)
    rows: List[Tuple[int, int, int, int, float]] = []
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for row in pool.map(eval_combo, combos):
                if row is not None:
                    rows.append(row)
    except Exception:
        pass

    if not rows:
        rows = _serial_search(
            cover, payload_bits, combos,
            embed_scope=embed_scope, virtual_mode=virtual_mode,
            n_pixels=n_pixels, n_channels=n_channels, payload_len=payload_len,
        )

    return rows


def compare_virtual_modes(
    cover: np.ndarray,
    payload_bits: Sequence[int],
    *,
    seed_candidates: Sequence[int],
    embed_scope: str,
    r_candidates: Sequence[int],
    virtual_modes: Sequence[str],
    max_workers: Optional[int] = None,
) -> List[Tuple[str, int, int, int, int, float, float]]:
    rows: List[Tuple[str, int, int, int, int, float, float]] = []
    for vmode in virtual_modes:
        best_seed, best_r, eval_rows = choose_best_seed_and_matrix_r(
            cover,
            payload_bits,
            seed_candidates=seed_candidates,
            embed_scope=embed_scope,
            virtual_mode=vmode,
            r_candidates=r_candidates,
            max_workers=max_workers,
        )
        seed_s, r, cap, flips, fpb = min(eval_rows, key=lambda x: (x[3], -x[1], x[0]))
        _, psnr_est = psnr_from_flips(np.asarray(cover).size, flips)
        rows.append((str(vmode), seed_s, r, cap, flips, fpb, psnr_est))
    return sorted(rows, key=lambda x: (x[4], -x[2], x[0]))
