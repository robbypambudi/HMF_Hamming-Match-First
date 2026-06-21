from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from hmf.flip import flip_u8
from hmf.hamming import (
    carrier_vector,
    group_bits,
    matrix_block_n,
    matrix_capacity,
    perm_indices,
    syndrome,
    syndrome_to_pos,
)
from hmf.plugin import get_plugin_by_mode, mask_matrices_for_modes, mask_matrix_for_mode
from hmf.types import EmbedStats
from hmf.virtual import effective_lsb_block


def count_required_flips(
    cover: np.ndarray,
    payload_bits: Sequence[int],
    *,
    seed: int,
    matrix_r: int,
    embed_scope: str = "ALL_CHANNELS",
    virtual_mode: str = "RGB_RELATION",
) -> int:
    carrier = carrier_vector(cover, embed_scope=embed_scope)
    n_pixels, n_channels = carrier.shape
    block_n = matrix_block_n(matrix_r)
    msg_k = matrix_r
    bits_per_block = msg_k * n_channels
    cap = matrix_capacity(n_pixels, matrix_r, n_channels)
    if len(payload_bits) > cap:
        raise ValueError(f"payload {len(payload_bits):,} bit melebihi kapasitas {cap:,} bit")

    payload_blocks = group_bits(payload_bits, bits_per_block).reshape(-1, n_channels, msg_k)
    perm = perm_indices(n_pixels, seed)
    plugin = get_plugin_by_mode(virtual_mode)
    n_flips = 0

    if plugin.is_adaptive:
        candidates = plugin.candidate_modes
        masks = mask_matrices_for_modes(cover, candidates)
        for bi, target_channels in enumerate(payload_blocks):
            idx_block = perm[bi * block_n : (bi + 1) * block_n]
            for ci, target in enumerate(target_channels):
                lsb_block = (carrier[idx_block, ci] & 1).astype(np.uint8)
                delta = syndrome(lsb_block, matrix_r) ^ target
                if syndrome_to_pos(delta) == 0:
                    continue
                can_avoid = any(
                    syndrome_to_pos(delta ^ syndrome(masks[cand][idx_block, ci].astype(np.uint8), matrix_r)) == 0
                    for cand in candidates
                )
                if not can_avoid:
                    n_flips += 1
        return n_flips

    mode = plugin.mode
    vmask = mask_matrix_for_mode(cover, mode)
    for bi, target_channels in enumerate(payload_blocks):
        idx_block = perm[bi * block_n : (bi + 1) * block_n]
        for ci, target in enumerate(target_channels):
            lsb_block = effective_lsb_block(carrier, idx_block, ci, virtual_mask=vmask)
            delta = syndrome(lsb_block, matrix_r) ^ target
            if syndrome_to_pos(delta) != 0:
                n_flips += 1
    return n_flips


def embed_hamming_matchfirst_lsb(
    cover: np.ndarray,
    payload_bits: Sequence[int],
    *,
    seed: int = 42,
    matrix_r: int = 4,
    embed_scope: str = "ALL_CHANNELS",
    virtual_mode: str = "RGB_RELATION",
) -> Tuple[np.ndarray, EmbedStats]:
    cover = np.asarray(cover, dtype=np.uint8)
    stego = cover.copy()
    carrier = carrier_vector(stego, embed_scope=embed_scope)

    n_pixels, n_channels = carrier.shape
    block_n = matrix_block_n(matrix_r)
    msg_k = matrix_r
    bits_per_block = msg_k * n_channels
    cap = matrix_capacity(n_pixels, matrix_r, n_channels)
    if len(payload_bits) > cap:
        raise ValueError(f"payload {len(payload_bits):,} bit melebihi kapasitas {cap:,} bit")

    payload_blocks = group_bits(payload_bits, bits_per_block).reshape(-1, n_channels, msg_k)
    perm = perm_indices(n_pixels, seed)
    plugin = get_plugin_by_mode(virtual_mode)
    mode = plugin.mode
    virtual_choices: Optional[List[int]] = [] if plugin.is_adaptive else None

    if plugin.is_adaptive:
        candidates = plugin.candidate_modes
        masks = mask_matrices_for_modes(stego, candidates)
    else:
        candidates = ()
        masks = {mode: mask_matrix_for_mode(stego, mode)}

    n_flips = 0
    n_blocks = payload_blocks.shape[0]
    n_no_flip = 0

    for bi, target_channels in enumerate(payload_blocks):
        idx_block = perm[bi * block_n : (bi + 1) * block_n]
        block_flips = 0

        for ci, target in enumerate(target_channels):
            if plugin.is_adaptive:
                lsb_block = (carrier[idx_block, ci] & 1).astype(np.uint8)
                delta = syndrome(lsb_block, matrix_r) ^ target
                fallback_pos = syndrome_to_pos(delta)
                chosen_id = candidates.index("LSB_ONLY")
                pos = fallback_pos

                if fallback_pos != 0:
                    for mode_id, cand in enumerate(candidates):
                        mask_block = masks[cand][idx_block, ci].astype(np.uint8)
                        if syndrome_to_pos(delta ^ syndrome(mask_block, matrix_r)) == 0:
                            chosen_id = mode_id
                            pos = 0
                            break

                assert virtual_choices is not None
                virtual_choices.append(chosen_id)
            else:
                lsb_block = effective_lsb_block(carrier, idx_block, ci, virtual_mask=masks[mode])
                pos = syndrome_to_pos(syndrome(lsb_block, matrix_r) ^ target)

            if pos == 0:
                continue

            flip_idx = int(idx_block[pos - 1])
            carrier[flip_idx, ci] = flip_u8(int(carrier[flip_idx, ci]))
            n_flips += 1
            block_flips += 1

        if block_flips == 0:
            n_no_flip += 1

    stats = EmbedStats(
        n_bits=len(payload_bits),
        n_blocks=n_blocks,
        n_no_flip_blocks=n_no_flip,
        n_flips=n_flips,
        virtual_choices=virtual_choices,
    )
    return stego, stats


def extract_hamming(
    stego: np.ndarray,
    payload_len: int,
    *,
    seed: int = 42,
    matrix_r: int = 4,
    embed_scope: str = "ALL_CHANNELS",
    virtual_mode: str = "RGB_RELATION",
    virtual_choices: Optional[Sequence[int]] = None,
) -> List[int]:
    carrier = carrier_vector(np.asarray(stego, dtype=np.uint8), embed_scope=embed_scope)
    n_pixels, n_channels = carrier.shape
    block_n = matrix_block_n(matrix_r)
    msg_k = matrix_r
    bits_per_block = msg_k * n_channels
    blocks = payload_len // bits_per_block + (1 if payload_len % bits_per_block else 0)

    perm = perm_indices(n_pixels, seed)
    plugin = get_plugin_by_mode(virtual_mode)
    mode = plugin.mode

    if plugin.is_adaptive:
        candidates = plugin.candidate_modes
        if virtual_choices is None:
            raise ValueError("virtual_choices wajib untuk ADAPTIVE_DUAL")
        choices = [int(x) for x in virtual_choices]
        expected = blocks * n_channels
        if len(choices) < expected:
            raise ValueError(f"virtual_choices kurang: {len(choices)} < {expected}")
        masks = mask_matrices_for_modes(stego, candidates)
    else:
        candidates = ()
        choices = []
        masks = {mode: mask_matrix_for_mode(stego, mode)}

    out: List[int] = []
    choice_i = 0
    for bi in range(blocks):
        idx_block = perm[bi * block_n : (bi + 1) * block_n]
        for ci in range(n_channels):
            if plugin.is_adaptive:
                mode_id = choices[choice_i]
                choice_i += 1
                if not 0 <= mode_id < len(candidates):
                    raise ValueError(f"virtual choice tidak valid: {mode_id}")
                mask = masks[candidates[mode_id]]
            else:
                mask = masks[mode]

            lsb_block = effective_lsb_block(carrier, idx_block, ci, virtual_mask=mask)
            out.extend(int(x) for x in syndrome(lsb_block, matrix_r).tolist())

    return out[:payload_len]
