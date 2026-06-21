from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from hmf.plugin import (
    adaptive_candidate_modes,
    mask_matrices_for_modes,
    mask_matrix_for_mode,
    normalize_mode,
)
from hmf.plugin.masks import (
    local_variance_mask,
    rgb_mask_matrix,
    rgb_relation_mask_block,
)

__all__ = [
    "adaptive_modes",
    "effective_lsb_block",
    "local_variance_mask",
    "rgb_mask_matrix",
    "rgb_relation_mask_block",
    "virtual_mask_matrices",
    "virtual_mask_matrix",
]


def virtual_mask_matrix(image: np.ndarray, *, virtual_mode: str) -> np.ndarray:
    return mask_matrix_for_mode(image, normalize_mode(virtual_mode))


def virtual_mask_matrices(image: np.ndarray, modes: Sequence[str]) -> Dict[str, np.ndarray]:
    return mask_matrices_for_modes(image, modes)


def effective_lsb_block(
    carrier: np.ndarray,
    idx_block: np.ndarray,
    channel_idx: int,
    *,
    virtual_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    ci = int(channel_idx)
    lsb = (carrier[idx_block, ci] & 1).astype(np.uint8)
    if virtual_mask is None:
        return lsb
    return lsb ^ virtual_mask[idx_block, ci].astype(np.uint8)


def adaptive_modes() -> tuple[str, ...]:
    return adaptive_candidate_modes("ADAPTIVE_DUAL")
