"""Key 5 — Adaptive dual (flip bits)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from hmf.plugin.base import AdaptiveVirtualPlugin


class AdaptiveDualPlugin(AdaptiveVirtualPlugin):
    key = 5
    label = "Adaptive dual (flip bits)"
    mode = "ADAPTIVE_DUAL"

    @property
    def candidate_modes(self) -> Tuple[str, ...]:
        return (
            "RGB_MARGIN_XOR_VARIANCE",
            "RGB_MARGIN",
            "RGB_RELATION",
            "LOCAL_VARIANCE",
            "LSB_ONLY",
        )

    def mask_matrix(self, image: np.ndarray) -> np.ndarray:
        raise ValueError(f"{self.mode} memakai mask per blok, bukan satu matrix global")
