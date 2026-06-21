"""Key 2 — Intra-block RGB margin."""

from __future__ import annotations

import numpy as np

from hmf.plugin.base import VirtualEmbeddingPlugin
from hmf.plugin.masks import rgb_mask_matrix


class RgbMarginPlugin(VirtualEmbeddingPlugin):
    key = 2
    label = "Intra-block RGB margin"
    mode = "RGB_MARGIN"

    def mask_matrix(self, image: np.ndarray) -> np.ndarray:
        return rgb_mask_matrix(image, margin=8)
