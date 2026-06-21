"""Key 3 — Inter-channel RGB correlation."""

from __future__ import annotations

import numpy as np

from hmf.plugin.base import VirtualEmbeddingPlugin
from hmf.plugin.masks import rgb_mask_matrix


class RgbRelationPlugin(VirtualEmbeddingPlugin):
    key = 3
    label = "Inter-channel RGB correlation"
    mode = "RGB_RELATION"

    def mask_matrix(self, image: np.ndarray) -> np.ndarray:
        return rgb_mask_matrix(image, margin=0)
