"""Key 4 — Joint margin-and-mean."""

from __future__ import annotations

import numpy as np

from hmf.plugin.base import VirtualEmbeddingPlugin
from hmf.plugin.masks import local_variance_mask, rgb_mask_matrix


class JointMarginMeanPlugin(VirtualEmbeddingPlugin):
    key = 4
    label = "Joint margin-and-mean"
    mode = "RGB_MARGIN_XOR_VARIANCE"

    def mask_matrix(self, image: np.ndarray) -> np.ndarray:
        return rgb_mask_matrix(image, margin=8) ^ local_variance_mask(image)
