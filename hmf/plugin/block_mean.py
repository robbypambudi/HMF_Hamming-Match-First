"""Key 1 — Block Mean Representation (local variance)."""

from __future__ import annotations

import numpy as np

from hmf.plugin.base import VirtualEmbeddingPlugin
from hmf.plugin.masks import local_variance_mask


class BlockMeanPlugin(VirtualEmbeddingPlugin):
    key = 1
    label = "Block Mean Representation"
    mode = "LOCAL_VARIANCE"

    def mask_matrix(self, image: np.ndarray) -> np.ndarray:
        return local_variance_mask(image)
