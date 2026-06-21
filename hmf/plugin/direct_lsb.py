"""Key 0 — Direct LSB."""

from __future__ import annotations

import numpy as np

from hmf.hamming import carrier_vector
from hmf.plugin.base import VirtualEmbeddingPlugin


class DirectLsbPlugin(VirtualEmbeddingPlugin):
    key = 0
    label = "Direct LSB"
    mode = "LSB_ONLY"

    def mask_matrix(self, image: np.ndarray) -> np.ndarray:
        carrier = carrier_vector(image)
        return np.zeros_like(carrier, dtype=np.uint8)
