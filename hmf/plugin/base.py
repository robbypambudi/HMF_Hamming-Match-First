"""Antarmuka dasar plugin virtual embedding."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class VirtualEmbeddingPlugin(ABC):
    """Satu metode virtual embedding (key paper + mode internal)."""

    key: int
    label: str
    mode: str

    @abstractmethod
    def mask_matrix(self, image: np.ndarray) -> np.ndarray:
        """Matrix mask virtual, bentuk sama dengan carrier_vector(image)."""

    @property
    def is_adaptive(self) -> bool:
        return False

    @property
    def candidate_modes(self) -> Tuple[str, ...]:
        return ()


class AdaptiveVirtualPlugin(VirtualEmbeddingPlugin):
    """Plugin adaptif: pilih kandidat mask per blok channel."""

    @property
    def is_adaptive(self) -> bool:
        return True

    @abstractmethod
    def mask_matrix(self, image: np.ndarray) -> np.ndarray:
        raise ValueError(f"{self.mode} memakai mask per blok, bukan satu matrix global")
