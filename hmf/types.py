from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EmbedStats:
    n_bits: int
    n_blocks: int
    n_no_flip_blocks: int
    n_flips: int
    n_no_flip_bits: int
    virtual_choices: Optional[List[int]] = None

    @property
    def n_flip_bits(self) -> int:
        return self.n_bits - self.n_no_flip_bits

    @property
    def flips_per_bit(self) -> float:
        return self.n_flips / max(self.n_bits, 1)

    @property
    def no_flip_bit_ratio(self) -> float:
        return self.n_no_flip_bits / max(self.n_bits, 1)
