from __future__ import annotations

from typing import List, Optional

import numpy as np


def dual_flip_options(value: int) -> List[int]:
    v = int(value)
    opts: List[int] = []
    if v > 0:
        opts.append(v - 1)
    if v < 255:
        opts.append(v + 1)
    return [x for x in opts if (x & 1) != (v & 1)]


def flip_u8(value: int, rng: Optional[np.random.Generator] = None) -> int:
    options = dual_flip_options(value)
    if not options:
        raise ValueError("tidak ada opsi flip uint8 yang valid")
    if rng is None:
        v = int(value)
        if v <= 0:
            return 1
        if v >= 255:
            return 254
        return v + 1 if (v & 1) == 0 else v - 1
    if len(options) == 1:
        return options[0]
    return options[int(rng.integers(0, len(options)))]
