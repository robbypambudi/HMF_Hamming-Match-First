# HMF — Hamming Matching First

LSB steganography based on **Hamming matrix embedding** with a match-first strategy and RGB virtual views.

## Paper keys (virtual mode)

| Key | Name | Mode |
|-----|------|------|
| 0 | Direct LSB | `LSB_ONLY` |
| 1 | Block Mean Representation | `LOCAL_VARIANCE` |
| 2 | Intra-block RGB margin | `RGB_MARGIN` |
| 3 | Inter-channel RGB correlation | `RGB_RELATION` |
| 4 | Joint margin-and-mean | `RGB_MARGIN_XOR_VARIANCE` |
| 5 | Adaptive dual (flip bits) | `ADAPTIVE_DUAL` |

Key 5 picks the best virtual view per block-channel; a physical ±1 flip is applied only when the virtual view alone is not enough.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Single embed run

```bash
python run_embed.py \
  --cover cover_images/Baboon_RGB.tiff \
  --payload payload/random-binary_10Kb.txt \
  --virtual-key 5
```

Important options:
- `--virtual-key 0..5` — select a paper mode (available keys follow registered plugins)
- `--no-parallel` — serial seed+r search
- `--workers N` — number of processes for seed+r search
- `--no-huffman` — disable Huffman compression

Output: `output/stego.tiff`, `output/key.json`, `output/extracted.txt`

## Batch evaluation (PSNR / SSIM)

```bash
python run_batch.py --virtual-key 5
```

Produces:
- `output/results_psnr_ssim.csv` — per-cell details
- `output/psnr_matrix.csv` — cover × payload matrix
- `output/ssim_matrix.csv`

Options:
- `--workers 2` — run up to N cells in parallel (each in its own subprocess)
- `--serial` — one cell at a time (safest, recommended on low-memory machines)
- `--cell-timeout 7200` — per-cell timeout in seconds (default 7200)
- `--parallel-search` — parallel seed+r search (threads) inside each cell
- `--search-workers 4` — thread count for `--parallel-search`

## Module layout

```
hmf/
  keys.py      — key → mode mapping (via plugin registry)
  plugin/      — one file per virtual embedding method
  io.py        — read/write images, payload, key JSON
  huffman.py   — optional payload compression
  hamming.py   — syndrome, capacity, carrier
  virtual.py   — virtual view mask wrapper
  embed.py     — embed & extract
  search.py    — seed+r optimization (multiprocessing spawn)
  metrics.py   — PSNR, SSIM
```

## Adding a virtual embedding plugin

Each virtual view method is implemented as a plugin under `hmf/plugin/`. Once registered, the new key is automatically available in `KEY_LABELS`, `--virtual-key`, embed/extract, and batch evaluation.

### 1. Create a plugin file

Add a new module, for example `hmf/plugin/my_method.py`:

```python
from __future__ import annotations

import numpy as np

from hmf.plugin.base import VirtualEmbeddingPlugin
from hmf.plugin.masks import rgb_mask_matrix  # shared utilities, optional


class MyMethodPlugin(VirtualEmbeddingPlugin):
    key = 6
    label = "My new method"
    mode = "MY_METHOD"

    def mask_matrix(self, image: np.ndarray) -> np.ndarray:
        # Output shape = carrier_vector(image): (n_pixel, n_channel), dtype uint8
        return rgb_mask_matrix(image, margin=4)
```

Required fields:
- `key` — paper key number (unique, must not conflict with other plugins)
- `label` — display name
- `mode` — internal mode name (uppercase, stored in `key.json`)
- `mask_matrix(image)` — virtual mask matrix; effective LSB = `LSB XOR mask`

Reusable mask utilities live in `hmf/plugin/masks.py` (`rgb_mask_matrix`, `local_variance_mask`, etc.).

### 2. Register in the registry

Import the plugin and add an instance to `_BUILTIN_PLUGINS` in `hmf/plugin/__init__.py`:

```python
from hmf.plugin.my_method import MyMethodPlugin

_BUILTIN_PLUGINS: tuple[VirtualEmbeddingPlugin, ...] = (
    DirectLsbPlugin(),
    # ...
    AdaptiveDualPlugin(),
    MyMethodPlugin(),
)
```

You do not need to change `keys.py`, `virtual.py`, or `embed.py` — they all read from the registry.

### 3. (Optional) Mode alias

If a mode has an alternative name, register an alias:

```python
register_mode_alias("MY_ALIAS", "MY_METHOD")
```

### Adaptive plugins

For methods like key 5 that choose a virtual view per block-channel, subclass `AdaptiveVirtualPlugin` and override `candidate_modes` (priority order of candidate modes):

```python
from hmf.plugin.base import AdaptiveVirtualPlugin

class MyAdaptivePlugin(AdaptiveVirtualPlugin):
    key = 7
    label = "My adaptive method"
    mode = "MY_ADAPTIVE"

    @property
    def candidate_modes(self) -> tuple[str, ...]:
        return ("MY_METHOD", "RGB_RELATION", "LSB_ONLY")
```

See `hmf/plugin/adaptive_dual.py` for a reference implementation.

### Verification

```bash
python3 -c "from hmf.plugin import sorted_plugins; print([(p.key, p.mode) for p in sorted_plugins()])"
python3 run_embed.py --virtual-key 6 --seed 42 --matrix-r 4 --no-parallel
```

## Multiprocessing notes

Parallel search uses threads (not child processes) to avoid nested `spawn` issues. BLAS threads are limited to 1 via environment variables.

Each batch cell runs in an **isolated subprocess**. If a cell crashes (segfault/OOM), the batch runner continues and writes `0` for that cell's metrics. Use `--serial` on low-memory machines; use `--workers 2` at most for parallel cells.

On persistent failure, numeric metrics are written as `0` in the CSV (`status` column records the error).
