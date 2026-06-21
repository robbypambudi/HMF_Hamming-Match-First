from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def read_payload_bits(path: Path) -> List[int]:
    text = path.read_text(encoding="utf-8")
    return [int(ch) for ch in text if ch in ("0", "1")]


def write_payload_bits(path: Path, bits: Sequence[int]) -> None:
    path.write_text("".join(str(int(b) & 1) for b in bits), encoding="utf-8")


def load_image(path: Path, mode: str = "L") -> np.ndarray:
    return np.array(Image.open(path).convert(mode), dtype=np.uint8)


def save_image(arr: np.ndarray, path: Path) -> None:
    arr = np.asarray(arr, dtype=np.uint8)
    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").save(path)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        Image.fromarray(arr, mode="RGB").save(path)
    else:
        raise ValueError("Gambar harus 2D (grayscale) atau 3D RGB")


def save_key(
    path: Path,
    *,
    seed: int,
    matrix_r: int,
    payload_len: int,
    shape: Tuple[int, ...],
    mode: str,
    embed_scope: str,
    virtual_mode: str = "RGB_RELATION",
    virtual_key: Optional[int] = None,
    virtual_choices: Optional[Sequence[int]] = None,
    compression: str = "none",
    huffman_freq: Optional[Dict[str, int]] = None,
    original_bit_len: Optional[int] = None,
) -> None:
    meta: Dict[str, Any] = {
        "version": "hmf_v1",
        "seed": int(seed),
        "matrix_r": int(matrix_r),
        "payload_len": int(payload_len),
        "shape": [int(x) for x in shape],
        "mode": str(mode),
        "embed_scope": str(embed_scope),
        "virtual_mode": str(virtual_mode),
        "compression": str(compression),
    }
    if virtual_key is not None:
        meta["virtual_key"] = int(virtual_key)
    if virtual_choices is not None:
        meta["virtual_choices"] = [int(x) for x in virtual_choices]
    if huffman_freq is not None:
        meta["huffman_freq"] = {str(k): int(v) for k, v in huffman_freq.items()}
    if original_bit_len is not None:
        meta["original_bit_len"] = int(original_bit_len)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_key(path: Path) -> Dict[str, Any]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    version = str(meta.get("version", ""))
    if version not in {"hmf_v1", "hamming_match_lsb_v1", "hamming_match_lsb_v2", "hamming_match_lsb_v3"}:
        raise ValueError(f"Versi key tidak didukung: {version}")

    meta.setdefault("embed_scope", "ALL_CHANNELS")
    meta.setdefault("virtual_mode", "RGB_RELATION")
    meta.setdefault("compression", "none")
    return meta
