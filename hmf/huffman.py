from __future__ import annotations

import heapq
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple


class _Node:
    __slots__ = ("sym", "freq", "left", "right")

    def __init__(
        self,
        *,
        sym: Optional[int] = None,
        freq: int = 0,
        left: Optional["_Node"] = None,
        right: Optional["_Node"] = None,
    ) -> None:
        self.sym = sym
        self.freq = int(freq)
        self.left = left
        self.right = right

    def min_sym(self) -> int:
        if self.sym is not None:
            return int(self.sym)
        left_sym = self.left.min_sym() if self.left else 256
        right_sym = self.right.min_sym() if self.right else 256
        return min(left_sym, right_sym)

    def __lt__(self, other: "_Node") -> bool:
        if self.freq != other.freq:
            return self.freq < other.freq
        return self.min_sym() < other.min_sym()


def pack_bits_to_bytes(bits: Sequence[int]) -> Tuple[bytes, int]:
    b = [int(x) & 1 for x in bits]
    n = len(b)
    out = bytearray()
    for i in range(0, n, 8):
        chunk = b[i : i + 8]
        while len(chunk) < 8:
            chunk.append(0)
        val = 0
        for j, bit in enumerate(chunk):
            val |= (bit & 1) << (7 - j)
        out.append(val)
    return bytes(out), n


def unpack_bytes_to_bits(data: bytes, n_bits: int) -> List[int]:
    out: List[int] = []
    for byte in data:
        for j in range(8):
            out.append((int(byte) >> (7 - j)) & 1)
    return out[:n_bits]


def _build_tree(freq: Counter) -> _Node:
    if not freq:
        raise ValueError("frekuensi kosong")
    if len(freq) == 1:
        sym, f = next(iter(freq.items()))
        return _Node(sym=int(sym) & 0xFF, freq=int(f))

    heap: List[_Node] = []
    for sym, f in sorted(freq.items(), key=lambda x: (x[0], x[1])):
        heapq.heappush(heap, _Node(sym=int(sym) & 0xFF, freq=int(f)))
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, _Node(freq=a.freq + b.freq, left=a, right=b))
    return heap[0]


def _codes(root: _Node) -> Dict[int, str]:
    out: Dict[int, str] = {}

    def walk(node: _Node, prefix: str) -> None:
        if node.left is None and node.right is None and node.sym is not None:
            out[int(node.sym) & 0xFF] = prefix if prefix else "0"
            return
        if node.left is not None:
            walk(node.left, prefix + "0")
        if node.right is not None:
            walk(node.right, prefix + "1")

    walk(root, "")
    return out


def huffman_encode_bytes(data: bytes, freq: Counter) -> List[int]:
    root = _build_tree(freq)
    table = _codes(root)
    bits: List[int] = []
    for byte in data:
        for ch in table[int(byte) & 0xFF]:
            bits.append(1 if ch == "1" else 0)
    return bits


def huffman_decode_bytes(bits: Sequence[int], freq: Counter, *, out_len: int) -> bytes:
    if out_len <= 0:
        return b""
    root = _build_tree(freq)
    if root.left is None and root.right is None and root.sym is not None:
        return bytes([int(root.sym) & 0xFF]) * out_len

    out = bytearray()
    node = root
    for bit in bits:
        b = int(bit) & 1
        if node.left is None and node.right is None:
            raise ValueError("bitstream tidak selaras dengan pohon Huffman")
        node = node.right if b else node.left
        if node.left is None and node.right is None and node.sym is not None:
            out.append(int(node.sym) & 0xFF)
            if len(out) >= out_len:
                break
            node = root
    if len(out) != out_len:
        raise ValueError(f"Huffman decode: dapat {len(out)} byte, diharapkan {out_len}")
    return bytes(out)


def maybe_huffman_compress_bits(payload_bits: Sequence[int]) -> Tuple[List[int], Dict[str, Any]]:
    raw = [int(b) & 1 for b in payload_bits]
    if not raw:
        return [], {"compression": "none", "original_bit_len": 0}

    packed, orig_bit_len = pack_bits_to_bytes(raw)
    freq: Counter = Counter(packed)
    if len(freq) <= 1:
        return raw, {"compression": "none", "original_bit_len": orig_bit_len}

    encoded = huffman_encode_bytes(packed, freq)
    if len(encoded) >= len(raw):
        return raw, {"compression": "none", "original_bit_len": orig_bit_len}

    return encoded, {
        "compression": "huffman",
        "original_bit_len": orig_bit_len,
        "huffman_freq": {str(k): int(v) for k, v in freq.items()},
        "packed_byte_len": len(packed),
    }


def huffman_expand_to_original_bits(embedded_bits: Sequence[int], meta: Dict[str, Any]) -> List[int]:
    if str(meta.get("compression", "none")) != "huffman":
        return [int(b) & 1 for b in embedded_bits]

    orig_len = int(meta["original_bit_len"])
    freq_map = meta.get("huffman_freq") or {}
    freq: Counter = Counter({int(k): int(v) for k, v in freq_map.items()})
    packed_len = (orig_len + 7) // 8
    decoded = huffman_decode_bytes(embedded_bits, freq, out_len=packed_len)
    return unpack_bytes_to_bits(decoded, orig_len)
