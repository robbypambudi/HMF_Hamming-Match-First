"""Hamming Matching First — steganografi LSB berbasis matrix embedding."""

from hmf.embed import count_required_flips, embed_hamming_matchfirst_lsb, extract_hamming
from hmf.hamming import carrier_vector, matrix_block_n, matrix_capacity
from hmf.huffman import huffman_expand_to_original_bits, maybe_huffman_compress_bits
from hmf.io import load_image, load_key, read_payload_bits, save_image, save_key, write_payload_bits
from hmf.keys import KEY_LABELS, KEY_TO_MODE, MODE_TO_KEY, resolve_mode
from hmf.plugin import VirtualEmbeddingPlugin, get_plugin_by_key, register_plugin, sorted_plugins
from hmf.metrics import psnr, psnr_from_flips, ssim
from hmf.search import choose_best_matrix_r, choose_best_seed_and_matrix_r, compare_virtual_modes
from hmf.types import EmbedStats

__all__ = [
    "EmbedStats",
    "KEY_LABELS",
    "KEY_TO_MODE",
    "MODE_TO_KEY",
    "carrier_vector",
    "choose_best_matrix_r",
    "choose_best_seed_and_matrix_r",
    "compare_virtual_modes",
    "count_required_flips",
    "embed_hamming_matchfirst_lsb",
    "extract_hamming",
    "huffman_expand_to_original_bits",
    "load_image",
    "load_key",
    "matrix_block_n",
    "matrix_capacity",
    "maybe_huffman_compress_bits",
    "psnr",
    "psnr_from_flips",
    "read_payload_bits",
    "VirtualEmbeddingPlugin",
    "get_plugin_by_key",
    "register_plugin",
    "resolve_mode",
    "sorted_plugins",
    "save_image",
    "save_key",
    "ssim",
    "write_payload_bits",
]

__version__ = "1.0.0"
