"""Registry plugin virtual embedding."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from hmf.plugin.base import AdaptiveVirtualPlugin, VirtualEmbeddingPlugin

_PLUGINS_BY_KEY: Dict[int, VirtualEmbeddingPlugin] = {}
_PLUGINS_BY_MODE: Dict[str, VirtualEmbeddingPlugin] = {}
_MODE_ALIASES: Dict[str, str] = {}


def register_plugin(plugin: VirtualEmbeddingPlugin) -> None:
    key = int(plugin.key)
    mode = str(plugin.mode).upper()
    if key in _PLUGINS_BY_KEY:
        raise ValueError(f"Key {key} sudah terdaftar: {_PLUGINS_BY_KEY[key].mode}")
    if mode in _PLUGINS_BY_MODE:
        raise ValueError(f"Mode {mode} sudah terdaftar")
    _PLUGINS_BY_KEY[key] = plugin
    _PLUGINS_BY_MODE[mode] = plugin


def register_mode_alias(alias: str, target_mode: str) -> None:
    _MODE_ALIASES[str(alias).upper()] = str(target_mode).upper()


def normalize_mode(mode: str) -> str:
    mode = str(mode).upper()
    return _MODE_ALIASES.get(mode, mode)


def get_plugin_by_key(key: int) -> VirtualEmbeddingPlugin:
    if key not in _PLUGINS_BY_KEY:
        raise ValueError(f"Key tidak dikenal: {key}")
    return _PLUGINS_BY_KEY[key]


def get_plugin_by_mode(mode: str) -> VirtualEmbeddingPlugin:
    mode = normalize_mode(mode)
    if mode not in _PLUGINS_BY_MODE:
        names = ", ".join(f"{p.key}={p.mode}" for p in sorted_plugins())
        raise ValueError(f"Mode tidak dikenal: {mode}. Pilihan: {names}")
    return _PLUGINS_BY_MODE[mode]


def sorted_plugins() -> List[VirtualEmbeddingPlugin]:
    return [_PLUGINS_BY_KEY[k] for k in sorted(_PLUGINS_BY_KEY)]


def key_labels() -> Dict[int, str]:
    return {p.key: p.label for p in sorted_plugins()}


def key_to_mode() -> Dict[int, str]:
    return {p.key: p.mode for p in sorted_plugins()}


def mode_to_key() -> Dict[str, int]:
    return {p.mode: p.key for p in sorted_plugins()}


def valid_modes() -> Tuple[str, ...]:
    return tuple(sorted(_PLUGINS_BY_MODE)) + tuple(sorted(_MODE_ALIASES))


def adaptive_candidate_modes(mode: str) -> Tuple[str, ...]:
    plugin = get_plugin_by_mode(mode)
    if not isinstance(plugin, AdaptiveVirtualPlugin):
        raise ValueError(f"Mode bukan adaptif: {mode}")
    return plugin.candidate_modes


def mask_matrix_for_mode(image: np.ndarray, mode: str) -> np.ndarray:
    plugin = get_plugin_by_mode(mode)
    if plugin.is_adaptive:
        raise ValueError(f"{plugin.mode} memakai mask per blok, bukan satu matrix global")
    return plugin.mask_matrix(image)


def mask_matrices_for_modes(image: np.ndarray, modes: Sequence[str]) -> Dict[str, np.ndarray]:
    return {normalize_mode(m): mask_matrix_for_mode(image, m) for m in modes}
