"""Pemetaan key paper ke mode virtual embedding (via plugin registry)."""

from __future__ import annotations

from typing import Dict, Tuple

from hmf.plugin import (
    adaptive_candidate_modes,
    get_plugin_by_key,
    get_plugin_by_mode,
    key_labels,
    key_to_mode,
    mode_to_key,
    normalize_mode,
    valid_modes,
)

KEY_LABELS: Dict[int, str] = key_labels()
KEY_TO_MODE: Dict[int, str] = key_to_mode()
MODE_TO_KEY: Dict[str, int] = mode_to_key()

# Urutan kandidat di mode ADAPTIVE_DUAL (disimpan di key sebagai virtual_choices).
ADAPTIVE_CANDIDATES: Tuple[str, ...] = adaptive_candidate_modes("ADAPTIVE_DUAL")


def resolve_mode(key_or_mode: int | str) -> str:
    """Terima angka key atau nama mode string."""
    if isinstance(key_or_mode, int):
        return get_plugin_by_key(key_or_mode).mode
    mode = normalize_mode(str(key_or_mode))
    valid = set(valid_modes())
    if mode not in valid:
        names = ", ".join(f"{k}={v}" for k, v in KEY_TO_MODE.items())
        raise ValueError(f"Mode tidak dikenal: {key_or_mode}. Pilihan: {names}")
    return get_plugin_by_mode(mode).mode


def key_from_mode(mode: str) -> int:
    mode = normalize_mode(mode)
    return get_plugin_by_mode(mode).key
