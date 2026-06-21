"""Plugin virtual embedding — tambah metode baru sebagai modul di folder ini."""

from hmf.plugin.adaptive_dual import AdaptiveDualPlugin
from hmf.plugin.base import AdaptiveVirtualPlugin, VirtualEmbeddingPlugin
from hmf.plugin.block_mean import BlockMeanPlugin
from hmf.plugin.direct_lsb import DirectLsbPlugin
from hmf.plugin.joint_margin_mean import JointMarginMeanPlugin
from hmf.plugin.registry import (
    adaptive_candidate_modes,
    get_plugin_by_key,
    get_plugin_by_mode,
    key_labels,
    key_to_mode,
    mask_matrices_for_modes,
    mask_matrix_for_mode,
    mode_to_key,
    normalize_mode,
    register_mode_alias,
    register_plugin,
    sorted_plugins,
    valid_modes,
)
from hmf.plugin.rgb_margin import RgbMarginPlugin
from hmf.plugin.rgb_relation import RgbRelationPlugin

_BUILTIN_PLUGINS: tuple[VirtualEmbeddingPlugin, ...] = (
    DirectLsbPlugin(),
    BlockMeanPlugin(),
    RgbMarginPlugin(),
    RgbRelationPlugin(),
    JointMarginMeanPlugin(),
    AdaptiveDualPlugin(),
)

for _plugin in _BUILTIN_PLUGINS:
    register_plugin(_plugin)

register_mode_alias("RGB_XOR_VARIANCE", "RGB_MARGIN_XOR_VARIANCE")

__all__ = [
    "AdaptiveDualPlugin",
    "AdaptiveVirtualPlugin",
    "BlockMeanPlugin",
    "DirectLsbPlugin",
    "JointMarginMeanPlugin",
    "RgbMarginPlugin",
    "RgbRelationPlugin",
    "VirtualEmbeddingPlugin",
    "adaptive_candidate_modes",
    "get_plugin_by_key",
    "get_plugin_by_mode",
    "key_labels",
    "key_to_mode",
    "mask_matrices_for_modes",
    "mask_matrix_for_mode",
    "mode_to_key",
    "normalize_mode",
    "register_mode_alias",
    "register_plugin",
    "sorted_plugins",
    "valid_modes",
]
