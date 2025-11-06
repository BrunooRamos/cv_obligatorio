"""
BILP: Brightness-Invariant Local Patterns for Person Re-Identification
"""

from .color import (
    extract_color_features,
    calibrate_color_ranges,
    rgb_to_log_chromaticity
)
from .texture import (
    extract_texture_features,
    build_gabor_bank
)
from .gating import (
    compute_gating_weight,
    compute_gating_weights_batch,
    optimize_gating_params,
    compute_per_stripe_gating
)
from .distance import (
    bilp_distance,
    compute_distance_matrix,
    compute_distance_matrix_fast,
    rank_gallery,
    compute_ap
)
from .utils import (
    normalize_l1,
    power_normalize,
    normalize_features,
    extract_bilp_descriptor,
    extract_bilp_batch,
    save_features,
    load_features
)

__all__ = [
    # Color
    'extract_color_features',
    'calibrate_color_ranges',
    'rgb_to_log_chromaticity',
    # Texture
    'extract_texture_features',
    'build_gabor_bank',
    # Gating
    'compute_gating_weight',
    'compute_gating_weights_batch',
    'optimize_gating_params',
    'compute_per_stripe_gating',
    # Distance
    'bilp_distance',
    'compute_distance_matrix',
    'compute_distance_matrix_fast',
    'rank_gallery',
    'compute_ap',
    # Utils
    'normalize_l1',
    'power_normalize',
    'normalize_features',
    'extract_bilp_descriptor',
    'extract_bilp_batch',
    'save_features',
    'load_features'
]
